# MERRA-2 M2T1NXAER downloader: count-based completion + URL-name saving
# - Stations run sequentially; per station: 2019_2020 then 2023_2024
# - Only .nc4/.nc data files are considered; README/metadata ignored
# - Save files using expected names from URL (FILENAME), never header names
# - Batch ends as soon as local_count >= expected_count (fast os.scandir)
# - Immediate abort on first error; up to MAX_PASSES_PER_BATCH for gaps
# - Requires .netrc/_netrc for NASA Earthdata Login

import os
import re
import sys
import time
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ===============================
# USER VARIABLES (edit TXT_DIRS)
# ===============================
STATIONS = ["DL009", "DL011", "DL022", "DL024", "DL029"]
PERIODS = ["2019_2020", "2023_2024"]

TXT_DIRS = {
    "2019_2020": r"MERRA-2/MERRA-2_data_raw/Delhi_2019_2020/Delhi_txt_2019_2020",
    "2023_2024": r"MERRA-2/MERRA-2_data_raw/Delhi_2023_2024/Delhi_txt_2023_2024"
}

MAX_WORKERS = 5
TIMEOUT = 10
MAX_RETRIES_PER_FILE = 3
MAX_PASSES_PER_BATCH = 3

# ===============================
# PATHS AND DISCOVERY
# ===============================
def txt_dir_for(period: str) -> Path:
    if period not in TXT_DIRS:
        raise FileNotFoundError(f"TXT_DIRS missing mapping for period: {period}")
    p = Path(TXT_DIRS[period])
    if not p.exists():
        raise FileNotFoundError(f"Linked-list directory not found: {p}")
    return p

def find_linked_list_file(station: str, period: str) -> Path:
    tdir = txt_dir_for(period)
    cands = list(tdir.glob(f"*_{station}_{period}.txt"))
    if not cands:
        raise FileNotFoundError(f"No linked-list file for {station} {period} in {tdir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def parse_station_period_from_txt(txt_file: Path) -> tuple[str, str, str]:
    name = txt_file.name
    m = re.search(r"_([A-Za-z0-9]+)_(\d{4}_\d{4})\.txt$", name)
    if not m:
        raise ValueError(f"Cannot parse station/period from list filename: {name}")
    station = m.group(1)
    period = m.group(2)
    return station, period, f"{station}_{period}"

def out_dir_from_txt(txt_file: Path) -> Path:
    station, period, suffix = parse_station_period_from_txt(txt_file)
    return txt_file.parent.parent / suffix

# ===============================
# URL AND NAME HELPERS
# ===============================
_fallback_counter = 0
_fallback_lock = threading.Lock()

def to_goldsmr(url: str) -> str:
    return url.replace("archive.gesdisc.eosdis.nasa.gov", "goldsmr4.gesdisc.eosdis.nasa.gov")

def get_filename_from_url(url: str) -> str:
    global _fallback_counter
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    filename_list = q.get("FILENAME") or q.get("filename")
    if filename_list:
        return filename_list[0].split("/")[-1]
    tail = Path(parsed.path).name
    if tail:
        return tail
    with _fallback_lock:
        _fallback_counter += 1
        return f"downloaded_file_{_fallback_counter}"

def is_data_name(name: str) -> bool:
    n = name.lower()
    return n.endswith(".nc4") or n.endswith(".nc")

def is_data_url(u: str) -> bool:
    return is_data_name(get_filename_from_url(u))

def dedupe_by_filename(urls: list[str]) -> list[str]:
    seen = set()
    out = []
    for u in urls:
        n = get_filename_from_url(u)
        if n in seen:
            continue
        seen.add(n)
        out.append(u)
    return out

def read_service_urls(txt_file: Path) -> list[str]:
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines and not lines[0].lower().startswith("http"):
        lines = lines[1:]
    urls = [to_goldsmr(u) for u in lines]
    urls = [u for u in urls if is_data_url(u)]
    urls = dedupe_by_filename(urls)
    return urls

# ===============================
# DOWNLOAD CORE
# ===============================
class BatchError(RuntimeError):
    pass

def get_direct_url(session: requests.Session, service_url: str, timeout: int) -> str:
    resp = session.get(service_url, allow_redirects=False, timeout=timeout)
    if resp.status_code in (301, 302, 303, 307, 308):
        loc = resp.headers.get("Location")
        if loc:
            return loc
    resp.raise_for_status()
    return service_url

def download_one(session: requests.Session, url: str, out_dir: Path, timeout: int, abort_evt: threading.Event):
    if abort_evt.is_set():
        return get_filename_from_url(url), "aborted"

    # Always use the expected filename from the URL (ensures exact match to list)
    expected_name = get_filename_from_url(url)
    out_path = out_dir / expected_name
    if out_path.exists():
        return expected_name, "skipped"

    last_exc = None
    for attempt in range(1, MAX_RETRIES_PER_FILE + 1):
        try:
            direct_url = get_direct_url(session, url, timeout=timeout)
            with session.get(direct_url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                # Ignore header filename; enforce URL-based expected_name for alignment
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if abort_evt.is_set():
                            return expected_name, "aborted"
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(out_path)
                return expected_name, "downloaded"
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES_PER_FILE:
                time.sleep(2 ** attempt)
            else:
                break
    return expected_name, f"error:{last_exc}"

def run_parallel_pass(urls: list[str], out_dir: Path, timeout: int, max_workers: int) -> tuple[dict, dict]:
    """
    Returns:
      status_by_name: {filename: status}
      reason_by_name: {filename: error_message_if_any}
    Aborts immediately on first error; dynamic one-line status.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    abort_evt = threading.Event()
    status_by_name, reason_by_name = {}, {}
    total = len(urls)
    done = downloaded = skipped = 0

    def print_status_line():
        msg = f"[{done}/{total}] downloaded={downloaded} skipped={skipped}"
        print("\r" + msg.ljust(80), end="", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download_one, session, url, out_dir, timeout, abort_evt): url for url in urls}
        try:
            for fut in as_completed(futs):
                url = futs[fut]
                try:
                    fname, status = fut.result()
                    status_by_name[fname] = status
                    done += 1
                    if status == "downloaded":
                        downloaded += 1
                    elif status == "skipped":
                        skipped += 1
                    print_status_line()

                    if status == "aborted":
                        continue
                    if status.startswith("error"):
                        print()
                        reason_by_name[fname] = status
                        abort_evt.set()
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise BatchError(f"HTTP/connection error for {fname}: {status}")
                    if status not in ("downloaded", "skipped"):
                        print()
                        reason_by_name[fname] = f"Unexpected status: {status}"
                        abort_evt.set()
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise BatchError(f"Unexpected status for {fname}: {status}")
                except Exception as e:
                    print()
                    fname = get_filename_from_url(url)
                    status_by_name[fname] = "error"
                    reason_by_name[fname] = f"{e}"
                    abort_evt.set()
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
        finally:
            session.close()
    print()
    return status_by_name, reason_by_name

# ===============================
# FAST COUNT + MISSING-ONLY LOGIC
# ===============================
def count_local_nc(out_dir: Path) -> int:
    # Fast count of .nc/.nc4 using os.scandir
    cnt = 0
    if not out_dir.exists():
        return 0
    with os.scandir(out_dir) as it:
        for entry in it:
            if entry.is_file() and is_data_name(entry.name):
                cnt += 1
    return cnt

def build_expected_map(urls: list[str]) -> dict:
    # filename -> url for .nc/.nc4
    return {get_filename_from_url(u): u for u in urls}

# ===============================
# BATCH ORCHESTRATION
# ===============================
def batch_download(station_hint: str, period_hint: str) -> None:
    print(f"\n=== Station {station_hint} | Period {period_hint} ===")
    txt_file = find_linked_list_file(station_hint, period_hint)
    out_dir = out_dir_from_txt(txt_file)

    # Expected (data-only, deduped)
    all_urls = read_service_urls(txt_file)
    if not all_urls:
        print("No data URLs (.nc/.nc4) in list; nothing to do.")
        return
    name_to_url = build_expected_map(all_urls)
    expected_count = len(name_to_url)

    # Fast skip by count
    local_count = count_local_nc(out_dir)
    if local_count >= expected_count:
        print(f"Already complete by count: {out_dir} ({local_count}/{expected_count})")
        return

    # Compute missing names locally and queue only those
    existing_names = set()
    if out_dir.exists():
        with os.scandir(out_dir) as it:
            for entry in it:
                if entry.is_file() and is_data_name(entry.name):
                    existing_names.add(entry.name)
    missing_names = sorted(n for n in name_to_url.keys() if n not in existing_names)
    remaining = {n: name_to_url[n] for n in missing_names}

    attempts = 0
    while attempts < MAX_PASSES_PER_BATCH:
        attempts += 1
        print(f"Pass {attempts} on {len(remaining)} files -> {out_dir}")
        if not remaining:
            print(f"Batch complete: {out_dir} ({max(local_count, expected_count)}/{expected_count})")
            return

        urls_this_pass = [remaining[name] for name in sorted(remaining.keys())]
        status_by_name, reason_by_name = run_parallel_pass(urls_this_pass, out_dir, TIMEOUT, MAX_WORKERS)

        # Abort immediately on any error
        any_error = any(st.startswith("error") for st in status_by_name.values())
        if any_error:
            for name, st in status_by_name.items():
                if st.startswith("error"):
                    print(f"Error: {name} -> {reason_by_name.get(name, st)}")
                    break
            raise BatchError("Terminating due to errors in this batch")

        # Recount; stop as soon as counts satisfy
        local_count = count_local_nc(out_dir)
        missing_count = max(0, expected_count - local_count)
        downloaded_now = sum(1 for st in status_by_name.values() if st == "downloaded")
        skipped_now = sum(1 for st in status_by_name.values() if st == "skipped")
        print(f"Pass {attempts} done | downloaded={downloaded_now} skipped={skipped_now} missing={missing_count}")

        if local_count >= expected_count:
            print(f"Batch complete: {out_dir} ({expected_count}/{expected_count})")
            return

        # If still short, compute exact missing names for the next pass
        existing_names.clear()
        if out_dir.exists():
            with os.scandir(out_dir) as it:
                for entry in it:
                    if entry.is_file() and is_data_name(entry.name):
                        existing_names.add(entry.name)
        missing_names = sorted(n for n in name_to_url.keys() if n not in existing_names)
        remaining = {n: name_to_url[n] for n in missing_names}

    raise BatchError(f"Failed to complete after {MAX_PASSES_PER_BATCH} passes; missing={expected_count - local_count}")

# ===============================
# TOP-LEVEL ORCHESTRATION
# ===============================
def run_all():
    # Validate TXT_DIRS
    for period in PERIODS:
        _ = txt_dir_for(period)

    for station in STATIONS:
        try:
            try:
                batch_download(station, "2019_2020")
            except FileNotFoundError as e:
                print(f"Missing list for {station} 2019_2020: {e}")
            try:
                batch_download(station, "2023_2024")
            except FileNotFoundError as e:
                print(f"Missing list for {station} 2023_2024: {e}")
        except BatchError as e:
            print(f"\nFATAL: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_all()
#    first code ends here h

# # Singular download code: 
# # Robust upgrade to main code
# import os
# import re
# import sys
# import time
# import requests
# from urllib.parse import urlparse, parse_qs
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading

# USERNAME = "amitjoshi"  # For reference only, auth handled by .netrc
# OUTPUT_DIR = r"MERRA-2/MERRA-2_data_raw/Delhi_2019_2020/DL011_2019_2020"
# URLS_FILE = r"MERRA-2/MERRA-2_data_raw/Delhi_2019_2020/Delhi_txt_2019_2020/subset_M2T1NXAER_5.12.4_20251027_151717_DL011_2019_2020.txt"
# MAX_WORKERS = 5  # Number of parallel downloads
# TIMEOUT = 10  # seconds for HTTP request timeout
# MAX_RETRIES = 3  # Number of retries per file

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Thread-safe counters
# downloaded_count = 0
# skipped_count = 0
# error_count = 0
# count_lock = threading.Lock()

# # Thread-safe counter for fallback filenames
# fallback_counter = 0
# counter_lock = threading.Lock()

# def get_filename_from_headers(headers):
#     content_disp = headers.get('Content-Disposition')
#     if content_disp:
#         filename_match = re.findall('filename="?([^\'";]+)"?', content_disp)
#         if filename_match:
#             return filename_match[0]
#     return None

# def get_filename_from_url(url):
#     global fallback_counter
#     parsed = urlparse(url)
#     query_params = parse_qs(parsed.query)
#     filename_list = query_params.get("FILENAME") or query_params.get("filename")
#     if filename_list:
#         return filename_list[0].split("/")[-1]
#     path_name = parsed.path.split("/")[-1]
#     if path_name:
#         return path_name
#     with counter_lock:
#         fallback_counter += 1
#         return f"downloaded_file_{fallback_counter}"

# def get_direct_url(session, service_url):
#     resp = session.get(service_url, allow_redirects=False, timeout=TIMEOUT)
#     if resp.status_code in [301, 302, 303, 307, 308]:
#         redirected_url = resp.headers.get("Location")
#         if redirected_url:
#             return redirected_url
#     resp.raise_for_status()
#     return service_url

# def print_status():
#     with count_lock:
#         msg = f"Downloaded: {downloaded_count} | Skipped: {skipped_count} | Errors: {error_count}"
#     print(msg.ljust(60), end="\r", flush=True)

# def download_file(session, service_url, retries=MAX_RETRIES, timeout=TIMEOUT):
#     global downloaded_count, skipped_count, error_count
#     for attempt in range(1, retries + 1):
#         try:
#             direct_url = get_direct_url(session, service_url)
#             with session.get(direct_url, stream=True, timeout=timeout) as r:
#                 r.raise_for_status()
#                 file_name = get_filename_from_headers(r.headers)
#                 if not file_name:
#                     file_name = get_filename_from_url(direct_url)
#                 out_path = os.path.join(OUTPUT_DIR, file_name)
#                 if os.path.exists(out_path):
#                     with count_lock:
#                         skipped_count += 1
#                     return file_name, "skipped"
#                 with open(out_path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=8192):
#                         if chunk:
#                             f.write(chunk)
#                 with count_lock:
#                     downloaded_count += 1
#                 return file_name, "downloaded"
#         except Exception as e:
#             if attempt == retries:
#                 with count_lock:
#                     error_count += 1
#                 print(f"\nError downloading {service_url} after {attempt} attempts: {e}")
#                 return service_url, "error"
#             else:
#                 wait_time = 2 ** attempt  # exponential backoff: 2, 4, 8 seconds
#                 print(f"\nAttempt {attempt} failed for {service_url}: {e}. Retrying in {wait_time}s...")
#                 time.sleep(wait_time)

# def main():
#     with open(URLS_FILE, "r") as f:
#         service_urls_raw = [line.strip() for line in f.readlines()[1:] if line.strip()]
#     # Replace archive URL with goldsmr4 URL as before
#     service_urls = [url.replace("archive.gesdisc.eosdis.nasa.gov", "goldsmr4.gesdisc.eosdis.nasa.gov") for url in service_urls_raw]

#     session = requests.Session()
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(download_file, session, url): url for url in service_urls}
#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as exc:
#                 with count_lock:
#                     global error_count
#                     error_count += 1
#                 print(f"\nException during download: {exc}")
#             print_status()
#     print()  # Newline to finalize status line
#     print(f"Download complete. Total: {len(service_urls)} | Downloaded: {downloaded_count} | Skipped: {skipped_count} | Errors: {error_count}")

# if __name__ == "__main__":
#     main()



