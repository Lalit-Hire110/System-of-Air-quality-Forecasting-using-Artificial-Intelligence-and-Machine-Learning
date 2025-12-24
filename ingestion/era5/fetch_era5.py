#!/usr/bin/env python3
"""
Fetch ERA5 data for specified stations and years.
Uses cdsapi for authentication with ~/.cdsapirc

Configurable modes (edit variables in the USER CONFIG section):
- TEST_MODE: lightweight test (uses only first station and month days 01/02)
- SELECT_STATIONS: list of station file_name strings to restrict processing (None = all)
- SELECT_YEARS: list of years to process (None uses defaults (see below))
- TEST_STATION: "STATION:YEAR" quick test (enables TEST_MODE)
- SKIP_EXISTING: if True skip station-year when both .nc and .csv exist
- MAX_WORKERS / REQUEST_DELAY: tuning parallel month downloads

Important behavior change:
- Pre-check logic now treats the *presence* of temp_*.nc / temp_*.zip / station_year_month.nc
  in RAW_DATA_DIR as proof that the month was already fetched and will skip re-submission.
  It DOES NOT attempt to open the file at pre-check time (so no netCDF backend is required to decide skipping).
- The assembly step (combining months into annual dataset) will open per-month files and may require
  an xarray netCDF backend (netCDF4 / h5netcdf) if processing/extracting is necessary.

Update (this version):
- When candidate temp files exist, the month is submitted to a worker which will *attempt to process the local candidate immediately*.
  This avoids skipping the month and later re-downloading variables individually (double-download).
"""
import os
import logging
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
import time
import calendar
from zipfile import ZipFile, BadZipFile
import mimetypes
import csv
import tempfile
import shutil
import concurrent.futures
import gzip

# === USER CONFIG ===
TEST_MODE = False              # set to False for full run
SELECT_STATIONS = ['DL009']  # e.g. ['DL009','DL010'] or None to process all from config
SELECT_YEARS = [2023]           # e.g. [2021,2022] or None to use defaults (see below)                                                                                                                                              
TEST_STATION = None          # e.g. "DL009:2021" for quick single station-year test (enables TEST_MODE)
SKIP_EXISTING = True              # if True skip station-year when both .nc and .csv exist
MAX_WORKERS = 2               # number of parallel month downloads per station-year (tune for speed)
REQUEST_DELAY = 1.0           # per-worker delay (s) after a successful month request

# Path to raw folder to look for existing monthly files.
# Default: relative data/raw/era5 in current working directory (portable).
# Override by setting ERA5_RAW_DIR environment variable to a full path.
RAW_DATA_DIR = Path(os.environ.get('ERA5_RAW_DIR', Path.cwd() / 'data' / 'raw' / 'era5')).expanduser().resolve()              
# === END USER CONFIG ===

# Good-file size thresholds (125 KB - 135 KB)
GOOD_MIN_BYTES = 125 * 1024
GOOD_MAX_BYTES = 135 * 1024

DOWNLOAD_LOG = "download_log.csv"

ERA5_NAME_MAP = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "tp": "total_precipitation",
    "ssr": "surface_solar_radiation_downwards",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "blh": "boundary_layer_height",
    "sp": "surface_pressure",
    "skt": "skin_temperature"
}

# === LOGGING ===
Path('logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/era5.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('era5_fetch')


# === HELPERS ===
def initialize_download_log():
    if not os.path.exists(DOWNLOAD_LOG):
        with open(DOWNLOAD_LOG, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['station_id', 'year', 'month', 'variables', 'status', 'timestamp', 'error_message'])


def log_download_status(station_id, year, month, variables, status, error_message=""):
    with open(DOWNLOAD_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([station_id, year, month, '|'.join(variables), status, datetime.now().isoformat(), error_message])


def is_valid_nc(filepath):
    """Basic quick sanity checks for a NetCDF file (file exists & size)."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
        return False
    mime = mimetypes.guess_type(filepath)[0]
    # object-store sometimes returns application/octet-stream or None
    return mime in ['application/x-netcdf', 'application/netcdf', 'application/octet-stream', None]


def is_file_good(filepath):
    """Check if file exists and has at least GOOD_MIN_BYTES."""
    return os.path.exists(filepath) and os.path.getsize(filepath) >= GOOD_MIN_BYTES


def round_to_grid(v, grid=0.25):
    return round(float(v) / grid) * grid


def adjust_bounding_box(station, min_size=0.25):
    """Create a symmetric bounding box around station center, snapped to ERA5 grid (0.25 deg)."""
    lat, lon = float(station['latitude']), float(station['longitude'])
    half = min_size / 2.0
    north, south = lat + half, lat - half
    east, west = lon + half, lon - half
    north = min(north, 90.0); south = max(south, -90.0)
    east = min(east, 180.0); west = max(west, -180.0)
    north = round_to_grid(north); south = round_to_grid(south)
    east = round_to_grid(east); west = round_to_grid(west)
    area = [north, west, south, east]
    logger.debug(f"Bounding box for station {station.get('file_name', '')}: {area}")
    return area


def load_config():
    """Load variables mapping and stations CSV"""
    with open('config/variables.yaml') as f:
        raw = yaml.safe_load(f)
    raw_vars = raw.get('era5', {})
    variables = {k: ERA5_NAME_MAP.get(v, v) for k, v in raw_vars.items()}
    stations = pd.read_csv('config/stations.csv')
    logger.info(f"Loaded {len(variables)} variables and {len(stations)} stations from config.")
    return variables, stations


def find_coord_name(ds, target='lat'):
    """Find latitude/longitude coordinate name heuristically (looks at coords, variables, dims)."""
    target = target.lower()
    for name in list(ds.coords) + list(ds.data_vars) + list(ds.dims):
        if target in name.lower():
            return name
    if target == 'lat':
        return 'latitude' if 'latitude' in ds.coords or 'latitude' in ds.data_vars else 'lat'
    else:
        return 'longitude' if 'longitude' in ds.coords or 'longitude' in ds.data_vars else 'lon'


def safe_open_dataset(path):
    """Open dataset with recommended engine and return xarray Dataset (with fallbacks)."""
    engines_to_try = ['netcdf4', 'h5netcdf']
    last_exc = None
    for eng in engines_to_try:
        try:
            return xr.open_dataset(path, decode_times=True, engine=eng)
        except Exception as e:
            logger.debug(f"Attempt with engine='{eng}' failed for {path}: {e}")
            last_exc = e
    try:
        return xr.open_dataset(path, decode_times=True)
    except Exception as e2:
        logger.error(f"Failed opening dataset {path} with xarray default engine: {e2}")
        if last_exc:
            raise RuntimeError(f"Tried engines {engines_to_try} and default; last error: {e2}; earlier: {last_exc}")
        else:
            raise e2


def _extract_zip_and_find_openable(zip_path):
    """
    Extract all members from zip_path into a temp dir and try opening each extracted file
    with safe_open_dataset. Return the path to the first file that opens successfully.
    If none can be opened, raise.
    """
    td = tempfile.mkdtemp(prefix="era5_zip_")
    try:
        with ZipFile(zip_path, 'r') as z:
            z.extractall(td)
    except BadZipFile:
        shutil.rmtree(td, ignore_errors=True)
        raise

    candidates = []
    for root, _, files in os.walk(td):
        for fn in files:
            candidates.append(os.path.join(root, fn))
    candidates.sort(key=lambda p: (0 if p.lower().endswith('.nc') else 1, p))

    for p in candidates:
        try:
            ds_test = safe_open_dataset(p)
            ds_test.close()
            return p
        except Exception as e:
            logger.debug(f"Attempt to open extracted file {p} failed: {e}")
            continue

    shutil.rmtree(td, ignore_errors=True)
    raise ValueError(f"No inner .nc (or openable member) found in zip {zip_path}")


def extract_zip_get_nc(zip_path, extract_to_dir):
    """Extract zip to temporary dir and return path to first .nc found."""
    return _extract_zip_and_find_openable(zip_path)


def get_nc_from_zip(zip_path):
    """Extract and return path to first openable .nc inside zip."""
    return _extract_zip_and_find_openable(zip_path)


def select_nearest_point(ds, lat_target, lon_target):
    """
    Robustly select nearest grid point to lat_target, lon_target.
    Handles 1D coords and 2D lat/lon grids.
    Returns point dataset (time dimension preserved if present).
    """
    lat_name = find_coord_name(ds, 'lat')
    lon_name = find_coord_name(ds, 'lon')

    if lat_name in ds.coords and lon_name in ds.coords:
        lat_var = ds.coords[lat_name]
        lon_var = ds.coords[lon_name]

        if getattr(lat_var, 'ndim', 1) == 1 and getattr(lon_var, 'ndim', 1) == 1:
            if lat_name != 'latitude' or lon_name != 'longitude':
                rename_map = {}
                if lat_name != 'latitude': rename_map[lat_name] = 'latitude'
                if lon_name != 'longitude': rename_map[lon_name] = 'longitude'
                ds = ds.rename(rename_map)
                lat_name, lon_name = 'latitude', 'longitude'
            point = ds.sel({lat_name: lat_target, lon_name: lon_target}, method='nearest')
            return point

        else:
            try:
                lat_arr = ds[lat_name].values
                lon_arr = ds[lon_name].values
            except Exception:
                lat_arr = ds[lat_name].data
                lon_arr = ds[lon_name].data

            lat_flat = lat_arr.ravel()
            lon_flat = lon_arr.ravel()
            dist2 = (lat_flat - float(lat_target))**2 + (lon_flat - float(lon_target))**2
            idx = int(np.argmin(dist2))
            shape = lat_arr.shape
            unr = np.unravel_index(idx, shape)
            lat_dims = ds[lat_name].dims
            isel_dict = {}
            if len(lat_dims) == 2:
                isel_dict[lat_dims[0]] = unr[0]
                isel_dict[lat_dims[1]] = unr[1]
            else:
                isel_dict[lat_dims[0]] = unr[0]
            point = ds.isel(isel_dict)
            return point

    for cand_lat in ['latitude', 'lat', 'y']:
        for cand_lon in ['longitude', 'lon', 'x']:
            if cand_lat in ds.variables and cand_lon in ds.variables:
                lat_arr = ds[cand_lat].values
                lon_arr = ds[cand_lon].values
                if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                    ds = ds.assign_coords({cand_lat: ds[cand_lat], cand_lon: ds[cand_lon]})
                    return ds.sel({cand_lat: lat_target, cand_lon: lon_target}, method='nearest')
                else:
                    lat_flat = lat_arr.ravel()
                    lon_flat = lon_arr.ravel()
                    dist2 = (lat_flat - float(lat_target))**2 + (lon_flat - float(lon_target))**2
                    idx = int(np.argmin(dist2))
                    unr = np.unravel_index(idx, lat_arr.shape)
                    lat_dims = ds[cand_lat].dims
                    isel_dict = {lat_dims[0]: unr[0], lat_dims[1]: unr[1]} if len(lat_dims) == 2 else {lat_dims[0]: unr[0]}
                    return ds.isel(isel_dict)

    raise ValueError("Could not find latitude/longitude coordinates in dataset.")


def _is_gzip_file(path):
    """Check gzip magic bytes"""
    try:
        with open(path, 'rb') as f:
            sig = f.read(2)
        return sig == b'\x1f\x8b'
    except Exception:
        return False


def _decompress_gzip_to_temp(path):
    td = tempfile.mkdtemp(prefix="era5_gz_")
    out_path = os.path.join(td, Path(path).stem)
    try:
        with gzip.open(path, 'rb') as gz_in, open(out_path, 'wb') as out_f:
            shutil.copyfileobj(gz_in, out_f)
        return out_path
    except Exception:
        shutil.rmtree(td, ignore_errors=True)
        raise


def process_downloaded_file(file_path, lat, lon):
    """
    Handle .zip or .nc, open dataset, select nearest gridpoint to (lat, lon),
    normalize dimension names (valid_time -> time), convert units and load into memory.
    """
    tmp_extracted = None
    tmp_work_path = None
    open_path = None
    try:
        open_path = str(file_path)

        # If file appears to be gzip, decompress to temp and try that first
        if _is_gzip_file(open_path):
            try:
                tmp_gz = _decompress_gzip_to_temp(open_path)
                tmp_extracted = tmp_gz
                open_path = tmp_gz
                logger.debug(f"Decompressed gzip candidate {file_path} -> {tmp_gz}")
            except Exception as e:
                logger.debug(f"Gzip decompress failed for {file_path}: {e}")

        # Try opening as NetCDF (with engine fallbacks)
        try:
            ds = safe_open_dataset(open_path)
        except Exception as e_open:
            logger.debug(f"Opening {open_path} directly failed: {e_open} — will try ZIP/container extraction.")
            try:
                extracted_candidate = None
                try:
                    extracted_candidate = _extract_zip_and_find_openable(open_path)
                except Exception as e_zip:
                    logger.debug(f"_extract_zip_and_find_openable failed for {open_path}: {e_zip}")
                    extracted_candidate = None

                if extracted_candidate:
                    tmp_work_path = extracted_candidate
                    tmp_extracted = tmp_work_path
                    ds = safe_open_dataset(tmp_work_path)
                else:
                    raise RuntimeError(f"Unable to open {file_path} as NetCDF and no inner openable member found.")
            except Exception as e2:
                logger.error(f"Failed opening dataset {file_path} with open and container-extract attempt: {e2}")
                raise

        # Normalize time coord name if needed
        if 'valid_time' in ds.coords and 'time' not in ds.coords:
            try:
                ds = ds.rename({'valid_time': 'time'})
            except Exception:
                pass

        ds_point = select_nearest_point(ds, lat, lon)

        if 'valid_time' in ds_point.coords and 'time' not in ds_point.coords:
            try:
                ds_point = ds_point.rename({'valid_time': 'time'})
            except Exception:
                pass

        ds_point = ds_point.load()

        # Convert units where necessary
        for var in ['t2m', '2m_temperature', 'd2m', '2m_dewpoint_temperature']:
            if var in ds_point.data_vars:
                ds_point[var] = ds_point[var] - 273.15

        for v in list(ds_point.data_vars):
            if 'ssr' in v.lower() or 'solar' in v.lower():
                ds_point[v] = ds_point[v] / 3600.0

        u_candidates = [n for n in ds_point.data_vars if ('u10' in n.lower() or '10m_u_component' in n.lower())]
        v_candidates = [n for n in ds_point.data_vars if ('v10' in n.lower() or '10m_v_component' in n.lower())]
        if u_candidates and v_candidates:
            uvar = u_candidates[0]
            vvar = v_candidates[0]
            ds_point = ds_point.assign(wind_speed=np.sqrt(ds_point[uvar] ** 2 + ds_point[vvar] ** 2))
            ds_point = ds_point.assign(
                wind_dir=(np.degrees(np.arctan2(-ds_point[uvar], -ds_point[vvar])) + 360) % 360
            )

        # attach grid coords if present
        try:
            if 'latitude' in ds_point.coords:
                grid_lat = float(ds_point['latitude'].values) if ds_point['latitude'].ndim == 0 else float(
                    ds_point['latitude'].isel({'time': 0}) if 'time' in ds_point.dims else ds_point['latitude'].values.flat[0]
                )
                ds_point = ds_point.assign_coords(grid_latitude=grid_lat)
            if 'longitude' in ds_point.coords:
                grid_lon = float(ds_point['longitude'].values) if ds_point['longitude'].ndim == 0 else float(
                    ds_point['longitude'].isel({'time': 0}) if 'time' in ds_point.dims else ds_point['longitude'].values.flat[0]
                )
                ds_point = ds_point.assign_coords(grid_longitude=grid_lon)
        except Exception:
            pass

        ds.close()

        # Cleanup any temporary extracted file
        if tmp_extracted:
            try:
                ptmp = Path(tmp_extracted)
                parent = ptmp.parent
                if parent.name.startswith("era5_zip_") or parent.name.startswith("era5_gz_"):
                    shutil.rmtree(parent, ignore_errors=True)
                else:
                    try:
                        os.remove(tmp_extracted)
                    except Exception:
                        pass
            except Exception:
                pass

        return ds_point

    except Exception:
        if tmp_extracted:
            try:
                ptmp = Path(tmp_extracted)
                parent = ptmp.parent
                if parent.name.startswith("era5_zip_") or parent.name.startswith("era5_gz_"):
                    shutil.rmtree(parent, ignore_errors=True)
                else:
                    try:
                        os.remove(tmp_extracted)
                    except Exception:
                        pass
            except Exception:
                pass
        raise


# Derived helpers for dataframe
def relative_humidity_from_t_and_td(t_degC, td_degC):
    """Magnus approximation - inputs in °C, returns percent RH"""
    a = 17.625
    b = 243.04
    sat_vp_t = 6.112 * np.exp(a * t_degC / (b + t_degC))
    sat_vp_td = 6.112 * np.exp(a * td_degC / (b + td_degC))
    rh = 100.0 * (sat_vp_td / sat_vp_t)
    return rh.clip(0, 100)


def add_derived_features_df(df):
    """Adds time conversions, wind speed, wind dir, RH, lags and rolling windows."""
    if 'time' not in df.columns and 'valid_time' in df.columns:
        df = df.rename(columns={'valid_time': 'time'})
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is None:
        df['time_utc'] = df['time'].dt.tz_localize('UTC')
    else:
        df['time_utc'] = df['time']
    df['time_ist'] = df['time_utc'].dt.tz_convert('Asia/Kolkata')

    if 'u10' in df.columns and 'v10' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10'] ** 2 + df['v10'] ** 2)
        df['wind_dir'] = (np.degrees(np.arctan2(-df['u10'], -df['v10'])) + 360) % 360

    t_col = (
        't2m'
        if 't2m' in df.columns
        else ('2m_temperature' if '2m_temperature' in df.columns else None)
    )
    td_col = (
        'd2m'
        if 'd2m' in df.columns
        else ('2m_dewpoint_temperature' if '2m_dewpoint_temperature' in df.columns else None)
    )

    if t_col and td_col:
        df[t_col] = df[t_col].astype(float)
        df[td_col] = df[td_col].astype(float)
        df['rh'] = relative_humidity_from_t_and_td(df[t_col], df[td_col])

    for base in [t_col, td_col, 'wind_speed', 'ssr', 'tp']:
        if base and base in df.columns:
            df[f'{base}_lag1'] = df[base].shift(1)
            df[f'{base}_rolling3'] = df[base].rolling(3, min_periods=1).mean()
            df[f'{base}_rolling24'] = df[base].rolling(24, min_periods=1).mean()

    return df


# === LOG / RESUME HELPERS ===
def read_successful_months_from_log(station_id, year):
    """Return set of months (ints) that are recorded as 'success' for station-year in DOWNLOAD_LOG."""
    if not os.path.exists(DOWNLOAD_LOG):
        return set()
    try:
        df = pd.read_csv(DOWNLOAD_LOG, dtype={'station_id': str, 'year': str, 'month': str, 'status': str})
    except Exception:
        return set()
    df = df[(df['station_id'] == str(station_id)) & (df['year'] == str(year)) & (df['status'] == 'success')]
    months = set()
    for m in df['month'].values:
        try:
            months.add(int(m))
        except Exception:
            try:
                months.add(int(str(m).lstrip("0")))
            except Exception:
                pass
    return months


def month_permanent_file(station_id, year, month):
    """Final per-month file name that script writes (zero-padded month)."""
    return str(Path(RAW_DATA_DIR) / f"{station_id}_{year}_{month:02d}.nc")


def month_temp_file_candidates(station_id, year, month):
    """
    Return list of candidate temp filenames to check (the service sometimes saves
    as temp_station_year_1.nc or temp_station_year_01.nc or .zip).
    Only existence/size is checked in pre-check — NO xarray reading here.
    """
    candidates = []
    base = Path(RAW_DATA_DIR)
    candidates.append(str(base / f"temp_{station_id}_{year}_{month}.nc"))     # temp_..._1.nc style
    candidates.append(str(base / f"temp_{station_id}_{year}_{month:02d}.nc"))  # temp_..._01.nc style
    candidates.append(str(base / f"{station_id}_{year}_{month:02d}.nc"))      # final-per-file (if present)
    candidates.append(str(base / f"temp_{station_id}_{year}_{month}.zip"))    # zip variant
    candidates.append(str(base / f"temp_{station_id}_{year}_{month:02d}.zip"))
    # Uncomment below if needed for suffixes
    # for p in base.glob(f"temp_{station_id}_{year}_{month}*"):
    #     candidates.append(str(p))
    return candidates


# === FETCH HELPERS ===
def fetch_variables_individually(client, station, year, month, variables):
    """Try fetching each variable separately and then merge them."""
    out = []
    station_id = station['file_name']
    lat = station['latitude']
    lon = station['longitude']
    for v in variables.values():
        try:
            area = adjust_bounding_box(station)
            days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
            if TEST_MODE:
                days = ['01', '02']
            times = [f"{h:02d}:00" for h in range(0, 24)]
            tmp = str(Path(RAW_DATA_DIR) / f"individual_{station_id}_{year}_{month}_{v}.nc")
            Path(tmp).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Fetching variable {v} individually for {station_id} {year}-{month:02d}")
            client.retrieve('reanalysis-era5-single-levels', {
                'product_type': 'reanalysis', 'format': 'netcdf', 'variable': [v],
                'year': str(year), 'month': f"{month:02d}", 'day': days,
                'time': times, 'area': area
            }).download(tmp)

            # If server returned a zip, extract
            if tmp.endswith(".zip") or (os.path.exists(tmp) and not is_valid_nc(tmp) and not tmp.lower().endswith('.nc')):
                try:
                    nc = get_nc_from_zip(tmp)
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                    tmp = nc
                except Exception:
                    pass

            if not is_valid_nc(tmp):
                logger.warning(f"Individual fetch produced invalid file {tmp}")
                log_download_status(station_id, year, month, [v], "failed", "invalid file")
                continue

            ds_point = process_downloaded_file(tmp, lat, lon)
            out.append(ds_point)
            log_download_status(station_id, year, month, [v], "success")
            try:
                os.remove(tmp)
            except Exception:
                pass
            time.sleep(1)
        except Exception as e:
            logger.error(f"Individual variable {v} failed for {station_id} {year}-{month:02d}: {e}")
            log_download_status(station_id, year, month, [v], "failed", str(e))
            time.sleep(1)
    if not out:
        return None
    merged = xr.merge(out, compat='override')
    return merged


def fetch_era5_data(station, year, variables, max_workers=3, request_delay=1.0):
    """
    Main fetch for a station-year. Downloads each month in parallel (max_workers),
    and then concatenates the monthly point datasets.
    If a month download fails, fallback to fetch_variables_individually for that month.

    Resume-supporting behavior:
     - Pre-checks only look for file existence (temp_*.nc/.zip or station_YYYY_MM.nc) in RAW_DATA_DIR and skip submission if present.
     - During assembly we try to open per-month files; if only temp variants exist we will attempt to extract/process them then.
    """
    station_id = station['file_name']
    lat = station['latitude']
    lon = station['longitude']
    months = [1] if TEST_MODE else range(1, 13)
    results = {}

    # compute months that are already recorded successful (log)
    succeeded_months = read_successful_months_from_log(station_id, year)

    def candidate_file_exists(m):
        for c in month_temp_file_candidates(station_id, year, m):
            try:
                if os.path.exists(c):
                    size = os.path.getsize(c)
                    if size >= GOOD_MIN_BYTES:
                        return c  # return first candidate path found that's big enough
                    else:
                        logger.info(f"[{station_id}-{year}-{m:02d}] candidate {c} too small ({size} bytes) — will re-download")
            except Exception:
                continue
        return None

    def fetch_month(m):
        cpath = candidate_file_exists(m)
        per_file = month_permanent_file(station_id, year, m)
        if cpath:
            logger.info(f"[{station_id}-{year}-{m:02d}] found existing candidate {cpath} — attempting local processing.")
            try:
                ds_point = process_downloaded_file(cpath, lat, lon)
                try:
                    if not (os.path.exists(per_file) and os.path.getsize(per_file) > 0):
                        ds_point.to_netcdf(per_file)
                        logger.info(f"[{station_id}-{year}-{m:02d}] saved per-month file {per_file} from candidate {cpath}")
                except Exception as e:
                    logger.warning(f"[{station_id}-{year}-{m:02d}] failed saving per-month file from candidate: {e}")
                log_download_status(station_id, year, m, list(variables.values()), "success", "processed existing candidate at fetch")
                return (m, ds_point, None)
            except Exception as e:
                logger.warning(f"[{station_id}-{year}-{m:02d}] processing existing candidate {cpath} failed: {e} — will try remote download/fallback.")

        client = cdsapi.Client()
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                area = adjust_bounding_box(station)
                days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, m)[1] + 1)]
                if TEST_MODE:
                    days = ['01', '02']
                times = [f"{h:02d}:00" for h in range(0, 24)]
                req = {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': list(variables.values()),
                    'year': str(year),
                    'month': f"{m:02d}",
                    'day': days,
                    'time': times,
                    'area': area
                }
                tmpf = str(Path(RAW_DATA_DIR) / f"temp_{station_id}_{year}_{m}.nc")
                Path(tmpf).parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"[{station_id}-{year}-{m:02d}] submitting request (attempt {attempt + 1})")
                client.retrieve('reanalysis-era5-single-levels', req).download(tmpf)
                if tmpf.endswith(".zip") or (os.path.exists(tmpf) and not is_valid_nc(tmpf)):
                    try:
                        nc = get_nc_from_zip(tmpf)
                        try:
                            os.remove(tmpf)
                        except Exception:
                            pass
                        tmpf = nc
                    except Exception:
                        logger.debug(f"[{station_id}-{year}-{m:02d}] could not extract zip; continue to validation")
                if not is_file_good(tmpf):
                    raise ValueError(f"Downloaded file too small ({os.path.getsize(tmpf) if os.path.exists(tmpf) else 'missing'} bytes)")
                ds_point = process_downloaded_file(tmpf, lat, lon)
                per_file = month_permanent_file(station_id, year, m)
                try:
                    ds_point.to_netcdf(per_file)
                    logger.info(f"[{station_id}-{year}-{m:02d}] saved per-month file {per_file}")
                except Exception as e:
                    logger.warning(f"[{station_id}-{year}-{m:02d}] failed to save per-month file: {e}")
                log_download_status(station_id, year, m, list(variables.values()), "success")
                try:
                    os.remove(tmpf)
                except Exception:
                    pass
                time.sleep(request_delay)
                return (m, ds_point, None)
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"[{station_id}-{year}-{m:02d}] attempt {attempt + 1} failed: {e}, retrying...")
                    if os.path.exists(tmpf):
                        try:
                            os.remove(tmpf)
                        except Exception as e_remove:
                            logger.debug(f"Could not remove {tmpf}: {e_remove}")
                    time.sleep(5)
                    continue
                else:
                    raise

    months_to_submit = []
    for m in months:
        per_file = month_permanent_file(station_id, year, m)
        if os.path.exists(per_file):
            try:
                size = os.path.getsize(per_file)
                if GOOD_MIN_BYTES <= size <= GOOD_MAX_BYTES:
                    logger.info(f"[{station_id}-{year}-{m:02d}] skipping submission — per-month file exists and is good ({per_file}, {size} bytes).")
                    results[m] = (None, None)
                    continue
                else:
                    logger.info(f"[{station_id}-{year}-{m:02d}] per-month file {per_file} exists but size {size} bytes not in good range -> will reprocess/download.")
            except Exception:
                logger.info(f"[{station_id}-{year}-{m:02d}] per-month file {per_file} exists but couldn't stat -> will reprocess/download.")
        if m in succeeded_months:
            logger.info(f"[{station_id}-{year}-{m:02d}] skipping submission — marked success in log.")
            results[m] = (None, None)
            continue
        months_to_submit.append(m)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_month = {executor.submit(fetch_month, m): m for m in months_to_submit}
        for fut in concurrent.futures.as_completed(future_to_month):
            m = future_to_month[fut]
            try:
                mm, ds_point, err = fut.result()
                results[mm] = (ds_point, err)
            except Exception as e:
                logger.error(f"Unexpected thread error for {station_id} {year} month {m}: {e}")
                results[m] = (None, e)

    all_month_ds = []
    for m in months:
        ds_point, err = results.get(m, (None, None))
        per_file = month_permanent_file(station_id, year, m)
        if os.path.exists(per_file):
            try:
                size = os.path.getsize(per_file)
                if GOOD_MIN_BYTES <= size <= GOOD_MAX_BYTES:
                    try:
                        ds_loaded = safe_open_dataset(per_file)
                        ds_loaded = ds_loaded.load()
                        all_month_ds.append(ds_loaded)
                        logger.info(f"[{station_id}-{year}-{m:02d}] loaded per-month file {per_file} for assembly.")
                        continue
                    except Exception as e:
                        logger.warning(f"[{station_id}-{year}-{m:02d}] failed to open per-month file {per_file}: {e}")
                else:
                    logger.info(f"[{station_id}-{year}-{m:02d}] per-month file {per_file} size {size} bytes not in good range -> will try candidates/fallback.")
            except Exception:
                logger.info(f"[{station_id}-{year}-{m:02d}] couldn't stat per-month file {per_file} -> will try candidates/fallback.")

        if ds_point is not None:
            try:
                if not (os.path.exists(per_file) and os.path.getsize(per_file) > 0):
                    ds_point.to_netcdf(per_file)
                    logger.info(f"[{station_id}-{year}-{m:02d}] saved per-month file (from in-memory ds) {per_file}")
            except Exception as e:
                logger.warning(f"[{station_id}-{year}-{m:02d}] unable to save per-month file: {e}")
            all_month_ds.append(ds_point)
            continue

        candidates = month_temp_file_candidates(station_id, year, m)
        found_and_processed = False
        for cand in candidates:
            if os.path.exists(cand):
                try:
                    size = os.path.getsize(cand)
                except Exception:
                    size = 0
                if size >= GOOD_MIN_BYTES:
                    try:
                        logger.info(f"[{station_id}-{year}-{m:02d}] processing existing candidate {cand} for assembly.")
                        ds_tmp = process_downloaded_file(cand, lat, lon)
                        try:
                            ds_tmp.to_netcdf(per_file)
                            logger.info(f"[{station_id}-{year}-{m:02d}] saved per-month file {per_file} from candidate {cand}")
                        except Exception as e:
                            logger.warning(f"[{station_id}-{year}-{m:02d}] couldn't save per-month file after processing candidate: {e}")
                        all_month_ds.append(ds_tmp)
                        log_download_status(station_id, year, m, list(variables.values()), "success", "processed existing candidate at assembly")
                        found_and_processed = True
                        break
                    except Exception as e:
                        logger.warning(f"[{station_id}-{year}-{m:02d}] failed processing candidate {cand}: {e}")
                        continue
                else:
                    logger.debug(f"[{station_id}-{year}-{m:02d}] candidate {cand} size {size} bytes is too small — skipping")
        if found_and_processed:
            continue

        logger.info(f"[{station_id}-{year}-{m:02d}] invoking per-variable fallback")
        client_fb = cdsapi.Client()
        fallback = fetch_variables_individually(client_fb, station, year, m, variables)
        if fallback is not None:
            try:
                fallback.to_netcdf(per_file)
                logger.info(f"[{station_id}-{year}-{m:02d}] saved per-month file (fallback) {per_file}")
            except Exception as e:
                logger.warning(f"[{station_id}-{year}-{m:02d}] failed to save fallback per-month file: {e}")
            log_download_status(station_id, year, m, list(variables.values()), "success")
            all_month_ds.append(fallback)
            logger.info(f"Per-variable fallback succeeded for {station_id} {year}-{m:02d}")
        else:
            logger.error(f"All attempts failed for {station_id} {year}-{m:02d}")

    if not all_month_ds:
        return None

    try:
        combined = xr.concat(all_month_ds, dim='time')
    except Exception:
        combined = xr.merge(all_month_ds, compat='override')
    return combined


# === MAIN ===
def main():
    initialize_download_log()
    variables, stations = load_config()

    global SELECT_STATIONS, SELECT_YEARS, TEST_MODE, TEST_STATION
    if isinstance(SELECT_STATIONS, str):
        SELECT_STATIONS = [s.strip() for s in SELECT_STATIONS.split(',') if s.strip()]
    if SELECT_STATIONS:
        stations = stations[stations['file_name'].isin(SELECT_STATIONS)]
        if stations.empty:
            logger.error(f"No stations matched SELECT_STATIONS={SELECT_STATIONS}")
            return

    if SELECT_YEARS:
        years = SELECT_YEARS if isinstance(SELECT_YEARS, list) else [int(y) for y in SELECT_YEARS]
    else:
        years = [2021] if TEST_MODE else [2021, 2022]

    if TEST_STATION:
        try:
            st_name, st_year = TEST_STATION.split(':')
            st_year = int(st_year)
        except Exception:
            logger.error("TEST_STATION format must be STATION:YEAR (e.g. DL009:2021)")
            return
        stations = stations[stations['file_name'] == st_name]
        if stations.empty:
            logger.error(f"No station '{st_name}' found in config/stations.csv")
            return
        years = [st_year]
        TEST_MODE = True
        logger.info(f"TEST_STATION enabled: {st_name} {st_year}")

    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path('data/processed/era5').mkdir(parents=True, exist_ok=True)

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")

    for _, st in stations.iterrows():
        for y in years:
            nc_out = str(Path(RAW_DATA_DIR) / f"{st.file_name}_{y}.nc")
            csv_out = f"data/processed/era5/{st.file_name}_{y}.csv"

            if SKIP_EXISTING and os.path.exists(nc_out) and os.path.exists(csv_out):
                try:
                    size_nc = os.path.getsize(nc_out)
                    if GOOD_MIN_BYTES <= size_nc <= GOOD_MAX_BYTES:
                        logger.info(f"Skipping {st.file_name} {y} — already downloaded and processed (annual NC size {size_nc} bytes).")
                        continue
                    else:
                        logger.info(f"Not skipping {st.file_name} {y}: annual NC {nc_out} size {size_nc} bytes not in good range -> will reprocess/update.")
                except Exception:
                    logger.info(f"Not skipping {st.file_name} {y}: couldn't stat {nc_out} -> will reprocess/update.")

            try:
                logger.info(f"Starting fetch for station {st.file_name} year {y}")
                ds = fetch_era5_data(st, y, variables, max_workers=MAX_WORKERS, request_delay=REQUEST_DELAY)
                if ds is None:
                    logger.warning(f"No data for {st.file_name} {y}")
                    continue

                ds.to_netcdf(nc_out)
                logger.info(f"Saved NetCDF: {nc_out}")

                df = ds.to_dataframe().reset_index()
                df = add_derived_features_df(df)

                df.to_csv(csv_out, index=False)
                logger.info(f"Saved CSV: {csv_out}")
            except Exception as e:
                logger.error(f"Top-level error for {st.file_name} {y}: {e}")
                log_download_status(st.file_name, y, "all", list(variables.values()), "failed", str(e))
  

if __name__ == "__main__":
    main()      