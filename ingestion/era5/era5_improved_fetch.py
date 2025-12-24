#!/usr/bin/env python3
"""
Fetch ERA5 data for specified stations and years.
Uses cdsapi for authentication with ~/.cdsapirc

FIXED VERSION: 
- Properly handles ZIP files that contain multiple NetCDF files
- Enhanced time coordinate detection and merging
- Better error handling and retry mechanisms
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
import requests

# === USER CONFIG ===
TEST_MODE = False              # set to False for full run
SELECT_STATIONS = ['DL022']  # e.g. ['DL009','DL010'] or None to process all from config
SELECT_YEARS = [2020]           # e.g. [2021,2022] or None to use defaults (see below)                                                                                                                                              
TEST_STATION = None          # e.g. "DL009:2021" for quick single station-year test (enables TEST_MODE)
SKIP_EXISTING = True              # if True skip station-year when both .nc and .csv exist
MAX_WORKERS = 12               # number of parallel month downloads per station-year (tune for speed)
REQUEST_DELAY = 1.0           # per-worker delay (s) after a successful month request

# Retry configuration
MAX_RETRIES = 3               # Maximum number of retry attempts for failed requests
RETRY_BACKOFF = 2.0           # Exponential backoff multiplier
INITIAL_RETRY_DELAY = 5.0     # Initial delay before first retry (seconds)

# Path to raw folder to look for existing monthly files.
# Default: relative data/raw/era5 in current working directory (portable).
# Override by setting ERA5_RAW_DIR environment variable to a full path.
RAW_DATA_DIR = Path(os.environ.get('ERA5_RAW_DIR', Path.cwd() / 'data' / 'raw' / 'era5')).expanduser().resolve()              
# === END USER CONFIG ===

# CONTENT-BASED validation instead of file size
MIN_EXPECTED_VARIABLES = 5     # At least 5 of our 9 variables should be present
MIN_EXPECTED_TIMESTEPS = 24    # At least 1 day of data (24 hours)

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


# === RETRY MECHANISM ===
def retry_with_backoff(func, max_retries=MAX_RETRIES, backoff_factor=RETRY_BACKOFF, initial_delay=INITIAL_RETRY_DELAY):
    """
    Retry decorator with exponential backoff for CDS API calls and network operations.
    """
    def wrapper(*args, **kwargs):
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if it's a CDS API error or network error
                error_str = str(e).lower()
                is_cds_error = any(keyword in error_str for keyword in [
                    'cds', 'server', 'http', 'request', 'connection', 'timeout', 'network'
                ])
                
                if is_cds_error or attempt < max_retries:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                     f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                else:
                    # For non-retryable errors, re-raise immediately
                    raise
        
        raise last_exception
    
    return wrapper


# === OPTIMIZED HELPERS ===
def initialize_download_log():
    if not os.path.exists(DOWNLOAD_LOG):
        with open(DOWNLOAD_LOG, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['station_id', 'year', 'month', 'variables', 'status', 'timestamp', 'error_message'])


def log_download_status(station_id, year, month, variables, status, error_message=""):
    with open(DOWNLOAD_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([station_id, year, month, '|'.join(variables), status, datetime.now().isoformat(), error_message])


def detect_file_type(filepath):
    """Quickly detect if file is NetCDF, ZIP, or corrupted using magic bytes"""
    if not os.path.exists(filepath):
        return 'missing'
    
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)  # Read more bytes for better detection
            
        # NetCDF magic numbers
        if header[:3] in [b'CDF', b'\x89HDF', b'\x0aHDF']:
            return 'valid_nc'
        # ZIP magic number
        elif header[:2] == b'PK':
            return 'zip_file'
        # GZIP magic number  
        elif header[:2] == b'\x1f\x8b':
            return 'gzip_file'
        # Check if it might be HTML/error response
        elif header[:4] == b'<!DO' or header[:4] == b'<htm' or header[:4] == b'<err':
            return 'html_error'
        elif b'error' in header.lower() or b'exception' in header.lower():
            return 'error_response'
        else:
            return 'unknown_corrupted'
            
    except Exception:
        return 'error'


def extract_and_merge_zip_files(zip_path):
    """
    Extract ZIP file containing multiple NetCDF files and merge them.
    Returns path to merged NetCDF file.
    """
    extract_to_dir = tempfile.mkdtemp(prefix="era5_zip_extract_")
    
    try:
        with ZipFile(zip_path, 'r') as z:
            # List all files in the ZIP
            file_list = z.namelist()
            logger.info(f"ZIP contains {len(file_list)} files: {file_list}")
            
            # Extract all files
            z.extractall(extract_to_dir)
            
            # Look for NetCDF files
            nc_files = []
            for root, _, files in os.walk(extract_to_dir):
                for file in files:
                    if file.lower().endswith('.nc'):
                        nc_files.append(os.path.join(root, file))
            
            if not nc_files:
                raise ValueError(f"No .nc files found in ZIP archive: {file_list}")
            
            logger.info(f"Found {len(nc_files)} NetCDF files in ZIP")
            
            # If there's only one file, return it directly
            if len(nc_files) == 1:
                return nc_files[0]
            
            # If there are multiple files, merge them
            datasets = []
            for nc_file in nc_files:
                try:
                    logger.info(f"Opening dataset: {os.path.basename(nc_file)}")
                    ds = xr.open_dataset(nc_file)
                    
                    # Log time information for debugging
                    time_coords = [coord for coord in ds.coords if 'time' in coord.lower()]
                    logger.info(f"  Time coordinates: {time_coords}")
                    for time_coord in time_coords:
                        logger.info(f"  {time_coord} shape: {ds[time_coord].shape}")
                    
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to open {nc_file}: {e}")
                    continue
            
            if not datasets:
                raise ValueError("No valid datasets could be opened from ZIP files")
            
            # Merge datasets with explicit handling of time coordinates
            logger.info(f"Merging {len(datasets)} datasets")
            
            # Find common time coordinate name
            time_coords = []
            for ds in datasets:
                for coord in ds.coords:
                    if 'time' in coord.lower():
                        time_coords.append(coord)
            
            if not time_coords:
                raise ValueError("No time coordinates found in datasets")
            
            # Use the most common time coordinate name
            time_coord = max(set(time_coords), key=time_coords.count)
            logger.info(f"Using time coordinate: {time_coord}")
            
            # Rename time coordinates to be consistent
            for i, ds in enumerate(datasets):
                for coord in ds.coords:
                    if 'time' in coord.lower() and coord != time_coord:
                        logger.info(f"Renaming {coord} to {time_coord} in dataset {i}")
                        ds = ds.rename({coord: time_coord})
                        datasets[i] = ds
            
            # Merge datasets
            merged_ds = xr.merge(datasets, compat='override', join='outer')
            
            # Ensure time coordinate is properly set
            if time_coord in merged_ds.coords:
                time_length = len(merged_ds[time_coord])
                logger.info(f"Merged dataset time dimension: {time_length} points")
                
                # If time dimension is empty, try to reconstruct it
                if time_length == 0:
                    logger.warning("Time dimension is empty, attempting to reconstruct...")
                    # Try to find time information from variables
                    for var_name, var_data in merged_ds.data_vars.items():
                        if 'time' in var_data.dims and len(var_data.dims) > 0:
                            time_dim = [dim for dim in var_data.dims if 'time' in dim.lower()]
                            if time_dim:
                                actual_time_length = var_data.shape[var_data.dims.index(time_dim[0])]
                                logger.info(f"Found time dimension in {var_name}: {actual_time_length}")
                                if actual_time_length > 0:
                                    # Reconstruct time coordinate
                                    year = int(os.path.basename(zip_path).split('_')[2])
                                    month = int(os.path.basename(zip_path).split('_')[3].split('.')[0])
                                    days_in_month = calendar.monthrange(year, month)[1]
                                    time_points = pd.date_range(
                                        start=f"{year}-{month:02d}-01", 
                                        end=f"{year}-{month:02d}-{days_in_month:02d} 23:00:00",
                                        freq='H'
                                    )
                                    if len(time_points) == actual_time_length:
                                        merged_ds = merged_ds.assign_coords({time_coord: time_points})
                                        logger.info(f"Reconstructed time coordinate with {len(time_points)} points")
                                        break
            
            # Save merged dataset to temporary file
            merged_path = os.path.join(extract_to_dir, "merged_data.nc")
            merged_ds.to_netcdf(merged_path)
            
            # Close all datasets
            for ds in datasets:
                ds.close()
            
            logger.info(f"Successfully merged {len(datasets)} datasets with time dimension: {len(merged_ds[time_coord]) if time_coord in merged_ds.coords else 'N/A'}")
            return merged_path
            
    except BadZipFile as e:
        shutil.rmtree(extract_to_dir, ignore_errors=True)
        raise ValueError(f"File is not a valid ZIP archive: {e}")
    except Exception as e:
        shutil.rmtree(extract_to_dir, ignore_errors=True)
        raise ValueError(f"Failed to extract and merge ZIP file: {e}")


def validate_file_content(filepath, expected_variables_count=9):
    """
    Validate file by checking actual content instead of just size.
    Returns (is_valid, variable_count, time_points, error_message)
    """
    if not os.path.exists(filepath):
        return False, 0, 0, "File does not exist"
    
    # Basic size check to avoid processing obviously corrupted files
    if os.path.getsize(filepath) < 1000:
        return False, 0, 0, "File too small (likely corrupted)"
    
    try:
        # First detect file type
        file_type = detect_file_type(filepath)
        logger.info(f"File type detected: {file_type}")
        
        # Handle ZIP files that have .nc extension
        if file_type == 'zip_file':
            try:
                # Extract and merge the ZIP files
                extracted_nc = extract_and_merge_zip_files(filepath)
                is_valid, var_count, time_points, error_msg = validate_file_content(extracted_nc, expected_variables_count)
                
                # Clean up extracted file and directory
                try:
                    temp_dir = os.path.dirname(extracted_nc)
                    if 'era5_zip_extract_' in temp_dir:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
                    
                return is_valid, var_count, time_points, error_msg
            except Exception as e:
                return False, 0, 0, f"ZIP extraction failed: {e}"
        
        # For regular NetCDF files, try to open with multiple engines
        ds = safe_open_dataset(filepath)
        
        # Count actual variables present
        variables_found = len(ds.data_vars)
        
        # Count time points - look for any time coordinate
        time_points = 0
        time_coords_found = []
        for coord in ds.coords:
            if 'time' in coord.lower():
                time_points = max(time_points, len(ds[coord]))
                time_coords_found.append(coord)
        
        # If no time coordinates found, check variable dimensions
        if time_points == 0:
            for var_name, var_data in ds.data_vars.items():
                for dim in var_data.dims:
                    if 'time' in dim.lower():
                        time_points = max(time_points, var_data.shape[var_data.dims.index(dim)])
        
        # Check for required coordinates
        has_lat = any(coord in ds.coords for coord in ['latitude', 'lat'])
        has_lon = any(coord in ds.coords for coord in ['longitude', 'lon'])
        
        ds.close()
        
        # Validation criteria - relaxed time points requirement for debugging
        is_valid = (variables_found >= MIN_EXPECTED_VARIABLES and 
                   has_lat and has_lon)
        
        logger.info(f"Content validation: {variables_found}/{expected_variables_count} variables, "
                   f"{time_points} time points, coords: lat={has_lat}, lon={has_lon}, time_coords={time_coords_found}")
        
        if not is_valid:
            error_msg = (f"Content validation failed: {variables_found} vars, {time_points} times, "
                        f"lat={has_lat}, lon={has_lon}, time_coords={time_coords_found}")
            return False, variables_found, time_points, error_msg
            
        return True, variables_found, time_points, ""
        
    except Exception as e:
        logger.error(f"Content validation error for {filepath}: {e}")
        return False, 0, 0, f"Failed to open file: {e}"


def handle_downloaded_file(filepath):
    """
    Immediately check what was downloaded and handle appropriately.
    Returns path to valid NetCDF file or None if failed.
    """
    if not os.path.exists(filepath):
        return None
        
    file_type = detect_file_type(filepath)
    size = os.path.getsize(filepath)
    
    logger.info(f"Downloaded file type: {file_type}, size: {size} bytes")
    
    # Check for error responses first
    if file_type in ['html_error', 'error_response']:
        logger.error(f"Downloaded file appears to be an error response: {file_type}")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1000 chars
                logger.error(f"Error response content: {content[:500]}...")
        except:
            pass
        try:
            os.remove(filepath)
        except:
            pass
        return None
    
    # Handle ZIP files (including those with .nc extension)
    if file_type == 'zip_file':
        logger.info(f"Downloaded file is a ZIP archive, extracting and merging...")
        try:
            # Extract and merge the ZIP files
            extracted_nc = extract_and_merge_zip_files(filepath)
            
            # Validate the extracted content
            is_valid, var_count, time_points, error_msg = validate_file_content(extracted_nc)
            if not is_valid:
                logger.warning(f"Extracted ZIP content invalid: {error_msg}")
                try:
                    temp_dir = os.path.dirname(extracted_nc)
                    if 'era5_zip_extract_' in temp_dir:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
                return None
            
            logger.info(f"ZIP extraction and merging successful: {var_count} variables, {time_points} time points")
            
            # Clean up the original ZIP file
            try:
                os.remove(filepath)
            except:
                pass
            
            return extracted_nc
            
        except Exception as e:
            logger.error(f"ZIP extraction failed: {e}")
            return None
            
    elif file_type == 'valid_nc':
        # Use content validation instead of size check
        is_valid, var_count, time_points, error_msg = validate_file_content(filepath)
        if is_valid:
            logger.info(f"NetCDF file validated: {var_count} variables, {time_points} time points")
            return filepath
        else:
            logger.warning(f"NetCDF file content invalid: {error_msg}")
            return None
            
    elif file_type == 'gzip_file':
        logger.info(f"Downloaded file is gzipped, decompressing...")
        try:
            decompressed_path = _decompress_gzip_to_temp(filepath)
            
            # Validate the decompressed content
            is_valid, var_count, time_points, error_msg = validate_file_content(decompressed_path)
            if not is_valid:
                logger.warning(f"Decompressed content invalid: {error_msg}")
                return None
                
            try:
                os.remove(filepath)
            except:
                pass
            return decompressed_path
            
        except Exception as e:
            logger.error(f"GZIP decompression failed: {e}")
            return None
            
    else:
        logger.error(f"Downloaded file is corrupted or unknown type: {file_type}")
        try:
            os.remove(filepath)
        except:
            pass
        return None


def is_file_good(filepath):
    """Check if file exists and passes content validation."""
    if not os.path.exists(filepath):
        return False
    
    # Use content-based validation instead of size
    is_valid, var_count, time_points, error_msg = validate_file_content(filepath)
    return is_valid


def cleanup_corrupted_files(station_id, year, month):
    """Remove any corrupted temporary files for a specific month"""
    candidates = month_temp_file_candidates(station_id, year, month)
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                file_type = detect_file_type(candidate)
                if file_type in ['unknown_corrupted', 'html_error', 'error_response']:
                    logger.info(f"Removing corrupted file: {candidate}")
                    os.remove(candidate)
            except Exception as e:
                logger.debug(f"Error checking file {candidate}: {e}")


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
            ds = xr.open_dataset(path, decode_times=True, engine=eng)
            
            # Log time information for debugging
            time_coords = [coord for coord in ds.coords if 'time' in coord.lower()]
            logger.info(f"Dataset time coordinates: {time_coords}")
            for time_coord in time_coords:
                logger.info(f"  {time_coord} shape: {ds[time_coord].shape}")
            
            return ds
        except Exception as e:
            logger.debug(f"Attempt with engine='{eng}' failed for {path}: {e}")
            last_exc = e
    try:
        ds = xr.open_dataset(path, decode_times=True)
        
        # Log time information for debugging
        time_coords = [coord for coord in ds.coords if 'time' in coord.lower()]
        logger.info(f"Dataset time coordinates (default engine): {time_coords}")
        for time_coord in time_coords:
            logger.info(f"  {time_coord} shape: {ds[time_coord].shape}")
            
        return ds
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
    return candidates


# === OPTIMIZED FETCH HELPERS ===
@retry_with_backoff
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
            
            # Download first, then process
            client.retrieve('reanalysis-era5-single-levels', {
                'product_type': 'reanalysis', 'format': 'netcdf', 'variable': [v],
                'year': str(year), 'month': f"{month:02d}", 'day': days,
                'time': times, 'area': area
            }).download(tmp)

            # Process after download completes
            processed_tmp = handle_downloaded_file(tmp)
            if not processed_tmp:
                logger.warning(f"Individual fetch produced invalid file {tmp}")
                log_download_status(station_id, year, month, [v], "failed", "invalid file")
                continue

            if not is_file_good(processed_tmp):
                logger.warning(f"Individual fetch produced invalid content {processed_tmp}")
                log_download_status(station_id, year, month, [v], "failed", "invalid content")
                continue

            ds_point = process_downloaded_file(processed_tmp, lat, lon)
            out.append(ds_point)
            log_download_status(station_id, year, month, [v], "success")
            try:
                if os.path.exists(processed_tmp):
                    os.remove(processed_tmp)
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


@retry_with_backoff
def fetch_month_data(client, station_id, year, month, variables, area, lat, lon, request_delay):
    """Fetch data for a single month with retry mechanism."""
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    if TEST_MODE:
        days = ['01', '02']
    times = [f"{h:02d}:00" for h in range(0, 24)]
    req = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': list(variables.values()),
        'year': str(year),
        'month': f"{month:02d}",
        'day': days,
        'time': times,
        'area': area
    }
    tmpf = str(Path(RAW_DATA_DIR) / f"temp_{station_id}_{year}_{month}.nc")
    Path(tmpf).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{station_id}-{year}-{month:02d}] submitting request")
    
    # Clean up any existing corrupted files first
    cleanup_corrupted_files(station_id, year, month)
    
    # Download first without blocking operations
    client.retrieve('reanalysis-era5-single-levels', req).download(tmpf)
    
    # Only AFTER download completes, check and process the file
    if not os.path.exists(tmpf):
        raise ValueError("Download failed - no file created")
    
    # Handle file type and processing after download
    processed_tmpf = handle_downloaded_file(tmpf)
    if not processed_tmpf:
        raise ValueError("Download failed - file content invalid")
    
    tmpf = processed_tmpf
    
    if not is_file_good(tmpf):
        raise ValueError("Downloaded file content validation failed")
    
    ds_point = process_downloaded_file(tmpf, lat, lon)
    per_file = month_permanent_file(station_id, year, month)
    try:
        ds_point.to_netcdf(per_file)
        logger.info(f"[{station_id}-{year}-{month:02d}] saved per-month file {per_file}")
    except Exception as e:
        logger.warning(f"[{station_id}-{year}-{month:02d}] failed to save per-month file: {e}")
    log_download_status(station_id, year, month, list(variables.values()), "success")
    try:
        if os.path.exists(tmpf):
            os.remove(tmpf)
    except Exception:
        pass
    time.sleep(request_delay)
    return (month, ds_point, None)


def fetch_era5_data(station, year, variables, max_workers=3, request_delay=1.0):
    """
    OPTIMIZED VERSION: Uses content-based validation instead of file size
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
                if os.path.exists(c) and is_file_good(c):
                    return c  # return first candidate path that passes content validation
            except Exception:
                continue
        return None

    def fetch_month(m):
        # Clean up any corrupted files before starting
        cleanup_corrupted_files(station_id, year, m)
        
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
        area = adjust_bounding_box(station)
        
        try:
            return fetch_month_data(client, station_id, year, m, variables, area, lat, lon, request_delay)
        except Exception as e:
            logger.error(f"[{station_id}-{year}-{m:02d}] all attempts failed: {e}")
            # Clean up any files that might have been created during failed attempt
            cleanup_corrupted_files(station_id, year, m)
            return (m, None, e)

    months_to_submit = []
    for m in months:
        per_file = month_permanent_file(station_id, year, m)
        if os.path.exists(per_file) and is_file_good(per_file):
            logger.info(f"[{station_id}-{year}-{m:02d}] skipping submission — per-month file exists and is valid.")
            results[m] = (None, None)
            continue
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

    # Original assembly logic preserved
    all_month_ds = []
    for m in months:
        ds_point, err = results.get(m, (None, None))
        per_file = month_permanent_file(station_id, year, m)
        if os.path.exists(per_file) and is_file_good(per_file):
            try:
                ds_loaded = safe_open_dataset(per_file)
                ds_loaded = ds_loaded.load()
                all_month_ds.append(ds_loaded)
                logger.info(f"[{station_id}-{year}-{m:02d}] loaded per-month file {per_file} for assembly.")
                continue
            except Exception as e:
                logger.warning(f"[{station_id}-{year}-{m:02d}] failed to open per-month file {per_file}: {e}")

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
            if os.path.exists(cand) and is_file_good(cand):
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
                if is_file_good(nc_out):
                    logger.info(f"Skipping {st.file_name} {y} — already downloaded and processed.")
                    continue

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