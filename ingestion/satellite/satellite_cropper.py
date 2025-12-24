"""
Satellite image cropper for AEP 3.0 pipeline.

This module handles cropping of INSAT satellite images for each monitoring station
based on their bounding box coordinates.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from tqdm import tqdm

# Import utility functions
from .utils import parse_insat_filename, convert_ist_to_utc


class SatelliteCropper:
    """
    Cropper for satellite images based on station locations.
    
    This class handles:
    - Finding satellite images for each station
    - Cropping images to station bounding boxes
    - Organizing cropped images by station and date
    """
    
    def __init__(self, buffer_km: float = 10.0):
        """
        Initialize the satellite cropper.
        
        Args:
            buffer_km: Buffer distance in kilometers for cropping
        """
        self.buffer_km = buffer_km
        self.logger = logging.getLogger(__name__)
    
    def crop_station_images(self, station_metadata: Dict[str, Any], 
                           raw_images_dir: Path, output_dir: Path) -> List[str]:
        """
        Crop satellite images for a specific station.
        
        Args:
            station_metadata: Station metadata dictionary
            raw_images_dir: Directory containing raw satellite images
            output_dir: Directory to save cropped images
            
        Returns:
            List of paths to cropped images
        """
        station_name = station_metadata['station_name']
        self.logger.info(f"Cropping images for station: {station_name}")
        
        # Get station bounds with buffer
        bounds = self._get_station_bounds(station_metadata)
        
        # Create output directory
        station_output_dir = output_dir / station_name
        station_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all satellite images
        satellite_images = self._find_satellite_images(raw_images_dir)
        
        if not satellite_images:
            self.logger.warning(f"No satellite images found in {raw_images_dir}")
            return []
        
        # Crop images with progress bar
        cropped_images = []
        from tqdm import tqdm
        
        for image_path in tqdm(satellite_images, desc=f"Cropping {station_name}", unit="image"):
            # Check if cropped image already exists
            cropped_filename = f"{image_path.stem}_cropped_{station_name}{image_path.suffix}"
            cropped_path = station_output_dir / cropped_filename
            
            if cropped_path.exists():
                # Skip if already cropped
                cropped_images.append(str(cropped_path))
                continue
            
            # Crop the image
            cropped_path = self._crop_single_image(image_path, bounds, station_output_dir)
            if cropped_path:
                cropped_images.append(cropped_path)
        
        self.logger.info(f"Successfully cropped {len(cropped_images)} images for {station_name}")
        return cropped_images
    
    def _get_station_bounds(self, station_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Get station bounding box with buffer.
        
        Args:
            station_info: Station metadata dictionary
            
        Returns:
            Dictionary containing bounding box coordinates
        """
        import math
        
        # Get original bounding box
        bbox = station_info['bounding_box']
        
        # Convert buffer from km to degrees (approximate)
        lat = station_info['latitude']
        buffer_deg_lat = self.buffer_km / 111.0
        buffer_deg_lon = self.buffer_km / (111.0 * math.cos(math.radians(lat)))
        
        # Apply buffer
        bounds = {
            'west_lon': bbox['west_lon'] - buffer_deg_lon,
            'east_lon': bbox['east_lon'] + buffer_deg_lon,
            'south_lat': bbox['south_lat'] - buffer_deg_lat,
            'north_lat': bbox['north_lat'] + buffer_deg_lat
        }
        
        # Ensure bounds are within valid ranges
        bounds['south_lat'] = max(-90, bounds['south_lat'])
        bounds['north_lat'] = min(90, bounds['north_lat'])
        bounds['west_lon'] = max(-180, bounds['west_lon'])
        bounds['east_lon'] = min(180, bounds['east_lon'])
        
        return bounds
    
    def _find_satellite_images(self, raw_data_dir: Path) -> List[Path]:
        """
        Find all satellite images in the raw data directory.
        
        Args:
            raw_data_dir: Directory containing raw satellite images
            
        Returns:
            List of paths to satellite image files
        """
        image_extensions = ['*.tif', '*.tiff', '*.img']
        satellite_images = []
        
        for ext in image_extensions:
            pattern = str(raw_data_dir / '**' / ext)
            images = glob.glob(pattern, recursive=True)
            satellite_images.extend([Path(img) for img in images])
        
        # Sort by filename for consistent processing
        satellite_images.sort()
        
        return satellite_images
    
    def _crop_single_image(self, image_path: Path, bounds: Dict[str, float],
                          output_dir: Path) -> Optional[str]:
        """
        Crop a single satellite image to station bounds.
        
        Args:
            image_path: Path to the satellite image
            bounds: Bounding box coordinates
            output_dir: Directory to save cropped image
            
        Returns:
            Path to cropped image if successful, None otherwise
        """
        try:
            import rasterio
            import geopandas as gpd
            from shapely.geometry import box
            from rasterio.mask import mask
            from rasterio.errors import RasterioIOError
            
            with rasterio.open(image_path) as src:
                # Get image bounds to check overlap
                img_bounds = src.bounds
                
                # Check if station bounds overlap with image bounds
                station_bbox = box(bounds['west_lon'], bounds['south_lat'],
                                 bounds['east_lon'], bounds['north_lat'])
                image_bbox = box(img_bounds.left, img_bounds.bottom, 
                               img_bounds.right, img_bounds.top)
                
                # Create GeoDataFrame for station bounds
                station_gdf = gpd.GeoDataFrame([1], geometry=[station_bbox], crs='EPSG:4326')
                image_gdf = gpd.GeoDataFrame([1], geometry=[image_bbox], crs=src.crs)
                
                # Reproject station bounds to image CRS for overlap check
                if src.crs != station_gdf.crs:
                    station_gdf_reproj = station_gdf.to_crs(src.crs)
                else:
                    station_gdf_reproj = station_gdf
                
                # Check for overlap between station and image bounds
                if not station_gdf_reproj.geometry.iloc[0].intersects(image_gdf.geometry.iloc[0]):
                    self.logger.debug(f"Station bounds do not overlap with image {image_path.name}. Skipping.")
                    return None
                
                # Crop the image using the reprojected geometry
                try:
                    cropped_image, cropped_transform = mask(src, station_gdf_reproj.geometry, crop=True)
                    
                    # Check if cropped image has valid data
                    if cropped_image.size == 0 or (hasattr(src, 'nodata') and src.nodata is not None and 
                                                  cropped_image.min() == src.nodata and cropped_image.max() == src.nodata):
                        self.logger.debug(f"Cropped image from {image_path.name} contains no valid data. Skipping.")
                        return None
                    
                except Exception as crop_error:
                    # Handle specific cropping errors gracefully
                    if "Input shapes do not overlap raster" in str(crop_error):
                        self.logger.debug(f"No overlap between station bounds and raster {image_path.name}. Skipping.")
                        return None
                    else:
                        raise crop_error
                
                # Create output filename with station name
                station_name = output_dir.name
                cropped_filename = f"{image_path.stem}_cropped_{station_name}{image_path.suffix}"
                output_path = output_dir / cropped_filename
                
                # Save cropped image
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=cropped_image.shape[1],
                    width=cropped_image.shape[2],
                    count=cropped_image.shape[0],
                    dtype=cropped_image.dtype,
                    crs=src.crs,
                    transform=cropped_transform
                ) as dst:
                    dst.write(cropped_image)
                
                return str(output_path)
                
        except ImportError:
            self.logger.warning("rasterio or geopandas not available. Skipping cropping.")
            return None
        except Exception as e:
            self.logger.error(f"Error cropping {image_path}: {e}")
            return None
    
    def _create_cropped_filename(self, original_path: Path, bounds: Dict[str, float]) -> str:
        """
        Create filename for cropped image.
        
        Args:
            original_path: Path to original image
            bounds: Bounding box coordinates
            
        Returns:
            Filename for cropped image
        """
        # Parse original filename
        filename_info = parse_insat_filename(original_path.name)
        
        if filename_info:
            # Create new filename with bounds information
            base_name = original_path.stem
            bounds_str = f"w{bounds['west_lon']:.3f}_e{bounds['east_lon']:.3f}_s{bounds['south_lat']:.3f}_n{bounds['north_lat']:.3f}"
            return f"{base_name}_cropped_{bounds_str}.tif"
        else:
            # Fallback filename
            return f"cropped_{original_path.name}"
    
    def get_image_timestamps(self, cropped_images: List[str]) -> Dict[str, datetime]:
        """
        Extract timestamps from cropped image filenames.
        
        Args:
            cropped_images: List of paths to cropped images
            
        Returns:
            Dictionary mapping image paths to timestamps
        """
        timestamps = {}
        
        for image_path in cropped_images:
            filename = Path(image_path).name
            filename_info = parse_insat_filename(filename)
            
            if filename_info and 'datetime' in filename_info:
                timestamps[image_path] = filename_info['datetime']
        
        return timestamps
    
    def validate_cropped_images(self, cropped_images: List[str]) -> Dict[str, Any]:
        """
        Validate cropped images and return statistics.
        
        Args:
            cropped_images: List of paths to cropped images
            
        Returns:
            Dictionary containing validation statistics
        """
        stats = {
            'total_images': len(cropped_images),
            'valid_images': 0,
            'invalid_images': 0,
            'file_sizes': [],
            'timestamps': []
        }
        
        for image_path in cropped_images:
            try:
                import rasterio
                
                with rasterio.open(image_path) as src:
                    # Check if image has valid data
                    data = src.read()
                    if data.size > 0 and not np.all(data == src.nodata):
                        stats['valid_images'] += 1
                        stats['file_sizes'].append(os.path.getsize(image_path))
                        
                        # Extract timestamp
                        filename_info = parse_insat_filename(Path(image_path).name)
                        if filename_info and 'datetime' in filename_info:
                            stats['timestamps'].append(filename_info['datetime'])
                    else:
                        stats['invalid_images'] += 1
                        
            except Exception as e:
                stats['invalid_images'] += 1
                self.logger.debug(f"Error validating {image_path}: {e}")
        
        return stats
    
    def cleanup_invalid_images(self, cropped_images: List[str]) -> List[str]:
        """
        Remove invalid cropped images.
        
        Args:
            cropped_images: List of paths to cropped images
            
        Returns:
            List of valid cropped image paths
        """
        valid_images = []
        
        for image_path in cropped_images:
            try:
                import rasterio
                
                with rasterio.open(image_path) as src:
                    data = src.read()
                    if data.size > 0 and not np.all(data == src.nodata):
                        valid_images.append(image_path)
                    else:
                        # Remove invalid image
                        os.remove(image_path)
                        self.logger.debug(f"Removed invalid image: {image_path}")
                        
            except Exception as e:
                # Remove corrupted image
                try:
                    os.remove(image_path)
                    self.logger.debug(f"Removed corrupted image: {image_path}")
                except:
                    pass
        
        return valid_images 