import xarray as xr
import glob
import os

# ✅ Correct Windows path handling
base_dir = r"C:\Users\Lalit Hire\OneDrive\Desktop\AEP_unified\data\raw\era5"

# Pattern for all monthly files
pattern = os.path.join(base_dir, "DL011_2021_*.nc")
monthly_files = sorted(glob.glob(pattern))

# Check if files exist
if not monthly_files:
    raise FileNotFoundError(f"No files found matching: {pattern}")

# Count timestamps + file sizes before merge
total_timestamps = 0
file_sizes = {}

for file in monthly_files:
    size_kb = os.path.getsize(file) / 1024  # size in KB
    file_sizes[file] = size_kb
    with xr.open_dataset(file) as ds:  # safer file handling
        if 'time' not in ds:
            raise KeyError(f"'valid_time' variable missing in {file}")
        total_timestamps += len(ds['time'])

# Print summary before merging
print(f"Found {len(monthly_files)} files for merging.\n")
print("File sizes (KB):")
for file, size in file_sizes.items():
    print(f"  {os.path.basename(file)}: {size:.1f} KB")

# Merge files safely
merged = xr.open_mfdataset(monthly_files, combine='by_coords')

# Count timestamps after merging
merged_timestamps = len(merged['time'])

print(f"\nTotal timestamps before merge: {total_timestamps}")
print(f"Timestamps after merge: {merged_timestamps}")

# Final verdict
if merged_timestamps < total_timestamps:
    print("⚠️ Warning: Some timestamps were lost during merging!")
else:
    print("✅ All timestamps preserved in the merged file.")
