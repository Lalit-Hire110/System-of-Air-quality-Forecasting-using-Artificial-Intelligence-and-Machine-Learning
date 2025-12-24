import os
import pandas as pd

# Base directory where state folders live
base_dir = r"C:\Users\Lalit Hire\OneDrive\Desktop\AEP_unified\MERRA-2_data_pre-processed\MERRA-2_data_processed"

# Output file
output_file = r"C:\Users\Lalit Hire\OneDrive\Desktop\AEP_unified\Final_MERRA2.csv"

# Station codes of interest
stations = {
    "Delhi": ["DL009", "DL011", "DL022", "DL024", "DL029"],
    "Haryana": ["HR001", "HR002", "HR003", "HR004", "HR009"],
    "Karnataka": ["KA001", "KA002", "KA007", "KA015", "KA018"],
    "Maharashtra": ["MH004", "MH007", "MH012", "MH026", "MH033"],
}

all_dataframes = []

# Loop over states and stations
for state, codes in stations.items():
    state_dir = os.path.join(base_dir, state)
    
    for code in codes:
        # Build prefix like DL009_2021_2022
        file_prefix = f"{code}_2021_2022"
        station_dir = os.path.join(state_dir, file_prefix)
        
        if not os.path.exists(station_dir):
            print(f"⚠️ Directory not found for {code} in {state}: {station_dir}")
            continue
        
        # Collect all CSV files from that folder
        for file in os.listdir(station_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(station_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    all_dataframes.append(df)
                    print(f"✔️ Loaded {file_path} with {len(df)} rows")
                except Exception as e:
                    print(f"❌ Failed to read {file_path}: {e}")

# Merge all together
if all_dataframes:
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"\n🎉 Merged file saved as: {output_file} ({len(merged_df)} rows)")
else:
    print("No dataframes found. Check paths and station list.")