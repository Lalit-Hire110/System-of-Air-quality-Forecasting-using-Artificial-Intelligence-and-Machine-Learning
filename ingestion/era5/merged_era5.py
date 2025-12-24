import pandas as pd
import os

# Path to folder containing all station CSV files
input_folder = r"Yearly_Merged_ERA5"   
output_file = r"C:\Users\Lalit Hire\OneDrive\Desktop\AEP_unified\Final_ERA5.csv"

# Collect all CSV files from the folder
all_files = []
for root, dirs, files in os.walk(input_folder):
    for f in files:
        if f.lower().endswith(".csv"):
            all_files.append(os.path.join(root, f))

# Merge data
merged_data = []
for file in all_files:
    try:
        df = pd.read_csv(file)

        # Get Station_ID from filename (everything before first "_")
        station_id = os.path.basename(file).split("_")[0]
        df.insert(0, "Station_ID", station_id)  # Add Station_ID as first column

        merged_data.append(df)
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# Combine into one dataframe
final_df = pd.concat(merged_data, ignore_index=True)

# Save to Excel
final_df.to_csv(output_file, index=False)
print(f"✅ Merged file saved as: {output_file}")

# Quick check
df_check = pd.read_csv(output_file)
print("Columns:", list(df_check.columns))
print("First 10 Station_ID values:")
print(df_check["Station_ID"].head(10))