import pandas as pd
import os

# -----------------------------
# CONFIG
# -----------------------------
MERRA2_FILE = r"C:\Users\Lalit Hire\OneDrive\Desktop\AEP_unified\Final_MERRA2.csv"
TIMEZONE = 'UTC'  # all MERRA2 timestamps are UTC

# -----------------------------
# LOAD FILE
# -----------------------------
print("Loading MERRA2 file...")
df = pd.read_csv(MERRA2_FILE)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Unique stations: {df['station_code'].nunique()}")

# -----------------------------
# TIMESTAMP PARSING
# -----------------------------
df['timestamp_utc'] = pd.to_datetime(df['time_utc'], errors='coerce', utc=True)
nat_count = df['timestamp_utc'].isna().sum()
print(f"NaT count in timestamp: {nat_count}")

# Show some problematic timestamps
if nat_count > 0:
    print("Sample problematic time_utc values:")
    print(df.loc[df['timestamp_utc'].isna(), 'time_utc'].unique()[:10])

# -----------------------------
# MINUTE DISTRIBUTION
# -----------------------------
df['minute'] = df['timestamp_utc'].dt.minute
minute_counts = df['minute'].value_counts()
print("\nTimestamp distribution by minute:")
print(minute_counts)

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
dup_count = df.duplicated(subset=['station_code', 'timestamp_utc']).sum()
print(f"\nDuplicate rows (station + timestamp): {dup_count}")

df = df.drop_duplicates(subset=['station_code', 'timestamp_utc'])

# -----------------------------
# HANDLE HALF-HOURLY -> HOURLY
# -----------------------------
# If model expects hourly data, floor 30-min timestamps to the hour
df['timestamp_hour'] = df['timestamp_utc'].dt.floor('h')

# -----------------------------
# DIAGNOSE MISSING HOURS PER STATION
# -----------------------------
stations = df['station_code'].unique()
for s in stations:
    df_s = df[df['station_code'] == s]
    start = df_s['timestamp_hour'].min()
    end = df_s['timestamp_hour'].max()
    full_hours = pd.date_range(start, end, freq='h', tz=TIMEZONE)
    missing_hours = full_hours.difference(df_s['timestamp_hour'])
    print(f"\nStation {s}:")
    print(f"Rows: {len(df_s)}, Unique hours: {df_s['timestamp_hour'].nunique()}")
    print(f"Expected total hours: {len(full_hours)}, Missing hours: {len(missing_hours)}")
    if len(missing_hours) > 0:
        print("Sample missing hours:", missing_hours[:10])

# -----------------------------
# REMOVE ANY REMAINING NaTs
# -----------------------------
df_clean = df.dropna(subset=['timestamp_utc'])

# -----------------------------
# SAVE CLEANED FILE
# -----------------------------
output_file = os.path.splitext(MERRA2_FILE)[0] + "_cleaned.csv"
df_clean.to_csv(output_file, index=False)
print(f"\nCleaned MERRA2 file saved to: {output_file}")

# -----------------------------
# SUMMARY
# -----------------------------
print("\n✅ Diagnosis & cleanup complete.")
print(f"Original rows: {len(df)}, Cleaned rows: {len(df_clean)}")
print("Duplicates removed, NaTs dropped, timestamps floored to hours.")
