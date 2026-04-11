import os # os lets the system navigate thorough files and reads them
import pandas as pd  # Pandas is used to load and analyse the CSV files

# Files paths to the squat training data csvs
squat_side_csv  = "../../data/squat/side.csv"
squat_front_csv = "../../data/squat/front.csv"

# Checks if both CSV files are created before doing anything
for csv_path in [squat_side_csv, squat_front_csv]:
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at: {csv_path}")
        print("Make sure squat_extraction has been run first")
        exit()

# Loads both CSVs into dataframes so that they can be analysed
side_df  = pd.read_csv(squat_side_csv)
front_df = pd.read_csv(squat_front_csv)

# Loops through both dataframes and print validation info
for name, df in [("Side", side_df), ("Front", front_df)]:
    print(f"\n{'='*40}")
    print(f"  {name} CSV Validation")
    print(f"{'='*40}")

    # Checks the total amount of rows
    print(f"\nTotal rows: {len(df)}")

    # Confirms if the column names are correct
    print(f"Columns: {list(df.columns)}")

    # Checks how many rows per label
    print(f"\n--- Rows per label ---")
    print(df['class'].value_counts())

    # Searches for any missing values which could lead to issues for later ML training
    print(f"\n--- Missing values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found")
    else:
        print(missing)

    # Makes sure the angle ranges are within a realistic range for a squat
    print(f"\n--- Angle ranges ---")
    angle_cols = [
        'left_knee_angle', 'right_knee_angle',
        'left_hip_angle', 'right_hip_angle',
        'left_trunk_angle', 'right_trunk_angle'
    ]
    for col in angle_cols:
        print(f"  {col}: min={df[col].min():.1f}  max={df[col].max():.1f}  mean={df[col].mean():.1f}")

    # Checks the knee and ankle distance values look realistic
    print(f"\n--- Distance features ---")
    for col in ['knee_distance', 'ankle_distance', 'knee_ankle_ratio']:
        print(f"  {col}: min={df[col].min():.3f}  max={df[col].max():.3f}  mean={df[col].mean():.3f}")

    # Prints a few sample rows
    print(f"\n--- Sample rows ---")
    print(df.head(3))

print(f"\n{'='*40}")
print("Data validated has been completed, data is ready to be used for Machine Learning")
print(f"{'='*40}")