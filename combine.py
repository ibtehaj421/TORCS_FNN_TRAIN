import pandas as pd
import numpy as np

def combine_datasets(file1_path, file2_path, output_path):
    # Read parsed_log.csv.xls (dataset.csv.txt structure)
    try:
        df1 = pd.read_csv(file1_path, on_bad_lines='skip')
        print(f"parsed_log.csv.xls: {df1.shape[0]} rows, {df1.shape[1]} columns")
        print(f"Columns: {df1.columns.tolist()}")
        print(f"Missing values:\n{df1.isna().sum()}")
    except Exception as e:
        print(f"Failed to read {file1_path} as CSV: {e}")
        try:
            df1 = pd.read_excel(file1_path, engine='openpyxl')
            print(f"parsed_log.csv.xls (Excel): {df1.shape[0]} rows, {df1.shape[1]} columns")
            print(f"Columns: {df1.columns.tolist()}")
            print(f"Missing values:\n{df1.isna().sum()}")
        except Exception as e:
            print(f"Error reading {file1_path} as Excel: {e}")
            return

    # Read New DataSet copy.csv (second.csv.txt structure)
    try:
        df2 = pd.read_csv(file2_path, on_bad_lines='skip')
        print(f"New DataSet copy.csv: {df2.shape[0]} rows, {df2.shape[1]} columns")
        print(f"Columns: {df2.columns.tolist()}")
        print(f"Missing values:\n{df2.isna().sum()}")
    except Exception as e:
        print(f"Error reading {file2_path}: {e}")
        return
    
    # Check for duplicate columns in input DataFrames
    df1_duplicates = df1.columns[df1.columns.duplicated()].tolist()
    df2_duplicates = df2.columns[df2.columns.duplicated()].tolist()
    if df1_duplicates:
        print(f"Error: Duplicate columns in parsed_log.csv.xls: {df1_duplicates}")
        return
    if df2_duplicates:
        print(f"Error: Duplicate columns in New DataSet copy.csv: {df2_duplicates}")
        return
    
    # Use columns from New DataSet copy.csv
    all_columns = [
        'Acceleration', 'Brake', 'Gear', 'Steer', 'Clutch', 'Focus', 'Meta', 'Angle',
        'CurLapTime', 'Damage', 'DistFromStart', 'DistRaced', 'Focus2', 'Fuel', 'Gear2',
        'LastLapTime', 'Opponents', 'RacePos', 'RPM', 'Speed X', 'Speed Y', 'Speed Z',
        'Track', 'TrackPos', 'WheelSpinVel', 'Z'
    ]
    
    # Verify that df2 has the expected columns
    missing_cols = [col for col in all_columns if col not in df2.columns]
    if missing_cols:
        print(f"Warning: Missing columns in New DataSet copy.csv: {missing_cols}")
        print("Please verify the file structure.")
        return
    
    # Define column mappings from parsed_log.csv.xls to New DataSet copy.csv
    column_mapping = {
        'angle': 'Angle',
        'curLapTime': 'CurLapTime',
        'damage': 'Damage',
        'distFromStart': 'DistFromStart',
        'distRaced': 'DistRaced',
        'fuel': 'Fuel',
        'gear': 'Gear',  # Vehicle state
        'lastLapTime': 'LastLapTime',
        'opponents': 'Opponents',
        'racePos': 'RacePos',
        'rpm': 'RPM',
        'speedX': 'Speed X',
        'speedY': 'Speed Y',
        'speedZ': 'Speed Z',
        'track': 'Track',
        'trackPos': 'TrackPos',
        'wheelSpinVel': 'WheelSpinVel',
        'z': 'Z',
        'focus': 'Focus',
        'client_focus': 'Focus2',
        'client_accel': 'Acceleration',
        'client_brake': 'Brake',
        'client_gear': 'Gear2',  # Control input; change to 'Gear' if Gear is control
        'client_steer': 'Steer',
        'client_clutch': 'Clutch',
        'client_meta': 'Meta'
    }
    
    # Rename columns in df1
    df1 = df1.rename(columns=column_mapping)
    
    # Check for duplicate columns after mapping
    df1_columns = df1.columns.tolist()
    df1_duplicates = [col for col in df1_columns if df1_columns.count(col) > 1]
    if df1_duplicates:
        print(f"Error: Duplicate columns in df1 after mapping: {df1_duplicates}")
        return
    
    # Select only the columns present in New DataSet copy.csv
    df1 = df1[[col for col in all_columns if col in df1.columns]]
    
    # Add missing columns to df1 with NaN
    for col in all_columns:
        if col not in df1.columns:
            df1[col] = np.nan
    
    # Reorder df1 columns to match all_columns
    df1 = df1[all_columns]
    
    # Ensure df2 has the correct columns
    df2 = df2[all_columns]
    
    # Check for duplicate columns before concatenation
    if df1.columns.duplicated().any() or df2.columns.duplicated().any():
        print("Error: Duplicate columns detected before concatenation")
        print(f"df1 columns: {df1.columns.tolist()}")
        print(f"df2 columns: {df2.columns.tolist()}")
        return
    
    # Concatenate the datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Check for empty rows
    empty_rows = combined_df[combined_df.isna().all(axis=1)]
    print(f"Number of completely empty rows: {len(empty_rows)}")
    partially_empty = combined_df[combined_df.isna().sum(axis=1) > len(all_columns) * 0.8]
    print(f"Number of rows with >80% missing values: {len(partially_empty)}")
    
    # Remove completely empty rows
    combined_df = combined_df.dropna(how='all')
    print(f"After removing empty rows: {combined_df.shape[0]} rows")
    
    # Save the combined dataset
    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")

def main():
    # File paths
    file1_path = 'parsed_log.csv.xls'  # dataset.csv.txt structure
    file2_path = 'New DataSet copy.csv'  # second.csv.txt structure
    output_path = 'combined_dataset.csv'
    
    # Combine the datasets
    combine_datasets(file1_path, file2_path, output_path)

if __name__ == '__main__':
    main()