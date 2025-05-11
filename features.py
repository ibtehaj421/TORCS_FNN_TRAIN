import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def extract_advanced_features(data):
    """Extract more sophisticated features from TORCS data"""
    # New dataframe for processed features
    processed_data = pd.DataFrame()
    
    # Basic features (direct from sensors)
    basic_cols = ['Angle', 'CurLapTime', 'Damage', 'DistFromStart', 'DistRaced', 
                  'Fuel', 'Gear', 'LastLapTime', 'RPM', 'Speed X', 'Speed Y', 'Speed Z', 
                  'TrackPos', 'Z']
    
    for col in basic_cols:
        processed_data[col] = data[col]
    
    # Process track sensors (19 range sensors)
    # Instead of just using mean, extract more meaningful features
    track_arrays = data['Track'].values
    
    # Distance to nearest obstacle in different sectors
    processed_data['track_left'] = [np.min(arr[:5]) if len(arr) >= 5 else np.min(arr) if len(arr) > 0 else 0.0 
                                   for arr in track_arrays]
    
    processed_data['track_center_left'] = [np.min(arr[5:9]) if len(arr) >= 9 else 0.0 
                                         for arr in track_arrays]
    
    processed_data['track_center'] = [arr[9] if len(arr) > 9 else 0.0 
                                    for arr in track_arrays]
    
    processed_data['track_center_right'] = [np.min(arr[10:14]) if len(arr) >= 14 else 0.0 
                                          for arr in track_arrays]
    
    processed_data['track_right'] = [np.min(arr[14:]) if len(arr) >= 15 else 0.0 
                                   for arr in track_arrays]
    
    # Calculate asymmetry (useful for detecting turns)
    processed_data['track_asymmetry'] = processed_data['track_left'] - processed_data['track_right']
    
    # Wheel spin velocity features
    wheel_arrays = data['WheelSpinVel'].values
    
    # Extract individual wheel velocities
    processed_data['wheel_fl'] = [arr[0] if len(arr) > 0 else 0.0 for arr in wheel_arrays]
    processed_data['wheel_fr'] = [arr[1] if len(arr) > 1 else 0.0 for arr in wheel_arrays]
    processed_data['wheel_rl'] = [arr[2] if len(arr) > 2 else 0.0 for arr in wheel_arrays]
    processed_data['wheel_rr'] = [arr[3] if len(arr) > 3 else 0.0 for arr in wheel_arrays]
    
    # Calculate wheel slip indicators (difference between left and right wheels)
    processed_data['wheel_front_diff'] = processed_data['wheel_fl'] - processed_data['wheel_fr']
    processed_data['wheel_rear_diff'] = processed_data['wheel_rl'] - processed_data['wheel_rr']
    
    # Focus sensors (5 focus sensors)
    focus_arrays = data['Focus'].values
    processed_data['focus_closest'] = [np.min(arr) if len(arr) > 0 else 0.0 for arr in focus_arrays]
    
    # Calculate speed features
    processed_data['speed_magnitude'] = np.sqrt(
        data['Speed X']**2 + data['Speed Y']**2 + data['Speed Z']**2
    )
    
    # Calculate lateral acceleration (useful for predicting understeer/oversteer)
    processed_data['lateral_g'] = data['Speed Y'] * data['Speed X'] / 9.81
    
    # Get absolute track position (distance from center regardless of side)
    processed_data['abs_track_pos'] = np.abs(data['TrackPos'])
    
    # Output columns remain the same
    output_cols = ['Acceleration', 'Brake', 'Gear', 'Steer']
    for col in output_cols:
        processed_data[col] = data[col]
    
    return processed_data, list(processed_data.columns[:-4]), output_cols

def process_dataset(file_path):
    """Load and process the dataset with advanced features"""
    # Load the CSV file
    dtypes = {
        'Track': str,
        'WheelSpinVel': str,
        'Focus': str,
        'Focus2': str,
        'Opponents': str,
    } #this is to address the type error i was getting
    data = pd.read_csv(file_path,dtype=dtypes)
    output_cols = ['Acceleration', 'Brake', 'Gear', 'Steer', 'Clutch']
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(output_cols):
        plt.subplot(1, len(output_cols), i+1)
        plt.hist(data[col], bins=50, log=True)  # Log scale to visualize rare values
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Count (log scale)')
    plt.tight_layout()
    plt.savefig('plots/output_distributions.png')
    plt.close()
    print(f"Initial dataset size: {len(data)}")
    originalLen = len(data)
    # Filter out static/idle rows
    print("Filtering out static/idle rows...")
    # Check for non-zero outputs
    data = data[
        (data['Acceleration'] != 0) | 
        (data['Brake'] != 0) | 
        (data['Steer'] != 0) | 
        (data['Gear'] != 0)
    ]
    
    
    data = data[data['Speed X'].abs() > 0.01]  # Adjust threshold as needed
    print(f"After Speed X filter: {len(data)} rows ({len(data)/originalLen*100:.2f}% remaining)")
    
    # Handle negative CurLapTime by shifting
    if (data['CurLapTime'] <= 0).any():
        print("Found negative CurLapTime values, shifting to positive...")
        data['CurLapTime'] = data['CurLapTime'] - data['CurLapTime'].min() + 1e-6
    
    # Skip WheelSpinVel and Focus filters for now
    # data = data[data['WheelSpinVel'] != '0 0 0 0']
    # data = data[data['Focus'] != '-1 -1 -1 -1 -1']
    
    if len(data) == 0:
        raise ValueError("All rows were filtered out! Check filter conditions or dataset content. "
                        "Consider relaxing filters or collecting new data.")
    
    print(f"Final dataset size after filtering: {len(data)}")
    
    # Save filtered dataset for inspection
    data.to_csv('filtered_dataset.csv', index=False)

    print(f"Dataset size after filtering: {len(data)}")
    # Parse multi-value columns
    data['Track'] = data['Track'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
    data['WheelSpinVel'] = data['WheelSpinVel'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
    data['Focus'] = data['Focus'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
    data['Focus2'] = data['Focus2'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
    
    # Extract advanced features
    processed_data, input_cols, output_cols = extract_advanced_features(data)
    
    # Normalize input and output
    X = processed_data[input_cols].values
    y = processed_data[output_cols].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    print(f"Processed dataset with {X.shape[1]} input features and {y.shape[1]} outputs")
    print(f"Input features: {input_cols}")
    
    return X_scaled, y_scaled, scaler_X, scaler_y, input_cols, output_cols

if __name__ == "__main__":
    # Example usage
    X_scaled, y_scaled, scaler_X, scaler_y, input_cols, output_cols = process_dataset('combined_dataset.csv')
    print(f"Dataset processed successfully: {X_scaled.shape[0]} samples")