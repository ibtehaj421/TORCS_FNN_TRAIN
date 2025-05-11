import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from features import extract_advanced_features
from model import TORCSLSTMController
import matplotlib.pyplot as plt

def prepare_sequence_data(X, y, sequence_length):
    """Prepare sequential data for LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scalers
    scaler_X_path = 'models/scaler_X_lstm.pkl'
    scaler_y_path = 'models/scaler_y_lstm.pkl'
    try:
        with open(scaler_X_path, 'rb') as f:
            scaler_X = joblib.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = joblib.load(f)
        print(f"Loaded scalers from {scaler_X_path} and {scaler_y_path}")
    except FileNotFoundError as e:
        print(f"Error: Scaler file not found: {e}")
        return
    
    # Load filtered dataset
    dataset_path = 'filtered_dataset.csv'
    print(f"Loading filtered dataset from {dataset_path}...")
    try:
        data = pd.read_csv(dataset_path, dtype={
            'Track': str,
            'WheelSpinVel': str,
            'Focus': str,
            'Focus2': str,
            'Opponents': str,
        }, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {dataset_path} not found")
        return
    
    print(f"Filtered dataset size: {len(data)}")
    
    if len(data) == 0:
        print("Error: Filtered dataset is empty")
        return
    
    # Analyze dataset
    print("Output column statistics:")
    print(data[['Acceleration', 'Brake', 'Gear', 'Steer']].describe())
    print("\nNon-zero output counts:")
    print((data[['Acceleration', 'Brake', 'Gear', 'Steer']] != 0).sum())
    print("\nSpeed X statistics:")
    print(data['Speed X'].describe())
    
    # Plot output distributions
    print("Plotting output distributions...")
    plt.figure(figsize=(12, 3))
    for i, col in enumerate(['Acceleration', 'Brake', 'Gear', 'Steer']):
        plt.subplot(1, 4, i+1)
        plt.hist(data[col], bins=50, log=True)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Count (log scale)')
    plt.tight_layout()
    plt.savefig('plots/filtered_output_distributions.png')
    plt.close()
    print("Saved output distributions to plots/filtered_output_distributions.png")
    
    # Parse multi-value columns
    try:
        data['Track'] = data['Track'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
        data['WheelSpinVel'] = data['WheelSpinVel'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
        data['Focus'] = data['Focus'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
        data['Focus2'] = data['Focus2'].apply(lambda x: np.array([float(v) for v in str(x).split()]))
    except Exception as e:
        print(f"Error parsing multi-value columns: {e}")
        return
    
    # Extract advanced features
    try:
        processed_data, input_cols, output_cols = extract_advanced_features(data)
    except Exception as e:
        print(f"Error extracting advanced features: {e}")
        return
    
    # Get raw X and y
    X = processed_data[input_cols].values
    y = processed_data[output_cols].values
    
    # Normalize using loaded scalers
    try:
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)
    except Exception as e:
        print(f"Error applying scalers: {e}")
        return
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Prepare sequences for LSTM
    sequence_length = 10
    print(f"Preparing sequential data with sequence length {sequence_length}...")
    try:
        X_train_seq, y_train_seq = prepare_sequence_data(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = prepare_sequence_data(X_val, y_val, sequence_length)
    except Exception as e:
        print(f"Error preparing sequence data: {e}")
        return
    
    # Convert to PyTorch tensors
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
    
    # Initialize model
    input_size = X_val_seq.shape[2]
    hidden_size = 128
    num_layers = 2
    output_size = 4  # Acceleration, Brake, Gear, Steer
    dropout = 0.5
    
    model = TORCSLSTMController(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    ).to(device)
    
    # Load saved model weights
    model_path = 'models/torcs_lstm_controller.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test the model on validation samples
    print("\nTesting model on validation samples:")
    model.eval()
    mae_scores = {col: [] for col in ['Acceleration', 'Brake', 'Gear', 'Steer']}
    with torch.no_grad():
        for i in range(min(5, len(X_val_tensor))):
            X_sample = X_val_tensor[i:i+1].to(device)
            y_true = y_val_tensor[i].cpu().numpy()
            
            y_pred = model(X_sample).cpu().numpy()
            y_pred = y_pred[0]
            
            y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(1, -1))[0]
            y_true_denorm = scaler_y.inverse_transform(y_true.reshape(1, -1))[0]
            
            # Apply physical constraints
            y_pred_denorm[1] = np.clip(y_pred_denorm[1], 0, 1)  # Brake: [0, 1]
            y_pred_denorm[3] = np.clip(y_pred_denorm[3], -1, 1)  # Steer: [-1, 1]
            y_pred_denorm[2] = np.round(y_pred_denorm[2])  # Gear: Integer
            y_true_denorm[1] = np.clip(y_true_denorm[1], 0, 1)  # Brake: [0, 1]
            y_true_denorm[3] = np.clip(y_true_denorm[3], -1, 1)  # Steer: [-1, 1]
            y_true_denorm[2] = np.round(y_true_denorm[2])  # Gear: Integer
            
            # Calculate MAE for each output
            for j, col in enumerate(['Acceleration', 'Brake', 'Gear', 'Steer']):
                mae = np.abs(y_pred_denorm[j] - y_true_denorm[j])
                mae_scores[col].append(mae)
            
            print(f"\nSample {i+1}:")
            print(f"Predicted: [Acceleration: {y_pred_denorm[0]:.4f}, "
                  f"Brake: {y_pred_denorm[1]:.4f}, "
                  f"Gear: {y_pred_denorm[2]:.4f}, "
                  f"Steer: {y_pred_denorm[3]:.4f}]")
            print(f"Actual:    [Acceleration: {y_true_denorm[0]:.4f}, "
                  f"Brake: {y_true_denorm[1]:.4f}, "
                  f"Gear: {y_true_denorm[2]:.4f}, "
                  f"Steer: {y_true_denorm[3]:.4f}]")
    
    # Print average MAE for each output
    print("\nMean Absolute Error (MAE) per output:")
    for col in mae_scores:
        avg_mae = np.mean(mae_scores[col])
        print(f"{col}: {avg_mae:.4f}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()