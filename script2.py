import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Import our custom modules
from features import process_dataset
from model import TORCSLSTMController, train_sequence_model, prepare_sequence_data

def main():
    print("===== Advanced TORCS Controller with LSTM =====")
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset path - update to your dataset path
    dataset_path = 'combined_dataset.csv'
    
    # Create output directory for saving models and plots
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Process the dataset with advanced feature engineering
    print("Processing dataset with advanced features...")
    X_scaled, y_scaled, scaler_X, scaler_y, input_cols, output_cols = process_dataset(dataset_path)
    print(f"Dataset processed: {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
    
    # Set LSTM sequence length (number of time steps to consider)
    sequence_length = 10  # Adjust based on your driving dynamics
    
    # Prepare sequences for LSTM
    print(f"Preparing sequential data with sequence length {sequence_length}...")
    X_train_seq, y_train_seq = prepare_sequence_data(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = prepare_sequence_data(X_val, y_val, sequence_length)
    print(f"X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    print(f"X_val_seq shape: {X_val_seq.shape}, y_val_seq shape: {y_val_seq.shape}")
    print(f"Sequential data shape: {X_train_seq.shape}, {y_train_seq.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
    
    # Initialize LSTM model
    input_size = X_train_seq.shape[2]  # Number of features
    hidden_size = 128  # LSTM hidden size
    num_layers = 2    # Number of LSTM layers
    output_size = 4   # 5 outputs: Acceleration, Brake, Gear, Steer, Clutch(no longer needed as value is always 0)
    
    print(f"Creating LSTM model with {input_size} inputs, {hidden_size} hidden units, "
          f"{num_layers} layers, and {output_size} outputs")
    
    model = TORCSLSTMController(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=0.5 # to prevent overfitting as i was observing bad values.
    )
    
    # Training parameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 1e-5 # add L2 regularization
    print(f"Training LSTM model for {num_epochs} epochs with batch size {batch_size}...")
    start_time = time.time()
    
    # Train the model
    model, train_losses, val_losses = train_sequence_model(
        model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
        device, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        sequence_length=sequence_length,weight_decay=weight_decay
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the trained model and scalers
    model_path = 'models/torcs_lstm_controller.pth'
    scaler_X_path = 'models/scaler_X_lstm.pkl'
    scaler_y_path = 'models/scaler_y_lstm.pkl'
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    
    # Save feature names for future reference
    pd.DataFrame({'feature_name': input_cols}).to_csv('models/input_features.csv', index=False)
    
    print(f"Model and scalers saved successfully at {model_path}")
    
    # Plot training and validation losses
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/lstm_training_curve.png')
    plt.close()
    
    # Test the model on some validation samples
    print("\nTesting model on validation samples:")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Select some random samples for testing
        num_samples = 5
        sample_indices = np.random.choice(len(X_val_tensor), num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            X_sample = X_val_tensor[idx].unsqueeze(0).to(device)  # Add batch dimension
            y_true = y_val_tensor[idx].cpu().numpy()
            
            # Get model prediction
            y_pred = model(X_sample).cpu().numpy()[0]
            
            # Denormalize predictions and actual values
            y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(1, -1))[0]
            y_true_denorm = scaler_y.inverse_transform(y_true.reshape(1, -1))[0]
            
            print(f"\nSample {i+1}:")
            print(f"  Predicted: [Accel: {y_pred_denorm[0]:.4f}, Brake: {y_pred_denorm[1]:.4f}, "
                  f"Gear: {y_pred_denorm[2]:.1f}, Steer: {y_pred_denorm[3]:.4f}, "
                  )
            print(f"  Actual:    [Accel: {y_true_denorm[0]:.4f}, Brake: {y_true_denorm[1]:.4f}, "
                  f"Gear: {y_true_denorm[2]:.1f}, Steer: {y_true_denorm[3]:.4f}, "
                  )
    
    print("\n===== Model training complete =====")
    print("To use this model in TORCS:")
    print("1. Run the TORCS simulator with the appropriate configuration")
    print("2. Execute the 'torcs_control.py' script which will use this trained model")
    print("3. The model will handle sequential inputs to make better driving decisions")

if __name__ == "__main__":
    main()