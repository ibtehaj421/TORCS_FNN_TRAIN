import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib
import time
# Define the FNN model
class TORCSController(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, output_size=5):
        super(TORCSController, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for output (regression)
        return x

# Function to parse space-separated multi-value columns
def parse_multi_value_column(col):
    return col.apply(lambda x: np.array([float(v) for v in str(x).split()]))

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Parse multi-value columns
    data['Track'] = parse_multi_value_column(data['Track'])
    data['WheelSpinVel'] = parse_multi_value_column(data['WheelSpinVel'])
    data['Focus'] = parse_multi_value_column(data['Focus'])
    data['Focus2'] = parse_multi_value_column(data['Focus2'])
    
    # Input and output columns
    input_cols = ['Angle', 'CurLapTime', 'Damage', 'DistFromStart', 'DistRaced', 'Fuel', 'Gear', 
                  'LastLapTime', 'RPM', 'Speed X', 'Speed Y', 'Speed Z', 'TrackPos', 'Z']
    output_cols = ['Acceleration', 'Brake', 'Gear', 'Steer', 'Clutch']
    
    # Process multi-value columns (use mean)
    data['track_mean'] = data['Track'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    data['wheelSpinVel_mean'] = data['WheelSpinVel'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    data['focus_mean'] = data['Focus'].apply(lambda x: np.mean(x[:5]) if len(x) >= 5 else np.mean(x) if len(x) > 0 else 0.0)
    data['focus2_mean'] = data['Focus2'].apply(lambda x: np.mean(x[:5]) if len(x) >= 5 else np.mean(x) if len(x) > 0 else 0.0)
    
    # Update input columns
    input_cols.extend(['track_mean', 'wheelSpinVel_mean', 'focus_mean'])
    
    # Extract inputs and outputs
    X = data[input_cols].values
    y = data[output_cols].values
    
    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    return X_tensor, y_tensor, scaler_X, scaler_y

# Training function
def train_model(model, X_train, y_train, X_val, y_val, device, num_epochs=100, batch_size=32, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # DataLoader for batching
    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time taken: {elapsed_time:.4f} seconds')

# Testing function
def test_model(model, X_val, y_val, scaler_y, device, num_samples=5):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # Select a few validation samples
        indices = np.random.choice(len(X_val), num_samples, replace=False)
        X_test = X_val[indices].to(device)
        y_test = y_val[indices].cpu().numpy()
        y_pred = model(X_test).cpu().numpy()
        
        # Denormalize predictions
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test = scaler_y.inverse_transform(y_test)
        
        # Print predictions vs. actual
        print("\nSample Predictions (Acceleration, Brake, Gear, Steer, Clutch):")
        for i in range(num_samples):
            print(f"Sample {i+1}:")
            print(f"  Predicted: {y_pred[i]}")
            print(f"  Actual:    {y_test[i]}")

def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # File path to your dataset
    dataset_path = 'combined_dataset.csv'
    
    # Load and preprocess data
    X_tensor, y_tensor, scaler_X, scaler_y = load_and_preprocess_data(dataset_path)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = TORCSController(input_size=17, hidden_size=128, output_size=5)
    
    # Train the model
    train_model(model, X_train, y_train, X_val, y_val, device, num_epochs=50, batch_size=32, learning_rate=0.001)
    
    # Test the model on a few validation samples
    test_model(model, X_val, y_val, scaler_y, device, num_samples=5)
    
    # Save the model and scalers
    torch.save(model.state_dict(), 'torcs_controller2.pth')
    joblib.dump(scaler_X, 'scaler_X2.pkl')
    joblib.dump(scaler_y, 'scaler_y2.pkl')
    print("Model and scalers saved successfully.")

if __name__ == '__main__':
    main()