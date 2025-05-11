import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import time
import matplotlib.pyplot as plt

# Define a sequence dataset for LSTM training
class TORCSSequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length=10):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) #- self.sequence_length
    
    def __getitem__(self, idx):
        # X_seq = self.X[idx:idx+self.sequence_length]
        # y_target = self.y[idx+self.sequence_length-1]  # Predict the last frame's output
        # return X_seq, y_target
        # Return a single sequence and its corresponding output
        X_seq = self.X[idx]  # Shape: (sequence_length, input_size)
        y_target = self.y[idx]  # Shape: (output_size,)
        return X_seq, y_target

# Advanced LSTM model for sequential data
class TORCSLSTMController(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=5, dropout=0.2):
        super(TORCSLSTMController, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]
        x = self.relu(lstm_out)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Multi-head model - separate predictions for different control aspects
class TORCSMultiHeadController(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(TORCSMultiHeadController, self).__init__()
        # Shared layers
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Specialized heads
        # Acceleration and braking (related controls)
        self.accel_brake_fc = nn.Linear(hidden_size, hidden_size//2)
        self.accel_out = nn.Linear(hidden_size//2, 1)  # Acceleration
        self.brake_out = nn.Linear(hidden_size//2, 1)  # Brake
        
        # Steering (requires special attention)
        self.steer_fc = nn.Linear(hidden_size, hidden_size//2)
        self.steer_out = nn.Linear(hidden_size//2, 1)  # Steering
        
        # Gear and clutch
        self.gear_clutch_fc = nn.Linear(hidden_size, hidden_size//2)
        self.gear_out = nn.Linear(hidden_size//2, 1)  # Gear
        self.clutch_out = nn.Linear(hidden_size//2, 1)  # Clutch
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Shared processing
        shared = self.relu(self.shared_fc1(x))
        shared = self.dropout(shared)
        shared = self.relu(self.shared_fc2(shared))
        shared = self.dropout(shared)
        
        # Acceleration and braking head
        accel_brake = self.relu(self.accel_brake_fc(shared))
        accel_brake = self.dropout(accel_brake)
        accel = self.accel_out(accel_brake)
        brake = self.brake_out(accel_brake)
        
        # Steering head
        steer = self.relu(self.steer_fc(shared))
        steer = self.dropout(steer)
        steer = self.steer_out(steer)
        
        # Gear and clutch head
        gear_clutch = self.relu(self.gear_clutch_fc(shared))
        gear_clutch = self.dropout(gear_clutch)
        gear = self.gear_out(gear_clutch)
        clutch = self.clutch_out(gear_clutch)
        
        # Combine outputs
        output = torch.cat([accel, brake, gear, steer, clutch], dim=1)
        return output

# Training function for sequence model
def train_sequence_model(model, X_train, y_train, X_val, y_val, device, 
                         num_epochs=100, batch_size=32, learning_rate=0.001,
                         sequence_length=10,weight_decay=0.0):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create sequence datasets
    train_dataset = TORCSSequenceDataset(X_train, y_train, sequence_length)
    val_dataset = TORCSSequenceDataset(X_val, y_val, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:

            #verify the input shape here for error values/
            if len(X_batch.shape) != 3:
                raise ValueError(f"EXpected X_batch to be 3D (batch_Size,seq_length,input_size)")
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if len(X_batch.shape) != 3:
                    raise ValueError(f"EXpected X_batch to be 3D (batch_Size,seq_length,input_size)")
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        # Average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        # Store for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f}s, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.close()
    
    return model, train_losses, val_losses

# Function to convert regular dataset to sequences for LSTM training
def prepare_sequence_data(X, y, sequence_length=10):
    """Convert flat data to sequences for LSTM training"""
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])  # Predict the last frame's output
    
    return np.array(X_seq), np.array(y_seq)

# Main function to train and evaluate the advanced model
def train_advanced_model(X_train, y_train, X_val, y_val, input_size, 
                         model_type='lstm', sequence_length=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Convert numpy arrays to PyTorch tensors
    if model_type == 'lstm':
        # For LSTM, we need sequences
        X_train_seq, y_train_seq = prepare_sequence_data(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = prepare_sequence_data(X_val, y_val, sequence_length)
        
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
        
        model = TORCSLSTMController(input_size=input_size, hidden_size=128, num_layers=2, output_size=5)
        
        model, train_losses, val_losses = train_sequence_model(
            model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
            device, num_epochs=50, batch_size=64, learning_rate=0.001, sequence_length=sequence_length
        )
    else:
        # For multi-head or other models
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        if model_type == 'multi_head':
            model = TORCSMultiHeadController(input_size=input_size, hidden_size=128)
        else:
            # Default to standard FNN
            model = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5)
            )
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(50):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
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
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            end_time = time.time()
            print(f'Epoch {epoch+1}/50, Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Time: {end_time - start_time:.2f}s')
    
    # Save model
    model_filename = f'torcs_controller_{model_type}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({model_type.upper()} model)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_curve_{model_type}.png')
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    # This would be called from the main training script
    pass