import torch
import torch.nn as nn
import torch.optim as optim

# Define a classical Feedforward Neural Network
class TimeSeriesEmulator(nn.Module):
    def __init__(self, input_size=16, output_size=500, hidden_dim=128):
        super(TimeSeriesEmulator, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)
        
        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since it's regression (predicting raw time series values)
        return x