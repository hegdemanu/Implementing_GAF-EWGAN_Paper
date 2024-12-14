

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """WD discriminator with Conv1D layers."""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # First Conv1D layer block
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16)
        )
        
        # Second Conv1D layer block
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16)
        )
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(16 * 11, 50)  # 11 is sequence length
        self.dense2 = nn.Linear(50, 50)
        self.dense3 = nn.Linear(50, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.leaky_relu(self.dense1(x))
        x = self.leaky_relu(self.dense2(x))
        x = self.dense3(x)
        return x
