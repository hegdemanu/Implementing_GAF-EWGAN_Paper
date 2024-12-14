

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """ConvLSTM cell implementation."""
    
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)

    def forward(self, x, h, c):
        """Forward pass."""
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        return ch, cc

class Generator(nn.Module):
    """WG Generator implementation with ConvLSTM layers."""
    
    def __init__(self):
        super(Generator, self).__init__()
        
        # ConvLSTM layers
        self.conv_lstm1 = ConvLSTMCell(3, 64, 3)  # Input: RGB channels
        self.conv_lstm2 = ConvLSTMCell(64, 64, 3)
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64 * 60 * 60, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, height, width, channels)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        h1, c1 = torch.zeros(batch_size, 64, 60, 60).to(x.device), \
                 torch.zeros(batch_size, 64, 60, 60).to(x.device)
        h2, c2 = torch.zeros(batch_size, 64, 60, 60).to(x.device), \
                 torch.zeros(batch_size, 64, 60, 60).to(x.device)
        
        # Process sequence through ConvLSTM layers
        for t in range(seq_len):
            h1, c1 = self.conv_lstm1(x[:, t], h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
        
        # Dense layers
        x = self.flatten(h2)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.relu(self.dense4(x))
        x = self.output(x)
        
        return x
