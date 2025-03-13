import torch.nn.functional as F
import torch.nn as nn

# Define a simple feedforward neural network for binary classification
class EEGClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=2):
        super(EEGClassifier, self).__init__()
        # First fully-connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Second fully-connected layer
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # Output layer: number of classes (here, 2 for binary classification)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        # Dropout to help with overfitting
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input x shape: (batch_size, 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No softmax needed with CrossEntropyLoss
        return x
    
class EEGNet(nn.Module):
    def __init__(self, 
                 num_channels=14, 
                 num_classes=2, 
                 temporal_filters=8, 
                 depthwise_filters=16, 
                 kernel_length=64, 
                 dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        # Temporal Convolution: learns frequency-specific filters
        self.temporal_conv = nn.Conv2d(1, temporal_filters, (1, kernel_length), padding=(0, kernel_length//2))
        self.bn1 = nn.BatchNorm2d(temporal_filters)
        
        # Depthwise Convolution: spatial filtering (each temporal filter separately)
        self.depthwise_conv = nn.Conv2d(temporal_filters, depthwise_filters, (num_channels, 1), groups=temporal_filters)
        self.bn2 = nn.BatchNorm2d(depthwise_filters)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Separable Convolution: further processing the feature maps
        self.separable_conv = nn.Conv2d(depthwise_filters, depthwise_filters, (1, 16), padding=(0, 8))
        self.bn3 = nn.BatchNorm2d(depthwise_filters)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(depthwise_filters * self._calc_final_width(), num_classes)
    
    def _calc_final_width(self):
        # Assume an input length that remains constant (here abstracted as 128 time points)
        # and calculate the output width after two pooling layers.
        # For example, with an input width=128:
        # After pool1: width = 128/4 = 32, after pool2: width = 32/8 = 4.
        return 4  # adjust based on your input signal length
    
    def forward(self, x):
        # x shape: (batch_size, num_channels) but we need to add spatial dimensions:
        # Reshape to (batch_size, 1, num_channels, time_points)
        # Assume input x is already in shape (batch_size, num_channels, time_points)
        x = x.unsqueeze(1)  # (batch_size, 1, num_channels, time_points)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = F.elu(x)
        
        # Depthwise convolution (spatial filtering)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Separable convolution
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ResidualMLP(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=2):
        super(ResidualMLP, self).__init__()
        # First layer transforms input to hidden_dim space
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Two subsequent layers with a residual connection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Output layer for classification
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch_size, 14)
        # Initial transformation
        out = F.relu(self.bn1(self.fc1(x)))
        
        # Save residual for later addition
        residual = out.clone()
        
        # Pass through two layers with non-linearities
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = F.relu(self.bn3(self.fc3(out)))
        out = self.dropout(out)
        
        # Add skip connection (residual)
        out = out + residual
        
        # Final classification layer (no softmax needed with CrossEntropyLoss)
        out = self.fc_out(out)
        return out