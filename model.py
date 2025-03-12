import torch.nn.functional as F
import torch.nn as nn

# Define a simple feedforward neural network for binary classification
class EEGClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=2):
        super(EEGClassifier, self).__init__()
        # First fully-connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Second fully-connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
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