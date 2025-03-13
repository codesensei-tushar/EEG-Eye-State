import torch
import torch.nn as nn
import torch.optim as optim
from model import EEGClassifier
from model import EEGNet
from model import ResidualMLP
from train_tqdm import train_model, evaluate_model
from dataloader import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# model = EEGClassifier(input_dim=14, hidden_dim=64, num_classes=2).to(device)
# model = EEGNet(num_channels=14, num_classes=2).to(device)
model = ResidualMLP(input_dim=14, hidden_dim=64, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print("Starting Training...")
model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)

print("Evaluating on Test Set...")
evaluate_model(model, test_loader, criterion, device)