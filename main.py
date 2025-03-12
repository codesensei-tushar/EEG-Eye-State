import torch
import torch.nn as nn
import torch.optim as optim
from model import EEGClassifier
from train import train_model, evaluate_model
from dataloader import train_loader, test_loader

    # Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

    # Initialize the model, criterion (loss function) and optimizer
model = EEGClassifier(input_dim=14, hidden_dim=64, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
print("Starting Training...")
model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)

    # Evaluate the model on the test set
print("Evaluating on Test Set...")
evaluate_model(model, test_loader, criterion, device)