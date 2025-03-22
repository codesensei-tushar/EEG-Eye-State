import torch
import torch.nn as nn
import torch.optim as optim
from model import EEGClassifier
from model import EEGNet
from model import ResidualMLP
from model import ResidualBlock
from model import RobustEEGClassifier
from model import EEG_LSTM 
from train_tqdm import train_model, evaluate_model
from dataloader import train_loader, test_loader, val_loader
from visulisation import plot_training_curves, plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_training_vs_validation_accuracy, plot_permutation_importance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# model = EEGClassifier(input_dim=14, hidden_dim=64, num_classes=2).to(device)
# model = EEGNet(num_channels=14, num_classes=2).to(device)
# model = ResidualMLP(input_dim=14, hidden_dim=64, num_classes=2).to(device)
model = RobustEEGClassifier(input_dim=14, hidden_dim=128, num_classes=2, num_blocks=3, dropout=0.1).to(device)
# model = EEG_LSTM(input_dim=14, hidden_dim=64, num_classes=2, num_layers=2, dropout=0.5).to(device)
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.0021, weight_decay=1e-4)

print("Starting Training...")
model, train_losses, train_accuracies,val_accuracies= train_model(model, train_loader,val_loader,criterion, optimizer, device, num_epochs=2)

print("Evaluating on Test Set...")
evaluate_model(model, test_loader, criterion, device)

plot_training_curves(train_losses, train_accuracies)
plot_training_vs_validation_accuracy(train_accuracies, val_accuracies)
# Get predictions
y_true, y_pred, y_scores = [], [], []
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_scores.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())  # Probabilities for class 1

# Plot confusion matrix & ROC Curve
plot_confusion_matrix(y_true, y_pred)
plot_roc_curve(y_true, y_scores)
# plot_permutation_importance(model, test_loader, device)
