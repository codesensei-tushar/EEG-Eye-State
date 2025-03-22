import torch
from tqdm import tqdm  # Progress bar for better visualization

def train_model(model, train_loader,val_loader, criterion, optimizer, device, num_epochs=20):
    model.train()  # Set model to training mode
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm to monitor progress per epoch
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Move data to the device with non_blocking=True (if pin_memory=True)
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()  # Reset gradients
            
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, target)  # Compute loss
            
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * data.size(0)
            
            # Compute batch accuracy
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        val_acc = evaluate_model(model, val_loader, criterion, device)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return model, train_losses, train_accuracies ,val_accuracies

def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients during evaluation
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}")
    return epoch_acc