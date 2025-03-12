import torch

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            # Move data to the chosen device (CPU/GPU)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()  # Reset gradients
            
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, target)  # Compute loss
            
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * data.size(0)
            
            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return model

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}")