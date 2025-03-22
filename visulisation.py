import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
import shap
import torch

# Function to plot training loss and accuracy
def plot_training_curves(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o', color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()

    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Closed", "Open"], yticklabels=["Closed", "Open"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Function to plot ROC Curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')  # Random guess line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
def plot_feature_importance(model, data_loader, device):
    model.eval()  # Set model to evaluation mode

    # Fetch a batch of data
    batch, _ = next(iter(data_loader))  # Extract only input features
    batch = batch.to(device)  # Move to the same device as model
    batch.requires_grad = True  # Enable gradients for SHAP

    # Use GradientExplainer
    explainer = shap.GradientExplainer(model, batch)

    # Compute SHAP values
    shap_values = explainer.shap_values(batch)

    # If SHAP returns a list, select the first element (for binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Convert batch to NumPy
    batch = batch.cpu().detach().numpy()

    # Ensure feature names match the number of features
    num_features = batch.shape[1]
    feature_names = [f"Channel {i+1}" for i in range(num_features)]

    # Plot SHAP summary
    shap.summary_plot(shap_values, batch, feature_names=feature_names)

def plot_training_vs_validation_accuracy(train_accuracies, val_accuracies):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o', linestyle='-')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker='s', linestyle='--', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy per Epoch")
    plt.legend()
    plt.grid()
    plt.show()


def plot_permutation_importance(model, data_loader, device):
    model.eval()  # Set model to evaluation mode

    # Extract one batch for evaluation
    batch, labels = next(iter(data_loader))
    batch, labels = batch.to(device), labels.cpu().numpy()

    # Convert batch to NumPy
    batch_np = batch.cpu().detach().numpy()

    # Define a prediction function that returns class probabilities
    def model_predict(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # Convert logits to probabilities
        return np.argmax(probs, axis=1)  # Return class predictions

    # Compute permutation importance
    result = permutation_importance(model_predict, batch_np, labels, scoring="accuracy", n_repeats=10, random_state=42)

    # Plot feature importance
    feature_importance = result.importances_mean
    feature_names = [f"Channel {i+1}" for i in range(batch.shape[1])]

    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, feature_importance, color="skyblue")
    plt.xlabel("Importance Score")
    plt.ylabel("EEG Channels")
    plt.title("Permutation Feature Importance")
    plt.show()
