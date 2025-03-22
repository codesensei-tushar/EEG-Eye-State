import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Step 1: Load the ARFF File and Create a DataFrame ---
with open('eeg+eye+state/EEG Eye State.arff', 'r') as f:
    dataset = arff.load(f)

# Get column names from the ARFF attributes
columns = [attr[0] for attr in dataset['attributes']]
# Create DataFrame (each row: 14 EEG channel readings + 1 label)
df = pd.DataFrame(dataset['data'], columns=columns)

# Optional: Show a few rows to verify
print("Dataset preview:")
print(df.shape)
print(df.head())
# print(df.iloc[:, -1].value_counts())
scaler = StandardScaler()
# Assuming df contains the raw data, apply scaling to the 14 EEG channels:
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

class EEGDataset(Dataset):
    def __init__(self, dataframe):
        # Separate features (first 14 columns) and labels (last column)
        self.X = dataframe.iloc[:, :-1].values.astype('float32')
        self.y = dataframe.iloc[:, -1].values.astype('int64')  # Ensure labels are integers

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get the sample and label
        sample = self.X[idx]
        label = self.y[idx]
        # Convert to torch tensors
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

# --- Step 3: Split the Data into Training and Testing Sets ---
temp_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, -1])
train_df, val_df = train_test_split(temp_df, test_size=0.1, random_state=42, stratify=temp_df.iloc[:, -1])

print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))
print("Validation set size:", len(val_df))
# Create Dataset instances for train and test sets
train_dataset = EEGDataset(train_df)
test_dataset = EEGDataset(test_df)
val_dataset = EEGDataset(val_df)
# print(train_df.iloc[:, -1].value_counts())
# print(test_df.iloc[:, -1].value_counts())
# --- Step 4: Create DataLoaders ---
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# --- Example: Iterating Through the DataLoader ---
for batch_idx, (data, target) in enumerate(train_loader):
    print(f"Batch {batch_idx+1} - data shape: {data.shape}, target shape: {target.shape}")
    # Break after first batch for demonstration purposes
    break

# 2. Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df.iloc[:, -1])  # Last column is the target label
plt.title("Class Distribution (Eyes Open vs Closed)")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# 3. EEG Channel Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, :-1].corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("EEG Channel Correlation Matrix")
plt.show()
