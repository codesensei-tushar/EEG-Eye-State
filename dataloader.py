import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

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

# --- Step 2: Define a Custom PyTorch Dataset ---
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
train_df, test_df = train_test_split(df, test_size=0.2, random_state=40, stratify=df.iloc[:, -1])
print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))

# Create Dataset instances for train and test sets
train_dataset = EEGDataset(train_df)
test_dataset = EEGDataset(test_df)
# print(train_df.iloc[:, -1].value_counts())
# print(test_df.iloc[:, -1].value_counts())
# --- Step 4: Create DataLoaders ---
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Example: Iterating Through the DataLoader ---
for batch_idx, (data, target) in enumerate(train_loader):
    print(f"Batch {batch_idx+1} - data shape: {data.shape}, target shape: {target.shape}")
    # Break after first batch for demonstration purposes
    break
