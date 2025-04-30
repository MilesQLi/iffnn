import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Import the IFFNN class
from iffnn import IFFNN

# --- 1. Generate Synthetic Data ---
n_samples = 1000
n_features = 20
n_classes = 3 # Try 2 for binary classification

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=10, # Number of informative features
    n_redundant=5,
    n_repeated=0,
    n_classes=n_classes,
    n_clusters_per_class=1,
    random_state=42
)

# --- 2. Preprocess Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create feature names (optional, but good for explanation)
feature_names = [f'feat_{i}' for i in range(n_features)]

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Convert to PyTorch Tensors
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train) # Keep as Long or Float depending on loss
X_valid_t = torch.from_numpy(X_valid).float()
y_valid_t = torch.from_numpy(y_valid)
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train_t, y_train_t)
valid_dataset = TensorDataset(X_valid_t, y_valid_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

my_class_names = [f'Type {chr(ord("A")+i)}' for i in range(n_classes)] # e.g., ['Type A', 'Type B', 'Type C']

# --- 3. Initialize IFFNN Model ---
input_size = n_features

# Example: Multi-class (n_classes = 3)
# Use default hidden layers, provide feature names

model = IFFNN(
    input_size=input_size,
    num_classes=n_classes,
    feature_names=feature_names,
    class_names=my_class_names, # Pass the defined class names
    hidden_sizes=None,
    activation='relu',
    device='auto'
)


# Example: Binary classification (if n_classes was 2, you'd set num_classes=1 here)
# if n_classes == 2:
#     # Convert y labels to 0 and 1 if they aren't already
#     # y_train_binary = torch.from_numpy((y_train > 0).astype(int)) # Example conversion
#     # y_valid_binary = ... etc ...
#     # Create new TensorDatasets and DataLoaders with binary labels
#     model = IFFNN(
#         input_size=input_size,
#         num_classes=1, # Crucial for binary setup with BCEWithLogitsLoss
#         feature_names=feature_names,
#         hidden_sizes=[30, 15, 15, 30] # Custom hidden layers
#     )

print(model)

# --- 4. Train the Model ---
history = model.train_model(
    train_loader=train_loader,
    valid_loader=valid_loader,
    num_epochs=30, # Adjust as needed
    learning_rate=0.001,
    save_path='best_iffnn_model.pth', # Optional: saves the best model
    print_every=5 # Print progress every 5 epochs
)

# --- 5. Evaluate the Model ---

model.evaluate_model(test_loader)

# Or you can evaluate manually:
model.eval()
n_correct = 0
n_samples = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        predictions = model.predict(batch_x) # Gets class labels
        n_correct += (predictions == batch_y).sum().item()
        n_samples += batch_y.size(0)

test_accuracy = 100.0 * n_correct / n_samples
print(f"\nTest Accuracy: {test_accuracy:.2f}%")


# --- 6. Get Explanations ---
print("\n--- Explaining first 2 samples from the test set ---")
# Get a small batch from the test set
test_iter = iter(test_loader)
x_sample_batch, y_sample_batch = next(test_iter)

# Explain the batch (e.g., first 2 samples)
explanations = model.explain(
    x_sample_batch[:2], # Explain first 2 samples
    top_n=5,          # Show top 5 features
    print_output=True # Print explanations to console
)

# You can also access the explanation data programmatically:
# print("\n--- Programmatic access to explanation data: ---")
# first_sample_explanation = explanations[0]
# print("Sample 0, Features:", first_sample_explanation['features'])
# print("Sample 0, Top features for Class 0:", first_sample_explanation['classes']['class_0'])