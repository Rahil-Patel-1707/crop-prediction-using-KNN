import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sys

# Load the dataset
df = pd.read_csv("Crop_Recommendation.csv")

# Separate features and target
feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
X = df[feature_cols]
y = df['Crop']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded  # Maintain class distribution
)

# Save datasets as CSV
# Convert arrays back to DataFrames for saving
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
y_train_df = pd.DataFrame(y_train, columns=['Crop_Encoded'])
y_test_df = pd.DataFrame(y_test, columns=['Crop_Encoded'])

X_train_df.to_csv("X_train.csv", index=False)
X_test_df.to_csv("X_test.csv", index=False)
y_train_df.to_csv("y_train.csv", index=False)
y_test_df.to_csv("y_test.csv", index=False)

# Generate summary report
output_file = "step5_train_test_split.txt"
with open(output_file, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 100)
    print("TRAIN-TEST SPLIT REPORT")
    print("=" * 100)
    print()
    
    print("-" * 100)
    print("SPLIT CONFIGURATION")
    print("-" * 100)
    print("  Method: sklearn.model_selection.train_test_split")
    print("  Test size: 0.2 (20%)")
    print("  Train size: 0.8 (80%)")
    print("  Random state: 42 (for reproducibility)")
    print("  Stratify: Yes (maintains class distribution)")
    print()
    
    print("-" * 100)
    print("ORIGINAL DATASET")
    print("-" * 100)
    print(f"  Total samples: {len(X_scaled)}")
    print(f"  Features: {X_scaled.shape[1]}")
    print(f"  Classes: {len(np.unique(y_encoded))}")
    print()
    
    print("-" * 100)
    print("TRAINING SET (80%)")
    print("-" * 100)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Samples: {len(X_train)}")
    print()
    
    print("  Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        crop_name = label_encoder.inverse_transform([cls])[0]
        percentage = (count / len(y_train)) * 100
        print(f"    {cls:2} ({crop_name:<15}): {count:3} samples ({percentage:5.2f}%)")
    print()
    
    print("-" * 100)
    print("TESTING SET (20%)")
    print("-" * 100)
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Samples: {len(X_test)}")
    print()
    
    print("  Class distribution in test set:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for cls, count in zip(unique_test, counts_test):
        crop_name = label_encoder.inverse_transform([cls])[0]
        percentage = (count / len(y_test)) * 100
        print(f"    {cls:2} ({crop_name:<15}): {count:3} samples ({percentage:5.2f}%)")
    print()
    
    print("-" * 100)
    print("SPLIT VERIFICATION")
    print("-" * 100)
    print(f"  Total samples (train + test): {len(X_train) + len(X_test)}")
    print(f"  Original total: {len(X_scaled)}")
    print(f"  Match: {'Yes' if len(X_train) + len(X_test) == len(X_scaled) else 'No'}")
    print()
    print(f"  Train ratio: {len(X_train) / len(X_scaled):.2f} (expected: 0.80)")
    print(f"  Test ratio: {len(X_test) / len(X_scaled):.2f} (expected: 0.20)")
    print()
    
    print("-" * 100)
    print("FILES SAVED")
    print("-" * 100)
    print("  1. X_train.csv - Training features (1760 x 7)")
    print("  2. X_test.csv - Testing features (440 x 7)")
    print("  3. y_train.csv - Training labels (1760 x 1)")
    print("  4. y_test.csv - Testing labels (440 x 1)")
    print()
    
    print("-" * 100)
    print("DATA CONSISTENCY CHECK")
    print("-" * 100)
    # Check for any data leakage (should be none)
    overlap = len(set(map(tuple, X_train)) & set(map(tuple, X_test)))
    print(f"  Overlapping samples between train and test: {overlap}")
    print(f"  Status: {'No leakage detected' if overlap == 0 else 'WARNING: Data leakage detected'}")
    print()
    
    # Verify shapes match
    print(f"  X_train shape {X_train.shape} matches y_train shape {y_train.shape}: {'Yes' if X_train.shape[0] == y_train.shape[0] else 'No'}")
    print(f"  X_test shape {X_test.shape} matches y_test shape {y_test.shape}: {'Yes' if X_test.shape[0] == y_test.shape[0] else 'No'}")
    print()
    
    print("=" * 100)
    print("END OF TRAIN-TEST SPLIT REPORT")
    print("=" * 100)
    
    sys.stdout = original_stdout

print(f"Train-test split complete.")
print(f"Report saved to: {output_file}")
print()
print("Dataset shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test:  {y_test.shape}")
print()
print("CSV files saved:")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_train.csv")
print("  - y_test.csv")
