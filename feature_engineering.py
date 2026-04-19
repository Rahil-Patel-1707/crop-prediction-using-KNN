import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys

# Load the dataset
df = pd.read_csv("Crop_Recommendation.csv")

output_file = "step4_feature_engineering.txt"
with open(output_file, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 100)
    print("FEATURE ENGINEERING FOR KNN MODEL")
    print("=" * 100)
    print()
    
    # Task 1: Separate X (features) and y (target)
    print("-" * 100)
    print("STEP 1: SEPARATE FEATURES (X) AND TARGET (y)")
    print("-" * 100)
    print()
    
    # Define feature columns (X) and target (y)
    feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    X = df[feature_cols]
    y = df['Crop']
    
    print("Feature columns (X):")
    print(f"  Shape: {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    print()
    
    print("Target column (y):")
    print(f"  Shape: {y.shape}")
    print(f"  Column: Crop")
    print()
    
    print("Sample of X (first 5 rows):")
    print(X.head().to_string())
    print()
    
    print("Sample of y (first 10 values):")
    print(y.head(10).to_list())
    print()
    
    # Task 2: Encode target using LabelEncoder
    print("-" * 100)
    print("STEP 2: TARGET ENCODING WITH LABELENCODER")
    print("-" * 100)
    print()
    print("Explanation: LabelEncoder converts categorical crop names into numeric values.")
    print("This is necessary because KNN (and most ML algorithms) requires numeric inputs.")
    print()
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("Unique crop labels (original):")
    unique_crops = label_encoder.classes_
    for i, crop in enumerate(unique_crops, 1):
        print(f"  {i}. {crop}")
    print(f"\nTotal unique crops: {len(unique_crops)}")
    print()
    
    print("Encoded values mapping:")
    print("  Original Label -> Encoded Value")
    for original, encoded in zip(unique_crops, range(len(unique_crops))):
        print(f"  {original:<12} -> {encoded}")
    print()
    
    print("Sample of encoded y (first 20 values):")
    print(f"  Original: {y.head(20).to_list()}")
    print(f"  Encoded:  {list(y_encoded[:20])}")
    print()
    
    # Task 3: Apply StandardScaler on features
    print("-" * 100)
    print("STEP 3: FEATURE SCALING WITH STANDARDSCALER")
    print("-" * 100)
    print()
    print("=" * 80)
    print("WHY SCALING IS CRITICAL FOR KNN")
    print("=" * 80)
    print()
    print("KNN is a distance-based algorithm. It calculates the distance between")
    print("data points to find the 'k' nearest neighbors. Without scaling, features")
    print("with larger value ranges will dominate the distance calculation.")
    print()
    print("Example from our dataset:")
    print(f"  - Rainfall range: {df['Rainfall'].min():.2f} to {df['Rainfall'].max():.2f}")
    print(f"  - pH_Value range: {df['pH_Value'].min():.2f} to {df['pH_Value'].max():.2f}")
    print(f"  - Nitrogen range: {df['Nitrogen'].min()} to {df['Nitrogen'].max()}")
    print()
    print("Without scaling, Rainfall (range ~278) would dominate over pH (range ~6)")
    print("even if pH is more important for crop prediction.")
    print()
    print("StandardScaler transforms each feature to have:")
    print("  - Mean = 0")
    print("  - Standard Deviation = 1")
    print()
    print("Formula: z = (x - mean) / standard_deviation")
    print()
    print("Benefits:")
    print("  1. All features contribute equally to distance calculations")
    print("  2. Improves KNN performance and convergence")
    print("  3. Makes model less sensitive to outliers in specific features")
    print("  4. Required for meaningful distance comparisons")
    print()
    
    # Apply StandardScaler (fit on full dataset temporarily for understanding)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("-" * 80)
    print("SCALING RESULTS")
    print("-" * 80)
    print()
    
    print("Before Scaling (X):")
    print("  Feature Statistics:")
    stats_before = X.describe().loc[['mean', 'std', 'min', 'max']]
    print(stats_before.to_string())
    print()
    
    print("After Scaling (X_scaled):")
    print("  Feature Statistics:")
    stats_after = X_scaled_df.describe().loc[['mean', 'std', 'min', 'max']]
    print(stats_after.round(6).to_string())
    print()
    
    print("Verification - All means are now ~0 and std ~1:")
    for col in feature_cols:
        mean_val = X_scaled_df[col].mean()
        std_val = X_scaled_df[col].std()
        print(f"  {col:<12}: mean = {mean_val:>10.6f}, std = {std_val:>10.6f}")
    print()
    
    # Task 4: Print feature names, unique crop labels, encoded values
    print("-" * 100)
    print("STEP 4: SUMMARY - FEATURES, LABELS, AND ENCODING")
    print("-" * 100)
    print()
    
    print("FEATURE NAMES (X):")
    print("-" * 40)
    for i, feature in enumerate(feature_cols, 1):
        print(f"  {i}. {feature}")
    print()
    
    print("UNIQUE CROP LABELS (y):")
    print("-" * 40)
    for i, crop in enumerate(unique_crops, 1):
        encoded_val = label_encoder.transform([crop])[0]
        print(f"  {i:2}. {crop:<15} (encoded: {encoded_val})")
    print()
    
    print("ENCODED VALUE DISTRIBUTION:")
    print("-" * 40)
    unique, counts = np.unique(y_encoded, return_counts=True)
    for encoded_val, count in zip(unique, counts):
        crop_name = label_encoder.inverse_transform([encoded_val])[0]
        print(f"  {encoded_val:2} ({crop_name:<15}): {count} samples")
    print()
    
    print("FINAL DATASET SUMMARY FOR KNN:")
    print("-" * 100)
    print(f"  Features (X):           {X_scaled.shape}")
    print(f"  Target (y):             {y_encoded.shape}")
    print(f"  Number of classes:      {len(unique_crops)}")
    print(f"  Features are scaled:    Yes (StandardScaler)")
    print(f"  Target is encoded:      Yes (LabelEncoder)")
    print()
    
    print("=" * 100)
    print("END OF FEATURE ENGINEERING REPORT")
    print("=" * 100)
    
    sys.stdout = original_stdout

print(f"Feature engineering report saved to: {output_file}")
