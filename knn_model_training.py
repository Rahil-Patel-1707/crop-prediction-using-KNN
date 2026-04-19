import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
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

# Split dataset
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded
)

output_file = "step6_model_training.txt"
with open(output_file, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 100)
    print("KNN MODEL TRAINING REPORT")
    print("=" * 100)
    print()
    
    # Task 1: Import and explain KNN
    print("-" * 100)
    print("STEP 1: KNEIGHBORSCLASSIFIER IMPORTED")
    print("-" * 100)
    print()
    print("=" * 80)
    print("HOW KNN WORKS (K-NEAREST NEIGHBORS)")
    print("=" * 80)
    print()
    print("KNN is a distance-based, instance-based learning algorithm:")
    print()
    print("1. NO EXPLICIT TRAINING PHASE:")
    print("   - KNN stores all training data points in memory")
    print("   - It is a 'lazy learner' - no model is built during training")
    print("   - All computation happens at prediction time")
    print()
    print("2. PREDICTION PROCESS:")
    print("   - For a new input, KNN calculates distance to ALL training points")
    print("   - Finds the 'k' closest neighbors (default k=5)")
    print("   - Predicts the majority class among these k neighbors")
    print()
    print("3. DISTANCE METRICS:")
    print("   - Default: Euclidean distance (L2 norm)")
    print("   - Formula: d = sqrt(sum((x2 - x1)^2))")
    print("   - Other options: Manhattan, Minkowski, Hamming")
    print()
    print("4. WHY KNN FOR CROP PREDICTION:")
    print("   - Crop recommendation is based on similar environmental conditions")
    print("   - Similar N, P, K, temperature, humidity, pH, rainfall -> same crop")
    print("   - KNN naturally groups similar agricultural conditions")
    print()
    
    # Task 2: Apply StandardScaler properly
    print("-" * 100)
    print("STEP 2: FEATURE SCALING (StandardScaler)")
    print("-" * 100)
    print()
    print("=" * 80)
    print("WHY SCALING IS MANDATORY FOR KNN")
    print("=" * 80)
    print()
    print("CRITICAL PRINCIPLE: KNN uses distance calculations. Features with")
    print("larger numerical ranges will dominate the distance metric if not scaled.")
    print()
    print("Example from our dataset (BEFORE scaling):")
    print("  - Rainfall:  mean=103.5, std=55.0, range=[20.2, 298.6]")
    print("  - pH_Value:  mean=6.5,   std=0.8,  range=[3.5, 9.9]")
    print()
    print("Problem: A 10-unit change in Rainfall is insignificant,")
    print("         but a 10-unit change in pH is impossible (max range is ~6).")
    print("         Rainfall would DOMINATE the distance calculation!")
    print()
    print("SOLUTION: StandardScaler (z-score normalization)")
    print("  - Formula: z = (x - mean) / std")
    print("  - Result: All features have mean=0, std=1")
    print("  - All features now contribute EQUALLY to distance")
    print()
    print("CORRECT SCALING PROCEDURE (to prevent data leakage):")
    print("  1. Fit scaler ONLY on X_train (learns mean/std from training data)")
    print("  2. Transform X_train (using learned parameters)")
    print("  3. Transform X_test (using SAME parameters from step 1)")
    print("  - NEVER fit on test data (prevents information leakage)")
    print()
    
    # Apply scaler correctly
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print("SCALING RESULTS:")
    print()
    print("  X_train (AFTER scaling):")
    train_means = np.mean(X_train, axis=0)
    train_stds = np.std(X_train, axis=0)
    for i, col in enumerate(feature_cols):
        print(f"    {col:<12}: mean = {train_means[i]:>8.4f}, std = {train_stds[i]:>8.4f}")
    print()
    print("  X_test (AFTER scaling - using train parameters):")
    test_means = np.mean(X_test, axis=0)
    test_stds = np.std(X_test, axis=0)
    for i, col in enumerate(feature_cols):
        print(f"    {col:<12}: mean = {test_means[i]:>8.4f}, std = {test_stds[i]:>8.4f}")
    print()
    
    # Task 3: Initialize model
    print("-" * 100)
    print("STEP 3: MODEL INITIALIZATION")
    print("-" * 100)
    print()
    print("Model: KNeighborsClassifier")
    print("Parameters:")
    print("  - n_neighbors = 5 (k=5, uses 5 nearest neighbors for prediction)")
    print("  - metric = 'minkowski' (default distance metric)")
    print("  - p = 2 (Euclidean distance, L2 norm)")
    print("  - weights = 'uniform' (all neighbors have equal vote)")
    print()
    
    knn = KNeighborsClassifier(n_neighbors=5)
    print("Model initialized successfully.")
    print()
    
    # Task 4: Train model
    print("-" * 100)
    print("STEP 4: MODEL TRAINING")
    print("-" * 100)
    print()
    print("Training data:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - Number of classes: {len(np.unique(y_train))}")
    print()
    
    # Fit the model
    knn.fit(X_train, y_train)
    
    print("Training process:")
    print("  - KNN stores training data in memory (lazy learning)")
    print("  - No model parameters are learned (unlike linear regression, neural networks)")
    print("  - Model is ready for predictions")
    print()
    
    print("Model attributes after fitting:")
    print(f"  - Classes: {knn.classes_}")
    print(f"  - Number of features: {knn.n_features_in_}")
    print(f"  - Effective metric: {knn.effective_metric_}")
    print()
    
    # Task 5: Confirmation
    print("-" * 100)
    print("STEP 5: TRAINING CONFIRMATION")
    print("-" * 100)
    print()
    print("=" * 60)
    print("  KNN MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  - Algorithm: K-Nearest Neighbors (KNN)")
    print(f"  - k (neighbors): 5")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Classes: {len(np.unique(y_train))}")
    print(f"  - Scaling applied: StandardScaler (fit on train, transform both)")
    print()
    print("Next steps:")
    print("  1. Make predictions on X_test")
    print("  2. Evaluate model performance (accuracy, precision, recall)")
    print("  3. Tune hyperparameters (k value, distance metric)")
    print()
    
    print("=" * 100)
    print("END OF MODEL TRAINING REPORT")
    print("=" * 100)
    
    sys.stdout = original_stdout

print(f"Model training complete.")
print(f"Report saved to: {output_file}")
print()
print("=" * 60)
print("KNN MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
