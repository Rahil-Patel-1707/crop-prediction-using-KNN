import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

print("=" * 80)
print("SAVING MODEL COMPONENTS")
print("=" * 80)
print()

# Load dataset
print("Step 1: Loading dataset...")
df = pd.read_csv("Crop_Recommendation.csv")
feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
X = df[feature_cols]
y = df['Crop']
print(f"  Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print()

# Encode target
print("Step 2: Encoding target labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"  Classes: {len(label_encoder.classes_)}")
print(f"  Class names: {list(label_encoder.classes_)}")
print()

# Scale features
print("Step 3: Fitting StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("  Scaler fitted on training data")
print()

# Train KNN model
print("Step 4: Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y_encoded)
print("  Model trained successfully")
print()

# Save components using joblib
print("Step 5: Saving model components with joblib...")
print()

# Save KNN model
model_filename = "knn_model.pkl"
joblib.dump(knn, model_filename)
print(f"  Saved KNN model: {model_filename}")
print(f"    File size: {os.path.getsize(model_filename):,} bytes")

# Save Scaler
scaler_filename = "scaler.pkl"
joblib.dump(scaler, scaler_filename)
print(f"  Saved Scaler: {scaler_filename}")
print(f"    File size: {os.path.getsize(scaler_filename):,} bytes")

# Save LabelEncoder
encoder_filename = "label_encoder.pkl"
joblib.dump(label_encoder, encoder_filename)
print(f"  Saved LabelEncoder: {encoder_filename}")
print(f"    File size: {os.path.getsize(encoder_filename):,} bytes")

print()

# Verify files exist
print("Step 6: Verification...")
files_to_check = [model_filename, scaler_filename, encoder_filename]
all_saved = True

for filename in files_to_check:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"  [OK] {filename} exists ({size:,} bytes)")
    else:
        print(f"  [ERROR] {filename} not found")
        all_saved = False

print()
print("=" * 80)
if all_saved:
    print("ALL MODEL COMPONENTS SAVED SUCCESSFULLY!")
else:
    print("ERROR: Some files were not saved properly")
print("=" * 80)
print()

# Test loading
print("Step 7: Testing model loading...")
try:
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    loaded_encoder = joblib.load(encoder_filename)
    
    # Test prediction with loaded components
    test_input = np.array([[90, 40, 40, 25, 80, 6.5, 200]])
    test_scaled = loaded_scaler.transform(test_input)
    test_pred = loaded_model.predict(test_scaled)
    test_crop = loaded_encoder.inverse_transform(test_pred)
    
    print(f"  [OK] Model loaded successfully")
    print(f"  [OK] Scaler loaded successfully")
    print(f"  [OK] LabelEncoder loaded successfully")
    print(f"  [OK] Test prediction: {test_crop[0]}")
    print()
    print("=" * 80)
    print("VERIFICATION SUCCESSFUL - MODEL IS READY FOR DEPLOYMENT")
    print("=" * 80)
    
except Exception as e:
    print(f"  [ERROR] Failed to load or test model: {e}")
    print()
    print("=" * 80)
    print("VERIFICATION FAILED")
    print("=" * 80)

print()
print("Saved files:")
print(f"  1. {model_filename} - Trained KNN classifier")
print(f"  2. {scaler_filename} - StandardScaler for feature preprocessing")
print(f"  3. {encoder_filename} - LabelEncoder for crop name encoding")
print()
print("Usage in production:")
print("  import joblib")
print("  model = joblib.load('knn_model.pkl')")
print("  scaler = joblib.load('scaler.pkl')")
print("  encoder = joblib.load('label_encoder.pkl')")
print()
