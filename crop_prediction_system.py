import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import sys

# Global variables for model components
_scaler = None
_label_encoder = None
_model = None
_feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

def train_and_save_model():
    """Train the KNN model and save components for reuse."""
    global _scaler, _label_encoder, _model
    
    # Load dataset
    df = pd.read_csv("Crop_Recommendation.csv")
    
    # Separate features and target
    X = df[_feature_cols]
    y = df['Crop']
    
    # Encode target
    _label_encoder = LabelEncoder()
    y_encoded = _label_encoder.fit_transform(y)
    
    # Scale features
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)
    
    # Train KNN model
    _model = KNeighborsClassifier(n_neighbors=5)
    _model.fit(X_scaled, y_encoded)
    
    return _scaler, _label_encoder, _model

def predict_crop(input_data):
    """
    Predict the recommended crop based on input environmental conditions.
    
    Parameters:
        input_data (list or array): [Nitrogen, Phosphorus, Potassium, Temperature, 
                                    Humidity, pH_Value, Rainfall]
    
    Returns:
        str: Recommended crop name
    
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If model is not trained
    """
    global _scaler, _label_encoder, _model
    
    # Check if model components are initialized
    if _scaler is None or _label_encoder is None or _model is None:
        raise RuntimeError("Model not trained. Call train_and_save_model() first.")
    
    # Validate input
    if not isinstance(input_data, (list, tuple, np.ndarray)):
        raise ValueError(f"Input must be a list, tuple, or numpy array. Got: {type(input_data)}")
    
    if len(input_data) != 7:
        raise ValueError(f"Input must have exactly 7 values (one per feature). Got: {len(input_data)}")
    
    # Check for valid numeric values
    try:
        input_numeric = [float(x) for x in input_data]
    except (ValueError, TypeError) as e:
        raise ValueError(f"All input values must be numeric. Error: {e}")
    
    # Check for negative values where inappropriate
    for i, (feature, value) in enumerate(zip(_feature_cols, input_numeric)):
        if value < 0:
            raise ValueError(f"{feature} cannot be negative. Got: {value}")
    
    # Check pH is within valid range
    ph_index = _feature_cols.index('pH_Value')
    if not (0 <= input_numeric[ph_index] <= 14):
        raise ValueError(f"pH_Value must be between 0 and 14. Got: {input_numeric[ph_index]}")
    
    # Step 1: Convert input to DataFrame with proper column names
    input_df = pd.DataFrame([input_numeric], columns=_feature_cols)
    
    # Step 2: Apply the same scaler used during training
    input_scaled = _scaler.transform(input_df)
    
    # Step 3: Predict using trained model
    prediction_encoded = _model.predict(input_scaled)
    
    # Step 4: Decode prediction back to crop name
    prediction_label = _label_encoder.inverse_transform(prediction_encoded)
    
    return prediction_label[0]

def predict_with_confidence(input_data):
    """
    Predict crop with confidence score and nearest neighbors info.
    
    Parameters:
        input_data (list or array): [Nitrogen, Phosphorus, Potassium, Temperature, 
                                    Humidity, pH_Value, Rainfall]
    
    Returns:
        dict: Contains 'crop', 'confidence', and 'nearest_neighbors'
    """
    global _scaler, _label_encoder, _model
    
    if _scaler is None or _label_encoder is None or _model is None:
        raise RuntimeError("Model not trained. Call train_and_save_model() first.")
    
    # Validate input
    if len(input_data) != 7:
        raise ValueError(f"Input must have exactly 7 values. Got: {len(input_data)}")
    
    # Convert and scale input
    input_df = pd.DataFrame([input_data], columns=_feature_cols)
    input_scaled = _scaler.transform(input_df)
    
    # Get prediction
    prediction_encoded = _model.predict(input_scaled)[0]
    prediction_label = _label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get prediction probabilities (confidence)
    probabilities = _model.predict_proba(input_scaled)[0]
    confidence = probabilities[prediction_encoded] * 100
    
    # Get nearest neighbors info
    distances, indices = _model.kneighbors(input_scaled)
    
    return {
        'crop': prediction_label,
        'confidence': round(confidence, 2),
        'nearest_distances': distances[0].tolist(),
        'top_3_alternatives': get_top_alternatives(probabilities, prediction_encoded)
    }

def get_top_alternatives(probabilities, predicted_class, n=3):
    """Get top N alternative crop predictions with their probabilities."""
    global _label_encoder
    
    # Get indices of top probabilities (excluding the predicted class)
    sorted_indices = np.argsort(probabilities)[::-1]
    alternatives = []
    
    for idx in sorted_indices:
        if idx != predicted_class and len(alternatives) < n:
            crop_name = _label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx] * 100
            alternatives.append((crop_name, round(prob, 2)))
    
    return alternatives

def main():
    """Main function to demonstrate the prediction system."""
    
    print("=" * 80)
    print("CROP PREDICTION SYSTEM")
    print("=" * 80)
    print()
    
    # Step 1: Train the model
    print("Step 1: Training KNN model...")
    train_and_save_model()
    print("Model trained successfully!")
    print()
    
    # Step 2: Example prediction
    print("Step 2: Making predictions with example inputs...")
    print()
    
    # Example 1: Rice conditions (from dataset pattern)
    example_1 = [90, 40, 40, 25, 80, 6.5, 200]
    print(f"Example 1 Input: {example_1}")
    print(f"  Features: N={example_1[0]}, P={example_1[1]}, K={example_1[2]}, "
          f"Temp={example_1[3]}°C, Humidity={example_1[4]}%, pH={example_1[5]}, Rainfall={example_1[6]}mm")
    
    try:
        result_1 = predict_crop(example_1)
        print(f"  Recommended Crop: {result_1}")
        
        # Detailed prediction with confidence
        detailed = predict_with_confidence(example_1)
        print(f"  Confidence: {detailed['confidence']}%")
        print(f"  Top alternatives: {detailed['top_3_alternatives']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 2: Different conditions (likely different crop)
    example_2 = [20, 20, 20, 30, 60, 7.0, 100]
    print(f"Example 2 Input: {example_2}")
    print(f"  Features: N={example_2[0]}, P={example_2[1]}, K={example_2[2]}, "
          f"Temp={example_2[3]}°C, Humidity={example_2[4]}%, pH={example_2[5]}, Rainfall={example_2[6]}mm")
    
    try:
        result_2 = predict_crop(example_2)
        print(f"  Recommended Crop: {result_2}")
        
        detailed = predict_with_confidence(example_2)
        print(f"  Confidence: {detailed['confidence']}%")
        print(f"  Top alternatives: {detailed['top_3_alternatives']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 3: Another set of conditions
    example_3 = [40, 60, 30, 20, 90, 5.5, 250]
    print(f"Example 3 Input: {example_3}")
    print(f"  Features: N={example_3[0]}, P={example_3[1]}, K={example_3[2]}, "
          f"Temp={example_3[3]}°C, Humidity={example_3[4]}%, pH={example_3[5]}, Rainfall={example_3[6]}mm")
    
    try:
        result_3 = predict_crop(example_3)
        print(f"  Recommended Crop: {result_3}")
        
        detailed = predict_with_confidence(example_3)
        print(f"  Confidence: {detailed['confidence']}%")
        print(f"  Top alternatives: {detailed['top_3_alternatives']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Step 3: Error handling demonstration
    print("Step 3: Error handling demonstration...")
    print()
    
    # Test invalid input
    print("Testing invalid inputs:")
    
    # Test 1: Wrong number of features
    try:
        predict_crop([1, 2, 3])  # Only 3 values instead of 7
    except ValueError as e:
        print(f"  Wrong input size: {e}")
    
    # Test 2: Negative value
    try:
        predict_crop([90, 40, 40, 25, 80, 6.5, -100])  # Negative rainfall
    except ValueError as e:
        print(f"  Negative value: {e}")
    
    # Test 3: Invalid pH
    try:
        predict_crop([90, 40, 40, 25, 80, 15.0, 200])  # pH > 14
    except ValueError as e:
        print(f"  Invalid pH: {e}")
    
    # Test 4: Non-numeric input
    try:
        predict_crop([90, 40, 40, 25, 80, "invalid", 200])
    except ValueError as e:
        print(f"  Non-numeric: {e}")
    
    print()
    print("=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
    print()
    print("Usage:")
    print("  from crop_prediction_system import train_and_save_model, predict_crop")
    print("  train_and_save_model()")
    print("  result = predict_crop([90, 40, 40, 25, 80, 6.5, 200])")
    print("  print(f'Recommended Crop: {result}')")
    print()

if __name__ == "__main__":
    # Save output to file
    output_file = "step8_prediction_system.txt"
    
    class Tee:
        """Class to write to both stdout and file."""
        def __init__(self, stdout, file):
            self.stdout = stdout
            self.file = file
        
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
        
        def flush(self):
            self.stdout.flush()
            self.file.flush()
    
    with open(output_file, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f)
        
        main()
        
        sys.stdout = original_stdout
    
    print(f"\nOutput also saved to: {output_file}")
