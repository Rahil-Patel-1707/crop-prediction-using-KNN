import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .info-box {
        background-color: #000000;
        color: #ffffff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-label {
        font-weight: bold;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>Crop Recommendation System</h1>", unsafe_allow_html=True)

# Information box
st.markdown("""
    <div class="info-box">
        <strong>About this system:</strong><br>
        Enter the soil and environmental conditions below to get the most suitable crop recommendation.
        This system uses a K-Nearest Neighbors (KNN) machine learning model trained on 2,200 crop samples.
    </div>
""", unsafe_allow_html=True)

# Load model components
@st.cache_resource
def load_model_components():
    """Load and cache the model components."""
    try:
        model = joblib.load("knn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoder = joblib.load("label_encoder.pkl")
        return model, scaler, encoder, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, encoder, error = load_model_components()

if error:
    st.markdown(f"""
        <div class="error-box">
            <strong> Error loading model:</strong><br>
            {error}<br>
            Please ensure the model files (knn_model.pkl, scaler.pkl, label_encoder.pkl) are in the current directory.
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Create input form
st.markdown("### Enter Soil and Environmental Conditions")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("<span class='feature-label'>🧪 Soil Nutrients (N-P-K)</span>", unsafe_allow_html=True)
    # Sliders improve UX by providing visual feedback and preventing invalid inputs
    # Range based on typical agricultural soil test values
    nitrogen = st.slider(
        "Nitrogen (N) - kg/ha",
        min_value=0,
        max_value=140,
        value=70,
        help="Amount of Nitrogen in soil (0-140 kg/ha). Typical range: 20-100"
    )
    
    phosphorus = st.slider(
        "Phosphorus (P) - kg/ha",
        min_value=0,
        max_value=150,
        value=75,
        help="Amount of Phosphorus in soil (0-150 kg/ha). Typical range: 10-80"
    )
    
    potassium = st.slider(
        "Potassium (K) - kg/ha",
        min_value=0,
        max_value=200,
        value=100,
        help="Amount of Potassium in soil (0-200 kg/ha). Typical range: 20-120"
    )

with col2:
    st.markdown("<span class='feature-label'>🌤️ Environmental Conditions</span>", unsafe_allow_html=True)
    # Slider ranges based on global agricultural climate data
    temperature = st.slider(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0,
        step=0.1,
        help="Average temperature in Celsius (0-50°C). Most crops: 15-35°C"
    )
    
    humidity = st.slider(
        "Humidity (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Relative humidity percentage (0-100%). Most crops: 50-85%"
    )
    
    ph = st.slider(
        "pH Value",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1,
        help="Soil pH level (0-14, optimal range 5.5-8.5 for most crops)"
    )
    
    rainfall = st.slider(
        "Rainfall (mm)",
        min_value=0,
        max_value=300,
        value=150,
        help="Annual rainfall in millimeters (0-300mm per growing season). Typical: 50-250mm"
    )

# Display current input summary
st.markdown("---")
st.markdown("#### Current Input Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric("Nitrogen", f"{nitrogen:.0f}")
with summary_col2:
    st.metric("Phosphorus", f"{phosphorus:.0f}")
with summary_col3:
    st.metric("Potassium", f"{potassium:.0f}")
with summary_col4:
    st.metric("pH", f"{ph:.1f}")

summary_col5, summary_col6, summary_col7 = st.columns(3)

with summary_col5:
    st.metric("Temperature", f"{temperature:.1f}°C")
with summary_col6:
    st.metric("Humidity", f"{humidity:.0f}%")
with summary_col7:
    st.metric("Rainfall", f"{rainfall:.0f}mm")

# Prediction button
st.markdown("---")

if st.button(" Predict Crop", key="predict_button"):
    # Collect input data
    input_data = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
    
    # Validate input
    errors = []
    
    # Validation - sliders inherently prevent out-of-range values
    # These checks serve as additional safety for any edge cases
    if nitrogen < 0 or nitrogen > 140:
        errors.append("Nitrogen must be between 0 and 140")
    if phosphorus < 0 or phosphorus > 150:
        errors.append("Phosphorus must be between 0 and 150")
    if potassium < 0 or potassium > 200:
        errors.append("Potassium must be between 0 and 200")
    if temperature < 0 or temperature > 50:
        errors.append("Temperature must be between 0 and 50")
    if humidity < 0 or humidity > 100:
        errors.append("Humidity must be between 0 and 100")
    if ph < 0 or ph > 14:
        errors.append("pH must be between 0 and 14")
    if rainfall < 0 or rainfall > 300:
        errors.append("Rainfall must be between 0 and 300")
    
    if errors:
        error_msg = "<br>".join([f"• {e}" for e in errors])
        st.markdown(f"""
            <div class="error-box">
                <strong>Invalid Input:</strong><br>
                {error_msg}
            </div>
        """, unsafe_allow_html=True)
    else:
        try:
            # Prepare input for prediction
            feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
            input_df = pd.DataFrame([input_data], columns=feature_cols)
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction_encoded = model.predict(input_scaled)
            prediction_label = encoder.inverse_transform(prediction_encoded)
            predicted_crop = prediction_label[0]
            
            # Get prediction probabilities for confidence
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = probabilities[prediction_encoded[0]] * 100
            
            # Get top 3 alternatives
            top_indices = np.argsort(probabilities)[::-1][:4]  # Top 4 (including prediction)
            alternatives = []
            for idx in top_indices:
                if idx != prediction_encoded[0]:
                    crop_name = encoder.inverse_transform([idx])[0]
                    prob = probabilities[idx] * 100
                    alternatives.append((crop_name, prob))
                    if len(alternatives) >= 3:
                        break
            
            # Display result
            st.markdown(f"""
                <div class="result-box">
                    <h2 style="color: #2e7d32; margin: 0;">Recommended Crop: {predicted_crop}</h2>
                    <p style="font-size: 18px; margin: 10px 0;">
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
                <div class="error-box">
                    <strong>Prediction Error:</strong><br>
                    An error occurred during prediction: {str(e)}
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
        <p>Model: K-Nearest Neighbors (k=5) | Accuracy: 97.95%</p>
    </div>
""", unsafe_allow_html=True)
