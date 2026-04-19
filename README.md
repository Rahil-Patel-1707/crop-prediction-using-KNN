# 🌱 Crop Recommendation System using KNN

## 📌 Project Overview

The **Crop Recommendation System** is a machine learning-based application that predicts the most suitable crop to grow based on soil and environmental conditions. The model is built using the **K-Nearest Neighbors (KNN)** algorithm, which classifies crops based on similarity with historical agricultural data.

This project demonstrates a complete **data science pipeline**, including data preprocessing, exploratory data analysis, model training, evaluation, and deployment using an interactive user interface.

---

## 🎯 Objective

To assist farmers and agricultural planners in selecting the most appropriate crop by analyzing key factors such as:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH level
* Rainfall

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Streamlit
* Joblib

---

## 🧠 Machine Learning Model

* Algorithm: **K-Nearest Neighbors (KNN)**
* Reason:

  * Simple and effective for classification problems
  * Works well with structured agricultural data
  * Uses distance-based learning for prediction

---

## 🔍 Features

* 📊 Data Cleaning and Preprocessing
* 📈 Exploratory Data Analysis with visualizations
* 🤖 Model Training using KNN
* 📉 Model Evaluation (Accuracy, Confusion Matrix, FP/FN analysis)
* 🌾 Real-time Crop Prediction
* 💾 Model Saving for reuse
* 🖥️ Interactive UI using Streamlit

## 🚀 How to Run the Project

1. Clone the repository:

```
git clone <your-repo-link>
cd Crop_Recommendation_KNN
```

2. Run the Streamlit app:

```
streamlit run app.py
```

---

## 🧪 Sample Input

```
N = 90  
P = 40  
K = 40  
Temperature = 25°C  
Humidity = 80%  
pH = 6.5  
Rainfall = 200 mm  
```

## ✅ Output

```
Recommended Crop: Rice
```

---

## 📊 Model Performance

* High accuracy achieved using KNN
* Evaluation includes:

  * Confusion Matrix
  * Precision, Recall, F1-score
  * False Positive & False Negative analysis

---

## 💡 Future Improvements

* Add multiple model comparison (SVM, Random Forest)
* Deploy as a web application
* Integrate real-time weather API
* Improve UI/UX design

---

## 📜 License

This project is for educational purposes.

---

## 👨‍💻 Author

Developed as part of a Data Science academic project.

