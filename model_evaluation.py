import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys
import os

# Create plots directory if not exists
os.makedirs("plots", exist_ok=True)

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

# Apply StandardScaler correctly
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Task 1: Predict using X_test
y_pred = knn.predict(X_test)

output_file = "step7_model_evaluation.txt"
with open(output_file, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 100)
    print("KNN MODEL EVALUATION REPORT")
    print("=" * 100)
    print()
    
    # Task 2: Compute accuracy
    print("-" * 100)
    print("STEP 1: PREDICTION & ACCURACY")
    print("-" * 100)
    print()
    print("Prediction details:")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Predictions made: {len(y_pred)}")
    print()
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ACCURACY SCORE: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("Interpretation:")
    if accuracy >= 0.95:
        print("  - EXCELLENT: Model correctly predicts crops with very high accuracy")
    elif accuracy >= 0.90:
        print("  - GOOD: Model performs well with reliable predictions")
    elif accuracy >= 0.80:
        print("  - MODERATE: Acceptable performance but room for improvement")
    else:
        print("  - NEEDS IMPROVEMENT: Consider tuning hyperparameters or feature engineering")
    print()
    
    # Confusion Matrix
    print("-" * 100)
    print("STEP 2: CONFUSION MATRIX")
    print("-" * 100)
    print()
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix Shape:", cm.shape)
    print()
    print("Confusion Matrix (Raw Counts):")
    print()
    
    # Print with crop names
    crop_names = [label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))]
    
    # Create DataFrame for better display
    cm_df = pd.DataFrame(cm, index=crop_names, columns=[f"Pred_{name}" for name in crop_names])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(cm_df.to_string())
    print()
    
    # Diagonal analysis
    print("Diagonal Analysis (Correct Predictions):")
    correct_per_class = np.diag(cm)
    for i, (crop, correct) in enumerate(zip(crop_names, correct_per_class)):
        total = np.sum(cm[i, :])
        class_accuracy = correct / total if total > 0 else 0
        print(f"  {crop:<15}: {correct:2}/{total:2} correct ({class_accuracy*100:5.1f}%)")
    print()
    
    # Misclassification analysis
    print("MISCLASSIFICATION PATTERNS:")
    print()
    misclass_count = 0
    for i in range(len(crop_names)):
        for j in range(len(crop_names)):
            if i != j and cm[i, j] > 0:
                misclass_count += cm[i, j]
                actual = crop_names[i]
                predicted = crop_names[j]
                count = cm[i, j]
                print(f"  - {actual} misclassified as {predicted}: {count} case(s)")
    print()
    print(f"Total misclassifications: {misclass_count} out of {len(y_test)} ({misclass_count/len(y_test)*100:.2f}%)")
    print()
    
    # Classification Report
    print("-" * 100)
    print("STEP 3: CLASSIFICATION REPORT")
    print("-" * 100)
    print()
    print("Metrics per class:")
    print("  - Precision: Of all predicted as X, how many were actually X")
    print("  - Recall: Of all actual X, how many were correctly predicted as X")
    print("  - F1-Score: Harmonic mean of precision and recall")
    print()
    
    report = classification_report(y_test, y_pred, target_names=crop_names, digits=4)
    print(report)
    print()
    
    # Task 4 & 5: Compute TP, FP, FN for each class
    print("-" * 100)
    print("STEP 4: TRUE POSITIVES, FALSE POSITIVES, FALSE NEGATIVES")
    print("-" * 100)
    print()
    
    print("Computing for multi-class classification:")
    print("  - TP (True Positive)  = Diagonal element (correctly predicted)")
    print("  - FP (False Positive) = Sum of column excluding diagonal (wrongly predicted as this class)")
    print("  - FN (False Negative) = Sum of row excluding diagonal (actual class missed)")
    print()
    
    # Calculate TP, FP, FN for each class
    tp_fp_fn_data = []
    for i in range(len(crop_names)):
        tp = cm[i, i]  # Diagonal element
        fp = np.sum(cm[:, i]) - tp  # Column sum minus diagonal
        fn = np.sum(cm[i, :]) - tp  # Row sum minus diagonal
        tp_fp_fn_data.append({
            'Class': crop_names[i],
            'TP': tp,
            'FP': fp,
            'FN': fn
        })
    
    # Create structured table
    print("-" * 70)
    print(f"{'Class':<15} | {'True Positive (TP)':<18} | {'False Positive (FP)':<19} | {'False Negative (FN)':<19}")
    print("-" * 70)
    
    for data in tp_fp_fn_data:
        print(f"{data['Class']:<15} | {data['TP']:>18} | {data['FP']:>19} | {data['FN']:>19}")
    
    print("-" * 70)
    print()
    
    # Task 6: Interpretation
    print("-" * 100)
    print("STEP 5: INTERPRETATION OF FP AND FN")
    print("-" * 100)
    print()
    
    print("=" * 80)
    print("WHAT DO FP AND FN MEAN IN CROP PREDICTION?")
    print("=" * 80)
    print()
    
    print("FALSE POSITIVE (FP):")
    print("  Definition: Model predicts a specific crop, but the actual crop is DIFFERENT.")
    print()
    print("  Example from our results:")
    # Find classes with FP > 0
    fp_cases = [(d['Class'], d['FP']) for d in tp_fp_fn_data if d['FP'] > 0]
    if fp_cases:
        for crop, fp_count in fp_cases:
            print(f"    - {crop}: {fp_count} FP cases")
            print(f"      -> Model predicted '{crop}' {fp_count} time(s) when it was actually a different crop")
    print()
    print("  Agricultural Impact:")
    print("    - Farmer might plant wrong crop based on recommendation")
    print("    - Wasted resources (seeds, fertilizer, water)")
    print("    - Potential yield loss due to unsuitable conditions")
    print()
    
    print("FALSE NEGATIVE (FN):")
    print("  Definition: Model FAILS to predict the correct crop (misses the actual crop).")
    print()
    print("  Example from our results:")
    # Find classes with FN > 0
    fn_cases = [(d['Class'], d['FN']) for d in tp_fp_fn_data if d['FN'] > 0]
    if fn_cases:
        for crop, fn_count in fn_cases:
            print(f"    - {crop}: {fn_count} FN cases")
            print(f"      -> Model missed identifying '{crop}' {fn_count} time(s)")
    print()
    print("  Agricultural Impact:")
    print("    - Farmer misses opportunity to plant optimal crop")
    print("    - Suboptimal yield from planting less suitable crop")
    print("    - Reduced profitability for farmer")
    print()
    
    print("=" * 80)
    print("DETAILED ANALYSIS OF FP AND FN CASES")
    print("=" * 80)
    print()
    
    # Analyze specific misclassifications contributing to FP/FN
    print("Classes with False Positives (model over-predicts):")
    for crop, fp_count in fp_cases:
        if fp_count > 0:
            # Find which crops were misclassified as this crop
            idx = crop_names.index(crop)
            contributors = []
            for i in range(len(crop_names)):
                if i != idx and cm[i, idx] > 0:
                    contributors.append((crop_names[i], cm[i, idx]))
            print(f"  - {crop} ({fp_count} FP):")
            for actual_crop, count in contributors:
                print(f"      * {count} case(s) of actual {actual_crop} predicted as {crop}")
    print()
    
    print("Classes with False Negatives (model under-predicts):")
    for crop, fn_count in fn_cases:
        if fn_count > 0:
            # Find what crops this was misclassified as
            idx = crop_names.index(crop)
            misclass_as = []
            for j in range(len(crop_names)):
                if j != idx and cm[idx, j] > 0:
                    misclass_as.append((crop_names[j], cm[idx, j]))
            print(f"  - {crop} ({fn_count} FN):")
            for predicted_crop, count in misclass_as:
                print(f"      * {count} case(s) predicted as {predicted_crop} instead")
    print()
    
    print("=" * 80)
    print("SUMMARY OF FP AND FN ANALYSIS")
    print("=" * 80)
    print()
    
    total_tp = sum(d['TP'] for d in tp_fp_fn_data)
    total_fp = sum(d['FP'] for d in tp_fp_fn_data)
    total_fn = sum(d['FN'] for d in tp_fp_fn_data)
    
    print(f"  Total True Positives (correct):  {total_tp:3d}")
    print(f"  Total False Positives (errors):  {total_fp:3d}")
    print(f"  Total False Negatives (missed):  {total_fn:3d}")
    print(f"  Total predictions:               {total_tp + total_fp:3d}")
    print()
    print(f"  Precision (TP / (TP + FP)):      {total_tp / (total_tp + total_fp):.4f}")
    print(f"  Recall (TP / (TP + FN)):         {total_tp / (total_tp + total_fn):.4f}")
    print()
    
    # Model Performance Summary
    print("-" * 100)
    print("STEP 6: OVERALL MODEL PERFORMANCE")
    print("-" * 100)
    print()
    print("=" * 80)
    print("OVERALL PERFORMANCE ASSESSMENT")
    print("=" * 80)
    print()
    
    # Calculate macro and weighted averages manually for clarity
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"  Overall Accuracy:     {accuracy*100:6.2f}%")
    print(f"  Macro Avg Precision:  {macro_precision*100:6.2f}%")
    print(f"  Macro Avg Recall:     {macro_recall*100:6.2f}%")
    print(f"  Macro Avg F1-Score:   {macro_f1*100:6.2f}%")
    print()
    
    # Identify best and worst performing classes
    class_f1_scores = [(crop_names[i], f1[i]) for i in range(len(crop_names))]
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Best Performing Classes (highest F1-scores):")
    for crop, score in class_f1_scores[:5]:
        print(f"  - {crop:<15}: F1 = {score:.4f}")
    print()
    
    print("Classes Needing Attention (lowest F1-scores):")
    for crop, score in class_f1_scores[-5:]:
        print(f"  - {crop:<15}: F1 = {score:.4f}")
    print()
    
    # Insights and Recommendations
    print("-" * 100)
    print("STEP 7: INSIGHTS & RECOMMENDATIONS")
    print("-" * 100)
    print()
    print("Key Findings:")
    print()
    
    if accuracy > 0.95:
        print("  1. EXCELLENT MODEL PERFORMANCE:")
        print("     - KNN with k=5 achieves >95% accuracy on crop prediction")
        print("     - Feature scaling and selection are effective")
        print()
        print("  2. CROP PREDICTION RELIABILITY:")
        print("     - The model reliably distinguishes between 22 different crops")
        print("     - Agricultural recommendations can be trusted")
        print()
    elif accuracy > 0.90:
        print("  1. GOOD MODEL PERFORMANCE:")
        print("     - KNN with k=5 achieves >90% accuracy")
        print("     - Minor improvements possible through hyperparameter tuning")
        print()
    
    # Check for confused crops
    confused_pairs = []
    for i in range(len(crop_names)):
        for j in range(len(crop_names)):
            if i != j and cm[i, j] >= 2:  # 2 or more misclassifications
                confused_pairs.append((crop_names[i], crop_names[j], cm[i, j]))
    
    if confused_pairs:
        print("  3. FREQUENTLY CONFUSED CROP PAIRS:")
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        for actual, predicted, count in confused_pairs[:5]:
            print(f"     - {actual} <-> {predicted}: {count} misclassifications")
        print("     These crops may have similar soil/climate requirements")
        print()
    
    print("  4. KNN SUITABILITY:")
    print("     - KNN works well for this dataset because:")
    print("       * Clear feature-crop relationships exist")
    print("       * Similar environmental conditions cluster together")
    print("       * 7 features provide sufficient discriminative power")
    print()
    
    print("Recommendations:")
    print("  - For production: Model is ready for crop recommendation system")
    print("  - For improvement: Try k=3 or k=7 to compare performance")
    print("  - For deployment: Include confidence scores with predictions")
    print()
    
    print("=" * 100)
    print("END OF MODEL EVALUATION REPORT")
    print("=" * 100)
    
    sys.stdout = original_stdout

# Task 3: Visualization - Confusion Matrix Heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=crop_names, yticklabels=crop_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - KNN Crop Prediction (k=5)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/06_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Model evaluation complete.")
print(f"Report saved to: {output_file}")
print(f"Confusion matrix heatmap saved to: plots/06_confusion_matrix.png")
print()
print("=" * 60)
print(f"ACCURACY: {accuracy*100:.2f}%")
print("=" * 60)
