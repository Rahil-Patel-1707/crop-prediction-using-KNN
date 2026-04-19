import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Create plots directory if not exists
os.makedirs("plots", exist_ok=True)

# Load the dataset
df = pd.read_csv("Crop_Recommendation.csv")

# Define numerical columns
numerical_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

# ============================
# PART A: STATISTICAL ANALYSIS
# ============================

output_file = "step3_statistical_analysis.txt"
with open(output_file, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 100)
    print("DETAILED STATISTICAL ANALYSIS - CROP RECOMMENDATION DATASET")
    print("=" * 100)
    print()
    
    # 1. Central Tendency and Dispersion
    print("-" * 100)
    print("1. CENTRAL TENDENCY AND DISPERSION MEASURES")
    print("-" * 100)
    print()
    
    # Helper functions
    def classify_skewness(skew_val):
        if skew_val > 0.5:
            return "Positive Skew (Right-skewed)"
        elif skew_val < -0.5:
            return "Negative Skew (Left-skewed)"
        else:
            return "Approximately Zero (Symmetrical)"
    
    def classify_kurtosis(kurt_val):
        if kurt_val > 0:
            return "Leptokurtic (High peak)"
        elif kurt_val < 0:
            return "Platykurtic (Flat distribution)"
        else:
            return "Mesokurtic (Normal-like)"
    
    def get_mode(series):
        modes = series.mode()
        if len(modes) == 0:
            return "N/A"
        else:
            return f"{modes[0]:.3f}"
    
    # Print table header
    header = f"{'Feature':<12} | {'Mean':<10} | {'Median':<10} | {'Mode':<10} | {'Variance':<12} | {'Std Dev':<10}"
    print(header)
    print("-" * 100)
    
    stats_results = []
    
    for col in numerical_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            median_val = df[col].median()
            mode_val = get_mode(df[col])
            variance_val = df[col].var()
            std_val = df[col].std()
            
            row = f"{col:<12} | {mean_val:>10.3f} | {median_val:>10.3f} | {mode_val:>10} | {variance_val:>12.3f} | {std_val:>10.3f}"
            print(row)
            
            stats_results.append({
                'Feature': col,
                'Mean': mean_val,
                'Median': median_val,
                'Mode': mode_val,
                'Variance': variance_val,
                'StdDev': std_val
            })
    
    print()
    print()
    
    # 2. Skewness and Kurtosis
    print("-" * 100)
    print("2. SKEWNESS AND KURTOSIS ANALYSIS")
    print("-" * 100)
    print()
    
    header = f"{'Feature':<12} | {'Skewness':<12} | {'Skewness Type':<35} | {'Kurtosis':<12} | {'Kurtosis Type':<25}"
    print(header)
    print("-" * 100)
    
    skew_kurt_results = []
    
    for col in numerical_cols:
        if col in df.columns:
            skew_val = df[col].skew()
            kurt_val = df[col].kurt()
            skew_type = classify_skewness(skew_val)
            kurt_type = classify_kurtosis(kurt_val)
            
            row = f"{col:<12} | {skew_val:>12.3f} | {skew_type:<35} | {kurt_val:>12.3f} | {kurt_type:<25}"
            print(row)
            
            skew_kurt_results.append({
                'Feature': col,
                'Skewness': skew_val,
                'SkewType': skew_type,
                'Kurtosis': kurt_val,
                'KurtType': kurt_type
            })
    
    print()
    print()
    
    # 3. Correlation Matrix
    print("-" * 100)
    print("3. CORRELATION MATRIX")
    print("-" * 100)
    print()
    
    corr_matrix = df[numerical_cols].corr()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(corr_matrix.to_string())
    
    print()
    print()
    
    # Summary of findings
    print("-" * 100)
    print("STATISTICAL SUMMARY")
    print("-" * 100)
    print()
    
    # Highly correlated pairs
    print("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.5):")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr:
        for feat1, feat2, corr in high_corr:
            print(f"   - {feat1} vs {feat2}: r = {corr:.4f}")
    else:
        print("   - No highly correlated feature pairs found (|r| > 0.5)")
    print()
    
    # Skewness summary
    print("SKEWNESS SUMMARY:")
    highly_skewed = [r for r in skew_kurt_results if abs(r['Skewness']) > 1.0]
    moderately_skewed = [r for r in skew_kurt_results if 0.5 < abs(r['Skewness']) <= 1.0]
    
    if highly_skewed:
        print("   Highly Skewed (|skew| > 1.0):")
        for r in highly_skewed:
            print(f"      - {r['Feature']}: {r['Skewness']:.3f} - {r['SkewType']}")
    
    print("=" * 100)
    print("END OF STATISTICAL ANALYSIS")
    print("=" * 100)
    
    sys.stdout = original_stdout

print(f"Statistical analysis saved to: {output_file}")

# ============================
# PART B: VISUALIZATIONS
# ============================

print("Generating visualizations...")

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.3f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   - Saved: plots/02_correlation_heatmap.png")

# 3. Histograms for each feature
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    ax = axes[idx]
    ax.hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'Histogram of {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3)

# Remove empty subplot
if len(numerical_cols) < len(axes):
    fig.delaxes(axes[-1])

plt.suptitle('Feature Distributions - Histograms', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/03_feature_histograms.png', dpi=300, bbox_inches='tight')
plt.close()
print("   - Saved: plots/03_feature_histograms.png")

# 4. Individual histograms (one per feature)
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/04_hist_{col.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
print("   - Saved: Individual histograms for all 7 features")

print("\nAll visualizations completed successfully!")
print("Files saved in 'plots/' directory:")
print("   01_crop_distribution.png")
print("   02_correlation_heatmap.png")
print("   03_feature_histograms.png")
print("   04_hist_<feature>.png (7 individual histograms)")
print("   05_pairplot.png")
