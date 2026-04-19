import pandas as pd
import sys

# Load the dataset
file_path = "Crop_Recommendation.csv"
df = pd.read_csv(file_path)

# Open output file
output_file = "step1_data_exploration.txt"
with open(output_file, "w") as f:
    # Redirect stdout to file
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 80)
    print("CROP RECOMMENDATION DATASET - INITIAL DATA EXPLORATION")
    print("=" * 80)
    print()
    
    # 1. First 10 rows
    print("-" * 80)
    print("FIRST 10 ROWS")
    print("-" * 80)
    print(df.head(10).to_string())
    print()
    
    # 2. Last 5 rows
    print("-" * 80)
    print("LAST 5 ROWS")
    print("-" * 80)
    print(df.tail(5).to_string())
    print()
    
    # 3. Dataset Shape
    print("-" * 80)
    print("DATASET SHAPE")
    print("-" * 80)
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print(f"Shape: {df.shape}")
    print()
    
    # 4. Column Names
    print("-" * 80)
    print("COLUMN NAMES")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    print()
    
    # 5. Data Types
    print("-" * 80)
    print("DATA TYPES")
    print("-" * 80)
    print(df.dtypes.to_string())
    print()
    # 10. Missing Values
    print("-" * 80)
    print("MISSING VALUES CHECK")
    print("-" * 80)
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing.to_string())
    print()
    print(f"Total missing values in dataset: {df.isnull().sum().sum()}")
    print()
    
    # 11. Duplicate Rows
    print("-" * 80)
    print("DUPLICATE ROWS CHECK")
    print("-" * 80)
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        print("\nDuplicate rows:")
        print(df[df.duplicated()].to_string())
    else:
        print("No duplicate rows found.")
    print()
    
    print("=" * 80)
    print("END OF DATA EXPLORATION REPORT")
    print("=" * 80)
    
    # Restore stdout
    sys.stdout = original_stdout

print(f"Data exploration complete. Output saved to: {output_file}")
