import pandas as pd
import numpy as np
import sys

# Load the dataset
file_path = "Crop_Recommendation.csv"
df = pd.read_csv(file_path)

# Open output file
output_file = "step2_data_cleaning.txt"
with open(output_file, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 80)
    print("CROP RECOMMENDATION DATASET - DATA CLEANING REPORT")
    print("=" * 80)
    print()
    
    # Store original shape
    shape_before = df.shape
    print("-" * 80)
    print("DATASET SHAPE BEFORE CLEANING")
    print("-" * 80)
    print(f"Number of Rows: {shape_before[0]}")
    print(f"Number of Columns: {shape_before[1]}")
    print(f"Shape: {shape_before}")
    print()
    
    # Step 1: Remove duplicate rows
    print("-" * 80)
    print("STEP 1: REMOVE DUPLICATE ROWS")
    print("-" * 80)
    duplicate_count_before = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count_before}")
    
    if duplicate_count_before > 0:
        df = df.drop_duplicates()
        print(f"Action: Removed {duplicate_count_before} duplicate rows.")
        print(f"Rows after removal: {df.shape[0]}")
    else:
        print("Action: No duplicate rows to remove.")
    print()
    
    # Step 2: Handle missing values
    print("-" * 80)
    print("STEP 2: HANDLE MISSING VALUES")
    print("-" * 80)
    print("Explanation: Using median to fill missing values because:")
    print("  - Median is robust to outliers (unlike mean)")
    print("  - Preserves the central tendency of the data")
    print("  - Appropriate for numeric agricultural measurements")
    print()
    
    missing_before = df.isnull().sum()
    total_missing_before = missing_before.sum()
    print(f"Missing values per column before cleaning:")
    print(missing_before.to_string())
    print(f"Total missing values: {total_missing_before}")
    
    if total_missing_before > 0:
        # Get numeric columns (exclude the target 'Crop' column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nFilling missing values in numeric columns using median:")
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  - {col}: filled {missing_count} values with median ({median_val})")
        print("\nMissing values after filling:")
        print(df.isnull().sum().to_string())
    else:
        print("\nAction: No missing values to handle.")
    print()
    
    # Step 3: Validate data ranges
    print("-" * 80)
    print("STEP 3: VALIDATE DATA RANGES")
    print("-" * 80)
    
    # Check for negative values
    print("Checking for negative values:")
    columns_to_check = ['Rainfall', 'Temperature', 'Humidity']
    negative_found = False
    
    for col in columns_to_check:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            print(f"  - {col}: {negative_count} negative values")
            if negative_count > 0:
                negative_found = True
                # Remove rows with negative values
                df = df[df[col] >= 0]
                print(f"    Action: Removed {negative_count} rows with negative {col}")
    
    if not negative_found:
        print("  Action: No negative values found in any column.")
    
    # Check pH range (0-14)
    print("\nChecking pH range (0-14):")
    ph_out_of_range = df[(df['pH_Value'] < 0) | (df['pH_Value'] > 14)]
    ph_invalid_count = len(ph_out_of_range)
    print(f"  - pH values outside 0-14 range: {ph_invalid_count}")
    
    if ph_invalid_count > 0:
        print(f"    Action: Removing {ph_invalid_count} rows with invalid pH values")
        df = df[(df['pH_Value'] >= 0) & (df['pH_Value'] <= 14)]
    else:
        print("  Action: All pH values are within valid range (0-14).")
    print()
    
    # Step 4: Convert feature columns to numeric
    print("-" * 80)
    print("STEP 4: CONVERT FEATURE COLUMNS TO NUMERIC")
    print("-" * 80)
    print("Feature columns (excluding target 'Crop'):")
    feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    
    for col in feature_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            new_dtype = df[col].dtype
            
            # Check if any values were coerced to NaN
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                print(f"  - {col}: {original_dtype} -> {new_dtype} (Warning: {nan_count} values converted to NaN)")
                # Fill with median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    Filled NaN values with median: {median_val}")
            else:
                print(f"  - {col}: {original_dtype} -> {new_dtype} (OK)")
    
    print("\nAll feature columns are now numeric.")
    print()
    
    # Final shape after cleaning
    shape_after = df.shape
    print("-" * 80)
    print("DATASET SHAPE AFTER CLEANING")
    print("-" * 80)
    print(f"Number of Rows: {shape_after[0]}")
    print(f"Number of Columns: {shape_after[1]}")
    print(f"Shape: {shape_after}")
    print()
    
    # Summary of changes
    print("-" * 80)
    print("CLEANING SUMMARY")
    print("-" * 80)
    rows_removed = shape_before[0] - shape_after[0]
    print(f"Rows removed: {rows_removed}")
    print(f"Rows changed: {shape_before[0]} -> {shape_after[0]}")
    print(f"Columns unchanged: {shape_before[1]}")
    print()
    
    print("-" * 80)
    print("FINAL DATA TYPES")
    print("-" * 80)
    print(df.dtypes.to_string())
    print()
    
    print("-" * 80)
    print("SAMPLE OF CLEANED DATA (First 5 rows)")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head().to_string())
    print()
    
    print("=" * 80)
    print("END OF DATA CLEANING REPORT")
    print("=" * 80)
    
    sys.stdout = original_stdout

print(f"Data cleaning complete. Output saved to: {output_file}")
