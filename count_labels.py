import os
import pandas as pd

INPUT_CSV = "taylor_swift_labeled_data_final_updated_with_addnl_data.csv"

def print_label_breakdown():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Target file '{INPUT_CSV}' not found in the current directory.")
        return

    # Load dataset
    df = pd.read_csv(INPUT_CSV)
    
    # Check if 'label' column exists
    if 'label' not in df.columns:
        print("❌ Error: The 'label' column does not exist in this CSV yet.")
        return
        
    # Clean up column data types and fill missing spaces with 'Unlabeled'
    df['label'] = df['label'].fillna("Unlabeled").astype(str).str.strip()
    df.loc[df['label'] == "", 'label'] = "Unlabeled"

    total_records = len(df)
    
    # Compute value counts
    counts = df['label'].value_counts()
    
    print("\n" + "="*40)
    print("      DATASET LABEL DISTRIBUTION      ")
    print("="*40)
    print(f"Total Rows in Dataset: {total_records}\n")
    
    # Display the structured counts table with percentages
    print(f"{'Label Category':<22} | {'Count':<6} | {'Percentage':<10}")
    print("-" * 45)
    
    for category, count in counts.items():
        percentage = (count / total_records) * 100
        print(f"{category:<22} | {count:<6} | {percentage:>5.1f}%")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    print_label_breakdown()