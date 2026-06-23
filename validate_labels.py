import os
import pandas as pd

INPUT_CSV = "taylor_swift_dataset.csv"

def interactive_validation():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Target file '{INPUT_CSV}' not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    if 'manually_verified' not in df.columns:
        df['manually_verified'] = False
        
    df['manually_verified'] = df['manually_verified'].fillna(False).astype(bool)
    df['label'] = df['label'].fillna("Unlabeled").astype(str).str.strip()
    text_col = 'text' if 'text' in df.columns else None

    unverified_df = df[df['manually_verified'] == False]
    total_unverified = len(unverified_df)
    
    if total_unverified == 0:
        print("🎉 All records have been manually verified!")
        return

    print("="*50)
    print("      INTERACTIVE DATASET VALIDATION LOOP      ")
    print("="*50)
    print(f"Remaining to verify: {total_unverified} / {len(df)} rows")
    print("Instructions:")
    print("  [Press Enter]  -> Accept AI label")
    print("  [Type 1,2,3,4] -> Override with a new category")
    print("  [Type 'q']     -> Save progress and quit safely\n")
    print("="*50 + "\n")

    shortcuts = {
        "1": "Updates",
        "2": "Community & Culture",
        "3": "Poll",
        "4": "Opinion"
    }

    verified_in_this_session = 0

    for idx in unverified_df.index:
        row = df.loc[idx]
        
        print(f"--- [Row Index: {idx}] ---")
        print(f"Title:    {row['title']}")
        
        print("\n--- Post Text Content ---")
        if text_col:
            body_content = str(row[text_col]).strip()
            if body_content and body_content.lower() != 'nan' and body_content != '':
                print(body_content)
            else:
                print("\033[33m[No body text - Title only post]\033[0m")
        print("-------------------------\n")
        
        print(f"AI Label: \033[1;36m{row['label']}\033[0m") 
        
        # Prints shortcuts inline right above the input prompt every single turn
        print("Shortcuts: \033[92m1\033[0m=Updates | \033[92m2\033[0m=Community & Culture | \033[92m3\033[0m=Poll | \033[92m4\033[0m=Opinion")
        
        while True:
            choice = input("Approve or Override? ").strip()
            
            if choice.lower() == 'q':
                print(f"\nExiting safely. Saved {verified_in_this_session} verifications.")
                return
                
            elif choice == "":
                df.at[idx, 'manually_verified'] = True
                verified_in_this_session += 1
                print("✅ Approved.\n")
                break
                
            elif choice in shortcuts:
                new_label = shortcuts[choice]
                df.at[idx, 'label'] = new_label
                df.at[idx, 'manually_verified'] = True
                verified_in_this_session += 1
                print(f"✏️  Overrode to: \033[1;33m{new_label}\033[0m\n")
                break
                
            else:
                print("Invalid option. Press Enter to accept, type 1-4 to override, or 'q' to quit.")

        df.to_csv(INPUT_CSV, index=False, encoding='utf-8')

    print(f"Success! Dataset fully updated!")

if __name__ == "__main__":
    interactive_validation()