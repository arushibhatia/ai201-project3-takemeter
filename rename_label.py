import pandas as pd

# Load the file to inspect the labels and update them
# I am assuming the file is "taylor_swift_labeled_data_final.csv" based on previous turns.
df = pd.read_csv('taylor_swift_labeled_data_final.csv')

# Show the unique labels to confirm what needs changing
unique_labels = df['label'].unique()
print(f"Current labels: {unique_labels}")

# Map 'Community & Culture' to 'Community'
df['label'] = df['label'].replace('Community & Culture', 'Community')

# Save the updated file
df.to_csv('taylor_swift_labeled_data_final_updated.csv', index=False)

# Verify the update
updated_unique_labels = df['label'].unique()
print(f"Updated labels: {updated_unique_labels}")