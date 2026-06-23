import pandas as pd

# Load the existing dataset
df = pd.read_csv('taylor_swift_dataset.csv')

# Combine 'title' and 'text' into a single 'text' column, handling potential NaN values in 'text'
df['text'] = df['title'].astype(str) + ' ' + df['text'].fillna('')

# Retain only the necessary columns 'text' and 'label'
df_final = df[['text', 'label']].copy()

# Add the 'notes' column as required
df_final['notes'] = ''

# Overwrite or save as a new file. The user said: 
# "Save your collected examples in a CSV file... The notebook handles... This file goes in your repo."
# and previously "save another file in this new format, don't overwrite existing file, it's good to have"
# So I will save this to 'taylor_swift_labeled_data_final.csv'
df_final.to_csv('taylor_swift_labeled_data_final.csv', index=False)

print("File 'taylor_swift_labeled_data_final.csv' saved successfully.")
print(df_final.head())