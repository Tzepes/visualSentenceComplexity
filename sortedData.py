import pandas as pd

# Read the CSV file
train_data = './data/train.csv'
train_sorted = './data/train_sorted.csv'
vaildation_data = './data/val.csv'
validation_sorted = './data/val_sorted.csv'

# Load the data into a DataFrame
df_train = pd.read_csv(train_data)
df_validation = pd.read_csv(vaildation_data)

# Sort the DataFrame by a specified column (e.g., 'column_name')
trainSorted_df = df_train.sort_values(by='score')
validationSorted_df = df_validation.sort_values(by='score')

# Write the sorted DataFrame to a new CSV file
trainSorted_df.to_csv(train_sorted, index=False)
validationSorted_df.to_csv(validation_sorted, index=False)

print(f"DataSorted")