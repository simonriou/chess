import pandas as pd

fens_file = '../data.nosync/input/features.csv'  # Path to the first CSV file
evals_file = '../data.nosync/input/scores.csv'  # Path to the second CSV file

# Load the CSV files
fens_df = pd.read_csv(fens_file)
evals_df = pd.read_csv(evals_file)

# Select the first 'm' rows
m = len(evals_df)
fens_subset = fens_df.head(m)

# Combine the 'FEN' and 'evaluation' columns
result_df = pd.DataFrame({
    'FEN': fens_subset['FEN'],
    'eval': evals_df['evaluation']
})

# Save the result to a new CSV file
result_df.to_csv('merged_fens_evals.csv', index=False)

print("CSV file 'merged_fens_evals.csv' has been created successfully.")