import pandas as pd

fens_file = '../data.nosync/input/final/features.csv'
evals_file = '../data.nosync/input/temp/scores_normalized.csv'

# Load the CSV files
fens_df = pd.read_csv(fens_file)
evals_df = pd.read_csv(evals_file)

# Extract indices and evaluations
selected_indices = evals_df['index']
evaluations = evals_df['evaluation'].values

# Select the corresponding FENs
fens_subset = fens_df.loc[selected_indices].reset_index(drop=True)

# Combine the 'FEN' and 'evaluation' columns
result_df = pd.DataFrame({
    'FEN': fens_subset['FEN'],
    'eval': evaluations
})

# Save the result to a new CSV file
result_df.to_csv('merged_fens_evals.csv', index=False)

print("CSV file 'merged_fens_evals.csv' has been created successfully.")