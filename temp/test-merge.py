import pandas as pd

# Load the CSV files
features_df = pd.read_csv('../data.nosync/input/final/features.csv')  # contains column 'FEN'
scores_df = pd.read_csv('../data.nosync/input/temp/scores_normalized.csv')  # contains columns 'evaluation' and 'index'
merged_df = pd.read_csv('../data.nosync/input/temp/merged_fens_evals.csv')  # contains columns 'FEN' and 'eval'

# Sanity check: ensure merged_df and scores_df have the same length
if len(scores_df) != len(merged_df):
    print(f"Length mismatch: scores_normalized.csv has {len(scores_df)} rows, "
          f"but merged_fens_evals.csv has {len(merged_df)} rows.")

# Check for mismatches
mismatches = []

for i, (merged_row, score_row) in enumerate(zip(merged_df.itertuples(index=False), scores_df.itertuples(index=False))):
    fen_index = score_row.index
    true_fen = features_df.iloc[fen_index].FEN
    merged_fen = merged_row.FEN
    if true_fen != merged_fen:
        mismatches.append((i, fen_index, merged_fen, true_fen))

# Output results
if mismatches:
    print(f"Found {len(mismatches)} mismatches:")
    for i, idx, merged_fen, true_fen in mismatches[:10]:  # Show only first 10 mismatches
        print(f"Row {i}: Index {idx} -> Merged FEN: {merged_fen} | Expected FEN: {true_fen}")
    if len(mismatches) > 10:
        print("... (more mismatches not shown)")
else:
    print("All FENs in the merged file match the expected FENs.")