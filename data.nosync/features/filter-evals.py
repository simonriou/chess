import pandas as pd
import numpy as np

def filter_eval_indices(df, eval_column='evaluation', threshold=5000, ratio=0.5):
    """
    Returns a list of indices to select a subset of evaluations from the DataFrame
    such that a given ratio are extreme (>|threshold|), with equal number of positive and negative extremes.

    Args:
        df (pd.DataFrame): Input DataFrame.
        eval_column (str): Name of the column containing centipawn evaluations.
        threshold (float): Threshold for considering an evaluation "extreme".
        ratio (float): Desired ratio of extreme evaluations in the final selection.

    Returns:
        list: List of selected indices.
    """
    high = df[df[eval_column] > threshold]
    low = df[df[eval_column] < -threshold]
    neutral = df[(df[eval_column] <= threshold) & (df[eval_column] >= -threshold)]

    max_extreme_count = min(len(high), len(low))
    total_target_size = int((2 * max_extreme_count) / ratio)

    if total_target_size > len(df):
        print("Warning: Not enough data to meet the ratio. Using all symmetric extremes and all available neutral samples.")
        selected_high = high.sample(max_extreme_count)
        selected_low = low.sample(max_extreme_count)
        selected_neutral = neutral
    else:
        neutral_needed = total_target_size - 2 * max_extreme_count
        selected_high = high.sample(max_extreme_count)
        selected_low = low.sample(max_extreme_count)
        selected_neutral = neutral.sample(neutral_needed)

    selected_indices = pd.concat([selected_high, selected_low, selected_neutral]).index.tolist()
    np.random.shuffle(selected_indices)
    return selected_indices

def main():
    df = pd.read_csv('../input/latest_scores.csv')
    df = df[df['evaluation'] != 0]

    # Filter evaluations
    indices_to_keep = filter_eval_indices(df)

    # Create a new DataFrame with the filtered evaluations and their indices
    filtered_df = df.loc[indices_to_keep].reset_index(drop=True)
    filtered_df['index'] = filtered_df.index
    filtered_df['evaluation'] = filtered_df['evaluation'].round(2)
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv('../input/temp/filtered_scores.csv', index=False)


if __name__ == "__main__":
    main()