import pandas as pd
import numpy as np
import os
import tqdm

def normalize_eval(eval_cp, max_cp=12000, mate_score=100000):
    """
    Maps centipawn and mate scores to the range [-1.0, 1.0].
    - Centipawn scores are scaled linearly to [-0.9, 0.9]
    - Mate scores (+/- 100000) are mapped to +/- 1.0
    """
    if eval_cp >= mate_score:
        return 1.0
    elif eval_cp <= -mate_score:
        return -1.0
    else:
        return max(-0.9, min(0.9, eval_cp / max_cp * 0.9))

def denormalize_eval(norm_eval, max_cp=20000, mate_score=100000):
    if norm_eval >= 1.0:
        return mate_score
    elif norm_eval <= -1.0:
        return -mate_score
    else:
        return norm_eval / 0.9 * max_cp

if __name__ == "__main__":
    # Import current scores
    eval_df = pd.read_csv('input/temp/scores.csv')

    # Normalize evaluation scores
    eval_df['evaluation'] = eval_df['evaluation'].astype(float)
    eval_df['evaluation'] = eval_df['evaluation'].apply(normalize_eval)

    # Save normalized scores
    eval_df.to_csv('input/temp/scores_normalized.csv', index=False)
    print("Normalized evaluation scores saved to .input/temp/scores_normalized.csv")