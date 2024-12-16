import re
import json
import h5py
import os
import numpy as np

def clean_pgn(file_path):
    """
    Removes time annotations from a PGN file and writes the cleaned file to a new file.

    Args
    - file_path: str, path to the PGN file to clean

    Returns
    - output_path: str, path to the cleaned PGN file
    """

    # Get the file name
    file_name = file_path.split('/')[-1]

    # Output path : cleaned/file_name_cleaned.pgn
    output_path = f'cleaned/{file_name.split(".")[0]}_cleaned.pgn'

    # Read the PGN file
    with open(file_path, 'r', encoding='utf-8') as f:
        pgn = f.read()
    
    # Use regex to remove time annotations
    cleaned_pgn = re.sub(r'\s*\{\[%emt[^}]*\]\}\s*', ' ', pgn)

    # Save the cleaned PGN file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_pgn)

    return output_path

def export_features_labels(set_name, features, labels, unroll=True):
    """
    Exports the features (board positions) and the labels (evaluation scores) to JSON or HDF5 files.

    Args:
    - set_name: str, name of the dataset
    - features: list of lists, where each list is a list of board positions
    - labels: list of lists, where each inner list is a list of evaluation scores
    - unroll: bool, whether to unroll the features and labels into a single list

    Returns:
    - features_path: str, path to the file containing the features
    - labels_path: str, path to the file containing the labels
    """
    # Unroll the features and labels
    if unroll:
        features = [fen for game in features for fen in game]
        labels = [label for game in labels for label in game]
    
    # Export the features and labels

    features_path = f'processed/{set_name}/features.json'
    labels_path = f'processed/{set_name}/labels.json'

    # Save features and labels as JSON
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=4)
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=4)

    return features_path, labels_path