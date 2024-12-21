import re
import json
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

def export_features_labels(set_name, file_nb, features, labels, unroll=True):
    """
    Exports the features (board positions) and the labels (evaluation scores) to JSON or HDF5 files.

    Args:
    - set_name: str, name of the dataset
    - file_nb: int, number of the file being processed
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

    features_path = f'processed/{set_name}/{file_nb}/features.json'
    labels_path = f'processed/{set_name}/{file_nb}/labels.json'

    # Save features and labels as JSON
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=4)
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=4)

    return features_path, labels_path

def fen_to_matrix(fen):
    """
    Convert a FEN string into an 8x8x12 matrix representation.
    
    Parameters:
        fen (str): FEN string describing the board state.
        
    Returns:
        np.ndarray: 8x8x12 tensor representing the board.
    """
    # Mapping pieces to channel indices
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Initialize an 8x8x12 matrix with zeros
    board_matrix = np.zeros((8, 8, 12), dtype=np.int8)
    
    # Split FEN into board part and metadata
    board_part, _ = fen.split(' ', 1)
    
    # Process each row in the FEN board part
    rows = board_part.split('/')
    for rank_idx, row in enumerate(rows):
        file_idx = 0
        for char in row:
            if char.isdigit():
                # Empty squares
                file_idx += int(char)
            else:
                # Map the piece to the appropriate channel
                channel = piece_to_channel[char]
                board_matrix[rank_idx, file_idx, channel] = 1
                file_idx += 1
    
    return board_matrix

def matrix_to_fen(matrix):
    """
    Convert an 8x8x12 matrix representation of a board into a FEN string.
    
    Parameters:
        matrix (np.ndarray): 8x8x12 tensor representing the board.
        
    Returns:
        str: FEN string describing the board state.
    """
    # Mapping channel indices to pieces
    channel_to_piece = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
    }
    
    # Initialize an empty FEN string
    fen = ''
    
    # Process each row in the board matrix
    for rank_idx in range(8):
        empty_squares = 0
        for file_idx in range(8):
            for channel, piece in channel_to_piece.items():
                if matrix[rank_idx, file_idx, channel] == 1:
                    if empty_squares > 0:
                        fen += str(empty_squares)
                        empty_squares = 0
                    fen += piece
                    break
            else:
                empty_squares += 1
        if empty_squares > 0:
            fen += str(empty_squares)
        if rank_idx < 7:
            fen += '/'
    
    return fen

def flatten_data(features, labels):
    """
    Flattens the features and labels.

    Args:
    - features: list of lists, where each list is a list of board positions
    - labels: list of lists, where each inner list is a list of evaluation scores

    Returns:
    - flat_features: np.array, flattened features
    - flat_labels: np.array, flattened labels
    """
    # Flatten the features and labels
    flat_features = np.array([fen for game in features for fen in game])
    flat_labels = np.array([label for game in labels for label in game])

    return flat_features, flat_labels

def normalize_evaluation(evaluation, max_centipawn=1000, max_mate_distance=10):
    """
    Normalize evaluation values to the range [-1, 1] with improved handling.

    Args:
        evaluation (dict): The evaluation dictionary with keys "type" and "value".
        max_centipawn (int): Maximum centipawn value for normalization.
        max_mate_distance (int): Maximum mate distance for normalization.

    Returns:
        float: Normalized evaluation value.
    """
    if "type" not in evaluation or "value" not in evaluation:
        raise ValueError("Invalid evaluation format. Must contain 'type' and 'value' keys.")
    
    if evaluation["type"] == "centipawn":
        # Normalize centipawn values to [-1, 1]
        centipawn = float(evaluation["value"])
        normalized = centipawn / max_centipawn
        return max(-1, min(1, normalized))  # Clip to [-1, 1]

    elif evaluation["type"] == "mate":
        # Normalize mate-in values smoothly to [-1, -0.8] or [0.8, 1]
        try:
            mate_in = abs(int(evaluation["value"][1:]))  # Extract mate distance
            normalized_mate = min(mate_in, max_mate_distance) / max_mate_distance
            if evaluation["value"].startswith("M-"):  # Black mate
                return max(-1, -1 + normalized_mate * 0.2)  # Map to [-1, -0.8]
            else:  # White mate
                return min(1, 1 - normalized_mate * 0.2)  # Map to [0.8, 1]
        except (ValueError, IndexError):
            raise ValueError("Invalid mate value format. Expected 'M' or 'M-<distance>'.")
    
    else:
        # Unknown evaluation type
        return 0.0

def split_pgn(input_pgn_file, num_files):
    """
    Splits a PGN file into smaller files, each containing an equal number of games.

    Args:
    - input_pgn_file: str, path to the input PGN file
    - num_files: int, number of files to split the games into

    Returns:
    - list of str, paths to the split PGN files
    """
    # Open the input PGN file
    with open(input_pgn_file, 'r', encoding='utf-8') as f:
        pgn_content = f.read()

    # Split the PGN content into individual games based on the [Event] tag
    games = pgn_content.split("[Event")

    print("Done splitting games.")

    # Add back the [Event] tag to each game, starting from the second element
    games = [f"[Event{game}" for game in games if game.strip()]

    # Calculate the number of games per file
    games_per_file = len(games) // num_files
    remaining_games = len(games) % num_files
    
    # Get the output directory
    output_dir = os.path.dirname(input_pgn_file) + f"/{input_pgn_file.split('/')[-1].split('.')[0]}_split"
    
    # List to hold the paths of the generated files
    output_files = []
    
    # Initialize the index to start from
    game_index = 0
    
    # Split and write the games into M files
    for i in range(num_files):
        # Determine the number of games for the current file (distribute the remaining games)
        num_games = games_per_file + (1 if i < remaining_games else 0)
        
        # Define the output file path
        output_file = os.path.join(output_dir, f'split_{i + 1}.pgn')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Write the selected games to the new PGN file
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("\n\n".join(games[game_index:game_index + num_games]))
        
        # Append the file path to the list
        output_files.append(output_file)
        
        # Update the game index for the next chunk
        game_index += num_games

    print(f"Successfully split {len(games)} games into {num_files} files.")
    
    # Return the list of output file paths
    return output_files