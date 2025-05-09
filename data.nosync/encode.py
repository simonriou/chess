import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tqdm import tqdm

def fen_to_input_tensor(fen: str):
    """
    Converts a FEN string to a (19, 8, 8) input tensor for a chess engine.
    The channels are the following:
    - 12 channels for pieces (6 types x 2 colors)
    - 1 channel for side to move (1 for white, 0 for black)
    - 4 channels for castling rights (K, Q, k, q)
    - 1 channel for en passant square (1-hot encoding or 0 if none)
    - 1 channel for halfmove clock (normalized)
    """
    # Initialize empty tensor of shape (C, 8, 8)
    input_tensor = np.zeros((19, 8, 8), dtype=np.float32)

    # Split FEN into its components
    parts = fen.split()

    # Piece placement (first part of FEN)
    board = parts[0].split('/')
    
    # 12 channels for pieces (6 types × 2 colors)
    piece_planes = {'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5}
    
    # Mapping for white and black pieces
    for row in range(8):
        row_str = board[row]
        col = 0
        for char in row_str:
            if char.isdigit():
                # Skip empty squares
                col += int(char)
            else:
                # Place the piece in the corresponding channel
                piece_index = piece_planes[char.lower()]
                if char.islower():
                    piece_index += 6  # Adjust for black pieces
                input_tensor[piece_index, row, col] = 1
                col += 1
    
    # Side to move (1 for white, 0 for black)
    side_to_move = 1 if parts[1] == 'w' else 0
    input_tensor[12] = side_to_move
    
    # Castling rights (4 channels)
    castling_rights = parts[2]
    castling_rights_map = {'K': 0, 'Q': 1, 'k': 2, 'q': 3}
    for right in castling_rights:
        if right in castling_rights_map:
            input_tensor[13 + castling_rights_map[right]] = 1
    
    # En passant square (1-hot encoding or 0 if none)
    en_passant = parts[3]
    if en_passant != '-':
        col = ord(en_passant[0]) - ord('a')
        row = 8 - int(en_passant[1])
        input_tensor[17, row, col] = 1
    
    # Halfmove clock (normalized)
    halfmove_clock = int(parts[4])
    halfmove_value = min(halfmove_clock, 100) / 100.0  # Clip if too large
    input_tensor[18] = halfmove_value

    return input_tensor

def save_to_tfrecord(tensors, file_name):
    with tf.io.TFRecordWriter(file_name) as writer:
        for tensor in tensors:
            # Serialize the tensor into a tf.train.Example
            example = tf.train.Example(features=tf.train.Features(feature={
                'features': tf.train.Feature(
                    float_list=tf.train.FloatList(value=tensor.flatten())
                )
            }))
            writer.write(example.SerializeToString())
    print(f"Data saved to {file_name}")

def main():
    # Load the FENs from the merged CSV file
    features_file = 'input/temp/merged_fens_evals.csv'
    df = pd.read_csv(features_file)
    fens = df['FEN'].tolist()
    evaluations = df['eval'].tolist()
    print(f"Loaded {len(fens)} FENs from {features_file}")

    # Convert FENs to input tensors
    tensors = np.array([fen_to_input_tensor(fen) for fen in tqdm(fens, desc="Converting FENs to tensors")])
    print(f"Converted {len(tensors)} FENs to tensors")

    # Save the tensors to a TFRecord file
    save_to_tfrecord(tensors, 'input/temp/merged_fens_evals.tfrecord')

if __name__ == "__main__":
    main()