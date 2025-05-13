import tensorflow as tf
import random
import chess
import numpy as np
import sys
import os
from loss_functions import loss_fn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data.nosync')))

from features.encode import fen_to_input_tensor

def denormalize_eval(norm_eval, max_cp=20000, mate_score=100000):
    if norm_eval >= 1.0:
        return mate_score
    elif norm_eval <= -1.0:
        return -mate_score
    else:
        return norm_eval / 0.9 * max_cp

# ==========================
# Parameters
# ==========================
INPUT_SHAPE = (8, 8, 19)  # Height, Width, Channels (for chess, each square has 19 features)
MODEL_PATH = '../models/chess_eval_model.keras'  # Path to your saved model

# ==========================
# Load Model
# ==========================
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'loss_fn': loss_fn})
    return model

# ==========================
# Evaluate a Random FEN
# ==========================
def evaluate_random_fen(model):
    # Generate a random position (FEN)
    random_fen = "8/3k4/8/8/4KQ2/8/8/8 w - - 0 1"  # This generates a starting position, for now (can replace with random generation)
    
    # Convert FEN to input tensor
    input_tensor = np.transpose(fen_to_input_tensor(random_fen), (1, 2, 0))
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    
    # Get the model's evaluation
    prediction = model.predict(input_tensor)
    denorm_eval = denormalize_eval(prediction[0][0])  # Denormalize the output
    
    # Output the evaluation
    print(f"Denormalized evaluation for FEN ({random_fen}): {denorm_eval}")
    print(f"Normalized evaluation for FEN ({random_fen}): {prediction[0][0]}")

# ==========================
# Main Function
# ==========================
def main():
    model = load_model()  # Load the pre-trained model
    evaluate_random_fen(model)  # Evaluate on a random FEN

if __name__ == "__main__":
    main()