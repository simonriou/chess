import tensorflow as tf
from utils import fen_to_features, matrix_to_fen, denormalize
import numpy as np

def eval_fen(feature, turn, model):
    """
    Evaluates a chess position given its FEN using a CNN model.

    Args:
        feature (matrix): The FEN of the chess position as a matrix.
        turn (int): The turn of the player to move.
        model (tf.keras.Model): The CNN model to use for evaluation.

    Returns:
        float: The evaluation of the position.
    """
    # Make a prediction
    prediction = model.predict([np.array([fen]), np.array([turn])])

    return prediction[0][0]

# FEN to evaluate
fen = '1K1q4/4q3/8/8/8/8/8/8 b - - 0 1'
fens = np.array([fen_to_features(fen)[0]])
turns = np.array([fen_to_features(fen)[1]])
features = np.array(fens)
turns = np.array(turns)

# Load the model
model = tf.keras.models.load_model('models/cnn_model14.keras')

# Evaluate the positions
for fen, turn in zip(fens, turns):
    print(f"Evaluating position: {matrix_to_fen(fen)}")
    evaluation = eval_fen(fen, turn, model)
    print(f"Evaluation: {denormalize(evaluation)/100:.2f}")