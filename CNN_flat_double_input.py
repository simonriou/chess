from utils import fen_to_features, flatten_data, normalize_evaluation, matrix_to_fen
from evaluate import evaluate_position

import os
import chess.engine
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Input, Concatenate, Activation

from sklearn.model_selection import train_test_split

# Import the data from processed/classical_2000_0124
set_path = 'processed/blitz_2000_23'
set_name = set_path.split('/')[-1]

features = []
labels = []

# Get every subfolder of set_path
subfolders = [f.path for f in os.scandir(set_path) if f.is_dir()]
for folder in subfolders:
    features_path = f"{folder}/features.json"
    labels_path = f"{folder}/labels.json"
    with open(features_path, 'r') as f:
        features += json.load(f)
    with open(labels_path, 'r') as f:
        labels += json.load(f)

features, labels = flatten_data(features, labels)

# Convert the FEN strings to matrices
matrix_features = [fen_to_features(fen)[0] for fen in features]
turn_features = [fen_to_features(fen)[1] for fen in features]

# Normalize the labels
normalized_labels = [normalize_evaluation(label, max_centipawn=1000, max_mate_distance = 5) for label in labels]

# Convert to np arrays for compatibility with Keras
matrix_features = np.array(matrix_features)
turn_features = np.array(turn_features)
matrix_features = np.array(matrix_features)
turn_features = np.array(turn_features)

# Split the data into training and testing sets
X_train_matrix, X_test_matrix, X_train_turn, X_test_turn, y_train, y_test = train_test_split(
    matrix_features, turn_features, normalized_labels, test_size=0.2, random_state=42
)

y_train = np.array(y_train)
y_test = np.array(y_test)

def build_model(input_shape=(8, 8, 12)):
    """
    Builds a CNN model to evualuate chess positions.

    Args:
    - input_shape: tuple, shape of the input tensor

    Returns:
    - keras.Model: Compiled CNN model
    """

    # Input for the board features
    board_input = Input(shape=input_shape, name='board_input')
    x = Conv2D(32, kernel_size=(3, 3), padding='same')(board_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)

    # Input for the turn scalar
    turn_input = Input(shape=(1,), name='turn_input')

    # Concatenate the board and turn inputs
    combined = Concatenate()([x, turn_input])

    # FC Layer 1
    x = Dense(128, activation='relu')(combined)
    
    # FC Layer 2
    x = Dense(64, activation='relu')(x)

    # Output layer
    output = Dense(1, activation='tanh')(x)

    model = Model(inputs=[board_input, turn_input], outputs=output)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['accuracy', 'mae']
    )

    return model

def compare_models(models):
    """
    Comapres the performance of three models on the test set and plots the results.
    Also exports the performances (Loss, MAE) to a JSON file (append and not overwrite).

    Args:
    - models: list of keras.Model, the models to compare

    Returns:
    - None
    """
    # Initialize a dictionary to store the results
    print("Comparing models...")
    results = {}
    performances = {}

    # Evaluate each model on the test set
    for i, model in enumerate(models):
        # Evaluate the model
        test_loss, test_accuracy, test_mae = model.evaluate([X_test_matrix, X_test_turn], y_test, batch_size=64)
        print(f"Model {i + 1} Test Loss: {test_loss} | Test Accuracy: {test_accuracy} | Test MAE: {test_mae}")
        performances[f"Model {i + 1} (double)"] = {
            "Test Loss": test_loss,
            "Test MAE": test_mae
        }
        results[f"Model {i + 1}"] = test_mae

    # Plot the results
    plt.bar(results.keys(), results.values())
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Comparison')
    plt.show()

    # Pick a random fen and calculate the evaluation for each model, then compare to the stockfish evaluation
    idx = np.random.randint(len(X_test_matrix))
    fen = X_test_matrix[idx]
    turn = X_test_turn[idx]
    print(f"Stockfish Evaluation: {y_test[idx]} | FEN: {fen} | Turn: {turn}")
    label = y_test[idx]

    predictions = []

    for model in models:
        predictions.append(model.predict([np.array([fen]), np.array([turn])])[0][0])

    print(f"Stockfish Evaluation: {label} | FEN: {matrix_to_fen(fen)}")
    for i, pred in enumerate(predictions):
        print(f"Model {i + 1} Prediction: {pred}")

    # Export the results to a JSON file
    with open('performances/single_vs_double.json', 'a') as f:
        json.dump(performances, f)
        f.write('\n')

def train_model(model, X_t, y_t, X_v, y_v, epochs=20, batch_size=32):
    """
    Trains the model and saves it to disk.

    Args:
    - model: keras.Model, the model to train
    - X_t: [np.array, np.array], the training features
    - y_t: np.array, the training labels
    - X_v: [np.array, np.array], the validation features
    - y_v: np.array, the validation labels
    - epochs: int, the number of epochs to train for
    - batch_size: int, the batch size

    Returns:
    - history: keras.callbacks.History, the training history
    """
    model.summary()
    print("Training model...")

    # Train the model
    history = model.fit(
        X_t, 
        y_t, 
        validation_data=(X_v, y_v), 
        epochs=epochs, 
        batch_size=batch_size
    )

    # Save the model
    model.save('models/cnn_model.keras')
    print("Model saved succesfully.")
    
    return history

m6 = tf.keras.models.load_model('models/cnn_model6.keras')
compare_models([m6])