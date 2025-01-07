from utils import fen_to_features, flatten_data, normalize_evaluation, matrix_to_fen, denormalize
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

    # Convolutional layer 1
    x = Conv2D(32, kernel_size=(7, 7), padding='same', kernel_initializer='he_normal')(board_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Convolutional layer 2
    x = Conv2D(64, kernel_size=(7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Flatten the output
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

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        loss='mse',
        metrics=['accuracy', 'mae']
    )

    return model

def compare_models(models):
    """
    Comapres the performance of three models on the test set and plots the results.
    Also exports the performances (Loss, MAE) to a JSON file (append and not overwrite).

    Args:
    - models: list of keras.Model, the models to compare (format is mX, mY, mZ)

    Returns:
    - None
    """
    # Initialize a dictionary to store the results
    print("Comparing models...")
    results = {}
    performances = {}

    # Evaluate each model on the test set
    for i, model in enumerate(models):
        # If it's the first model, batch size is 32, else 64
        batch_size = 32 if i == 0 else 64
        # Evaluate the model
        test_loss, test_accuracy, test_mae = model.evaluate([X_test_matrix, X_test_turn], y_test, batch_size=64)
        print(f"Model {i + 1} Test Loss: {test_loss} | Test Accuracy: {test_accuracy} | Test MAE: {test_mae}")
        performances[f"Model {i + 1} ({batch_size})"] = {
            "Test Loss": test_loss,
            "Test MAE": test_mae,
            "Test Accuracy": test_accuracy
        }
        results[f"Model {i + 1}"] = test_mae

    # Plot the results
    plt.bar(results.keys(), results.values())
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Comparison')
    plt.show()

    # Pick a few random fens that have > 0.5 or < -0.5 eval and calculate the evaluation for each model, then compare to the stockfish evaluation
    indices = np.where((y_test > 0.5) | (y_test < -0.5))[0][:5]
    fens = X_test_matrix[indices]
    turns = X_test_turn[indices]
    labels = y_test[indices]
    
    predictions = []

    # For each random fen and turn
    for fen, turn in zip(fens, turns):
        fen_pred = []
        for model in models: # For each model
            fen_pred.append(model.predict([np.array([fen]), np.array([turn])])[0][0]) # Predict the evaluation
        predictions.append(fen_pred)

    # Display the results
    for i, (fen, turn, label, preds) in enumerate(zip(fens, turns, labels, predictions)):
        print(f"Stockfish Evaluation: {label} (Real: {denormalize(label)/100:.2f}) | FEN: {matrix_to_fen(fen)} | Turn: {turn}")
        for j, pred in enumerate(preds):
            print(f"Model {j + 13} Prediction: {pred:.2f} | Real eval: {denormalize(pred)/100:.2f}")

    # Export the results to a JSON file
    with open('performances/third_layer.json', 'w') as f:
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

# Load the models
m13 = tf.keras.models.load_model('models/cnn_model13.keras')
m14 = tf.keras.models.load_model('models/cnn_model14.keras')

models = [m13, m14]

compare_models(models)

# # Build the model
# model = build_model()

# # Train the model
# history = train_model(model, [X_train_matrix, X_train_turn], y_train, [X_test_matrix, X_test_turn], y_test, epochs=20, batch_size=32)