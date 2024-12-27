from utils import fen_to_matrix, flatten_data, normalize_evaluation, matrix_to_fen
from evaluate import evaluate_position

import os
import chess.engine
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation

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
matrix_features = [fen_to_matrix(fen) for fen in features]
# Normalize the labels
normalized_labels = [normalize_evaluation(label, max_centipawn=1000, max_mate_distance = 5) for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(matrix_features, normalized_labels, test_size=0.2, random_state=42)

def build_model(input_shape=(8, 8, 12)):
    """
    Builds a CNN model to evualuate chess positions.

    Args:
    - input_shape: tuple, shape of the input tensor

    Returns:
    - kera.Model: Compiled CNN model
    """
    model = Sequential([
        # Conv. Layer 1
        Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),

        # Conv. Layer 2
        Conv2D(64, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        # Flatten the feature maps
        Flatten(),

        # FC Layer 1
        Dense(128, activation='relu'),

        # FC Layer 2
        Dense(64, activation='relu'),

        # Output layer
        Dense(1, activation='tanh')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def compare_models(models):
    """
    Comapres the performance of three models on the test set and plots the results.

    Args:
    - models: list of keras.Model, the models to compare

    Returns:
    - None
    """
    # Initialize a dictionary to store the results
    print("Comparing models...")
    results = {}

    # Evaluate each model on the test set
    for i, model in enumerate(models):
        # Evaluate the model
        test_loss, test_mae = model.evaluate(np.array(X_test), np.array(y_test), batch_size=64)
        print(f"Model {i + 1} Test Loss: {test_loss}")
        results[f"Model {i + 1}"] = test_mae

    # Plot the results
    plt.bar(results.keys(), results.values())
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Comparison')
    plt.show()

    # Pick a random fen and calculate the evaluation for each model, then compare to the stockfish evaluation
    idx = np.random.randint(len(X_test))
    fen = X_test[idx]
    label = y_test[idx]

    predictions = []

    for model in models:
        predictions.append(model.predict(np.array([fen]))[0][0])

    print(f"Stockfish Evaluation: {label} | FEN: {matrix_to_fen(fen)}")
    for i, pred in enumerate(predictions):
        print(f"Model {i + 1} Prediction: {pred}")

    # Evaluate a specific position
    mate_fen = np.array([fen_to_matrix('1nkr3r/1ppqb3/4b3/1P1ppp2/Q5p1/2PPB1P1/P2NP1B1/R3NRK1 w - ')])
    predictions = []
    for model in models:
        predictions.append(model.predict(mate_fen)[0][0])

    print(f"FEN: 1nkr3r/1ppqb3/4b3/1P1ppp2/Q5p1/2PPB1P1/P2NP1B1/R3NRK1 w - ")
    for i, pred in enumerate(predictions):
        print(f"Model {i + 1} Prediction: {pred}")

def train_model(X_train, y_train, epochs=50, batch_size=64):
    """
    Trains a CNN model on the training data.

    Args:
    - model: keras.Model, the model to train
    - X_train: np.array, training features
    - y_train: np.array, training labels
    - epochs: int, number of epochs to train
    - batch_size: int, batch size for training

    Returns:
    - keras.Model: trained model
    """
    model = build_model()

    # Train the model
    history = model.fit(
        np.array(X_train), np.array(y_train),
        batch_size=64,
        epochs=50,
        validation_split=0.2
    )

    # Save the model
    model.save('models/cnn_model.keras')
    print("Model saved succesfully.")

m1 = tf.keras.models.load_model('models/cnn_model1.keras')
m2 = tf.keras.models.load_model('models/cnn_model2.keras')
m3 = tf.keras.models.load_model('models/cnn_model3.keras')
m4 = tf.keras.models.load_model('models/cnn_model4.keras')
m5 = tf.keras.models.load_model('models/cnn_model5.keras')

models = [m1, m2, m3, m4, m5]

compare_models(models)