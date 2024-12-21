from utils import fen_to_matrix, flatten_data, normalize_evaluation
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
# Find the max centipawn evaluation (max of labels that have 'type': 'centpawn')
max_eval = max([label for label in labels if label['type'] == 'centipawn'], key=lambda x: x['value'])['value']
print(f"Max evaluation: {max_eval}")

# Find the max mate distance (max of labels that have 'type': 'mate'), last character of the string
max_mate = max([label for label in labels if label['type'] == 'mate'], key=lambda x: x['value'])['value']
max_mate = int(max_mate[-1])
print(f"Max mate distance: {max_mate}")

# Convert the FEN strings to matrices
matrix_features = [fen_to_matrix(fen) for fen in features]
# Normalize the labels
normalized_labels = [normalize_evaluation(label, max_centipawn = max_eval, max_mate_distance = max_mate) for label in labels]

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

def compare_models(model1, model2, model3):
    """
    Comapres the performance of three models on the test set.

    Args:
    - model1: keras.Model, first model to compare
    - model2: keras.Model, second model to compare
    - model3: keras.Model, third model to compare

    Returns:
    - None
    """
    # Evaluate the models
    test_loss1, test_mae1 = model1.evaluate(np.array(X_test), np.array(y_test), batch_size=64)
    test_loss2, test_mae2 = model2.evaluate(np.array(X_test), np.array(y_test), batch_size=64)
    test_loss3, test_mae3 = model3.evaluate(np.array(X_test), np.array(y_test), batch_size=64)

    print(f"Model 1 - Test Loss: {test_loss1}, Test MAE: {test_mae1}")
    print(f"Model 2 - Test Loss: {test_loss2}, Test MAE: {test_mae2}")
    print(f"Model 3 - Test Loss: {test_loss3}, Test MAE: {test_mae3}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.bar(['Model 1', 'Model 2', 'Model 3'], [test_mae1, test_mae2, test_mae3])
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Comparison')
    plt.show()

# Check if models were saved
try:
    # model1 = tf.keras.models.load_model('models/cnn_model1.keras')
    # model2 = tf.keras.models.load_model('models/cnn_model2.keras')
    # model3 = tf.keras.models.load_model('models/cnn_model3.keras')
    # print("Models loaded succesfully.")

    # compare_models(model1, model2, model3)

    # # Pick a random fen and calculate the evaluation for each model, then compare to the stockfish evaluation
    # idx = np.random.randint(len(X_test))
    # fen = X_test[idx]
    # label = y_test[idx]
    # pred1 = model1.predict(np.array([fen]))[0][0]
    # pred2 = model2.predict(np.array([fen]))[0][0]
    # pred3 = model3.predict(np.array([fen]))[0][0]

    # print(f"Stockfish Evaluation: {label} | FEN: {fen}")
    # print(f"Model 1 Prediction: {pred1}")
    # print(f"Model 2 Prediction: {pred2}")
    # print(f"Model 3 Prediction: {pred3}")

    model = tf.keras.models.load_model('models/cnn_model.keras')
except:
    print("No model found. Building a new model.")

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

# Evaluate the model
# test_loss, test_mae = model.evaluate(np.array(X_test), np.array(y_test), batch_size=64)

# print(f"Test Loss: {test_loss}")
# print(f"Test Mean Absolute Error (MAE): {test_mae}") 