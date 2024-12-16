from utils import fen_to_matrix, flatten_data, normalize_evaluation
from evaluate import evaluate_position

import chess.engine
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation

from sklearn.model_selection import train_test_split

# Import the data from processed/classical_2000_0124
features_path = 'processed/classical_2000_0124/features.json'
labels_path = 'processed/classical_2000_0124/labels.json'

with open(features_path, 'r') as f:
    features = json.load(f)
with open(labels_path, 'r') as f:
    labels = json.load(f)

features, labels = flatten_data(features, labels)

# Convert the FEN strings to matrices
matrix_features = [fen_to_matrix(fen) for fen in features]
# Normalize the labels
normalized_labels = [normalize_evaluation(label) for label in labels]

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

# Check if a model was saved
try:
    model = tf.keras.models.load_model('models/cnn_model.keras')
    print("Model loaded successfully.")
except:
    print("No model found. Building a new model.")

    model = build_model()

    # Train the model
    history = model.fit(
        np.array(X_train), np.array(y_train),
        batch_size=64,
        epochs=10,
        validation_split=0.2
    )

    # Save the model
    model.save('models/cnn_model.keras')

# Evaluate the model
test_loss, test_mae = model.evaluate(np.array(X_test), np.array(y_test), batch_size=64)

print(f"Test Loss: {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")