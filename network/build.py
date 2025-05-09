import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters):
    skip = x
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.ReLU()(x)
    return x

def build_model(input_shape=(19, 8, 8), num_blocks=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(inputs)

    for _ in range(num_blocks):
        x = residual_block(x, 128)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='tanh')(x)

    return models.Model(inputs=inputs, outputs=output)

if __name__ == "__main__":
    model = build_model()
    model.summary()
    # Save the model
    model.save('../models/cnn_resblocks_model.keras')
    print("Model saved as cnn_resblocks_model.keras")