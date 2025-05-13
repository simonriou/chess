import tensorflow as tf
from tensorflow.keras import regularizers
import os
from loss_functions import loss_fn

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ==========================
# Parameters
# ==========================
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 500
TRAIN_SPLIT = 0.80
AUTOTUNE = tf.data.AUTOTUNE
INPUT_SHAPE = (8, 8, 19)  # Height, Width, Channels

# ==========================
# TFRecord Parsing Function
# ==========================
def parse_example(example_proto):
    feature_description = {
        'tensor': tf.io.FixedLenFeature([], tf.string),
        'eval': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    tensor = tf.io.parse_tensor(parsed['tensor'], out_type=tf.float32)
    tensor = tf.reshape(tensor, INPUT_SHAPE)
    label = tf.cast(parsed['eval'], tf.float32)
    return tensor, label

# ==========================
# Dataset Loading
# ==========================
def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = dataset.map(parse_example, num_parallel_calls=AUTOTUNE)
    return dataset

# ==========================
# Create Train/Val Splits
# ==========================
def split_dataset(dataset, total_size, train_ratio):
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    dataset = dataset.shuffle(buffer_size=total_size, reshuffle_each_iteration=True)
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    return train_ds, val_ds

# ==========================
# Prepare Datasets
# ==========================
def prepare_dataset(ds):
    return ds.cache().batch(BATCH_SIZE).repeat().prefetch(AUTOTUNE)

# ==========================
# Model Architecture
# ==========================
def residual_block(x, filters, downsample=False):
    shortcut = x

    stride = 2 if downsample else 1

    # First conv
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=stride, padding='same', activation=None,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Second conv
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding='same', activation=None,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Adjust shortcut if dimensions differ
    if downsample or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), strides=stride, padding='same', activation=None,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4)
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_model():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)

    # Initial conv
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation=None,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Residual blocks with increasing filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    # Global pooling and dense head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    return tf.keras.Model(inputs, outputs)

# ==========================
# Training Script
# ==========================
def main():
    tfrecord_files = ["../data.nosync/input/temp/merged_features.tfrecord"]
    total_examples = 109384

    raw_dataset = load_dataset(tfrecord_files)
    train_raw, val_raw = split_dataset(raw_dataset, total_examples, TRAIN_SPLIT)

    train_ds = prepare_dataset(train_raw)
    val_ds = prepare_dataset(val_raw)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=1e-5, clipnorm=1.0),
        loss=loss_fn,
        metrics=['mae']
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, verbose=1
    )

    steps_per_epoch = total_examples // BATCH_SIZE

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[lr_scheduler, early_stop],
        steps_per_epoch=steps_per_epoch,
        validation_steps=total_examples // BATCH_SIZE,
        verbose=1,
    )

    # Optionally save the model
    model.save("chess_eval_model.keras")

if __name__ == "__main__":
    main()