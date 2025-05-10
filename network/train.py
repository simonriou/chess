import tensorflow as tf
from build import build_model
import os

# ==========================
# Parameters
# ==========================
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
TRAIN_SPLIT = 0.9
AUTOTUNE = tf.data.AUTOTUNE
INPUT_SHAPE = (8, 8, 19)

# ==========================
# TFRecord Parsing Function
# ==========================
def parse_tfrecord(example_proto):
    feature_description = {
        'tensor': tf.io.FixedLenFeature([], tf.string),
        'eval': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    tensor = tf.io.parse_tensor(parsed['tensor'], out_type = tf.float32)
    tensor = tf.reshape(tensor, INPUT_SHAPE)
    label = tf.cast(parsed['eval'], tf.float32)
    return tensor, label

# ==========================
# Dataset Loading
# ==========================
def load_dataset(filenames):
    dataset = tf.data.TFRecordDatatset(filenames)
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
def prepare_datasets(ds):
    return ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ==========================
# Model Architecture - defined in build.py
# ==========================

# ==========================
# Training Script
# ==========================
def main():
    tfrecord_files = ["placeholder"]
    total_examples = 10000  # Placeholder for the total number of examples

    raw_dataset = load_dataset(tfrecord_files)
    train_raw, val_raw = split_dataset(raw_dataset, total_examples, TRAIN_SPLIT)

    train_ds = prepare_datasets(train_raw)
    val_ds = prepare_datasets(val_raw)

    model = build_model()
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss = 'mse',
        metrics = ['mae']
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, verbose=1
    )

    model.fil(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[lr_scheduler, early_stop]
    )

    model.save('../models/cnn_resblocks_model.keras')

if __name__ == "__main__":
    main()