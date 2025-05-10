import tensorflow as tf
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ==========================
# Parameters
# ==========================
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
TRAIN_SPLIT = 0.95
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
def build_model():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(inputs)
    
    # Residual blocks
    for _ in range(10):
        shortcut = x
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)

    return tf.keras.Model(inputs, outputs)

# ==========================
# Training Script
# ==========================
def main():
    tfrecord_files = ["../data.nosync/input/temp/merged_features.tfrecord"]
    total_examples = 102607

    raw_dataset = load_dataset(tfrecord_files)
    train_raw, val_raw = split_dataset(raw_dataset, total_examples, TRAIN_SPLIT)

    train_ds = prepare_dataset(train_raw)
    val_ds = prepare_dataset(val_raw)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True
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