import tensorflow as tf
import numpy as np

TFRECORD_FILE = '../data.nosync/input/temp/merged_features.tfrecord'  # Path to your merged TFRecord
TENSOR_SHAPE = (8, 8, 19)          # Adapt to match your encoding

# Define feature schema
feature_description = {
    'tensor': tf.io.FixedLenFeature([], tf.string),
    'eval': tf.io.FixedLenFeature([], tf.float32),
}

def parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    tensor = tf.io.parse_tensor(parsed['tensor'], out_type=tf.float32)
    tensor = tf.reshape(tensor, TENSOR_SHAPE)
    return tensor, parsed['eval']

# Load TFRecord
dataset = tf.data.TFRecordDataset(TFRECORD_FILE, compression_type='GZIP')
dataset = dataset.map(parse_example)

# Inspect first N samples
N = 5
for i, (tensor, eval_score) in enumerate(dataset.take(N)):
    tensor_np = tensor.numpy()
    print(f"--- Sample {i+1} ---")
    print(f"Eval: {eval_score.numpy():.4f}")
    print(f"Tensor shape: {tensor_np.shape}")
    print(f"Tensor min/max: {tensor_np.min():.2f} / {tensor_np.max():.2f}")
    print(f"Tensor mean: {tensor_np.mean():.4f}")
    print("Slice (first 8x8 plane):")
    print(tensor_np[:, :, 0])
    print()

print(f"✅ Successfully read and parsed {N} entries.")