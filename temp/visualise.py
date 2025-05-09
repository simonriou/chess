import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
RECORD_PATH = '../data.nosync/input/temp/merged_fens_evals.tfrecord'
TENSOR_SHAPE = (19, 8, 8)

# Define the feature description
feature_description = {
    'features': tf.io.FixedLenFeature([np.prod(TENSOR_SHAPE)], tf.float32),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

def load_first_tensor(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    
    for parsed_record in parsed_dataset.take(1):
        flat_tensor = parsed_record['features'].numpy()
        return flat_tensor.reshape(TENSOR_SHAPE)

def plot_tensor(tensor):
    # Plot piece planes
    piece_labels = [
        "P", "R", "N", "B", "Q", "K", 
        "p", "r", "n", "b", "q", "k"
    ]
    
    fig, axs = plt.subplots(4, 5, figsize=(15, 10))
    axs = axs.flatten()

    for i in range(12):
        axs[i].imshow(tensor[i], cmap='Greys')
        axs[i].set_title(f'Piece: {piece_labels[i]}')
        axs[i].axis('off')

    axs[12].imshow(tensor[12], cmap='Blues')
    axs[12].set_title("Side to move")
    axs[12].axis('off')

    for i in range(4):
        axs[13 + i].imshow(tensor[13 + i], cmap='Oranges')
        axs[13 + i].set_title(f"Castling {i}")
        axs[13 + i].axis('off')

    axs[17].imshow(tensor[17], cmap='Purples')
    axs[17].set_title("En passant")
    axs[17].axis('off')

    axs[18].imshow(np.tile(tensor[18], (8, 8)), cmap='Reds')
    axs[18].set_title("Halfmove clock")
    axs[18].axis('off')

    # Hide unused plots if any
    for i in range(19, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    tensor = load_first_tensor(RECORD_PATH)
    plot_tensor(tensor)