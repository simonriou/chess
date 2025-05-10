import tensorflow as tf
import pandas as pd

INPUT_TFRECORD = '../data.nosync/input/temp/encoded_features.tfrecord'
EVALS_CSV = '../data.nosync/input/temp/scores.csv'
OUTPUT_TFRECORD = '../data.nosync/input/temp/merged_features.tfrecord'

TENSOR_SHAPE = (8, 8, 19)

evals = pd.read_csv(EVALS_CSV, skiprows=1, header=None).squeeze().astype('float32').tolist()
assert isinstance(evals, list)

options = tf.io.TFRecordOptions(compression_type="GZIP")
writer = tf.io.TFRecordWriter(OUTPUT_TFRECORD, options=options)

raw_dataset = tf.data.TFRecordDataset(INPUT_TFRECORD, compression_type="GZIP")

def parse_example(example_proto):
    feature_description = {
        'features': tf.io.FixedLenFeature([TENSOR_SHAPE[0] * TENSOR_SHAPE[1] * TENSOR_SHAPE[2]], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    tensor = tf.reshape(parsed['features'], TENSOR_SHAPE)
    return tensor

parsed_dataset = raw_dataset.map(parse_example)

for i, (tensor, eval_value) in enumerate(zip(parsed_dataset, evals)):
    serialized_tensor = tf.io.serialize_tensor(tensor).numpy()
    example = tf.train.Example(features=tf.train.Features(feature={
        'tensor': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized_tensor])),
        'eval': tf.train.Feature(
            float_list=tf.train.FloatList(value=[eval_value])
        )
    }))
    writer.write(example.SerializeToString())

writer.close()

print(f"Merged TFRecord file created at: {OUTPUT_TFRECORD}")