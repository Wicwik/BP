import tensorflow.compat.v2 as tfc
import tensorflow as tf

import pathlib
import numpy as np
import PIL.Image

tf.compat.v1.enable_eager_execution()

data_dir = pathlib.Path('../../datasets/celeba')
tfrecords = list(data_dir.glob('*'))

raw_dataset = tf.data.TFRecordDataset([str(path) for path in tfrecords])

print(raw_dataset)

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)


image_feature_description = {
    'data': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_dataset.map(_parse_image_function)
print(parsed_image_dataset)

for image_features in parsed_image_dataset:
  image_raw = image_features['data'].numpy()
  display.display(display.Image(data=image_raw))
  break
