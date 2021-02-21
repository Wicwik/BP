import tensorflow.compat.v2 as tfc
import tensorflow_datasets as tfds
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

ds = tfds.load('celeb_a')