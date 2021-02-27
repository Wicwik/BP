import tensorflow.compat.v2 as tfc
import tensorflow as tf

import pathlib
import numpy as np
import pandas as pd
import IPython.display as display

tf.compat.v1.enable_eager_execution()

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
        
    if isinstance(value, str):
        value = str.encode(value)
        
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_labels(img_name, df):
    row = df.loc[df['filename'] == img_name]
    
    labels = {}
    
    for rowname in row:
        value = row[rowname].values[0]
        
        if isinstance(value, str):
            labels['filename'] = _bytes_feature(value)
        else:
            labels[rowname] = _int64_feature(value)
            
    return labels

def create_example(img_string, labels):
    img_shape = tf.image.decode_png(img_string).shape
    
    feature = {
        'height': _int64_feature(img_shape[0]),
        'width': _int64_feature(img_shape[1]),
        'depth': _int64_feature(img_shape[2]),
    }
    
    feature = dict(feature, **labels)
    feature['raw'] = _bytes_feature(img_string)
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


data_dir = pathlib.Path('../../datasets/celeba')
img_dir = pathlib.Path(str(data_dir) + '/img_align_celeba_png')
attr_dir = pathlib.Path(str(data_dir) + '/attr_celeba.csv')

df = pd.read_csv(attr_dir)
img_paths = list(img_dir.glob('*'))

examples = []
counter = 0
for img_path in img_paths:
    img_string = open(img_path, 'rb').read()
    examples.append(create_example(img_string, get_labels(str(img_path).split('/')[-1], df)))
    counter += 1
    print('%d / 202599\r' % (counter), end='', flush=True)