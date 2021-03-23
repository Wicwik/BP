from classifier import ImageCls
from custom_gen import StyleGANGenerator

import os
import pathlib

# TFRECORDS = ROOT + '/datasets/celeba/tfrecords_with_attributes'
# ATTR = ROOT + '/datasets/celeba/attributes/attr_celeba.csv'
# DATASET_SIZE = 202599
# BATCH_SIZE = 64
# EPOCHS = 2
# TRAIN_N = '4'
 

# df = pd.read_csv(ATTR)

# cls = ImageCls(IMAGE_SIZE, TESTED_ATTR)
# cls.train_from_tfrecord(TFRECORDS, df.columns, DATASET_SIZE, BATCH_SIZE, EPOCHS, TRAIN_N)
# cls.load_checkpoint(ROOT + '/condgen/classifier/training_checkpoints/training_3/cp-0001.ckpt')

# cls.predict(img_paths)

_root_path =  os.path.dirname(os.path.realpath(__file__)) + '/..'
_stylegan_ffhq_f_gdrive_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl'
_checkpoint_path = _root_path + '/condgen/classifier/training_checkpoints/training_1/glasses_modelv1.h5'
_test_dir = _root_path + '/datasets/celeba/test_images/png'
_test_img_paths = list(pathlib.Path(_test_dir).glob('*'))

_image_size = [218, 178]
_tested_attr = 'Eyeglasses'

generator = StyleGANGenerator(_stylegan_ffhq_f_gdrive_url)
classifier = ImageCls(_image_size, _tested_attr)


imgs = generator.get_images(64)
classifier.load_checkpoint(_checkpoint_path)
classifier.predict_from_arrays(imgs)