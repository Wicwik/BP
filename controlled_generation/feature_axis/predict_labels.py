import os
import glob
import h5py
import PIL.Image

import numpy as np

from classifier2 import ImageCls

import cv2

_root_path =  os.path.dirname(os.path.realpath(__file__)) + '/../..'
_checkpoint_path = _root_path + '/controlled_generation/classifier2/training_checkpoints/training_1_eyeglasses/model_20210325_093540.h5'

_path_sample_y = './assets/samples_2/'
_path_sample_jpg = './assets/samples_jpg/'
_x_file_pattern = 'sample_*.jpg'
_z_file_pattern = 'sample_*_z.npy'
_filename_y = 'sample_y.h5'

_image_size = [128, 128]
_tested_attr = 'Eyeglasses'

x_files = glob.glob(os.path.join(_path_sample_jpg, _x_file_pattern))
x_files.sort()

z_files = glob.glob(os.path.join(_path_sample_jpg, _z_file_pattern))
z_files.sort()

assert len(x_files) == len(z_files), 'same number of Z and X'

_cls = ImageCls(_image_size, _tested_attr)
model = _cls.make_model()
model.load_weights(_checkpoint_path)


y_list = []
batch_size = 64
img_batch_list = []
x_files_used = x_files
n = len(x_files_used)
save_freq = 2048

for i, file in enumerate(x_files_used):
	img = np.asarray(PIL.Image.open(file))
	img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
	img_batch_list.append(img)

	if (i%batch_size) == (batch_size-1) or i == (n-1):
		print('Labeling {} out of {} imgs.'.format(i+1, n))
		img_batch = np.stack(img_batch_list, axis=0)
		y = model.predict(img_batch, batch_size=batch_size)
		y_list.append(y)
		img_batch_list = []

		if i % save_freq == 0:
			y_concat = np.concatenate(y_list, axis=0)
			y_file = os.path.join(_path_sample_y, _filename_y)

			with h5py.File(y_file, 'w') as f:
				f.create_dataset('y', data=y_concat)

y_concat = np.concatenate(y_list, axis=0)
y_file = os.path.join(_path_sample_y, _filename_y)

with h5py.File(y_file, 'w') as f:
	f.create_dataset('y', data=y_concat)