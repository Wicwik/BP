import os
import glob
import h5py
import PIL.Image

import numpy as np

from classifier import ImageCls
import matplotlib.pyplot as plt

_root_path =  os.path.dirname(os.path.realpath(__file__)) + '/../..'
_checkpoint_path = _root_path + '/condgen/classifier/training_checkpoints/training_1/glasses_modelv1.h5'


_path_sample_jpg = './assets/samples_jpg/'
_x_file_pattern = 'sample_*.jpg'
_z_file_pattern = 'sample_*_z.npy'
_filename_y = 'sample_y.h5'

_image_size = [218, 178]
_tested_attr = 'Eyeglasses'

x_files = glob.glob(os.path.join(_path_sample_jpg, _x_file_pattern))
x_files.sort()

z_files = glob.glob(os.path.join(_path_sample_jpg, _z_file_pattern))
z_files.sort()

assert len(x_files) == len(z_files), 'same number of Z and X'

cls = ImageCls(_image_size, _tested_attr)
cls.load_checkpoint(_checkpoint_path)
model = cls.model

y_list = []
batch_size = 25
img_batch_list = []
x_files_used = x_files
n = len(x_files_used)

for i, file in enumerate(x_files_used):
	img = np.asarray(PIL.Image.open(file))
	img_batch_list.append(img)

	if (i%batch_size) == (batch_size-1) or i == (n-1):
		print('Labeling {} out of {} imgs.'.format(i+1, n))
		img_batch = np.stack(img_batch_list, axis=0)
		y = model.predict(img_batch, batch_size=batch_size)
		y_list.append(y)

		for j, print_img in enumerate(img_batch_list):
			ax = plt.subplot(5, 5, j + 1)
			plt.imshow(print_img / 255.0)
			plt.title(y_list[-1][j])
			plt.axis('off')

		plt.show()

		img_batch_list = []

