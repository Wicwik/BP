import os
import glob
import h5py
import pickle
import PIL.Image

import numpy as np

_path_sample_pkl = './assets/samples_paper/'
_path_sample_jpg = './assets/samples_paper/'

pkl_files = glob.glob(os.path.join(_path_sample_pkl, '*.pkl'))
pkl_files.sort()

filename_sample_z = 'sample_z.h5'

z_list = []
i = 0

for file in pkl_files:
	print(file)
	with open(file, 'rb') as f:
		pkl_content = pickle.load(f)

	x = pkl_content['x']
	z = pkl_content['z']

	x = x*255

	n = x.shape[0]
	for j in range(n):
		save_path = os.path.join(_path_sample_jpg, 'sample_{:0>6}'.format(i))

		PIL.Image.fromarray(x[j].astype(np.uint8)).save(save_path + '.jpg')
		np.save(save_path + '_z.npy', z[j])
		i += 1
	z_list.append(z)


z_concat = np.concatenate(z_list, axis=0)
file_sample_z = os.path.join(_path_sample_jpg, filename_sample_z)

with h5py.File(file_sample_z, 'w') as f:
	f.create_dataset('z', data=z_concat)