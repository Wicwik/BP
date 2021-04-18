import os
import time
import glob
import h5py
import pickle

import numpy as np
import pandas as pd

import axis

_root_path =  os.path.dirname(os.path.realpath(__file__)) + '/../..'
_path_sample_jpg = './assets/samples_2/'
_path_feature = './assets/results/feature_eyeglasses/'

_normalize_feature_direction = True

attr_path = _root_path + '/datasets/celeba/attributes/attr_celeba.csv'
y_filename = 'sample_y.h5'
z_filename = 'sample_z.h5'


def gen_time_str():
	return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

y_file = os.path.join(_path_sample_jpg, y_filename)
z_file = os.path.join(_path_sample_jpg, z_filename)

with h5py.File(y_file, 'r') as f:
	y = f['y'][:]

with h5py.File(z_file, 'r') as f:
	z = f['z'][:]

attr_df = pd.read_csv(attr_path)
y_name = attr_df.columns.to_list()[16]

feature_slope = axis.find_feature_axis(z, y)

if _normalize_feature_direction:
    feature_direction = axis.normalize_feature_axis(feature_slope)
else:
    feature_direction = feature_slope


feature_direction_file = os.path.join(_path_feature, 'feature_direction_{}.pkl'.format(gen_time_str()))
dict_to_save = {'direction': feature_direction, 'name': y_name}

with open(feature_direction_file, 'wb') as f:
	pickle.dump(dict_to_save, f)


# feature_direction_file = glob.glob(os.path.join(_path_feature, 'feature_direction_*.pkl'))[-1]

# with open(feature_direction_file, 'rb') as f:
	# feature_direction_dict = pickle.load(f)


# feature_direction = feature_direction_dict['direction']
# feature_name = np.array(feature_direction_dict['name'])

# en_z, len_y = feature_direction.shape

# feature_direction_disentangled = axis.disentangle_feature_axis_by_idx(feature_direction, idx_base=range(len_y//4), idx_target=None)

# axis.plot_feature_cos_sim(feature_direction_disentangled, feature_name=None)