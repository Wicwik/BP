import os 
import glob
import pickle

import numpy as np

from custom_gen import StyleGANGenerator
import axis

_path_feature = './assets/results/feature_eyeglasses/'
_pattern_feature = 'feature_direction_*.pkl'
_stylegan_ffhq_f_gdrive_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl'

rnd = np.random.RandomState()
gen = StyleGANGenerator(_stylegan_ffhq_f_gdrive_url)

feature_direction_file = glob.glob(os.path.join(_path_feature, 'feature_direction_*.pkl'))[-1]
with open(feature_direction_file, 'rb') as f:
	feature_direction_dict = pickle.load(f)

feature_direction = feature_direction_dict['direction']
feature_names = np.array(feature_direction_dict['name'])
# n_feature = 1


# feature_lock_status = np.zeros(n_feature).astype('bool')
# feature_direction_disentangled = axis.disentangle_feature_axis_by_idx(feature_direction, idx_base=np.flatnonzero(feature_lock_status))




import matplotlib.pyplot as plt
latent = rnd.randn(1, *gen.Gs.input_shape[1:])
img1 = gen.get_images(latent)

latent += feature_direction[:, 0]
img2 = gen.get_images(latent)

ax = plt.subplot(1, 2, 1)
plt.imshow(img1[0]/255)
plt.axis("off")

ax = plt.subplot(1, 2, 2)
plt.imshow(img2[0]/255)
plt.axis("off")
plt.show()

# for i in range(25):
# 	latent = rnd.randn(1, *gen.Gs.input_shape[1:])
# 	latent += feature_direction[:, 0]
# 	img = gen.get_images(latent)

# 	ax = plt.subplot(5, 5, i + 1)
# 	plt.imshow(img[0]/255)
# 	plt.axis("off")

# plt.show()