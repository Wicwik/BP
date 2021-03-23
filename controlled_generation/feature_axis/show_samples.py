import os
import pickle

_path_sample = './assets/samples'

with open(os.path.join(_path_sample, 'stylegan2-ffhq-{:0>6d}.pkl'.format(48688)), 'rb') as f:
	temp = pickle.load(f)

import matplotlib.pyplot as plt
plt.imshow(temp['x'][0]); plt.show()