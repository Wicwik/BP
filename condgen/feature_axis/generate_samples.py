from custom_gen import StyleGANGenerator

import cv2
import time
import numpy as np
import pickle
import os
import tensorflow.compat.v2 as tf

_stylegan_ffhq_f_gdrive_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl'
_path_sample = './assets/samples_paper'

_batch_size = 8
_n_batch = 24
_image_size = [218, 178]

rnd = np.random.RandomState()
gen = StyleGANGenerator(_stylegan_ffhq_f_gdrive_url)

for i in range(_n_batch):
	sample = i * _batch_size

	start = time.time()

	latents = rnd.randn(_batch_size, *gen.Gs.input_shape[1:])
	imgs = gen.get_images(latents)
	#imgs = np.asarray([cv2.resize(img, dsize=(178, 218), interpolation=cv2.INTER_CUBIC) for img in imgs])

	imgs = imgs/255

	with open(os.path.join(_path_sample, 'stylegan2-ffhq-{:0>6d}.pkl'.format(sample)), 'wb') as f:
		pickle.dump({'z': latents, 'x': imgs}, f)
	
	stop = time.time()

	print('Batch ' + str(i) + ' out of ' + str(_n_batch) + '. Currently generated ' + str(sample) + ' images. Duration: ' + str(stop-start), end='\r')