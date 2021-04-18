import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow.keras.layers as layers

import os
import pathlib
import time

import PIL

import cv2

_root_path =  os.path.dirname(os.path.realpath(__file__)) + '/../..'
_dataset_attr_path = _root_path + '/datasets/celeba/attributes/attr_celeba.csv'
_dataset_img_dir = _root_path + '/datasets/celeba/img_align_celeba_png/'
_saves_dir = 'training_checkpoints/training_1_eyeglasses_blond_hair'

_image_size = [128, 128]
_attr = ['Eyeglasses', 'Male']

_test_dir = pathlib.Path(_root_path + '/datasets/celeba/test_images/png')

def get_dataset_info(img_dir, attr_path, echo=False):
	df = pd.read_csv(attr_path, index_col=0)
	#df.replace({-1: 0}, inplace=True)

	img_names = list(pathlib.Path(img_dir).glob('*'))
	img_names = [img.name for img in img_names]
	img_names.sort()

	if echo:
		print(df.head(3))
		print(df.tail(3))
		print(img_names[:3])
		print(img_names[-3:])

	assert df.shape[0] == len(img_names), 'same number of images as attributes entries'
	assert set(img_names) == set(df.index.tolist()), 'sets are equal'

	return img_names, df


class ImageCls:
	def __init__(self, image_size, attributes):
		self.image_size = image_size
		self.attributes = attributes
		self._train_img_names = []
		self._trained_checkpoint_path = ''

	def make_model(self, echo=False):
		output_size = len(self.attributes)

		initial_learning_rate = 0.01
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
		)

		base_model = tf.keras.applications.mobilenet.MobileNet(
			include_top=False, 
			input_shape=(*self.image_size, 3),
			alpha=1,
			depth_multiplier=1,
			dropout=0.001,
			weights='imagenet',
			input_tensor=None,
			pooling=None)

		for layer in base_model.layers:
			layer.trainable = False

		inputs = tf.keras.layers.Input([*self.image_size, 3])
		x = tf.keras.applications.mobilenet.preprocess_input(inputs)
		x = base_model(x)
		x = layers.GlobalAveragePooling2D(data_format='channels_last', name='fc0_pooling')(x)
		x = layers.Dense(8, activation='relu', name='fc1_dense')(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		outputs = layers.Dense(output_size, activation='tanh', name='fc2_dense')(x)

		model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mean_squared_error', metrics=["accuracy"])

		if echo:
			print(model.summary())

		return model


	def train(self, img_names, num_img, df):
		model = self.make_model()

		x_all, y_all = self.load_data_batch(img_names, num_img, df, total_imgs=2**10)

		model.fit(x=x_all, y=y_all, batch_size=128, epochs=50, verbose=1, validation_split=0.125, shuffle=True)

		name_model_save = os.path.join(_saves_dir, 'model_{}.h5'.format(self.gen_time_str()))
		self._trained_checkpoint_path = name_model_save
		model.save(filepath=name_model_save)

		return model

	def get_data_sample(self, img_names, num_img, df, img_idx=None, img_name=None, plot=False):
		if img_name is None:
			if img_idx is None:
				img_idx = np.random.randint(num_img)

			img_name = img_names[img_idx]

		img = np.asarray(PIL.Image.open(os.path.join(_dataset_img_dir, img_name)))
		img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
		labels = df.loc[img_name, _attr]

		if plot:
			import matplotlib.pyplot as plt
			print(labels)
			print('image file name: {}'.format(img_name))
			plt.imshow(img)
			plt.show()

		x = img
		y = np.array(labels)

		return x, y

	def load_data_batch(self, img_names, num_img, df, total_imgs=None):
		x_lst, y_lst = [], []

		if total_imgs is None:
			img_filenames_select = img_names
		else:
			img_filenames_select = np.random.choice(img_names, total_imgs, replace=False)
			self._train_img_names = img_filenames_select

		import progressbar
		bar = progressbar.ProgressBar(maxval=len(img_filenames_select), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

		print('Loading images from directory: {}'.format(_dataset_img_dir))
		bar.start()
		for i, img_filename in enumerate(img_filenames_select):
			bar.update(i+1)
			x, y = self.get_data_sample(img_names, num_img, df, img_name=img_filename)
			x_lst.append(x)
			y_lst.append(y)
		bar.finish()

		x_batch = np.stack(x_lst, axis=0)
		x_ready = x_batch
		
		y_batch = np.stack(y_lst, axis=0)
		y_ready = np.array(y_batch, dtype='float32')

		return x_ready, y_ready

	def gen_time_str(self):
		return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def traincls(classifier):
	img_filenames, df_attr = get_dataset_info(_dataset_img_dir, _dataset_attr_path)
	num_img, _ = df_attr.shape

	classifier.train(img_filenames, num_img, df_attr)

def predictcls(classifier):
	model = classifier.make_model()
	model.load_weights(classifier._trained_checkpoint_path)

	img_paths = list(_test_dir.glob('*'))
	img_paths.sort()

	# img_batch = []
	# for filename in img_paths:
	# 	img = np.asarray(PIL.Image.open(filename))
	# 	img_batch.append(img)
	# imgs = np.stack([cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for img in img_batch], axis=0)

	img_filenames, df_attr = get_dataset_info(_dataset_img_dir, _dataset_attr_path)
	num_img, _ = df_attr.shape

	imgs, _ = classifier.load_data_batch(classifier._train_img_names, num_img, df_attr)
	y = model.predict(imgs)

	import matplotlib.pyplot as plt
	plt.figure(figsize=(10, 10))
	for n in range(25):
		ax = plt.subplot(5, 5, n + 1)

		img = imgs[n] / 255.0
		plt.imshow(img)
		plt.title(y[n])
		plt.axis("off")

	plt.show()

classifier = ImageCls(_image_size, _attr)

traincls(classifier)
predictcls(classifier)