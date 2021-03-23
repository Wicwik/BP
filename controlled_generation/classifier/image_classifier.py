''' Invariant: .tfrecords have to be saved by datasets/tfrecordmaker '''
import tensorflow.compat.v2 as tf2
import tensorflow as tf
import numpy as np
import pandas as pd

import os
import pathlib

from functools import partial
import matplotlib.pyplot as plt

#tf.compat.v1.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


class ImageCls:
	def __init__(self, image_size, attribute):
		self.image_size = image_size
		self.attribute =  attribute

		self.strategy = self._tpu_check()

		initial_learning_rate = 0.01
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
		)

		with self.strategy.scope():
			self.model = self._make_model(lr_schedule)

	history = None

	def train_from_tfrecord(self, tfrecords_dir, labels, dataset_size, batch_size, epochs, train_mark):
		train_size = int(0.7 * dataset_size)
		val_size = int(0.15 * dataset_size)
		test_size = int(0.15 * dataset_size)

		checkpoint_path = "training_checkpoints/training_" + train_mark + "/cp-{epoch:04d}.ckpt"

		checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
		    filepath=checkpoint_path,
		    monitor='sparse_categorical_accuracy',
		    verbose=1,  
		    save_best_only=True,
		    save_freq=5*batch_size
		  
		)

		early_stopping_cb = tf.keras.callbacks.EarlyStopping(
		    patience=10, 
		    restore_best_weights=True,
		    monitor="loss"
		)

		tfrecords = list(pathlib.Path(tfrecords_dir).glob('*'))

		full_dataset = tf.data.TFRecordDataset([str(path) for path in tfrecords])
		full_dataset = full_dataset.shuffle(2048)

		train_dataset = full_dataset.take(train_size)
		test_dataset = full_dataset.skip(train_size)
		val_dataset = test_dataset.skip(val_size)
		test_dataset = test_dataset.take(test_size)

		train_dataset = self._make_dataset(train_dataset, labels)
		test_dataset = self._make_dataset(test_dataset, labels)
		val_dataset = self._make_dataset(val_dataset, labels)

		image_batch, label_batch = next(iter(train_dataset))
		self._show_batch(image_batch, label_batch)


		self.history = self.model.fit(
		    train_dataset,
		    epochs=epochs,
		    validation_data=val_dataset,
		    callbacks=[checkpoint_cb, early_stopping_cb],
		)


	def load_checkpoint(self, path):
		self.model.load_weights(path)

	def predict_from_arrays(self, imgs_array):
		imgs = []
		for img in imgs_array:
		    img = tf.image.resize(img, self.image_size, method='bilinear')
		    imgs.append(img)

		self._show_test(imgs)

	def predict(self, img_paths):
		imgs = []
		for img_path in img_paths:
		    img_string = open(img_path, 'rb').read()
		    img = tf.image.decode_png(img_string, channels=3)
		    img = tf.image.resize(img, self.image_size, method='bilinear')
		    imgs.append(img)

		self._show_test(imgs)
		##model.predict(img_array)

	def _tpu_check(self):
		try:
		    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
		    print("Device:", tpu.master())
		    tf.config.experimental_connect_to_cluster(tpu)
		    tf.tpu.experimental.initialize_tpu_system(tpu)
		    strategy = tf.distribute.experimental.TPUStrategy(tpu)
		except:
		    strategy = tf.distribute.get_strategy()
		
		print("Number of replicas:", strategy.num_replicas_in_sync)
		return strategy

	def _decode_image(self, raw_img):
	    img = tf.image.decode_png(raw_img, channels=3)
	    img = tf.cast(img, tf.float32)
	    img = tf.reshape(img, [*self.image_size, 3])
	    return img

	# parses tfrecord exmaple to tuples (tensor, label)
	def _parse_images_function(self, example, labels, attribute_name):
	    image_feature_description = {
	        'height': tf.io.FixedLenFeature([], tf.int64),
	        'width': tf.io.FixedLenFeature([], tf.int64),
	        'depth': tf.io.FixedLenFeature([], tf.int64),
	        'raw': tf.io.FixedLenFeature([], tf.string),
	    }
	    
	    for col in labels:
	        if col == 'filename':
	            image_feature_description[col] = tf.io.FixedLenFeature([], tf.string)
	        else:
	            image_feature_description[col] = tf.io.FixedLenFeature([], tf.int64)
	    
	    example = tf.io.parse_single_example(example, image_feature_description)
	    img = self._decode_image(example['raw'])
	    lbl = example[attribute_name]
	    
	    return img, lbl


	def _parse_dataset(self, dataset, labels):
	    return dataset.map(partial(self._parse_images_function, labels=labels, attribute_name=self.attribute))

	def _make_dataset(self, parsed_image_dataset, labels):
	    parsed_image_dataset = self._parse_dataset(parsed_image_dataset, labels)
	    parsed_image_dataset = parsed_image_dataset.prefetch(buffer_size=AUTOTUNE)
	    parsed_image_dataset = parsed_image_dataset.batch(BATCH_SIZE)
	    return parsed_image_dataset

	def _make_model(self, lr_schedule):
	    base_model = tf.keras.applications.Xception(
	        input_shape=(*self.image_size, 3), 
	        include_top=False, 
	        weights="imagenet"
	    )

	    base_model.trainable = False

	    inputs = tf.keras.layers.Input([*self.image_size, 3])
	    x = tf.keras.applications.xception.preprocess_input(inputs)
	    x = base_model(x)
	    x = tf.keras.layers.GlobalAveragePooling2D()(x)
	    x = tf.keras.layers.Dense(8, activation="relu")(x)
	    x = tf.keras.layers.Dropout(0.7)(x)
	    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

	    model = tf.keras.Model(inputs=inputs, outputs=outputs)

	    model.compile(
	        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
	        loss='binary_crossentropy',
	        metrics=[tf2.keras.metrics.SparseCategoricalAccuracy()],
	    )

	    return model

	def _show_batch(self, image_batch, label_batch):
	    plt.figure(figsize=(10, 10))
	    for n in range(25):
	        ax = plt.subplot(5, 5, n + 1)
	        plt.imshow(image_batch[n] / 255.0)

	        if label_batch[n].numpy():
	            plt.title("GLASSES")
	        else:
	            plt.title("NO GLASSES")
	        plt.axis("off")
	    plt.show()

	def _show_test(self, image_batch):
		plt.figure(figsize=(10, 10))
		for n in range(25):
			ax = plt.subplot(5, 5, n + 1)
			img = image_batch[n] / 255.0
			img = img.eval(session=tf.compat.v1.Session()) 
			plt.imshow(img)
			img_array = tf.expand_dims(image_batch[n], axis=0)
			plt.title(self.model.predict(img_array,  steps=1)[0])
			print_img
			
		plt.show()
