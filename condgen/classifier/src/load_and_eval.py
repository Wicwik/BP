import tensorflow.compat.v2 as tfc
import tensorflow as tf

import os
import pathlib
import numpy as np
import pandas as pd
import IPython.display as display

from functools import partial
import matplotlib.pyplot as plt


IMAGE_SIZE = [218, 178]
TEST_DIR = pathlib.Path('./test_images')

tf.compat.v1.enable_eager_execution()
tf.enable_eager_execution()

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "glasses_model.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

def make_model():
    base_model = tf.keras.applications.Xception(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    return model

def show_batch(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(len(image_batch)):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        plt.axis("off")
    plt.show()

model = make_model()
model.load_weights('/home/rbelanec/Documents/BP/condgen/classifier/training_2/cp-0001.ckpt')

img_paths = list(TEST_DIR.glob('*'))
print(img_paths)

imgs = []
for img_path in img_paths:
    img_string = open(img_path, 'rb').read()
    img = tf.image.decode_jpeg(img_string, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method='bilinear')
    imgs.append(img)

show_batch(imgs)

def show_batch_predictions(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(len(image_batch)):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        img_array = tf.expand_dims(image_batch[n], axis=0)
        plt.title(model.predict(img_array)[0])
        plt.axis("off")
    plt.show()

show_batch_predictions(imgs)