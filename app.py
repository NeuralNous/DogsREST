from urllib.request import urlopen

import PIL
import numpy
from flask import Flask
from flask import request
from PIL import Image
import requests
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
from tensorflow.keras import layers
import random

ds = tf.keras.preprocessing.image_dataset_from_directory('static/Data', image_size=(256, 256))
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])
aug_ds = ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),
    (tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)),
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.fit(aug_ds, epochs=100)

app = Flask(__name__)

@app.route('/classify')
def classify():
    link = request.args.get('link')
    img = PIL.Image.open(urlopen(link))
    img = img.resize((256, 256), PIL.Image.ANTIALIAS)
    ans = model.predict(numpy.array(img.getdata()).reshape(1, 256, 256, 3))
    return str(ans)


if __name__ == '__main__':

    app.run()
