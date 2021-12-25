import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

dataset = "fmnist"
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 归一化
X_train = X_train / 255.
X_test = X_test / 255.

dnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation="relu"),
    # tf.keras.layers.Dense(256, activation="relu"),
    # tf.keras.layers.Dense(128, activation="relu"),
    # tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
dnn_model.summary()

loss = tf.keras.losses.SparseCategoricalCrossentropy()
epochs = 1

dnn_model.compile(optimizer=tf.optimizers.Adam(), loss=loss, metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=epochs)

dnn_model.evaluate(X_test, y_test)

dnn_model.save("model/dnn_model_" + dataset + "_epochs_" + str(epochs) + ".h5")

