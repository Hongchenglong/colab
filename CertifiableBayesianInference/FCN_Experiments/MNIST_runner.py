# Author: Matthew Wicker

# Description: Minimal working example of training and saving
# a BNN trained with Bayes by backprop (BBB)
# can handle any Keras model
import sys, os
from pathlib import Path

path = Path(os.getcwd())
sys.path.append(str(path.parent))

import BayesKeras
import BayesKeras.optimizers as optimizers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

arg = {'eps': 0.11, 'lam': 0.25, 'rob': 0, 'opt': 'BBB', 'gpu': '-1'}

eps = arg['eps']
lam = arg['lam']
rob = arg['rob']
optim = arg['opt']
gpu = arg['gpu']
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# 划分训练集，测试集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.
# X_train = X_train.astype("float32").reshape(-1, 28 * 28)
# X_test = X_test.astype("float32").reshape(-1, 28 * 28)

# X_train = X_train[0:10000]
# y_train = y_train[0:10000]

# # 构建Sequential模型
# model = Sequential()
# # 构建Dense隐藏层并添加到模型中
# # 添加具有512个神经元的Dense隐藏层，使用relu激活函数
# model.add(Dense(512, activation="relu", input_shape=(None, 28 * 28)))
# # 添加具有10个神经元的Dense隐藏层，使用softmax激活函数
# model.add(Dense(10, activation="softmax"))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(784, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

inf = 2
full_covar = False
if (optim == 'VOGN'):
    # was 0.25 for a bit
    inf = 2
    learning_rate = 0.35
    decay = 0.0
    opt = optimizers.VariationalOnlineGuassNewton()
elif (optim == 'BBB'):
    inf = 10
    learning_rate = 0.45
    decay = 0.0
    opt = optimizers.BayesByBackprop()
elif (optim == 'SWAG'):
    learning_rate = 0.01
    decay = 0.0
    opt = optimizers.StochasticWeightAveragingGaussian()
elif (optim == 'SWAG-FC'):
    learning_rate = 0.01
    decay = 0.0
    full_covar = True
    opt = optimizers.StochasticWeightAveragingGaussian()
elif (optim == 'SGD'):
    learning_rate = 1.0
    decay = 0.0
    opt = optimizers.StochasticGradientDescent()
elif (optim == 'NA'):
    inf = 2
    learning_rate = 0.001
    decay = 0.0
    opt = optimizers.NoisyAdam()
elif (optim == 'ADAM'):
    learning_rate = 0.00001
    decay = 0.0
    opt = optimizers.Adam()
elif (optim == 'HMC'):
    learning_rate = 0.075
    decay = 0.0
    inf = 250
    linear_schedule = False
    opt = optimizers.HamiltonianMonteCarlo()

# Compile the model to train with Bayesian inference
if (rob == 0):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
elif (rob != 0):
    loss = BayesKeras.optimizers.losses.robust_crossentropy_loss

dnn_layer = 1
epochs = 20
bayes_model = opt.compile(model, dnn_layer=dnn_layer, epochs=epochs,
                          loss_fn=loss, learning_rate=learning_rate,
                          batch_size=128, linear_schedule=True,
                          decay=decay, robust_train=rob, inflate_prior=inf,
                          burn_in=3, steps=25, b_steps=20, epsilon=eps, rob_lam=lam)  # , preload="SGD_FCN_Posterior_1")
# steps was 50
# Train the model on your data
bayes_model.train(X_train, y_train, X_test, y_test)

# Save your approxiate Bayesian posterior
path = "%s_FCN_Posterior_%s_epochs%s_dnnLayer%s" % (optim, rob, epochs, dnn_layer)
bayes_model.save(path)
print(path + "训练完成")

