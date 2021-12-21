import sys, os
from pathlib import Path

path = Path(os.getcwd())
sys.path.append(str(path.parent))

import BayesKeras
from BayesKeras import PosteriorModel
from BayesKeras import analyzers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

arg = {'eps': 0.11, 'lam': 0.25, 'rob': 2, 'opt': 'BBB', 'gpu': '1'}
# arg = {'eps': 0.11, 'lam': 0.25, 'rob': 2, 'opt': 'NA', 'gpu': '1'}

# robustness mode, 0:标准 1:IBP 2:PGD
rob = arg['rob']
# the variational inference method
opt = arg['opt']
inference = opt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train / 255.
X_test = X_test / 255.
# X_train = X_train.astype("float32").reshape(-1, 28*28)
X_test = X_test.astype("float32").reshape(-1, 28*28)
# np.set_printoptions(threshold=np.inf)
print("X_test: %s\n" % X_test[:2])

model = PosteriorModel("%s_FCN_Posterior_%s_mine" % (inference, rob))
from tqdm import trange

loss = tf.keras.losses.SparseCategoricalCrossentropy()

num_images = 500

accuracy = tf.keras.metrics.Accuracy()
preds = model.prearg(X_test[0:500])  # np.argmax(model.prearg(np.asarray(adv)), axis=1)
y_pred = np.argmax(preds, axis=1)  # 最大元素的索引值
y_true = y_test[0:500]
accuracy.update_state(y_pred, y_true)
# fgsm = accuracy.result()
print("%s Accuracy: " % (inference), accuracy.result())

accuracy = tf.keras.metrics.Accuracy()
# 投影梯度下降
adv = analyzers.PGD(model, X_test[0:500], eps=0.1, loss_fn=loss, num_models=10)
preds = model.prearg(adv)  # np.argmax(model.prearg(np.asarray(adv)), axis=1)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:500])
# fgsm = accuracy.result()
print("FGSM Robustness: ", accuracy.result())

accuracy = tf.keras.metrics.Accuracy()
preds = analyzers.chernoff_bound_verification(model, X_test[0:100], 0.1, y_test[0:100], confidence=0.80)
# print(preds.shape)
# print(np.argmax(preds, axis=1).shape)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:100])
print("Chernoff Lower Bound (IBP): ", accuracy.result())

"""
p = 0
for i in trange(100, desc="Computing FGSM Robustness"):
    this_p = analyzers.massart_bound_check(model, np.asarray([X_test[i]]), 0.075, y_test[i])
    print(this_p)
    p += this_p
print("Massart Lower Bound (IBP): ", p/100.0)
"""
