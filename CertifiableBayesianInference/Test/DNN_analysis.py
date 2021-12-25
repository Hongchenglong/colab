import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

dataset = "fmnist"
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.

dnn_model = tf.keras.models.load_model("model/dnn_model_" + dataset + ".h5")
dnn_model.summary()

pred = dnn_model.predict(X_test)
y_pred = np.argmax(pred, axis=1)
y_true = y_test
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(y_pred, y_true)
print("accuracy: ", accuracy.result())

# 残差 = 真实值-预测值
residual_error = []
y_true_len = len(y_true)
for i in range(y_true_len):
    zero_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_true_list = zero_list[:]
    y_true_list[y_true[i]] = 1
    y_pred_list = zero_list[:]
    y_pred_list[y_pred[i]] = 1

    y_true_list_len = len(y_true_list)
    residual = [y_true_list[i] - y_pred_list[i] for i in range(y_true_list_len)]
    residual_error.append(residual)

# print("residual_error: ", residual_error[:100])

columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
residual_df = pd.DataFrame(columns=columns, data=residual_error)
print(residual_df.head())
residual_df.to_csv('model/residual_df_' + dataset + '.csv', encoding='utf-8')
