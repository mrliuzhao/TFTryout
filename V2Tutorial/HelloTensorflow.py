import tensorflow as tf
from tensorflow.compat.v1.keras.datasets.mnist import load_data
import numpy as np

# 导入MNIST数据集
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, y_train), (x_test, y_test) = load_data(r'D:\PythonWorkspace\TFTryout\mnist.npz')

# 转换为小数
# x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train / 255.0
x_test = x_test / 255.0
# x_train_pro = np.resize(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
# x_test_pro = np.resize(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
x_train_pro = np.reshape(x_train, (x_train.shape[0], -1))
x_test_pro = np.reshape(x_test, (x_test.shape[0], -1))

# 使用keras Sequential模型
model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练并验证模型
model.fit(x_train_pro, y_train, batch_size=64, epochs=5)
model.evaluate(x_test_pro,  y_test, verbose=2)



