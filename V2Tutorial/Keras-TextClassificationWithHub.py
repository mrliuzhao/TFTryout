import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from  IPython import display

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# 使用tensorflow-datasets获取数据
# 将训练集分割成 60% 和 40%，从而最终我们将得到 15,000 个训练样本
# 10,000 个验证样本以及 25,000 个测试样本。
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

print(type(train_data))  # tensorflow.python.data.ops.dataset_ops.PrefetchDataset
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)

# 从tensorflow-hub中导入已经训练好的embedding层
url = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
embedding_layer = hub.KerasLayer(url, input_shape=[],
                                 dtype=tf.string, trainable=True)

# 再组装完整的网络模型
model = tf.keras.Sequential()
model.add(embedding_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print('abcdefg')
        print('.', end='')


# 训练模型，输入为DataSets时并不需要y和batch_size
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=0,
                    callbacks=[PrintDot()])

# 最后评估模型
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
