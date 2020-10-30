from tensorflow import keras
import matplotlib.pyplot as plt

# import imdb dataset
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path=r'.\imdb.npz', num_words=10000)

# 一个映射单词到整数索引的词典，单词为Key，其index为value
word_index = imdb.get_word_index(path=r'.\imdb_word_index.json')

# 将前三个索引保留给控制符
word_index_pro = {k: (v + 3) for k, v in word_index.items()}
word_index_pro["<PAD>"] = 0
word_index_pro["<START>"] = 1
word_index_pro["<UNK>"] = 2  # unknown
word_index_pro["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index_pro.items()])


def decode_review(text: list) -> str:
    '''
    将index向量翻译为文本

    :param text: index向量
    :return: 文本字符
    '''
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# 不同的评论长度不同
print((len(train_data[0]), len(train_data[1])))
# 标准化所有评论的长度
train_data_pro = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index_pro["<PAD>"],
                                                            padding='post')
test_data_pro = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index_pro["<PAD>"],
                                                           padding='post')
print((len(train_data_pro[0]), len(train_data_pro[1])))

# 定义模型
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
print(model.summary())

# 定义损失函数、优化器和测量指标
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 分离训练集和验证集
x_val = train_data_pro[:10000]
partial_x_train = train_data_pro[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 模型训练，返回值中包括训练过程中的详细记录
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=30,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
print(history.params.keys())  # 记录了模型中的各种参数，如batch_size, epochs
print(history.history.keys())  # 记录了训练过程中每一轮（epoch）的信息，如loss, accuracy
print(history.model is model)  # 模型对象

# 绘制训练过程中loss变化
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练过程中accuracy变化
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.clf()   # 清除数字
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# 测试模型
results = model.evaluate(test_data_pro,  test_labels, verbose=2)
print(results)



