import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import time

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

# df = pd.read_csv(r'.\data\sz.000063-07d.csv')
df = pd.read_csv(r'.\data\sh.600519-07d.csv')
# 拼接需要的13个特征列
df_use = pd.concat([df['open'], df['close'], df['high'], df['low'],
                    df['preclose'], df['volume'], df['amount'],
                    df['turn'], df['pctChg'], df['peTTM'],
                    df['pbMRQ'], df['psTTM'], df['pcfNcfTTM']], axis=1)
df_use = np.asarray(df_use, dtype=np.float32)
# 特征值归一化
scaler = MinMaxScaler().fit(df_use)
df_use = scaler.transform(df_use)

# 特征 1300 * 13，每行为一个样本
features = df_use
f_num = features.shape[1]

# 预测的收盘价并不归一化，预测收盘价取对数
labels = np.asarray(df['close'], dtype=np.float32)
labels = np.log(labels)
labels = np.reshape(labels, (-1, 1))  # 1300, 1


# 生成数据集，取前TRAIN_NUM个为训练集，后面的为测试集
def data_generate(data, actual_values, train_num, time_step):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for i in range(len(data) - time_step):
        if i < train_num:
            train_features.append(data[i: (i + time_step), :])
            train_labels.append(actual_values[i + time_step, :])
        else:
            test_features.append(data[i: (i + time_step), :])
            test_labels.append(actual_values[i + time_step, :])
    return np.array(train_features, dtype=np.float32), np.array(train_labels, dtype=np.float32),\
           np.array(test_features, dtype=np.float32), np.array(test_labels, dtype=np.float32)


TRAIN_NUM = 1000
TIME_STEPS = 5
train_x, train_y, test_x, test_y = data_generate(features, labels, TRAIN_NUM, TIME_STEPS)
# train_x.shape = 1000, 5, 13  TRAIN_NUM, TIME_STEPS, featureNum

BATCH_SIZE = 500
HIDDEN_SIZE = 13
global_step = tf.Variable(0)
decay_steps = TRAIN_NUM // BATCH_SIZE
learning_rate = tf.train.exponential_decay(0.5, global_step, 500, 0.99, staircase=True)


# 定义LSTM模型
def lstm_model(x, y, is_train, lstmcells, weights, bias):
    outputs, _ = tf.nn.dynamic_rnn(lstmcells, x, dtype=tf.float32)
    # outputs.shape = 32, 5, 50   batchsize, timestep, hiddensize
    output = outputs[:, -1, :]
    # output.shape = 32, 50   batchsize, hiddensize

    # weights.shape = 50, 1
    prediction = tf.matmul(output, weights) + bias  # batchsize, 1
    # 不在训练时仅返回预测值
    if not is_train:
        return prediction, None, None

    # 使用MSE作为损失函数，y.shape = batchsize, 1
    loss = tf.reduce_mean(tf.square(output - y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)
    return prediction, loss, train_op


x = tf.placeholder(dtype=tf.float32, shape=(None, TIME_STEPS, f_num), name='x')
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')
outputW = tf.Variable(tf.random_normal([HIDDEN_SIZE, 1], stddev=1, dtype=tf.float32), name='outputWeights',
                      dtype=tf.float32)
outputB = tf.Variable(tf.random_normal([1, 1], stddev=1, dtype=tf.float32), name='outputBias', dtype=tf.float32)

# 定义多层LSTM
LSTM_LAYER = 1
cell = tf.nn.rnn_cell.MultiRNNCell([
    tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    for _ in range(LSTM_LAYER)])
prediction, loss, train_op = lstm_model(x, y, True, cell, outputW, outputB)

# TRAIN_STEPS = 30001
TRAIN_STEPS = 201
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
start = time.time()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(TRAIN_STEPS):
        # 随机抽样
        randIdx = np.random.randint(0, TRAIN_NUM, BATCH_SIZE)
        _, mse, lr = sess.run([train_op, loss, learning_rate],
                              feed_dict={x: train_x[randIdx, :, :],
                                         y: train_y[randIdx, :]})
        if j % 100 == 0:
            print("After training %s times..." % j)
            print("MSE: %.5f" % mse)
            print("learning rate: %.5f" % lr)
            end = time.time()
            print("Cost %.2f s" % (end - start))
            start = end

    # 最后计算测试集
    prediction, _, _ = lstm_model(x, y, False, cell, outputW, outputB)
    preval = []
    for k in range(len(test_x)):
        logprice = sess.run(prediction, feed_dict={x: test_x[k:(k+1), :, :]})
        preval.append(logprice[0][0])

    plt.figure(figsize=(16, 9))
    preval = np.exp(preval).ravel()
    test_realval = np.exp(test_y).ravel()
    train_realval = np.exp(train_y).ravel()

    xval = np.arange(0, len(test_realval) + len(train_realval), 1)
    plt.plot(xval[:len(train_realval)], train_realval, label='Train Data')
    plt.plot(xval[len(train_realval):], test_realval, label='Real Test Data')
    plt.plot(xval[len(train_realval):], preval, label='Predictions Test')
    plt.legend()
    plt.show()

