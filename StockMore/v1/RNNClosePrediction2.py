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

# 特征 1227 * 13，每行为一个样本
features = df_use

# 预测的收盘价并不归一化，预测收盘价取对数
labels = np.asarray(df['close'], dtype=np.float32)
labels = np.log(labels)  # 1227,
labels = np.reshape(labels, (-1, 1))
f_num = features.shape[1]  # 13


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
TIME_STEPS = 10
train_x, train_y, test_x, test_y = data_generate(features, labels, TRAIN_NUM, TIME_STEPS)
# train_x.shape = 1000, 5, 13  TRAIN_NUM, TIME_STEPS, featureNum


# 定义多层RNN模型
# 第一层20个隐藏状态
l1_hiddensize = 25
with tf.variable_scope('rnnlayer1'):
    tf.get_variable(name='h0', shape=(1, l1_hiddensize),
                    dtype=tf.float32, trainable=False,
                    initializer=tf.constant_initializer(0.0))
    tf.get_variable(name='weights', shape=((f_num + l1_hiddensize), l1_hiddensize),
                    dtype=tf.float32, trainable=True,
                    initializer=tf.random_normal_initializer(stddev=1.0))
    tf.get_variable(name='bias', shape=(1, l1_hiddensize),
                    dtype=tf.float32, trainable=True,
                    initializer=tf.constant_initializer(0.0))
# 第二层10个隐藏状态
l2_hiddensize = 13
with tf.variable_scope('rnnlayer2'):
    tf.get_variable(name='h0', shape=(1, l2_hiddensize),
                    dtype=tf.float32, trainable=False,
                    initializer=tf.constant_initializer(0.0))
    tf.get_variable(name='weights', shape=((l1_hiddensize + l2_hiddensize), l2_hiddensize),
                    dtype=tf.float32, trainable=True,
                    initializer=tf.random_normal_initializer(stddev=1.0))
    tf.get_variable(name='bias', shape=(1, l2_hiddensize),
                    dtype=tf.float32, trainable=True,
                    initializer=tf.constant_initializer(0.0))

# 输出全连接网络
with tf.variable_scope('fclayer'):
    tf.get_variable(name='weights', shape=(l2_hiddensize, 1),
                    dtype=tf.float32, trainable=True,
                    initializer=tf.random_normal_initializer(stddev=1.0))
    tf.get_variable(name='bias', shape=(1, 1),
                    dtype=tf.float32, trainable=True,
                    initializer=tf.constant_initializer(0.0))


# 定义推断函数，x.shape = batchsize, TIME_STEPS, featureNum
def infer(x, y, activFunc, is_train):
    # 外层循环为遍历batch
    batch_output = None
    for i in range(x.shape[0]):
        batch = x[i, :, :]
        true_value = y[i:(i+1), :]
        # 内层循环为遍历timeSteps序列
        lastState1 = None
        lastState2 = None
        for j in range(x.shape[1]):
            with tf.variable_scope("rnnlayer1", reuse=True):
                rnn1_h0 = tf.get_variable(name='h0', dtype=tf.float32)
                rnn1_w = tf.get_variable(name='weights', dtype=tf.float32)
                rnn1_b = tf.get_variable(name='bias', dtype=tf.float32)
                if j == 0:
                    concatInput1 = tf.concat([rnn1_h0, batch[j:(j + 1), :]], 1)
                else:
                    concatInput1 = tf.concat([lastState1, batch[j:(j + 1), :]], 1)
                rnn1_output = tf.matmul(concatInput1, rnn1_w) + rnn1_b
                rnn1_output = activFunc(rnn1_output)  # 1 * 20
                lastState1 = rnn1_output

            with tf.variable_scope("rnnlayer2", reuse=True):
                rnn2_h0 = tf.get_variable(name='h0', dtype=tf.float32)
                rnn2_w = tf.get_variable(name='weights', dtype=tf.float32)
                rnn2_b = tf.get_variable(name='bias', dtype=tf.float32)
                if j == 0:
                    concatInput2 = tf.concat([rnn2_h0, rnn1_output], 1)
                else:
                    concatInput2 = tf.concat([lastState2, rnn1_output], 1)
                rnn2_output = tf.matmul(concatInput2, rnn2_w) + rnn2_b
                rnn2_output = activFunc(rnn2_output)  # 1 * 10
                lastState2 = rnn2_output

            # 最后一个序列的输出再计算损失
            if j == (x.shape[1] - 1):
                with tf.variable_scope('fclayer', reuse=True):
                    out_w = tf.get_variable(name='weights', dtype=tf.float32)
                    out_b = tf.get_variable(name='bias', dtype=tf.float32)
                    output = tf.matmul(rnn2_output, out_w) + out_b
                    if batch_output is None:
                        batch_output = output
                    else:
                        batch_output = tf.concat([batch_output, output], 0)
                # mse = tf.reduce_mean(tf.square(output - true_value))
                # tf.add_to_collection('totalLoss', mse)
    if not is_train:
        return batch_output, None, None

    # total_loss = tf.add_n(tf.get_collection('totalLoss'))
    total_loss = tf.reduce_mean(tf.square(batch_output - y))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(total_loss)
    return batch_output, total_loss, train_op


# 定义激活函数
# active_func = tf.nn.sigmoid
active_func = tf.nn.tanh
# active_func = tf.nn.relu

BATCH_SIZE = 50
x = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, TIME_STEPS, f_num), name='x')
y = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 1), name='y')

_, loss, train = infer(x, y, active_func, True)
trainTimes = 5
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
start = time.time()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(trainTimes):
        # tf.get_default_graph().clear_collection('totalLoss')
        # 随机抽样
        # randIdx = np.random.randint(0, TRAIN_NUM, BATCH_SIZE)
        # _, l = sess.run([train, loss], feed_dict={x: train_x[randIdx, :, :],
        #                                           y: train_y[randIdx, :]})
        # if j % 100 == 0:
        #     print("After training %s times..." % j)
        #     print("MSE: %.5f" % l)
        #     end = time.time()
        #     print("Cost %.2f s" % (end - start))
        #     start = end

        # 按顺序送样
        for k in range(TRAIN_NUM - BATCH_SIZE):
            startIdx = k
            endIdx = startIdx + BATCH_SIZE
            _, l = sess.run([train, loss], feed_dict={x: train_x[startIdx:endIdx, :, :],
                                                      y: train_y[startIdx:endIdx, :]})
        print("After training %s times..." % (j+1))
        print("MSE: %.5f" % l)
        end = time.time()
        print("Cost %.2f s" % (end - start))
        start = end

    # 最后计算测试集
    x1 = tf.placeholder(dtype=tf.float32, shape=(1, TIME_STEPS, f_num), name='x')
    y1 = tf.placeholder(dtype=tf.float32, shape=(1, 1), name='y')
    prediction, _, _ = infer(x1, y1, active_func, False)
    preval = []
    for k in range(len(test_x)):
        logprice = sess.run(prediction, feed_dict={x1: test_x[k:(k+1), :, :]})
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

