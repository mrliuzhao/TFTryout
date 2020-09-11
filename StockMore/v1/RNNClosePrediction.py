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
df_use = np.asarray(df_use, dtype=np.float64)
# 特征值归一化
scaler = MinMaxScaler().fit(df_use)
df_use = scaler.transform(df_use)
# df_use.shape = (1227, 13)

# 特征 1227 * 13，每行为一个样本
features = df_use

# 预测的收盘价并不归一化，预测收盘价取对数
labels = np.asarray(df['close'], dtype=np.float64)
labels = np.log(labels)  # 1227,
labels = np.reshape(labels, (-1, 1))

# 取前900个为训练集，后面的为测试集
sample_num = 1000
series_num = 5
train_features = features[:sample_num, :]  # 900 * 13
test_features = features[(sample_num - series_num):, :]  # 322 * 13
f_num = train_features.shape[1]  # 13
# 预测后一天的收盘价
train_labels = labels[1:sample_num+1]   # 900,1
test_labels = labels[sample_num:]    # 322,

# 定义激活函数
# active_func = tf.nn.sigmoid
active_func = tf.nn.tanh
# active_func = tf.nn.relu

# 状态维度  1 * 6
# state_num = 6
state_num = f_num
h0 = tf.zeros((1, state_num), dtype=tf.float64, name='h0')
x = tf.placeholder(dtype=tf.float64, shape=(series_num, f_num), name='x')
y = tf.placeholder(dtype=tf.float64, shape=(series_num, 1), name='y')

# 循环神经网络8个神经元  19(13+6) * 8
# rnn1_node = 8
# rnn1W = tf.Variable(tf.random_normal([f_num + state_num, rnn1_node], stddev=1, dtype=tf.float64), name='rnn1Weights', dtype=tf.float64)
# rnn1B = tf.Variable(tf.random_normal([1, rnn1_node], stddev=1, dtype=tf.float64), name='rnn1Bias', dtype=tf.float64)

# 循环神经网络6个神经元  8 * 6
# rnn2W = tf.Variable(tf.random_normal([rnn1_node, state_num], stddev=1, dtype=tf.float64), name='rnn2Weights', dtype=tf.float64)
rnn2W = tf.Variable(tf.random_normal([f_num + state_num, state_num], stddev=1, dtype=tf.float64), name='rnn2Weights', dtype=tf.float64)
rnn2B = tf.Variable(tf.random_normal([1, state_num], stddev=1, dtype=tf.float64), name='rnn2Bias', dtype=tf.float64)

# 输出网络6个神经元  6 * 1
outputW = tf.Variable(tf.random_normal([state_num, 1], stddev=1, dtype=tf.float64), name='outputWeights', dtype=tf.float64)
outputB = tf.Variable(tf.random_normal([1, 1], stddev=1, dtype=tf.float64), name='outputBias', dtype=tf.float64)

# 按序列循环输出并累计损失
lastState = None
lastOutput = None
for i in range(series_num):
    # 按列拼接状态和输入向量  1 * 19
    single_feature = x[i:(i+1), :]
    if i == 0:
        # concatInput = tf.concat([single_feature, h0], 1)
        concatInput = tf.concat([h0, single_feature], 1)
    else:
        # concatInput = tf.concat([single_feature, lastState], 1)
        concatInput = tf.concat([lastState, single_feature], 1)
    # rnnOutput = tf.matmul(concatInput, rnn1W) + rnn1B
    # rnnOutput = active_func(rnnOutput)
    # lastState = tf.matmul(rnnOutput, rnn2W) + rnn2B
    lastState = tf.matmul(concatInput, rnn2W) + rnn2B
    lastState = active_func(lastState)

    # 计算输出及损失
    # output = tf.matmul(lastState, outputW) + outputB
    # single_label = y[i:(i+1), :]
    # mse = tf.reduce_mean(tf.square(output - single_label))
    # tf.add_to_collection('totalLoss', mse)
    if i == (series_num - 1):
        output = tf.matmul(lastState, outputW) + outputB
        single_label = y[i:(i + 1), :]
        mse = tf.reduce_mean(tf.square(output - single_label))
        tf.add_to_collection('totalLoss', mse)
        lastOutput = output

total_loss = tf.add_n(tf.get_collection('totalLoss'))
train = tf.train.GradientDescentOptimizer(0.01).minimize(total_loss)
# 使用指数衰减法设置学习率
# batchsize = 20
# global_step = tf.Variable(0)
# decay_steps = sample_num // batchsize
# learning_rate = tf.train.exponential_decay(0.3, global_step, 500, 0.99, staircase=True)
# train = tf.train.GradientDescentOptimizer(learning_rate)\
#                 .minimize(mse, global_step=global_step)

trainTimes = 3001
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
start = time.time()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(trainTimes):
        # 完整遍历一遍数据集
        for i in range(sample_num - series_num + 1):
            sess.run(train, feed_dict={x: train_features[i:(i+series_num), :],
                                       y: train_labels[i:(i+series_num), :]})
        # 每完整遍历数据集100次检查一次
        if j % 100 == 0:
            randIdx = random.randint(0, (sample_num - series_num + 1))
            print("After training %s times..." % j)
            testmse = sess.run(total_loss, feed_dict={
                x: train_features[randIdx:(randIdx+series_num), :],
                y: train_labels[randIdx:(randIdx+series_num), :]})
            print("MSE: ")
            print(testmse)
            end = time.time()
            print("Cost %.2f s" % (end - start))
            start = end

    # 最后计算测试集
    preval = []
    for i in range(test_features.shape[0] - series_num):
        logprice = sess.run(lastOutput, feed_dict={x: test_features[i:(i + series_num), :]})
        preval.append(logprice[0][0])

    plt.figure(figsize=(16, 9))
    preval = np.exp(preval).ravel()
    test_realval = np.exp(test_labels).ravel()
    train_realval = np.exp(train_labels).ravel()

    xval = np.arange(0, len(test_realval) + len(train_realval), 1)
    plt.plot(xval[:len(train_realval)], train_realval, label='Train Data')
    plt.plot(xval[len(train_realval):], test_realval, label='Real Test Data')
    plt.plot(xval[len(train_realval):], preval, label='Predictions Test')
    plt.legend()
    plt.show()

