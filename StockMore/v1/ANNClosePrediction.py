import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

# df = pd.read_csv(r'.\data\sh.600519-07d.csv')
df = pd.read_csv(r'.\data\sz.000063-07d.csv')
# df = pd.read_csv(r'.\data\sh.600519-03d.csv')
# df = pd.read_csv(r'.\data\sh.600519-01d.csv')
# 提取年月日作为特征
df['year'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.year
df['month'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.month
df['day'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.day
# 再计算与一个具体日期的差值作为特征
df['dd'] = (pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce') - pd.to_datetime("2000-01-01", format='%Y-%m-%d', errors='coerce')).dt.days
# 标记上每周第几天、月末前3天、季末前7天
df['dw'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.dayofweek
df['monthEnd'] = (df['day'] >= 28).astype('int')
temp = (df['month'] == 3) | (df['month'] == 6) | (df['month'] == 9) | (df['month'] == 12)
df['seasonEnd'] = ((df['day'] >= 24) & temp).astype('int')

# 拼接需要的特征列  1300 * 15
# df_use = pd.concat([df['open'], df['close'], df['high'], df['low'],
#                     df['volume'], df['amount'], df['turn'], df['pctChg'],
#                     df['peTTM'], df['pbMRQ'], df['pcfNcfTTM'],
#                     df['year'], df['month'], df['day'], df['dd'],
#                     df['dw'], df['monthEnd'], df['seasonEnd']], axis=1)
df_use = pd.concat([df['volume'], df['amount'], df['turn'], df['pctChg'],
                    df['peTTM'], df['pbMRQ'], df['pcfNcfTTM'],
                    df['year'], df['month'], df['day'], df['dd'],
                    df['dw'], df['monthEnd'], df['seasonEnd']], axis=1)
# df_use = np.asarray(df_use, dtype=np.float64)
# 特征值归一化
scaler = MinMaxScaler().fit(df_use)
df_use = scaler.transform(df_use)

# 记录下nan的行作为预测用  15 * 5
predict = df_use[df['close_delay'].isna()].T
# 非Nan行作为训练集
# 特征 15 * 1295，每列为一个样本
features = df_use[np.bitwise_not(df['close_delay'].isna())].T

# 预测的收盘价并不归一化，预测收盘价取对数
labels = np.asarray(df['close_delay'].loc[np.bitwise_not(df['close_delay'].isna())], dtype=np.float64)
labels = np.log(labels)
labels = np.reshape(labels, (1, -1))  # 整形为  1 * 1295

# 取前1000个为训练集，后面的为测试集
train_features = features[:, :1000]
test_features = features[:, 1000:]
f_num = train_features.shape[0]
sample_num = train_features.shape[1]
train_labels = labels[:, :1000]
test_labels = labels[:, 1000:]

# 定义激活函数
active_func = tf.nn.sigmoid
# active_func = tf.nn.relu

# 输入为 14*n，每列为一个样本；输出为 1*n，每列为对应预测值
x = tf.placeholder(dtype=tf.float64, shape=(f_num, None), name='x')
y = tf.placeholder(dtype=tf.float64, shape=(1, None), name='y')

# 第一层8个神经元  8 * 14
l1_node = 8
layer1W = tf.Variable(tf.random_normal([l1_node, f_num], stddev=1, dtype=tf.float64), name='layer1Weights', dtype=tf.float64)
layer1B = tf.Variable(tf.random_normal([l1_node, 1], stddev=1, dtype=tf.float64), name='layer1Bias', dtype=tf.float64)
l1Output = tf.matmul(layer1W, x) + layer1B
l1Output = active_func(l1Output)

# 第二层6个神经元  6 * 8
l2_node = 6
layer2W = tf.Variable(tf.random_normal([l2_node, l1_node], stddev=1, dtype=tf.float64), name='layer2Weights', dtype=tf.float64)
layer2B = tf.Variable(tf.random_normal([l2_node, 1], stddev=1, dtype=tf.float64), name='layer2Bias', dtype=tf.float64)
l2Output = tf.matmul(layer2W, l1Output) + layer2B
l2Output = active_func(l2Output)

# 第三层6个神经元  6 * 6
l3_node = 6
layer3W = tf.Variable(tf.random_normal([l3_node, l2_node], stddev=1, dtype=tf.float64), name='layer3Weights', dtype=tf.float64)
layer3B = tf.Variable(tf.random_normal([l3_node, 1], stddev=1, dtype=tf.float64), name='layer3Bias', dtype=tf.float64)
l3Output = tf.matmul(layer3W, l2Output) + layer3B
l3Output = active_func(l3Output)

# 输出层1个神经元，无激活输出 1 * 10
outputW = tf.Variable(tf.random_normal([1, l3_node], stddev=1, dtype=tf.float64), name='outputWeights', dtype=tf.float64)
outputB = tf.Variable(tf.random_normal([1, 1], stddev=1, dtype=tf.float64), name='outputBias', dtype=tf.float64)
final_output = tf.matmul(outputW, l3Output) + outputB
pricePredict = tf.exp(final_output, name='PricePrediction')

# 使用MSE作为损失函数
mse = tf.reduce_mean(tf.square(final_output - y))
# train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)
# 使用指数衰减法设置学习率
batchsize = 20
global_step = tf.Variable(0)
decay_steps = sample_num // batchsize
learning_rate = tf.train.exponential_decay(0.3, global_step, 500, 0.99, staircase=True)
train = tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(mse, global_step=global_step)

# 预测一次
s1 = tf.constant(test_features, dtype=tf.float64)
l1 = tf.matmul(layer1W, s1) + layer1B
l1 = active_func(l1)
l2 = tf.matmul(layer2W, l1) + layer2B
l2 = active_func(l2)
l3 = tf.matmul(layer3W, l2) + layer3B
l3 = active_func(l3)
f = tf.matmul(outputW, l3) + outputB

# 将计算图写入日志
# writer = tf.summary.FileWriter(r"D:\PythonWorkspace\TFTryout\logs", tf.get_default_graph())
# writer.close()

# 开始训练
saver = tf.train.Saver()
train_times = 50000
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_times):
        # 轮流送样
        # startIdx = (i * batchsize) % sample_num
        # endIdx = startIdx + batchsize
        # tempX = features[:, startIdx:endIdx]
        # tempY = labels[:, startIdx:endIdx]
        # sess.run(train, feed_dict={x: tempX,
        #                            y: tempY})
        # 随机抽样
        # randIdx = np.random.randint(0, features.shape[1], batchsize)
        # sess.run(train, feed_dict={x: features[:, randIdx],
        #                            y: labels[:, randIdx]})
        # 考虑随机而不重复！

        # 全量训练
        sess.run(train, feed_dict={x: train_features,
                                   y: train_labels})

        # 每训练1000次检查一次
        if i % 1000 == 0:
            randIdx = np.random.randint(0, train_features.shape[1], 100)
            print("After training %s times..." % i)
            testmse = sess.run(mse, feed_dict={x: train_features[:, randIdx],
                                               y: train_labels[:, randIdx]})
            print("MSE: ")
            print(testmse)
            # print("learning_rate: %.15f" % sess.run(learning_rate))

    l1w = sess.run(layer1W)
    l1b = sess.run(layer1B)
    l2w = sess.run(layer2W)
    l2b = sess.run(layer2B)
    l3w = sess.run(layer3W)
    l3b = sess.run(layer3B)
    ow = sess.run(outputW)
    ob = sess.run(outputB)

    # 持久化训练后模型，保存为ckpt格式
    # saver.save(sess, r'.\models\sh.600519-closeprice-7d.ckpt')

    # 保存为pb模式
    # graph_def = tf.get_default_graph().as_graph_def()
    # output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['PricePrediction'])
    # with tf.gfile.GFile(r'.\models\sh.600519-closeprice-7d.pb', 'wb') as f:
    #     f.write(output_graph.SerializeToString())

    # 训练后再喂100组数据观察损失MSE
    # print("After training %s times..." % train_times)
    # randIdx = np.random.randint(0, features.shape[1], 100)
    # testout = sess.run(final_output,
    #                    feed_dict={x: features[:, randIdx],
    #                               y: labels[:, randIdx]})
    # testmse = sess.run(mse, feed_dict={x: features[:, randIdx],
    #                                    y: labels[:, randIdx]})
    # print("Output: ")
    # print(np.exp(testout))
    # print("MSE: ")
    # print(testmse)
    # #  再检测  4-28 -- 5-7 日后的开盘价
    # print("Prediction Test: ")
    # predictout = sess.run(f)
    # predictout = np.exp(predictout)
    # print(predictout)

    # 全量观察预测值与真实值
    plt.figure(figsize=(16, 9))
    preval = sess.run(f)
    preval = np.exp(preval).ravel()
    test_realval = np.exp(test_labels).ravel()
    train_realval = np.exp(train_labels).ravel()

    xval = np.arange(0, len(test_realval) + len(train_realval), 1)
    plt.plot(xval[:len(train_realval)], train_realval, label='Train Data')
    plt.plot(xval[len(train_realval):], test_realval, label='Real Test Data')
    plt.plot(xval[len(train_realval):], preval, label='Predictions Test')
    plt.legend()
    plt.show()

