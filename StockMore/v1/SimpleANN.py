import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import os
from sklearn.preprocessing import MinMaxScaler

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')


# df = pd.read_csv(r'.\data\sh.600519-7d.csv')
df = pd.read_csv(r'.\data\sh.600519-3d.csv')

# 拼接需要的特征列
df_use = pd.concat([df['open'], df['close'], df['high'], df['low'],
                    df['volume'], df['amount'], df['turn'], df['pctChg'],
                    df['peTTM'], df['pbMRQ'], df['pcfNcfTTM']], axis=1)
# 归一化
scaler = MinMaxScaler().fit(df_use)
df_use = scaler.transform(df_use)

# 记录下nan的行作为预测用  11 * 5
#  4-28 -- 5-7
predict = df_use[df['close_delay'].isna()].T
# 非Nan行为训练集，训练特征归一化
# 特征 11 * 1295，每列为一个样本
features = df_use[np.bitwise_not(df['close_delay'].isna())].T

# predict = np.asarray([predict['open'], predict['close'],
#                       predict['high'], predict['low'],
#                       predict['volume'], predict['amount'],
#                       predict['turn'], predict['pctChg'],
#                       predict['peTTM'], predict['pbMRQ'],
#                       predict['pcfNcfTTM']], dtype=np.float64)

# 预测的收盘价并不归一化
labels = np.asarray(df['close_delay'].loc[np.bitwise_not(df['close_delay'].isna())], dtype=np.float64)
labels = np.reshape(labels, (1, -1))  # 1 * 1295

# 输入为11 * n
x = tf.placeholder(dtype=tf.float64, shape=(11, None), name='x')
y = tf.placeholder(dtype=tf.float64, shape=(1, None), name='y')

# 第一层50个神经元  50 * 11
layer1W = tf.Variable(tf.random_normal([50, 11], stddev=1, dtype=tf.float64), name='layer1Weights', dtype=tf.float64)
layer1B = tf.Variable(tf.random_normal([50, 1], stddev=1, dtype=tf.float64), name='layer1Bias', dtype=tf.float64)
l1Output = tf.matmul(layer1W, x) + layer1B
l1Output = tf.nn.sigmoid(l1Output)
# l1Output = tf.nn.tanh(l1Output)
# l1Output = tf.nn.relu(l1Output)

# 第二层30个神经元  30 * 50
layer2W = tf.Variable(tf.random_normal([30, 50], stddev=1, dtype=tf.float64), name='layer2Weights', dtype=tf.float64)
layer2B = tf.Variable(tf.random_normal([30, 1], stddev=1, dtype=tf.float64), name='layer2Bias', dtype=tf.float64)
l2Output = tf.matmul(layer2W, l1Output) + layer2B
# l2Output = tf.nn.sigmoid(l2Output)
# l2Output = tf.nn.tanh(l2Output)
l2Output = tf.nn.relu(l2Output)

# 第三层20个神经元  20 * 30
layer3W = tf.Variable(tf.random_normal([20, 30], stddev=1, dtype=tf.float64), name='layer2Weights', dtype=tf.float64)
layer3B = tf.Variable(tf.random_normal([20, 1], stddev=1, dtype=tf.float64), name='layer2Bias', dtype=tf.float64)
l3Output = tf.matmul(layer3W, l2Output) + layer3B
# l3Output = tf.nn.sigmoid(l3Output)
l3Output = tf.nn.relu(l3Output)

# 第四层10个神经元   10 * 20
layer4W = tf.Variable(tf.random_normal([10, 20], stddev=1, dtype=tf.float64), name='layer2Weights', dtype=tf.float64)
layer4B = tf.Variable(tf.random_normal([10, 1], stddev=1, dtype=tf.float64), name='layer2Bias', dtype=tf.float64)
l4Output = tf.matmul(layer4W, l3Output) + layer4B
l4Output = tf.nn.sigmoid(l4Output)
# l4Output = tf.nn.relu(l4Output)

# 输出层1个神经元，无激活输出 1 * 10
outputW = tf.Variable(tf.random_normal([1, 10], stddev=1, dtype=tf.float64), name='outputWeights', dtype=tf.float64)
outputB = tf.Variable(tf.random_normal([1, 1], stddev=1, dtype=tf.float64), name='outputBias', dtype=tf.float64)
final_output = tf.matmul(outputW, l4Output) + outputB

# 使用MSE作为损失函数
mse = tf.reduce_mean(tf.square(final_output - y))
train = tf.train.GradientDescentOptimizer(0.1).minimize(mse)
# 使用指数衰减法设置学习率
# global_step = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(0.5, global_step, 100, 0.96, staircase=False)
# train = tf.train.GradientDescentOptimizer(learning_rate)\
#                 .minimize(mse, global_step=global_step)

# 预测一次
s1 = tf.constant(predict, dtype=tf.float64)
l1 = tf.matmul(layer1W, s1) + layer1B
l1 = tf.sigmoid(l1)
l2 = tf.matmul(layer2W, l1) + layer2B
l2 = tf.sigmoid(l2)
l3 = tf.matmul(layer3W, l2) + layer3B
l3 = tf.sigmoid(l3)
l4 = tf.matmul(layer4W, l3) + layer4B
l4 = tf.sigmoid(l4)
f = tf.matmul(outputW, l4) + outputB

# 开始训练
train_times = 10000
batchsize = 20
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_times):
        # 轮流送样
        # startIdx = (i * batchsize) % (features.shape[1])
        # endIdx = startIdx + batchsize
        # tempX = features[:, startIdx:endIdx]
        # tempY = labels[:, startIdx:endIdx]
        # 随机抽样
        randIdx = np.random.randint(0, features.shape[1], batchsize)
        sess.run(train, feed_dict={x: features[:, randIdx],
                                   y: labels[:, randIdx]})
        # 每训练100次检查一次
        if i % 100 == 0:
            randIdx = np.random.randint(0, features.shape[1], 100)
            print("After training %s times..." % i)
            testmse = sess.run(mse, feed_dict={x: features[:, randIdx],
                                               y: labels[:, randIdx]})
            print("MSE: ")
            print(testmse)
            # print("learning_rate: %s" % sess.run(learning_rate))

    # l1w = sess.run(layer1W)
    # l1b = sess.run(layer1B)
    # l2w = sess.run(layer2W)
    # l2b = sess.run(layer2B)
    # ow = sess.run(outputW)
    # ob = sess.run(outputB)

    # 训练后再喂100组数据观察损失MSE
    print("After training %s times..." % train_times)
    testout = sess.run(final_output,
                       feed_dict={x: features[:, 0:100],
                                  y: labels[:, 0:100]})
    testmse = sess.run(mse, feed_dict={x: features[:, 0:100],
                                       y: labels[:, 0:100]})
    print("Output: ")
    print(testout)
    print("MSE: ")
    print(testmse)
    #  再检测  4-28 -- 5-7 日后的开盘价
    print("Prediction Test: ")
    predictout = sess.run(f)
    print(predictout)
