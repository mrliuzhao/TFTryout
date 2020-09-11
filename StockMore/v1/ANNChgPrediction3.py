import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

# 要选择的特征列
feature_column = ['volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'pcfNcfTTM', 'year', 'month', 'day', 'dd', 'dw', 'monthEnd', 'seasonEnd', 'open', 'close', 'high', 'low']
f_num = len(feature_column)

# 定义神经网络
# 定义激活函数
active_func = tf.nn.sigmoid

# 输入为 14*n，每列为一个样本；输出为 1*n，每列为对应预测值
x = tf.placeholder(dtype=tf.float64, shape=(f_num, None), name='x')
y = tf.placeholder(dtype=tf.float64, shape=(1, None), name='y')

# 第一层8个神经元
l1_node = 8
layer1W = tf.Variable(tf.random_normal([l1_node, f_num], stddev=1, dtype=tf.float64), name='layer1Weights', dtype=tf.float64)
layer1B = tf.Variable(tf.random_normal([l1_node, 1], stddev=1, dtype=tf.float64), name='layer1Bias', dtype=tf.float64)
l1Output = tf.matmul(layer1W, x) + layer1B
l1Output = active_func(l1Output)

# 第二层6个神经元
l2_node = 6
layer2W = tf.Variable(tf.random_normal([l2_node, l1_node], stddev=1, dtype=tf.float64), name='layer2Weights', dtype=tf.float64)
layer2B = tf.Variable(tf.random_normal([l2_node, 1], stddev=1, dtype=tf.float64), name='layer2Bias', dtype=tf.float64)
l2Output = tf.matmul(layer2W, l1Output) + layer2B
l2Output = active_func(l2Output)

# 第三层6个神经元
l3_node = 6
layer3W = tf.Variable(tf.random_normal([l3_node, l2_node], stddev=1, dtype=tf.float64), name='layer3Weights', dtype=tf.float64)
layer3B = tf.Variable(tf.random_normal([l3_node, 1], stddev=1, dtype=tf.float64), name='layer3Bias', dtype=tf.float64)
l3Output = tf.matmul(layer3W, l2Output) + layer3B
l3Output = active_func(l3Output)

# 输出层1个神经元
outputW = tf.Variable(tf.random_normal([1, l3_node], stddev=1, dtype=tf.float64), name='outputWeights', dtype=tf.float64)
outputB = tf.Variable(tf.random_normal([1, 1], stddev=1, dtype=tf.float64), name='outputBias', dtype=tf.float64)
final_output = tf.matmul(outputW, l3Output) + outputB

# 使用MSE作为损失函数
mse = tf.reduce_mean(tf.square(final_output - y))
# 使用指数衰减法设置学习率
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.3, global_step, 200, 0.98, staircase=True)
train = tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(mse, global_step=global_step)

samples_name = ['sz.300463',
                'sh.601318',
                'sz.000063',
                'sz.000651']

count = 1
plt.figure(figsize=(32, 18))
pred_arr = []
for name in samples_name:
    # df = pd.read_csv(r'.\data\%s-10d-raiseprob.csv' % name)
    df = pd.read_csv(r'.\data\%s-07d.csv' % name)
    # 提取年月日作为特征
    df['year'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.year
    df['month'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.month
    df['day'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.day
    # 再计算与一个具体日期的差值作为特征
    df['dd'] = (pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce') - pd.to_datetime("2010-01-01", format='%Y-%m-%d', errors='coerce')).dt.days
    # 标记上每周第几天、月末前3天、季末前7天
    df['dw'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.dayofweek
    df['monthEnd'] = (df['day'] >= 28).astype('int')
    temp = (df['month'] == 3) | (df['month'] == 6) | (df['month'] == 9) | (df['month'] == 12)
    df['seasonEnd'] = ((df['day'] >= 24) & temp).astype('int')

    df_use = pd.concat([df[feature_column]], axis=1)
    scaler = MinMaxScaler().fit(df_use)
    df_use = scaler.transform(df_use)

    # 非Nan行作为训练集
    features = df_use[np.bitwise_not(df['close_delay'].isna())].T
    # 预测复合涨跌幅
    close_delay = np.asarray(df['close_delay'])
    close = np.asarray(df['close'])
    # delta = 100 * ((close_delay - close) / close)
    delta = (close_delay - close) / close
    delta = delta[~np.isnan(delta)]
    labels = np.reshape(delta, (1, -1))

    # 划分训练集和测试集
    train_size = int(0.9 * features.shape[1])
    train_features = features[:, :train_size]
    test_features = features[:, train_size:]
    f_num = train_features.shape[0]
    sample_num = train_features.shape[1]
    train_labels = labels[:, :train_size]
    test_labels = labels[:, train_size:]

    # 开始训练
    train_times = 100000
    # train_times = 10000
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_times):
            # 全量训练
            sess.run(train, feed_dict={x: train_features,
                                       y: train_labels})

            # 每训练100次检查一次
            if i % 1000 == 0:
                randIdx = np.random.randint(0, sample_num, 100)
                print("After training %s times..." % i)
                testmse = sess.run(mse, feed_dict={x: train_features[:, randIdx],
                                                   y: train_labels[:, randIdx]})
                print("MSE: ")
                print(testmse)

        # 全量观察预测值与真实值
        preval = sess.run(final_output, feed_dict={x: test_features})
        preval = preval.ravel()
        test_realval = test_labels.ravel()
        train_realval = train_labels.ravel()

        plt.subplot(2, 2, count)
        plt.xticks([])
        plt.xlabel('Time')
        plt.ylabel('Composite Price Change')
        xval = np.arange(0, len(test_realval) + len(train_realval), 1)
        plt.plot(xval[:len(train_realval)], train_realval, label='Train Data')
        plt.plot(xval[len(train_realval):], test_realval, label='Real Value')
        plt.plot(xval[len(train_realval):], preval, label='Predictions')
        plt.legend()
        plt.title('Predict Composite Price Change of %s' % name)

        pred_arr.append(preval)
        count += 1

plt.show()

