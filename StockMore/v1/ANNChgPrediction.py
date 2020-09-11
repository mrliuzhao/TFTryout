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
# df = pd.read_csv(r'.\data\sh.600519-3d.csv')
df = pd.read_csv(r'.\data\sh.600519-1d.csv')
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

# 标记涨或跌
df['updown'] = (df['pctChg_delay'] >= 0.0).astype('int')

# 拼接需要的特征列
df_use = pd.concat([df['open'], df['close'], df['high'], df['low'],
                    df['volume'], df['amount'], df['turn'], df['pctChg'],
                    df['peTTM'], df['pbMRQ'], df['pcfNcfTTM'],
                    df['year'], df['month'], df['day'], df['dd'],
                    df['dw'], df['monthEnd'], df['seasonEnd']], axis=1)
# df_use = pd.concat([df['volume'], df['amount'], df['turn'], df['pctChg'],
#                     df['peTTM'], df['pbMRQ'], df['pcfNcfTTM'],
#                     df['year'], df['month'], df['day'], df['dd'],
#                     df['dw'], df['monthEnd'], df['seasonEnd']], axis=1)
# df_use = np.asarray(df_use, dtype=np.float64)
# 特征值归一化
scaler = MinMaxScaler().fit(df_use)
df_use = scaler.transform(df_use)

# 记录下nan的行作为预测用  n * 18
predict = df_use[df['pctChg_delay'].isna()]
# 非Nan行作为训练集
# 特征 n * 18，每行为一个样本
features = df_use[np.bitwise_not(df['pctChg_delay'].isna())]
f_num = features.shape[1]
sample_num = features.shape[0]

# 预测涨或跌
labels = np.asarray(df['updown'].loc[np.bitwise_not(df['pctChg_delay'].isna())], dtype=np.int64)


# 定义一个构建神经网络的通用函数
def build_nn(input_num, nodes_nums):
    for ln in range(len(nodes_nums)):
        scope_name = 'layer' + str(ln)
        with tf.variable_scope(scope_name):
            if ln == 0:
                shape = [input_num, nodes_nums[ln]]
            else:
                shape = [nodes_nums[ln-1], nodes_nums[ln]]
            w = tf.get_variable(name='weights', shape=shape,
                                dtype=tf.float64, trainable=True,
                                initializer=tf.random_normal_initializer(stddev=1.0))
            tf.add_to_collection('r2losses', tf.nn.l2_loss(w))
            tf.get_variable(name='bias', shape=nodes_nums[ln],
                            dtype=tf.float64, trainable=True,
                            initializer=tf.constant_initializer(0.0))


# 定义一个推断函数
def infer(input_tensor, ma_cls, act_func, nodes_nums):
    last_output = None
    if ma_cls is None:
        for j in range(len(nodes_nums)):
            scope_name = 'layer' + str(j)
            with tf.variable_scope(scope_name, reuse=True):
                w = tf.get_variable(name='weights', dtype=tf.float64)
                b = tf.get_variable(name='bias', dtype=tf.float64)
                if last_output is None:
                    last_output = act_func(tf.matmul(input_tensor, w) + b)
                else:
                    last_output = act_func(tf.matmul(last_output, w) + b)
    else:
        for j in range(len(nodes_nums)):
            scope_name = 'layer' + str(j)
            with tf.variable_scope(scope_name, reuse=True):
                w = tf.get_variable(name='weights', dtype=tf.float64)
                b = tf.get_variable(name='bias', dtype=tf.float64)
                w_avg = ma_cls.average(w)
                b_avg = ma_cls.average(b)
                if last_output is None:
                    last_output = act_func(tf.matmul(input_tensor, w_avg) + b_avg)
                else:
                    last_output = act_func(tf.matmul(last_output, w_avg) + b_avg)
    return last_output


# 定义激活函数
active_func = tf.nn.sigmoid
# active_func = tf.nn.relu

# 输入为 n * fnum，每行为一个样本；输出为 n个0-1的值，
x = tf.placeholder(dtype=tf.float64, shape=(None, f_num), name='x')
y = tf.placeholder(dtype=tf.int64, shape=None, name='y')

# 定义神经网络结构
layers = [10, 8, 8, 6, 2]
build_nn(f_num, layers)

# final_output = infer(x, None, tf.nn.sigmoid, layers)
final_output = infer(x, None, tf.nn.relu, layers)
correct_pre = tf.equal(tf.arg_max(final_output, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

# 使用SoftMax+交叉熵作为损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=final_output, name='CrossEntropy')
cross_entropy = tf.reduce_mean(cross_entropy)

# 使用指数衰减法设置学习率
# global_step = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(0.5, global_step, 200, 0.98, staircase=True)
# train = tf.train.GradientDescentOptimizer(learning_rate)\
#                 .minimize(cross_entropy, global_step=global_step)
train = tf.train.GradientDescentOptimizer(0.1)\
                .minimize(cross_entropy)

# 预测一次
# f = infer(predict, None, tf.nn.sigmoid, layers)
f = infer(predict, None, tf.nn.relu, layers)

# 开始训练
train_times = 100000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_times):
        # 全量训练
        sess.run(train, feed_dict={x: features,
                                   y: labels})

        # 每训练100次检查一次
        if i % 100 == 0:
            randIdx = np.random.randint(0, sample_num, 100)
            acc = sess.run(accuracy, feed_dict={x: features[randIdx, :],
                                                y: labels[randIdx]})
            entro = sess.run(cross_entropy, feed_dict={x: features[randIdx, :],
                                                       y: labels[randIdx]})
            print("After training %d times, Accuracy is %.5f, Cross Entropy is %.5f"
                  % (i, acc, entro))
            # print("Learning Rate is %.10f" % (sess.run(learning_rate)))

    # 训练后再喂1000组数据观察正确率
    randIdx = np.random.randint(0, sample_num, 1000)
    acc = sess.run(accuracy, feed_dict={x: features[randIdx, :],
                                        y: labels[randIdx]})
    print("Finally, after training %d times, Accuracy is %.5f" % (train_times, acc))

    #  再检测  4-28 -- 5-7 日后的开盘价
    print("Prediction Test: ")
    print(sess.run(f))

