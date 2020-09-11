import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.datasets.mnist import load_data
import time
import os

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

(x_train, y_train), (x_test, y_test) = load_data(r'D:\PythonWorkspace\TFTryout\mnist.npz')

# 输入均为平面2维向量，即2*1矩阵
# s1 = tf.constant([[1.0], [2.0]])
# s2 = tf.constant([[2.5], [5.0]])
# ans1 = tf.constant([[7.0]])
# ans2 = tf.constant([[8.0]])
# 通过PlaceHolder作为存放输入数据的地方，以提高计算图的利用
# batchSize = 2
x = tf.placeholder(dtype=tf.float32, shape=(2, None), name='x')
y = tf.placeholder(dtype=tf.float32, shape=(1, None), name='y')

# 第1层隐藏层为3个神经元，故权重为一个3*2的矩阵，偏置为3*1
layer1W = tf.Variable(tf.random_normal([3, 2], stddev=1), name='layer1Weights')
layer1B = tf.Variable(tf.random_normal([3, 1], stddev=1), name='layer1Bias')
l1Output = tf.matmul(layer1W, x) + layer1B
l1Output = tf.sigmoid(l1Output)

# 第2层隐藏层为2个神经元，故权重为一个2*3的矩阵，偏置为2*1
layer2W = tf.Variable(tf.random_normal([2, 3], stddev=1), name='layer2Weights')
layer2B = tf.Variable(tf.random_normal([2, 1], stddev=1), name='layer2Bias')
l2Output = tf.matmul(layer2W, l1Output) + layer2B
l2Output = tf.nn.sigmoid(l2Output)

# 输出层仅有一个神经元，故权重为一个1*2的矩阵，偏置为1*1
outputW = tf.Variable(tf.random_normal([1, 2], stddev=1), name='outputWeights')
outputB = tf.Variable(tf.random_normal([1, 1], stddev=1), name='outputBias')
fOutput1 = tf.matmul(outputW, l2Output) + outputB

# 定义差距和优化方法，并记录迭代轮数
global_step = tf.Variable(0, trainable=False)
mse = tf.reduce_mean(tf.square(fOutput1 - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse, global_step=global_step)

# 加入L2正则化控制模型复杂度，注意TensorFlow2.0之后完全弃用contrib包！
# tf.add_to_collection('totalLoss', tf.contrib.layers.l2_regularizer(0.5)(layer1W))
# tf.add_to_collection('totalLoss', tf.contrib.layers.l2_regularizer(0.5)(layer2W))
# tf.add_to_collection('totalLoss', tf.contrib.layers.l2_regularizer(0.5)(outputW))
# tf.add_to_collection('totalLoss', mse)
# total_loss = tf.add_n(tf.get_collection('totalLoss'))
# train = tf.train.GradientDescentOptimizer(0.01).minimize(total_loss)

# 使用指数衰减法设置学习率
# global_step = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=False)
# train = tf.train.GradientDescentOptimizer(learning_rate)\
#                 .minimize(mse, global_step=global_step)

# 预测一次
s1 = tf.constant([[1.1], [2.1]])
l1 = tf.matmul(layer1W, s1) + layer1B
l1 = tf.sigmoid(l1)
l2 = tf.matmul(layer2W, l1) + layer2B
l2 = tf.sigmoid(l2)
f = tf.matmul(outputW, l2) + outputB

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(layer1W))
    print(sess.run(layer1B))
    print(sess.run(l1Output, feed_dict={x: [[1.0, 2.5],
                                            [2.0, 5.0]],
                                        y: [[7.0, 9.0]]}))
    print(sess.run(fOutput1, feed_dict={x: [[1.0, 2.5],
                                            [2.0, 5.0]],
                                        y: [[7.0, 9.0]]}))
    print(sess.run(mse, feed_dict={x: [[1.0, 2.5],
                                       [2.0, 5.0]],
                                   y: [[7.0, 9.0]]}))
    sess.run(train, feed_dict={x: [[1.0, 2.5],
                                   [2.0, 5.0]],
                               y: [[7.0, 9.0]]})
    print("After training once...")
    print("Global Step = " + str(sess.run(global_step)))
    print(sess.run(fOutput1, feed_dict={x: [[1.0, 2.5],
                                            [2.0, 5.0]],
                                        y: [[7.0, 9.0]]}))
    print(sess.run(mse, feed_dict={x: [[1.0, 2.5],
                                       [2.0, 5.0]],
                                   y: [[7.0, 9.0]]}))
    start = time.perf_counter()
    for i in range(10000):
        sess.run(train, feed_dict={x: [[1.0, 2.5],
                                       [2.0, 5.0]],
                                   y: [[7.0, 9.0]]})

    print("After training 10000 times...")
    print("Global Step = " + str(sess.run(global_step)))
    print(sess.run(fOutput1, feed_dict={x: [[1.0, 2.5],
                                            [2.0, 5.0]],
                                        y: [[7.0, 9.0]]}))
    print(sess.run(mse, feed_dict={x: [[1.0, 2.5],
                                       [2.0, 5.0]],
                                   y: [[7.0, 9.0]]}))
    end = time.perf_counter()
    print('training time: %.10f s' % (end - start))  # 8.2002708000 s

    print("Prediction: ")
    print(sess.run(f))

    # s1 = tf.constant([[1.0], [2.0]])
    # s2 = tf.constant([[2.5], [5.0]])
    # ans1 = tf.constant([[7.0]])
    # ans2 = tf.constant([[8.0]])






