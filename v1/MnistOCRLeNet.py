import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.datasets.mnist import load_data
import time
import os
import numpy as np
from tensorflow.python.framework import graph_util

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

# 读取数据
(x_train, y_train), (x_test, y_test) = load_data(r'D:\PythonWorkspace\TFTryout\mnist.npz')
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000, )
print(x_test.shape)  # (10000, 28, 28)
print(y_test.shape)  # (10000, )
Train_Num = x_train.shape[0]
Test_Num = x_test.shape[0]
channels = 1
height = x_train.shape[1]
width = x_train.shape[2]
# 注意需要将输入的矩阵也进行reshape
x_train = np.reshape(x_train, (Train_Num, height, width, channels))
x_test = np.reshape(x_test, (Test_Num, height, width, channels))

batchsize = 1000
# x = tf.placeholder(dtype=tf.float64, shape=(batchsize, height, width, channels), name='x')
x = tf.placeholder(dtype=tf.float64, shape=(None, height, width, channels), name='x')
y_lab = tf.placeholder(dtype=tf.int64, shape=None, name='y')

# 抽取一部分作为验证集
valid_idx = np.random.randint(0, Train_Num, batchsize)
x_validate = x_train[valid_idx, :]
y_validate = y_train[valid_idx]


# ---------------------------- 开始定义LeNet -------------------------------
# 定义第1层——卷积层，卷积核大小为 5*5*1 * depth (相当于depth个卷积核)
with tf.variable_scope("layer1-conv"):
    conv1_depth = 16
    tf.get_variable(name='weights', shape=[5, 5, channels, conv1_depth],
                    dtype=tf.float64, trainable=True,
                    initializer=tf.random_normal_initializer(stddev=1.0))
    # 对应depth个卷积层1的偏置
    tf.get_variable(name='bias', shape=[conv1_depth],
                    dtype=tf.float64, trainable=True,
                    initializer=tf.constant_initializer(0.0))

# 第2层为最大值池化层，不用额外定义
# 定义卷积层2的卷积核，大小为 5*5*1 * depth
with tf.variable_scope("layer2-conv"):
    conv2_depth = 32
    tf.get_variable(name='weights', shape=[5, 5, channels, conv2_depth],
                    dtype=tf.float64, trainable=True,
                    initializer=tf.random_normal_initializer(stddev=1.0))
    tf.get_variable(name='bias', shape=[conv2_depth],
                    dtype=tf.float64, trainable=True,
                    initializer=tf.constant_initializer(0.0))

# 第4层为最大值池化层，不用额外定义


# 定义一个构建全连接层神经网络的通用函数
def build_nn(input_count, nodes):
    for ln in range(len(nodes)):
        scope_name = 'fclayer' + str(ln)
        with tf.variable_scope(scope_name):
            if ln == 0:
                shape = [input_count, nodes[ln]]
            else:
                shape = [nodes[ln-1], nodes[ln]]
            tf.get_variable(name='weights', shape=shape,
                            dtype=tf.float64, trainable=True,
                            initializer=tf.random_normal_initializer(stddev=1.0))
            tf.get_variable(name='bias', shape=nodes[ln],
                            dtype=tf.float64, trainable=True,
                            initializer=tf.constant_initializer(0.0))


input_num = 5 * 5 * conv2_depth
# fcNodes = [120, 84, 10]
fcNodes = [512, 10]
build_nn(input_num, fcNodes)


# ---------------------------- 定义前向传播函数 -------------------------------
def infer(input_tensor, act_func, train):
    # 卷积层1
    with tf.variable_scope("layer1-conv", reuse=True):
        l1w = tf.get_variable(name='weights', dtype=tf.float64)
        l1b = tf.get_variable(name='bias', dtype=tf.float64)
        # 使用SAME模式、各方向步长均为1进行卷积，最后输出为 batchSize * 28 * 28 * conv1_depth
        conv1_out = tf.nn.conv2d(input_tensor, filter=l1w, strides=[1, 1, 1, 1], padding='SAME')
        conv1_out = tf.nn.bias_add(conv1_out, l1b)
        conv1_out = act_func(conv1_out)

    # 最大值池化层1
    # 使用VALID模式进行 2 * 2 的最大值池化，输出为 batchSize * 14 * 14 * conv1_depth
    pool1_out = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 卷积层2
    with tf.variable_scope("layer2-conv", reuse=True):
        l2w = tf.get_variable(name='weights', dtype=tf.float64)
        l2b = tf.get_variable(name='bias', dtype=tf.float64)
        # 使用VALID模式、各方向步长均为1进行卷积，最后输出则为 batchSize * 10 * 10 * conv2_depth
        conv2_out = tf.nn.conv2d(pool1_out, filter=l2w, strides=[1, 1, 1, 1], padding='VALID')
        conv2_out = tf.nn.bias_add(conv2_out, l2b)
        conv2_out = act_func(conv2_out)

    # 最大值池化层2
    # 使用VALID模式进行 2 * 2 的最大值池化，输出为 batchSize * 5 * 5 * conv2_depth
    pool2_out = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # reshape池化层2的输出，“拉直”后用于全连接层
    pool2_shape = pool2_out.get_shape().as_list()
    # print("Shape of Pool 2: " + str(pool2_shape))
    fcinput = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    # print("FC Input Number: " + str(fcinput))
    reshaped = tf.reshape(pool2_out, [pool2_shape[0], fcinput])

    # 最后添加几个全连接层，神经元数量分别为 120, 84, 10
    last_output = None
    for j in range(len(fcNodes)):
        scope_name = 'fclayer' + str(j)
        with tf.variable_scope(scope_name, reuse=True):
            w = tf.get_variable(name='weights', dtype=tf.float64)
            b = tf.get_variable(name='bias', dtype=tf.float64)
            if last_output is None:
                last_output = tf.matmul(reshaped, w) + b
            else:
                last_output = tf.matmul(last_output, w) + b
            # 不是最后一层再进行激活！！！
            if j < len(fcNodes) - 1:
                last_output = act_func(last_output)
                # 若为训练阶段，全连接层中的输出再进行dropout
                if train:
                    last_output = tf.nn.dropout(last_output, rate=0.5)
    return last_output


# 计算预测值  n * 10
# y_pre = infer(x, tf.nn.relu, True)
y_pre = infer(x, tf.nn.sigmoid, True)

# 使用SoftMax+交叉熵作为损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_lab, logits=y_pre, name='CrossEntropy')
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_lab, logits=y_pre, name='CrossEntropy')
cross_entropy = tf.reduce_mean(cross_entropy)

# 使用指数衰减法设置学习率
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.8, global_step, Train_Num//batchsize, 0.99, staircase=False)
trainStep = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(cross_entropy, global_step=global_step)
# trainStep = tf.train.GradientDescentOptimizer(0.01)\
#                     .minimize(cross_entropy)

# 计算预测结果和正确率
# y_pre_ma = infer(x, tf.nn.relu, False)
y_pre_ma = infer(x, tf.nn.sigmoid, False)
correct_pre = tf.equal(tf.arg_max(y_pre_ma, 1), y_lab)
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

train_times = 50000
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    start = time.perf_counter()
    for i in range(train_times):
        randIdx = np.random.randint(0, Train_Num, batchsize)
        sess.run(trainStep, feed_dict={x: x_train[randIdx, :],
                                       y_lab: y_train[randIdx]})
        # 每1000轮训练后检测一下正确率
        if i % 1000 == 0:
            acc = sess.run(accuracy, feed_dict={x: x_validate,
                                                y_lab: y_validate})
            cro_ent = sess.run(cross_entropy, feed_dict={x: x_validate,
                                                         y_lab: y_validate})
            print("After training %d steps, accuracy is %g. Cost %f s" %
                  (i, acc, (time.perf_counter() - start)))
            print("Cross Entropy is %f." % cro_ent)
            print("Global Step is %f." % sess.run(global_step))
            print("Learning Rate is %f." % sess.run(learning_rate))
            start = time.perf_counter()

    # 训练结束后在测试集上进行正确率验证
    acc = sess.run(accuracy, feed_dict={x: x_test,
                                        y_lab: y_test})
    print("After training %d steps, accuracy on test dataset is %g"
          % (train_times, acc))




