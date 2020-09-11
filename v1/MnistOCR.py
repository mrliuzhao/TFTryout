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
Class_Num = 10
# 生成符合神经网络需求的数据集，将二维图片“拉直”
x_train = x_train.reshape((Train_Num, -1))
Input_Num = x_train.shape[1]
x_test = x_test.reshape((Test_Num, -1))
# 抽取一部分作为验证集
valid_idx = np.random.randint(0, Train_Num, 10000)
x_validate = x_train[valid_idx, :]
y_validate = y_train[valid_idx]


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
                    last_output = tf.matmul(input_tensor, w) + b
                else:
                    last_output = tf.matmul(last_output, w) + b
                # 不是最后一层再进行激活！！！
                if j < len(nodes_nums) - 1:
                    last_output = act_func(last_output)
    else:
        for j in range(len(nodes_nums)):
            scope_name = 'layer' + str(j)
            with tf.variable_scope(scope_name, reuse=True):
                w = tf.get_variable(name='weights', dtype=tf.float64)
                b = tf.get_variable(name='bias', dtype=tf.float64)
                w_avg = ma_cls.average(w)
                b_avg = ma_cls.average(b)
                if last_output is None:
                    last_output = tf.matmul(input_tensor, w_avg) + b_avg
                else:
                    last_output = tf.matmul(last_output, w_avg) + b_avg
                # 不是最后一层再进行激活！！！
                if j < len(nodes_nums) - 1:
                    last_output = act_func(last_output)
    return last_output


batchsize = 1000
x = tf.placeholder(dtype=tf.float64, shape=(None, Input_Num), name='x')
y_lab = tf.placeholder(dtype=tf.int64, shape=None, name='y')

# 定义全连接神经网络结构
layers = [100, 50, 10]
build_nn(Input_Num, layers)

# 只有一个隐藏层
# Hidden_node = 500
# layer1W = tf.Variable(tf.random_normal([Input_Num, Hidden_node], stddev=0.1), name='layer1Weights', dtype=tf.float32)
# layer1B = tf.Variable(tf.random_normal([Hidden_node], stddev=0.1), name='layer1Bias', dtype=tf.float32)
# # 输出层
# outputW = tf.Variable(tf.random_normal([Hidden_node, Class_Num], stddev=0.1), name='OutputWeights', dtype=tf.float32)
# outputB = tf.Variable(tf.random_normal([Class_Num], stddev=0.1), name='OutputBias', dtype=tf.float32)

# 计算预测值  n * 10
y_pre = infer(x, None, tf.nn.sigmoid, layers)
# y_pre = infer(x, None, tf.nn.relu, layer1W, layer1B, outputW, outputB)

# 使用SoftMax+交叉熵作为损失函数
# 若结果仅属于一个分类，sparse_softmax_cross_entropy_with_logits函数可以进一步加速运算
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_lab, logits=y_pre, name='CrossEntropy')
cross_entropy = tf.reduce_mean(cross_entropy)
# loss = cross_entropy + tf.nn.l2_loss(layer1W)
# r2losses = tf.add_n(tf.get_collection('r2losses'))

# 使用指数衰减法设置学习率
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.8, global_step, Train_Num//batchsize, 0.99, staircase=False)
trainStep = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(cross_entropy, global_step=global_step)

# 设置滑动平均类，并对所有参数进行滑动平均
ema_avg = tf.train.ExponentialMovingAverage(0.99, global_step)
emaStep = ema_avg.apply(tf.trainable_variables())
# 计算滑动平均后的预测值  n * 10
y_pre_ma = infer(x, ema_avg, tf.nn.sigmoid, layers)
# y_pre_ma = infer(x, ema_avg, tf.nn.relu, layer1W, layer1B, outputW, outputB)
# 根据滑动平均后预测值来计算正确率
# correct_pre = tf.equal(tf.arg_max(y_pre, 1), y_lab)
correct_pre = tf.equal(tf.arg_max(y_pre_ma, 1), y_lab)
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

# 结合训练和滑动平均为一轮训练的操作步骤
group_operaion = tf.group(trainStep, emaStep)

train_times = 50000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    start = time.perf_counter()
    for i in range(train_times):
        randIdx = np.random.randint(0, Train_Num, batchsize)
        sess.run(group_operaion, feed_dict={x: x_train[randIdx, :],
                                            y_lab: y_train[randIdx]})
        # 每1000轮训练后检测一下正确率
        if i % 1000 == 0:
            acc = sess.run(accuracy, feed_dict={x: x_validate,
                                                y_lab: y_validate})
            print("After training %d steps, accuracy is %g. Cost %f s." %
                  (i, acc, (time.perf_counter() - start)))
            start = time.perf_counter()

    # 训练结束后在测试集上进行正确率验证
    acc = sess.run(accuracy, feed_dict={x: x_test,
                                        y_lab: y_test})
    print("After training %d steps, accuracy on test dataset is %g"
          % (train_times, acc))

    # 将训练后的模型本地化保存
    saver = tf.train.Saver()
    # 持久化训练后模型，保存为ckpt格式
    # saver.save(sess, r'.\mnistOCR.ckpt')

    # 保存为pb模式
    # graph_def = tf.get_default_graph().as_graph_def()
    # output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['PricePrediction'])
    # with tf.gfile.GFile(r'.\mnistOCR.pb', 'wb') as f:
    #     f.write(output_graph.SerializeToString())









