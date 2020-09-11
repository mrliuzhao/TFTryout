# import tensorflow.compat.v1 as tf
import tensorflow as tf
import os

tf1 = tf.compat.v1

# 取消TensorFlow2.0特性
tf1.disable_v2_behavior()

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# 默认将Tensor创建在默认图中
a = tf1.constant([1.0, 2.0], name="a")
print(a.graph is tf1.get_default_graph())

# 新建计算图并在该图中添加变量
g1 = tf1.Graph()
with g1.as_default():
    v1 = tf1.Variable([0], name='v')

g2 = tf1.Graph()
with g2.as_default():
    v2 = tf1.Variable([1], name='v')

# 分别展示不同计算图中的同一个变量
with tf1.Session(graph=g1) as sess:
    sess.run(tf1.global_variables_initializer())
    print(sess.run(v1))

with tf1.Session(graph=g2) as sess:
    sess.run(tf1.global_variables_initializer())
    print(sess.run(v2))

tf.nn.softmax_cross_entropy_with_logits()



