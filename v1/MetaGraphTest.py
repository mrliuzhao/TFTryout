import tensorflow.compat.v1 as tf
import os

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

v1 = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32), name='v1')
v2 = tf.Variable(tf.constant(3.3, shape=[1], dtype=tf.float32), name='v2')
add_res = tf.add(v1, v2, name='xaddy')

saver = tf.train.Saver()
saver.export_meta_graph(r'.\metagraph.ckpt.meta.json', as_text=True)

