import tensorflow.compat.v1 as tf

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name='add')

# 将计算图写入日志
writer = tf.summary.FileWriter(r"D:\PythonWorkspace\TFTryout\eventlog", tf.get_default_graph())
writer.close()


