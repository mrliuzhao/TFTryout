import tensorflow.compat.v1 as tf
import time
import numpy as np

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

# 输入均为平面2维向量，即2*1矩阵
# s1 = tf.constant(np.asarray([[1.0, 3.0], [2.0, 4.0]]).tolist())
# s2 = tf.constant([[2.5], [5.0]])
# s3 = tf.constant([[2.5, 5.0]])
# colAdd = tf.add(s1, s2)
# rowAdd = tf.add(s1, s3)

labels = tf.constant([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]], dtype=tf.float32)
predict = tf.constant([
    [1.8231245e-05, 3.4142386e-06, 1.0052040e-03, 3.9710379e-05, 4.0504027e-02, 2.3459836e-03, 9.5440400e-01,
     6.9793086e-06, 1.7153780e-04, 1.5009168e-03],
    [9.6701011e-03, 7.9407764e-04, 2.6054555e-03, 2.3177522e-01, 2.4770171e-04, 6.6697145e-01, 6.6636648e-04,
     2.5453877e-02, 5.5039514e-02, 6.7761424e-03],
    [1.0400537e-04, 1.5225104e-04, 4.4337459e-04, 4.9136155e-03, 3.8367616e-05, 4.3424472e-04, 1.7848268e-05,
     9.8544717e-01, 2.1695295e-04, 8.2321083e-03],
    [9.7521937e-01, 6.3683251e-06, 1.2908342e-03, 4.3456050e-04, 3.6698242e-05, 1.5766388e-02, 6.1185737e-03,
     1.2010672e-04, 8.6697319e-04, 1.4013739e-04],
    [1.2070185e-02, 8.3794759e-04, 9.1251898e-01, 5.0891329e-02, 1.0247771e-05, 1.8554430e-02, 4.6057504e-04,
     8.9228277e-05, 4.5117592e-03, 5.5323635e-05]], dtype=tf.float32)

clip_predict = tf.clip_by_value(predict, 1e-10, 1.0)
log_predict = tf.log(clip_predict)
multi = labels * log_predict
reducesum = -tf.reduce_sum(multi, reduction_indices=1)
reducemean = tf.reduce_mean(reducesum)
armax = tf.argmax(predict, 1)

with tf.Session() as sess:
    print('origin prediction:', sess.run(predict))
    print('clipped prediction:', sess.run(clip_predict))
    print('logged prediction:', sess.run(log_predict))
    print('multiply:', sess.run(multi))
    print('reduce sum:', sess.run(reducesum))
    print('reduce mean:', sess.run(reducemean))
    print('armax:', sess.run(armax))

# loss = tf.reduce_mean(
#     -tf.reduce_sum(labels * tf.log(tf.clip_by_value(output, 1e-10, 1.0)), reduction_indices=1))
