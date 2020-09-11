import tensorflow.compat.v1 as tf
import os
from tensorflow.python.platform import gfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 控制日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()
tf.device('/gpu:0')

df = pd.read_csv(r'.\data\sh.600519-07d.csv')
df['year'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.year
df['month'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.month
df['day'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.day
df['dd'] = (pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce') - pd.to_datetime("2000-01-01", format='%Y-%m-%d', errors='coerce')).dt.days
df['dw'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.dayofweek
df['monthEnd'] = (df['day'] >= 28).astype('int')
temp = (df['month'] == 3) | (df['month'] == 6) | (df['month'] == 9) | (df['month'] == 12)
df['seasonEnd'] = ((df['day'] >= 24) & temp).astype('int')
df_use = pd.concat([df['volume'], df['amount'], df['turn'], df['pctChg'],
                    df['peTTM'], df['pbMRQ'], df['pcfNcfTTM'],
                    df['year'], df['month'], df['day'], df['dd'],
                    df['dw'], df['monthEnd'], df['seasonEnd']], axis=1)
# 特征值归一化
scaler = MinMaxScaler().fit(df_use)
df_use = scaler.transform(df_use)
predict = df_use[df['close_delay'].isna()].T

with tf.Session() as sess:
    # 从pb文件中读取计算图
    with gfile.FastGFile(r'.\models\sh.600519-closeprice-7d.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    input_x = sess.graph.get_tensor_by_name('x:0')
    pred = sess.graph.get_tensor_by_name('PricePrediction:0')
    print(sess.run(pred, feed_dict={input_x: predict}))
    # print(sess.run(pred))





