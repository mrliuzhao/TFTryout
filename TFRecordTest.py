from tensorflow.compat.v1.keras.datasets.mnist import load_data
import tensorflow as tf
import cv2


# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 读取数据
(x_train, y_train), (x_test, y_test) = load_data(r'D:\PythonWorkspace\TFTryout\mnist.npz')
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000, )
print(x_test.shape)  # (10000, 28, 28)
print(y_test.shape)  # (10000, )
img1_be4 = x_train[0]
img1_be4 = cv2.resize(img1_be4, (320, 320))

filename = r'D:\PythonWorkspace\TFTryout\mnist-train.tfrecords'
writer = tf.io.TFRecordWriter(filename)
for i in range(x_train.shape[0]):
    # 图像转化为byte字符串保存
    img_raw = x_train[i].tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'shape': _int64_feature(x_train.shape[1]),
            'img_raw': _bytes_feature(img_raw),
            'label': _int64_feature(y_train[i])
        }
    ))

    writer.write(example.SerializeToString())

writer.close()

# 尝试读取TFRecord
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
    print(example.features.feature['img_raw'].bytes_list.value)

# Create a description of the features.
feature_description = {
    'shape': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'img_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


img1_be4 = x_train[0]
img1_be4 = cv2.resize(img1_be4, (320, 320))


