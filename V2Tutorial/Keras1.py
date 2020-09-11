# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('train images shape: ' + str(train_images.shape))  # (60000, 28, 28)
print('train labels shape: ' + str(train_labels.shape))  # (60000,)
print('test images shape: ' + str(test_images.shape))  # (10000, 28, 28)
print('test labels shape: ' + str(test_labels.shape))  # (10000,)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[10])
plt.colorbar()
plt.grid(False)
plt.show()



