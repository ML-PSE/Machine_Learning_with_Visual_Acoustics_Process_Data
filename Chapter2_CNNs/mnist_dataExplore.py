"""
MNIST data explore
"""

# packages
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from numpy.random import seed
seed(2)
tf.random.set_seed(2)

#%% load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)

# reshape each image's 2D data into a 3D shape with 1 channel
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

fig, ax = plt.subplots(1, 10, figsize=(10,2))
for i in range(10):
    ax[i].imshow(x_train[i])
    ax[i].axis('off')

plt.show()

