"""
MNIST classification 
"""

# packages
import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 

from numpy.random import seed
seed(2)
tf.random.set_seed(2)

#%% load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print('training input data shape: ', x_train.shape), print('test input data shape: ', x_test.shape)
print('training output data shape: ', y_train.shape), print('test output data shape: ', y_test.shape)

# reshape each image's 2D data into a 3D shape with 1 channel (i.e., of shape 28 X 28 X 1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

fig, ax = plt.subplots(1, 10, figsize=(10,2))
for i in range(10):
    ax[i].imshow(x_train[i])
    ax[i].axis('off')

plt.show()
                
#%% scale pixel values to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#%% define CNN
lenet5 = keras.Sequential([
    keras.layers.Conv2D(6, (5,5), padding='same', input_shape=[28, 28, 1], activation='tanh'),
    keras.layers.AveragePooling2D((2,2)),

    keras.layers.Conv2D(16, (5,5), padding='valid', activation='tanh'),
    keras.layers.AveragePooling2D((2,2)),

    keras.layers.Conv2D(120, (5,5), padding='valid', activation='tanh'),
    keras.layers.Flatten(),
    keras.layers.Dense(84, activation='tanh'),
    keras.layers.Dense(10, activation='softmax') # original LeNet-5 had used RBF activation
])

lenet5.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lenet5.summary()

#%% fit model
history = lenet5.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

#%% evaluate model
lenet5.evaluate(x_test, y_test)

fig, ax = plt.subplots(1, 10, figsize=(10,2))
for i in range(10): # first 10 test images
    img = x_test[i]
    softmax_probabilities = lenet5.predict(np.expand_dims(img, 0)) # adding the batch dimenson before predicting 
    label_pred = np.argmax(softmax_probabilities)
    
    ax[i].imshow(img)
    ax[i].set_title(f'Pred: {label_pred}'), ax[i].axis('off')
plt.show()