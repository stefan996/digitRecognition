#libraries for neural network
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# libraries for images
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the data from MNIST
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# define activation function
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

activation_function = sigmoid

# define model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation_function),   
    keras.layers.Dense(50, activation_function),
    keras.layers.Dense(10, activation_function),
    keras.layers.Dropout(0.05)
])

# compile our model
model.compile(optimizer = tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model 
model.fit(train_images, train_labels, epochs = 50)
# save the model
model.save("../model/my_model.h5")

# return loss value and accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Accuracy:', test_acc)












