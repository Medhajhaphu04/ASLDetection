# Importing the Keras libraries and packages
import pickle

import matplotlib
import keras
from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
from pasta.augment import inline
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, Reshape, Cropping2D
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import cv2
from sklearn.preprocessing import LabelBinarizer

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dict = pickle.load(open('D:/ASLMiniProject/data_final.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# data = data.reshape(-1, 42)
import numpy as np

# Assuming your input data is stored in a variable called 'input_data'
num_samples = data.shape[0]

# Reshape input data to match the input shape expected by VGG16
reshaped_input = np.zeros((num_samples, 32, 32, 3))
for i in range(num_samples):
    # Repeat the landmarks to create RGB values
    rgb_values = np.repeat(data[i].reshape((42, 1)), 3, axis=1)

    # Resize the RGB values to match the input shape of VGG16
    reshaped_input[i] = np.resize(rgb_values, (32, 32, 3))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(reshaped_input, labels, test_size=0.2, stratify=labels)
num_classes = len(np.unique(labels))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#
# Normalize the pixel values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Initialising vgg16
classifier_vgg16 = VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
# don't train existing weights for vgg16
for layer in classifier_vgg16.layers:
    layer.trainable = False

    # VGG16
    classifier1 = classifier_vgg16.output  # head mode
    classifier1 = Flatten()(classifier1)  # adding layer of flatten
    classifier1 = Dense(units=256, activation='relu')(classifier1)
    classifier1 = Dropout(0.6)(classifier1)
    classifier1 = Dense(units=26, activation='softmax')(classifier1)

    model = Model(inputs=classifier_vgg16.input, outputs=classifier1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summary of vgg16
model.summary()

# fit the model
# it will take some time to train
# vgg16
history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test))

# save the model

# f = open('vgg16modelepoch15.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

score = model.evaluate(x=x_test, y=y_test, verbose=0)
print('Accuracy for test images:', round(score[1] * 100, 3), '%')
