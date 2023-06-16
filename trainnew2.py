import pickle

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dict = pickle.load(open('D:/ASLMiniProject/data_final.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# print("Hand_landmarks: ", data.shape)

data = data.reshape(-1, 42)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# label_binarizer = LabelBinarizer()
#
# y_train = label_binarizer.fit_transform(y_train)
# y_test = label_binarizer.fit_transform(y_test)
#
# Convert labels to categorical one-hot encoding
num_classes = len(np.unique(labels))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#
# Normalize the pixel values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# # Define the CNN architecture
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(data.shape[1])))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(28, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(42,)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((7, 6, 1), input_shape=(42,)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model on test data
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


import matplotlib.pyplot as plt

# Assuming you have the accuracy and loss values from model training
accuracy = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(accuracy) + 1)

# Plot accuracy
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# save the model

f = open('finalmodelepoch15.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# model_json = model.to_json()
# with open("modelcnnepoch10.json", "w") as json_file:
#     json_file.write(model_json)
# model.save('modelcnnepoch10.h5')
