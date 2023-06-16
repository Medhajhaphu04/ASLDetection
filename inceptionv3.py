import pickle

import keras.utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, concatenate
from keras.models import Model
import numpy as np

# Step 1: Dataset
data_dict = pickle.load(open('D:/ASLMiniProject/data_final.pickle', 'rb'))

landmarks = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Step 2: Data Preprocessing
# Preprocess the hand landmarks based on your specific requirements
data = landmarks.reshape(-1, 42)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
num_classes = len(np.unique(labels))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#
# Normalize the pixel values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Step 3: Model Architecture
# Create input layers for labels and hand landmarks
input_labels = Input(shape=(num_classes,))  # num_classes is the number of ASL classes
input_landmarks = Input(shape=(42,))  # num_landmarks is the number of landmarks

# Concatenate the inputs
x = concatenate([input_labels, input_landmarks])

# Add dense layers for classification
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[input_labels, input_landmarks], outputs=predictions)

# Step 4: Model Training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    [y_train, x_train],
    y_train,
    batch_size=16,
    epochs=15,
    validation_split=0.2
)

# Step 5: Model Evaluation
# Evaluate the performance of the trained model on the test set
loss, accuracy = model.evaluate([y_test, x_test], y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


f = open('inceptionmodelepoch15.p', 'wb')
pickle.dump({'model': model}, f)
f.close()