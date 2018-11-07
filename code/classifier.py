import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

features = np.load("../processed data/phr_train.npy")
labels = np.load("../processed data/sen_train.npy")


def dense_net(data):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(data.shape[1],)))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = dense_net(features)
model.fit(x=features, y=labels, batch_size=32, epochs=25, verbose=1)
keras_file = "../models/classifer.h5"
tf.keras.models.save_model(model, keras_file)
