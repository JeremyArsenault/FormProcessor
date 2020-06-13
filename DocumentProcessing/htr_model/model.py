import numpy as np
import cv2
import pandas as pd

import tensorflow.compat.v2 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_datasets as tfds

import argparse
import pickle

IMG_HEIGHT = 28
IMG_WIDTH = 28

# build and return model
def build_model():
    model = Sequential([
        Conv2D(16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D(),
        Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(62, activation='softmax'),
    ])
    return model

# compile model
def compile_model(model):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

def preprocess_mnist_img(img):
    return np.reshape(np.absolute(img-255)/255, (28, 28, 1))


def load_data():
    X_train, y_train = tfds.as_numpy(tfds.load('emnist', split='train', batch_size=-1, shuffle_files=True, as_supervised=True))
    X_test, y_test = tfds.as_numpy(tfds.load('emnist', split='test', batch_size=-1, shuffle_files=True, as_supervised=True))

    X_train = np.absolute(X_train-255)/255
    X_test = np.absolute(X_test-255)/255

    lex_key = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    lex = []
    for char in range(len(lex_key)):
        val = np.zeros(len(lex_key))
        val[char] = 1
        lex.append(val)
    lex = np.array(lex)

    y_train_temp = []
    for i in range(len(y_train)):
        y_train_temp.append(lex[y_train[i]])
    y_train = np.array(y_train_temp)

    y_test_temp = []
    for i in range(len(y_test)):
        y_test_temp.append(lex[y_test[i]])
    y_test = np.array(y_test_temp)

    return X_train, y_train, X_test, y_test 

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def predict(img, possible_chars):
    pass

# train model
def train():
    X_train, y_train, X_test, y_test = load_data()
    model = build_model()
    model.summary()
    compile_model(model)

    batch_size = 128
    num_epoch = 10
    #model training
    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              verbose=1,
              validation_data=(X_test, y_test))

    model.save('models/emnist')
    return history

if __name__=="__main__":
    import matplotlib.pyplot as plt         
    # train the model
    history = train()

    with open('data/history', 'wb') as f:
            pickle.dump(history.history, f)