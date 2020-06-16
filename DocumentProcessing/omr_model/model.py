import numpy as np
import cv2

import tensorflow.compat.v2 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_datasets as tfds

import pickle

import matplotlib.pyplot as plt

IMG_HEIGHT = 28
IMG_WIDTH = 28

# build and return model
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D(),
        Dropout(0.4), 
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'),
        BatchNormalization(),       
        Flatten(),
        Dropout(0.3),
        Dense(2, activation='softmax'),
    ])
    return model

def load_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model

# compile model
def compile_model(model):
    model.compile(optimizer='adam',
              loss=tf.keras.losses. sparse_categorical_crossentropy,
              metrics=['accuracy'])

def preprocess_img(img):
    WIDTH = 28
    HEIGHT = 28

    if img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255

    img = np.reshape(cv2.resize(img, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA), (1,28,28,1))

    return img


def load_data():
    X_train = np.genfromtxt('../../../OMR_dataset/data/X_train.csv', delimiter=',')/255
    X_test = np.genfromtxt('../../../OMR_dataset/data/X_test.csv', delimiter=',')/255
    y_train = np.genfromtxt('../../../OMR_dataset/data/y_train.csv', delimiter=',')
    y_test = np.genfromtxt('../../../OMR_dataset/data/y_test.csv', delimiter=',')

    X_train = np.reshape(X_train, (1200, 28, 28, 1))
    X_test = np.reshape(X_test, (300, 28, 28, 1))

    return X_train, y_train, X_test, y_test 

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def predict(img, model):
    img = preprocess_img(img)
    return model.predict(img)[0]

# train model
def train():
    X_train, y_train, X_test, y_test = load_data()

    datagen = ImageDataGenerator(
        zoom_range=0.2,
        shear_range=10,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1)
    datagen.fit(X_train)

    model = build_model()
    model.summary()
    compile_model(model)

    batch_size = 32
    num_epoch = 5
    #model training
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
              epochs=num_epoch,
              verbose=1,
              validation_data=(X_test, y_test))

    model.save('models/checkbox')
    return history

if __name__=="__main__":
    import matplotlib.pyplot as plt         
    # train the model
    history = train()

    with open('data/checkbox', 'wb') as f:
            pickle.dump(history.history, f)
