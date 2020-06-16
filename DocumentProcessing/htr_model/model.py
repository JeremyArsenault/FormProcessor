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
        Conv2D(32, kernel_size=(4,4), padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.3),

        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),        
        Flatten(),
        Dropout(0.3),
#        Dense(128, activation='relu'),
        Dense(38, activation='softmax'),
    ])
    return model

def load_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model

# compile model
def compile_model(model):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

def preprocess_img(img):
    WIDTH = 28
    HEIGHT = 28

    crop_factor = 2
    img = img[crop_factor:-crop_factor, crop_factor:-crop_factor]

    if img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.absolute((img.T/255) - 1)

    if img.shape[0]>img.shape[1]:
        scale_factor = float(HEIGHT) / float(img.shape[0])
    else:
        scale_factor = float(WIDTH) / float(img.shape[1])
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    new_img = np.zeros((WIDTH, HEIGHT))
    new_img[:resized.shape[0], :resized.shape[1]] = resized

    new_img = np.reshape(new_img, (1,28,28,1))
    return new_img

def preprocess_mnist_img(img):
    return np.reshape(np.absolute(img-255)/255, (28, 28, 1))


def load_data():
    X_train, y_train = tfds.as_numpy(tfds.load('emnist', split='train', batch_size=-1, shuffle_files=True, as_supervised=True))
    X_test, y_test = tfds.as_numpy(tfds.load('emnist', split='test', batch_size=-1, shuffle_files=True, as_supervised=True))

    X_train = np.absolute(X_train-255)/255
    X_test = np.absolute(X_test-255)/255

    X_train_spaces = np.zeros((X_train.shape[0]//100, 28, 28, 1))
    y_train_spaces = np.zeros((y_train.shape[0]//100,), dtype=int)+62
    X_test_spaces = np.zeros((X_test.shape[0]//100, 28, 28, 1))
    y_test_spaces = np.zeros((y_test.shape[0]//100,), dtype=int)+62

    dot_img = np.zeros((28, 28, 1))
    dot_img[12:13, 12:13] = dot_img[12:13, 12:13] + 1
    X_train_dots = np.array([dot_img for i in range(X_train.shape[0]//100)])
    y_train_dots = np.zeros((y_train.shape[0]//100,), dtype=int)+63
    X_test_dots = np.array([dot_img for i in range(X_test.shape[0]//100)])
    y_test_dots = np.zeros((y_test.shape[0]//100,), dtype=int)+63

    X_train = np.concatenate((X_train, X_train_spaces, X_train_dots), axis=0)
    y_train = np.concatenate((y_train, y_train_spaces, y_train_dots), axis=0)
    X_test = np.concatenate((X_test, X_test_spaces, X_test_dots), axis=0)
    y_test = np.concatenate((y_test, y_test_spaces, y_test_dots), axis=0)

    # key: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ."
    lex = [0,1,2,3,4,5,6,7,8,9, # digits -> digits
        10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35, # uppercase -> uppercase
        10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35, # lowercase -> uppercase
        36,37] # spaces + dots

    for i in range(len(y_train)):
        y_train[i] = lex[y_train[i]]

    for i in range(len(y_test)):
        y_test[i] = lex[y_test[i]]

    y_train = y_train.astype(float)    
    y_test = y_test.astype(float)  

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
        zoom_range=0.1,
        shear_range=10,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1)
    datagen.fit(X_train)

    model = build_model()
    #model - load_model('models/emnist3/')
    model.summary()
    compile_model(model)

    batch_size = 264
    num_epoch = 15
    #model training
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
              epochs=num_epoch,
              verbose=1,
              validation_data=(X_test, y_test))

    model.save('models/emnist-merge')
    return history

if __name__=="__main__":
    import matplotlib.pyplot as plt         
    # train the model
    history = train()

    with open('data/emnist-merge', 'wb') as f:
            pickle.dump(history.history, f)
