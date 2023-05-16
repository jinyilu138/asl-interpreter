import os

from matplotlib import pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout, Dense
import keras
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import h5py

global train_dir
global test_dir
global model
global X_train
global Y_train
global X_test
global Y_test


def load_data():
    # loading training data
    global train_dir
    global test_dir
    global X_train
    global Y_train
    global X_test
    global Y_test

    base_dir = '/Users/jinyilu/Desktop/asl'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                   'M': 12,
                   'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
                   'Y': 24,
                   'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}
    images = []
    labels = []
    size = 64, 64
    for folder in os.listdir(train_dir):
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(labels_dict[folder])

    images = np.array(images)
    images = images.astype('float32') / 255.0

    labels = keras.utils.to_categorical(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2)

    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    print('Loaded', len(X_test), 'images for testing', 'Test data shape =', X_test.shape)

    return X_train, X_test, Y_train, Y_test


def create_model():
    global model
    model = tf.keras.Sequential()
    model.add(Conv2D(16, kernel_size=[3, 3], padding='same', activation='relu', input_shape=(64, 64, 3)))
    # 62 x 62 x 3
    model.add(MaxPool2D(pool_size=[2, 2]))
    # 31 x 31 x 3

    model.add(Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
    # 30 X 30 X 3
    model.add(MaxPool2D(pool_size=[3, 3]))
    # 15 x 15 x 3

    model.add(Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'))
    # 14 x 14 x 3
    model.add(MaxPool2D(pool_size=[3, 3]))
    # 7 x 7 x 3

    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(29, activation='softmax'))

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

    model.summary()
    tf.keras.models.save_model('/Users/jinyilu/Desktop/asl/model.h5')

    return model


def fit_model():
    model_hist = model.fit(X_train, Y_train, batch_size=64, epochs=5, validation_split=0.1)
    return model_hist


def validate(h):
    acc = h.history['accuracy']
    val_acc = h.history['val_accuracy']

    loss = h.history['loss']
    val_loss = h.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('training + test accuracy')
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('training + test loss')
    plt.figure()


def main():
    load_data()
    create_model()
    h = fit_model()
    validate(h)


if __name__ == "__main__":
    main()

