from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from skimage import transform
from skimage.io import imread
from os import listdir
from os.path import join
import numpy as np


def train_detector(train_gt, train_img_dir, fast_train=False):
    imageSize = 50
    inputData = np.zeros((len(train_gt), imageSize, imageSize, 3))
    checkData = np.zeros((len(train_gt), 28)).astype(float)

    for i in train_gt:
        numbI = int(i.split('.')[0])
        img = imread(join(train_img_dir, i)).astype(float)
        checkData[numbI] = np.array(train_gt[i])
        checkData[numbI][::2] *= (imageSize / img.shape[1])
        checkData[numbI][1::2] *= (imageSize / img.shape[0])
        img = transform.resize(img, [imageSize, imageSize, 3])
        inputData[numbI] = img

    expectedValue = inputData.mean(axis=(0, 1, 2))
    std = inputData.var(axis=(0, 1, 2))
    inputData = (inputData - expectedValue) / std

    model = Sequential()
    stSize = imageSize
    model.add(Conv2D(stSize, (3, 3), padding='same', input_shape=(imageSize, imageSize, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(stSize, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(stSize * 2, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(stSize * 2, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(stSize * 4, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(stSize * 4, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1800, activation='relu'))
    model.add(Dense(900, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28))

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    print(model.summary())

    num_epochs = 100
    if (fast_train):
        num_epochs = 1
    model.fit(inputData, checkData, batch_size=30, epochs=num_epochs, validation_split=0.1, shuffle=True)

    model.save("facepoints_model.hdf5")


def detect(model, test_img_dir):
    imageSize = 50
    dir_files_list = listdir(test_img_dir)

    inputData = np.zeros((len(dir_files_list), imageSize, imageSize, 3))
    changeSize = np.zeros((len(dir_files_list), 2))

    for i, filename in enumerate(dir_files_list):
        img = imread(join(test_img_dir, filename)).astype(float)
        changeSize[i] = np.array([img.shape[0], img.shape[1]])
        img = transform.resize(img, [imageSize, imageSize, 3])
        inputData[i] = img

    expectedValue = inputData.mean(axis=(0, 1, 2))
    std = inputData.var(axis=(0, 1, 2))
    inputData = (inputData - expectedValue) / std

    y = model.predict(inputData)
    for i in range(y.shape[0]):
        y[i, ::2] *= (changeSize[i][1] / imageSize)
        y[i, 1::2] *= (changeSize[i][0] / imageSize)

    return {filename: list(map(int, y[i])) for i, filename in enumerate(dir_files_list)}