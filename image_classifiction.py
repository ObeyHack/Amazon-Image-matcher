# we want to train an NN to classify image to classes based on the dataset provided.
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

arr_testX_testy = []
arr_datasets = ['data1.npy', 'data2.npy', 'data3.npy']
_model = None


def load_dataset(data_path):
    """
        -----------------------------------------------------
        Load the dataset and split it into training and testing data.
        we should do it for each dataset, and than train the model on each dataset.
        -----------------------------------------------------
        return trainX, trainy, testX, testy
    """
    # 30 percent of the data is used for testing
    data = np.load(data_path)
    # Y is on column 2 all the other
    # the image is in column 3
    X = data[:, 3]
    y = data[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return trainX, y_train, testX, y_test


def initialize_model():
    """
        -----------------------------------------------------
        Initialize the model. The model should be a Convolutional Neural Network.
        the model shoud classify the image to 140 classes.
        -----------------------------------------------------
        return model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(140, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def get_trained_model_on_datasets():
    """
        -----------------------------------------------------
        Perform image classification.
        we
        -----------------------------------------------------
        return accuracy
    """

    def train_model(model, trainx, trainy):
        """
            -----------------------------------------------------
            Train the model on the given dataset.
            -----------------------------------------------------
            return trained model
        """
        model.fit(trainx, trainy, epochs=10, validation_data=(testX, testy))
        return model

    model = initialize_model()
    for file in arr_datasets:
        trainX, trainy, testX, testy = load_dataset(file)
        model = train_model(model, trainX, trainy)
        arr_testX_testy.append((testX, testy))
    _model = model
    # print weights
    print(model.get_weights())
    return model


def get_accuracy_on_datasets():
    """
        -----------------------------------------------------
        Perform image classification.
        first of all, we need to train the model on the datasets provided.
        -----------------------------------------------------
        return accuracy
    """

    def evaluate_model(model, testX, testy):
        """
            -----------------------------------------------------
            Evaluate the model on the given dataset.
            -----------------------------------------------------
            return accuracy
        """
        test_loss, test_accuracy = model.evaluate(testX, testy)
        return test_accuracy

    accuracy = []
    for testX, testy in arr_testX_testy:
        accuracy.append(
            evaluate_model(_model, testX, testy))  # Expected type '{evaluate}', got 'None' beacuse need train
    return accuracy


def classify_image(image_path):
    """
        -----------------------------------------------------
        Classify the image.
        -----------------------------------------------------
        return class
    """
    img = image.load_img(image_path, target_size=(224, 224, 3))
    img = image.img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    prediction = _model.predict(img)
    return prediction


def run_init():
    """
        -----------------------------------------------------
        Initialize the model.
        -----------------------------------------------------
        return model
    """
    # get trained model on datasets
    model = get_trained_model_on_datasets()
    # get accuracy on datasets
    accuracy = get_accuracy_on_datasets()
