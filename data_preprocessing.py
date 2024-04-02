# data_preprocessing.py
import pandas as pd
import numpy as np
from keras.utils import to_categorical

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    X_train = train_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(train_data['label'].values)
    X_test = test_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y_test = to_categorical(test_data['label'].values)
    return X_train, y_train, X_test, y_test
