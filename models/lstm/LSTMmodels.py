'''
models.py

This python script contains the various LSTM networks
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Imports for model components
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanAbsoluteError
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def build_lstm(time_steps, features, nprams=100, activation='relu', optimizer='adam', loss='mae', metrics=['mae']):
	model = Sequential()
	model.add(LSTM(nprams, activation=activation, input_shape=(time_steps, features)))
	model.add(Dense(10))
	model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
	model.summary()
	return model

def train_model(model,train_X,train_y,tensorboard_callback,epochs=300):
	model.fit(train_X,train_y,epochs=epochs,callbacks=[tensorboard_callback])
	return model

