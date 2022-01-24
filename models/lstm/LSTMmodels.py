'''
LSTMmodels.py

This python script contains the various LSTM networks
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Imports for model components
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def build_lstm(time_steps, features, nprams=100, outputs=1, activation='relu', optimizer='adam', loss='mae', metrics=['mae']):
	model = Sequential()
	model.add(LSTM(nprams, activation=activation, input_shape=(time_steps, features)))
	model.add(Dense(20, activation=activation))
	model.add(Dense(1))
	model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
	model.summary()
	return model

def train_model(model,train_X,train_y,tensorboard_callback,epochs=300):
	model.fit(train_X,train_y,epochs=epochs,callbacks=[tensorboard_callback])
	return model

def plot_accuracy(history):
	plt.plot(history.history['accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

'''
GRID SEARCH CODE

# Using KerasClassifier wrapper 
keras_estimator = KerasRegressor(build_fn=build_univariate_non_stacked, verbose=1)

# Defining parameters for the gridserach
param_grid = {
#     'epochs': [10,100,300,],
    'lstm_nparams':[15,50],
#     'n_steps': [3, 6, 15],
#     'optimizer': ['RMSprop','Adam','Adamax','sgd']
}

kfold_splits = 5

# Defining GridSearch
grid = GridSearchCV(estimator=keras_estimator,
                    verbose=-1,
                    return_train_score=True,
                    cv=kfold_splits,
                    param_grid=param_grid,
#                     scoring="neg_mean_absolute_error",
)

# Fitting GridSearch
grid_result = grid.fit(india_cases_train_X, india_cases_train_y, )

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Writing the gridsearch results to file
file1 = open("univariate_non_stacked_india.txt", "w")
file1.write("mean,stdev,pram")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    file1.write("%f,%f,%r" % (mean, stdev, param))
    file1.write("\n")
file1.close()

sorted(grid_result.cv_results_.keys())
grid_result.cv_results_['split2_train_score']
'''