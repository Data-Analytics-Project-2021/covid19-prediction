'''
evaluate.py

This python script evaluates the trained LSTM networks
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
import preprocessing

def make_predictions(model, test_X, timesteps,features,days=None):	
	X_data = test_X[0:timesteps]
	if days==None:
		itr = test_X.size-timesteps
	else:
		itr=days
	X_data = X_data.reshape(1,timesteps,features)
	while itr:
		forecast = model.predict(X_data[:,-timesteps:])
		X_data = np.append(X_data,np.array([forecast]),axis=1)
		itr-=1
	return X_data

def make_predictions_short(model, test_X, timesteps,features,days=None):
	X_data, y_data = preprocessing.lstm_data_transform(test_X,test_X,timesteps)
	forecast = model.predict(X_data)
	return forecast, y_data

def evaluate(y, y_hat):
	mape = MeanAbsolutePercentageError()
	mape_res = mape(y, y_hat).numpy()
	mae = MeanAbsoluteError()
	mae_res = mae(y, y_hat).numpy()
	rmse = RootMeanSquaredError()
	rmse_res = rmse(y,y_hat).numpy()
	return [mape_res, mae_res, rmse_res]

def plot_fore_test(test, fore, title):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    ax.plot(test, color='blue', label='Test')
    ax.plot(fore, color='red', label='Forecast')
    ax.legend(loc='best')
    plt.title(title)
    plt.show()