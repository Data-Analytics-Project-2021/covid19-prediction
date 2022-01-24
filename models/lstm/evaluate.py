'''
evaluate.py

This python script evaluates the trained LSTM networks
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError

def make_predictions(model, test_X, timesteps,features,days=None):
	print("hello")
	X_data = test_X[0:timesteps]
	if days==None:
		itr = test_X.size-timesteps
	else:
		itr=days
	print("itr size:",itr)
	X_data = X_data.reshape(1,timesteps,features)
	print("X_data shape",X_data.shape)
	print("X_data:",X_data)
	print("X_data input:",X_data[:,-timesteps:])

	while itr:
		forecast = model.predict(X_data[:,-timesteps:])
		# forecast = forecast.reshape(1,timesteps,features)
		print("forecast:", np.array([forecast]))
		X_data = np.append(X_data,np.array([forecast]),axis=1)
		print("In loop:",X_data)
		itr-=1
	return X_data

def evaluate(y, y_hat):
	mape = MeanAbsolutePercentageError()
	mape_res = mape(y, y_hat).numpy()
	mae = MeanAbsoluteError()
	mae_res = mae(y, y_hat).numpy()
	rmse = RootMeanSquaredError()
	rmse_res = rmse(y,y_hat).numpy()
	return [mape_res, mae_res, rmse_res]