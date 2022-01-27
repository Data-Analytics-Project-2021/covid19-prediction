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
import LSTMmodels

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

def make_predictions_short(model, train, test, timesteps,features,tensorboard_callback,days=None,epochs=10):
	# TODO: delete test split, add 1 to train and resplit in loop
	
	temp_train = train
	# print(temp_train)
	if days==None:
		itr=test.size-timesteps

	forecast=[]
	y_data=np.array([])
	print("y_data",y_data)
	for i in range(0,itr):
		# print(test[i])
		temp_train=np.append(temp_train,[test[i]],axis=0)
		print(temp_train.flatten()[-1])
		y_data = np.append(y_data,temp_train[-1],axis=0)
		# print(temp_train)
		X_train, y_train = preprocessing.lstm_data_transform(temp_train,temp_train,timesteps)
		if(i%14==0):
			model = LSTMmodels.train_model(model,X_train,y_train,tensorboard_callback,epochs=epochs)
		print(temp_train[-15:-1])
		forecast.append(model.predict(temp_train[-15:-1].reshape(1,timesteps,features)))

	return forecast, y_data, model

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