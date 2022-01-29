'''
preprocessing.py

This python script contains the data ingestion and 
preprocessing steps for the LSTM network.  
'''

import os
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Step 1: Ingestion
def ingestion():
	# Get the path of the current working directory
	curPath = os.getcwd()
	# Appened the parent directory to the current path to step out of the current folder
	parentDir = os.path.abspath(os.path.join(curPath, os.pardir))
	print("Parent Directory", parentDir)
	# Save the path to all of the datasets
	india_cases_path = os.path.join(parentDir, "../cleaned_datasets/india/daily_cases_india.csv")
	india_vacc_path = os.path.join(parentDir, "../cleaned_datasets/india/cum_vacc_india_cleaned.csv")
	usa_cases_path = os.path.join(parentDir, "../cleaned_datasets/usa/daily_cases_usa.csv")
	usa_vacc_path = os.path.join(parentDir, "../cleaned_datasets/usa/cum_vacc_usa_cleaned.csv")

	# Quick check to make sure the path exists
	print("Path:", india_cases_path)
	print("Exists:", os.path.exists(india_cases_path))

	# Load the data as a pandas dataframe
	india_cases_df = pd.read_csv(india_cases_path)
	india_vacc_df =  pd.read_csv(india_vacc_path)

	usa_cases_df = pd.read_csv(usa_cases_path)
	usa_vacc_df = pd.read_csv(usa_vacc_path)

	# Rename columns to keep consistent name pattern
	india_vacc_df.rename(columns = {'date':'Date'}, inplace = True)
	usa_vacc_df.rename(columns = {'date':'Date'}, inplace = True)

	# Visualize the datasets
	print('India Cases:\n',india_cases_df.head(),'\n')
	print('India Vacc:\n',india_vacc_df.head(),'\n')

	print('USA Cases:\n',usa_cases_df.head(),'\n')
	print('USA Vacc:\n',usa_vacc_df.head(),'\n')

	return (india_cases_df, india_vacc_df, usa_cases_df, usa_vacc_df)

# Step 2: Pre-Processing

def univariate(india_cases_df, usa_cases_df):
	# Select only the Confirmed column for univariate analysis
	# Selecting from the first index because the 0th index is NaN
	india_cases_df = india_cases_df[["Confirmed"]][1:]
	usa_cases_df = usa_cases_df[["Confirmed"]][1:]

	# Visualize the datasets
	print('India Cases:\n',india_cases_df.head(),'\n')
	print('USA Cases:\n',usa_cases_df.head(),'\n')

	return (india_cases_df, usa_cases_df)

def multivariate(india_cases_df, india_vacc_df, usa_cases_df, usa_vacc_df):
	# Merge datasets based on date
	merged_india_df = pd.merge(india_cases_df, india_vacc_df, how='outer', on='Date')
	merged_india_df.drop(['Recovered', 'Deaths'], axis=1, inplace=True)
	print('India:\n',merged_india_df.head())

	merged_usa_df = pd.merge(usa_cases_df, usa_vacc_df, how='outer', on='Date')
	merged_usa_df.drop(['Recovered', 'Deaths'], axis=1, inplace=True)
	print('USA:\n',merged_usa_df.head())

	return (merged_india_df, merged_usa_df)

def dropNull(india_multi, usa_multi):
	india_multi = india_multi.dropna()
	usa_multi = usa_multi.dropna()
	print('India:\n',india_multi.head())
	print('USA:\n',usa_multi.head())
	return (india_multi,usa_multi)

def normalize(india_cases_uni, usa_cases_uni, india_multi, usa_multi):
	# Normalize the univariate data
	india_cases_mean = india_cases_uni.mean()
	india_cases_std = india_cases_uni.std()

	usa_cases_mean = usa_cases_uni.mean()
	usa_cases_std = usa_cases_uni.std()


	india_cases_normalized_df = (india_cases_uni-india_cases_mean)/india_cases_std
	usa_cases_normalized_df = (usa_cases_uni-usa_cases_mean)/usa_cases_std

	# Visualize the datasets
	print('India Cases univariate:\n',india_cases_normalized_df.head(),'\n')
	print('USA Cases univariate:\n',usa_cases_normalized_df.head(),'\n')

	# Normalize the multivariate data
	india_date = india_multi[['Date']]
	india_multi.drop(['Date'], axis=1, inplace=True)

	usa_date = usa_multi[["Date"]]
	usa_multi.drop(['Date'], axis=1, inplace=True)

	india_multi_mean = india_multi.mean()
	india_multi_std = india_multi.std()

	usa_multi_mean = usa_multi.mean()
	usa_multi_std = usa_multi.std()

	india_multi_normalized_df = (india_multi-india_multi_mean)/india_multi_std
	usa_multi_normalized_df = (usa_multi-usa_multi_mean)/usa_multi_std

	# Add the date column back
	pd.merge(india_multi, india_date, left_index=True, right_index=True)
	pd.merge(usa_multi, usa_date, left_index=True, right_index=True)
	# Visualize the datasets
	print('India Cases multivariate:\n',india_multi_normalized_df.head(),'\n')
	print('USA Cases multivariate:\n',usa_multi_normalized_df.head(),'\n')

	return (india_cases_normalized_df, usa_cases_normalized_df,
		india_multi_normalized_df, usa_multi_normalized_df,india_cases_mean,india_cases_std,usa_cases_mean,
		usa_cases_std,india_multi_mean,india_multi_std,
		usa_multi_mean, usa_multi_std)

def de_normalize(data,mean,std):
	denormalized_d = data*std+mean
	return denormalized_d

def split(india, usa, test_size):
	india_train, india_test = train_test_split(india, test_size=test_size, shuffle=False)
	usa_train, usa_test = train_test_split(usa, test_size=test_size, shuffle=False)

	# Visualize splits
	print('India:\n',india_train,'\n')
	print('USA:\n',usa_train,'\n')
	return (india_train, india_test, usa_train, usa_test)

def toNumpy(india_uni_train, india_uni_test, usa_uni_train, usa_uni_test):
	india_uni_train, india_uni_test = india_uni_train.to_numpy(), india_uni_test.to_numpy()
	usa_uni_train, usa_uni_test = usa_uni_train.to_numpy(), usa_uni_test.to_numpy()
	return (india_uni_train, india_uni_test, usa_uni_train, usa_uni_test)

def lstm_data_transform(x_data, y_data, num_steps=5):
	""" Changes data to the format for LSTM training 
		for sliding window approach.
	"""
	# Prepare the list for the transformed data
	X, y = list(), list()
	# Loop of the entire data set
	for i in range(x_data.shape[0]):
		# compute a new (sliding window) index
		end_ix = i + num_steps
		# if index is larger than the size of the dataset, we stop
		if end_ix >= x_data.shape[0]:
		    break
		# Get a sequence of data for x
		seq_X = x_data[i:end_ix]
		# Get only the last element of the sequency for y
		seq_y = y_data[end_ix]
		# Append the list with sequencies
		X.append(seq_X)
		y.append(seq_y)
	# Make final arrays
	x_array = np.array(X)
	y_array = np.array(y)
	return x_array, y_array