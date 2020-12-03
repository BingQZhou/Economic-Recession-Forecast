################################################################################
# LSTM
# Code snippet by Bingqi Zhou
# Nov 2020
################################################################################

import numpy as np
import pandas as pd
import datetime
import timeit
import os
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import *

# Encapsulate an LSTM network. It involves training and testing
class Serialized_LSTM(object):
    def __init__(self, train_data, test_data, yield_train, yield_test,
                 lag,
                 neurons,
                 input_size,
                 output_size,
                 nb_epoch,
                 batch_size,
                 plot_title, plot_x_title, plot_y_title,
                convert, verbal):
        ####### Data Transformation ##############################
        # transform data to be stationary
        train_raw_values = train_data.values
        diff_values = self.difference(train_raw_values, 1)

        # transform data to be supervised learning
        train_supervised = self.timeseries_to_supervised(diff_values, lag, input_size)
        # select number of columns with respect to train_size
        input_idx = list(range(input_size))
        input_idx.append(-1)
        train = train_supervised.iloc[:, input_idx].values

        # transform data to be stationary
        test_raw_values = test_data.values
        diff_values = self.difference(test_raw_values, 1)

        # transform data to be supervised learning
        supervised = self.timeseries_to_supervised(diff_values, lag, input_size)
        test = supervised.iloc[:, input_idx].values
        
        ###### Train Data ####################
        # transform the scale of the data
        scaler, train_scaled, test_scaled = self.scale(train, test)

        # fit the model
        lstm_model = self.fit_lstm(train_scaled, batch_size, nb_epoch, neurons, output_size)
        # forecast the entire training dataset to build up state for forecasting
        predictions = list()
        for i in range(len(train_scaled)):
            # make one-step forecast
            X, y = train_scaled[i, -input_size-1:-1], train_scaled[i, -1]
            yhat = self.forecast_lstm(lstm_model, input_size, X)
            # invert scaling
            yhat = self.invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = self.inverse_difference(history = train_raw_values, yhat = yhat, interval = len(train_scaled)-i+lag)
            # store forecast
            predictions.append(yhat)
            
        # if conversion is required
        if convert:
            conversion_rate = yield_rate_convert(yield_train[input_size:-lag])
            conv_predictions = np.array(predictions) * np.array(conversion_rate)
            rmse = np.sqrt(mean_squared_error(train_raw_values[input_size+lag:], conv_predictions))
            print('Train RMSE: %.3f' % rmse)
        else:
            rmse = np.sqrt(mean_squared_error(train_raw_values[input_size+lag:], predictions))
            print('Train RMSE: %.3f' % rmse)
        
        ###### Test Data #########################
        predictions = list()
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i, -input_size-1:-1], test_scaled[i, -1]
            yhat = self.forecast_lstm(lstm_model, input_size, X)
            # invert scaling
            yhat = self.invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = self.inverse_difference(history = test_raw_values, yhat = yhat, interval = len(test_scaled)-i+lag)
            # store forecast
            predictions.append(yhat)
            expected = test_raw_values[input_size + i]
            if verbal:
                print('Predicted=%f, Expected=%f' % (yhat, expected))
            
        if convert:
            conversion_rate = yield_rate_convert(yield_test[input_size:-lag])
            conv_predictions = np.array(predictions) * np.array(conversion_rate)
            rmse = np.sqrt(mean_squared_error(test_raw_values[input_size+lag:], conv_predictions))
            ######### Plotting
            # line plot of observed vs predicted
            plot_pred(test_raw_values[input_size+lag:], conv_predictions, plot_title, plot_x_title, plot_x_title)
            print('Test RMSE: %.3f' % rmse)
        else:
            rmse = np.sqrt(mean_squared_error(test_raw_values[input_size+lag:], predictions))
            print('Test RMSE: %.3f' % rmse)
            ######### Plotting
            # line plot of observed vs predicted
            plot_pred(test_raw_values[input_size+lag:], predictions, plot_title, plot_x_title, plot_x_title)
    
    # frame a sequence as a supervised learning problem
    def timeseries_to_supervised(self, data, lag, input_size):
        original = pd.DataFrame(data)
        columns = [original.shift(i) for i in range(input_size)]
        # shift to make X and y corresponds
        columns.append(original.shift(-lag))
        df = pd.concat(columns, axis=1)
        df.fillna(0, inplace=True)
        # only return those rows with full data
        return df.iloc[input_size-1:-lag]
        
    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    # invert differenced value
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]

    # scale train and test data to [-1, 1]
    def scale(self, train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    # inverse scaling for a forecasted value
    def invert_scale(self, scaler, X, value):
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    # fit an LSTM network to training data
    def fit_lstm(self, train, batch_size, nb_epoch, neurons, output_size):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(output_size))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs = nb_epoch, batch_size = batch_size, verbose=0, shuffle = False)
        return model

    # make a one-step forecast
    def forecast_lstm(self, model, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        return np.sum(yhat[0,0])