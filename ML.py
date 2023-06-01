from datetime import datetime

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import Sequential


df = pd.read_csv('Data/TotalGas.csv')

df['Date'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['Date']]

class DeepModelTS():
    
    #class to create a deep time series model
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        Y_var: str,
        lag: int,
        LSTM_layer_depth: int, 
        epochs=10, 
        batch_size=32,
        train_test_split=0.8
    ):

        self.data = data 
        self.Y_var = Y_var 
        self.lag = lag 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split
        self.early_stopping = EarlyStopping(
            monitor='val_loss',  # track validation loss
            patience=10,         # stop after 10 epochs with no improvement
            restore_best_weights=True  # restore the best weights found during training
        )

    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        
        #method to create X and Y matrix from a time series list for the training of 
        #deep learning models 
        
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                X.append(ts[i:(i + lag)])

        X, Y = np.asanyarray(X), np.asanyarray(Y)
        
        # Reshaping the X array to an LSTM input shape 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y      
       

    def create_data_for_NN(
        self,
        use_last_n=None
        ):
         #method to create data for the neural network model
    
        #extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        #Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        #X matrix will hold the lags of Y 
        X, Y = self.create_X_Y(y, self.lag)

        #creating training and test sets 
        X_train = np.asarray(X)
        X_test = []

        Y_train = np.asarray(Y)
        Y_test = []

        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)
            X_train = X[:(len(X) - index)]
            X_test = X[-index:]     
            
            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]

        return X_train, X_test, Y_train, Y_test
    

    def LSTModel(self):
        
        #method to fit the LSTM model 
        
        #Getting the data 
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()

        #Defining the model
        model = Sequential()
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.lag, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        #defining the model parameter dict 
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False
        }

        if self.train_test_split > 0:
            keras_dict.update({
                'validation_data': (X_test, Y_test)
            })

        #fit the model 
        model.fit(
            **keras_dict
        )

        #saving the model to the class 
        self.model = model

        return model
    

    def predict(self) -> list:
        
        # method to predict using the test data used in creating the class
        
        yhat = []

        if(self.train_test_split > 0):
        
            #getting the last n time series 
            _, X_test, _, _ = self.create_data_for_NN()        

            #making the prediction list 
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat
    

    def predict_n_ahead(self, n_ahead: int):
        
        # method to predict n time steps ahead
            
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)        

        #making  prediction list 
        yhat = []

        for _ in range(n_ahead):
            #making the prediction
            fc = self.model.predict_on_batch(X)
            yhat.append(fc)

            #creating a new input matrix for forecasting
            X = np.append(X, fc)

            #ommiting the first variable
            X = np.delete(X, 0)

            #reshaping for the next iteration
            X = np.reshape(X, (1, len(X), 1))

        return yhat        
