import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from utils import *
from sklearn.linear_model import LogisticRegression as LogRes

class log_reg(): 
    
    def __init__(self, data, response, regularizer = 1.0): 
        """
        Initialized and trains the logistic regression model with 
        the given data and resposne. 
        
        Keyword Arguments: 
        data - the input matrix 
        reponse - the output vector
        """
        self.x = data 
        self.y = response 
        #Set Seed for Reproducibility
        self.model = LogRes(random_state = 462021, C = regularizer).fit(self.x, self.y)
        
    def predict(self, x): 
        """
        Returns predicted probabilities NOT Classes for 
        the model. Tuning may be needed for optimal results for the 
        decision rule. 
        
        Keyword Arguments: 
        x - the data to generate predictions on 
        """
        return self.model.predict_proba(x)[:,1]


class feed_forward(): 
    
    def __init__(self, data, response, width = 32): 
        """
        Initializes a feed_forward nueral network for this 
        specfic problem of credit classification. 
        
        Keyword Arguments: 
        
        data - the input matrix 
        response - the output response vector
        """
        self.x = data 
        self.y = response 
        self.model = feed_forward.build_model(self.x, self.y, width=width)
        
        
    def build_model(x, y, suppress = False, width = 20): 
        """
        Builds a feed forward nueral network with a hidden layer 
        of 20 nuerons. 
        
        Keyword Arguments: 
        
        x - the input data being used to build to model
        y - the response vector model is going to be trained on 
        suppress - if true will not print out model summary
        width - the size of the hidden layer
        """

        model = keras.Sequential()
        model.add(layers.Dense(width, input_dim = x.shape[1], 
                            kernel_initializer = "uniform", 
                            activation = "relu", ))
        model.add(layers.Dense(1, kernel_initializer = "normal", activation = "sigmoid"))
        model.compile(loss = "binary_crossentropy", optimizer = "adam")
        if not suppress: 
            model.summary()
        return model


    def train(self, epochs = 20): 
        """
        Trains the model for a certain number of epochs. Default 
        is 20. 
        """
        self.model.fit(self.x, self.y, epochs = epochs, verbose = 0)
        
        
    def predict(self, x): 
        """
        Generates Predictions of 0 or 1. 
        """
        y_prob = self.model.predict(x)[:,0]
        yhat = decide(y_prob, 0.5)
        
        return yhat

    def predict_proba(self, x):
        """
        Generates predicted probabilites. 
        """ 
        return self.model.predict(x)[:,0]