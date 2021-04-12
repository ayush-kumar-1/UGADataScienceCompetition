#!/usr/bin/env python
# coding: utf-8

# UGA Data Science Competition
# Authors: Ayush Kumar, Chole Phelps, Faisal Hossian, Nicholas Sung

# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score

# Logistic Regression Model Class
class log_reg():
    
    def __init__(self, x, y): 
        """
        Constructs a logistic regression model from numpy arrays of x variables and a response. 
        
        Keyword Arguments: 
        x: Numpy Array - the input matrix, assumed to have no bias column
        y: Numpy Vector - the response variable 
        k: Integer - the number of parameters in the model
        n: Integer - the number of observations in the training set
        """
        self.x = x
        self.y = y
        self.k = self.x.shape[1] - 1
        self.n = self.x.shape[0]
        
    def predict(self):
        """Predicts the outcome of x based on the model using scaled data, if no argument is used
        then training data is used to predict, should return raw probability not a 1 or 0.
        
        Keyword Arguments:
        x - the input matrix, assumed to have no bias column
        y - the response vector
        """
        # Create linear regression object
        classifier = LogisticRegression()

        # Split the data into training/testing sets (train 80%, test 20%)
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2)

        # Train the model using the training set
        classifier.fit(X_train, y_train)

        # Make predictions using the testing set
        predictions = classifier.predict(X_test)

        return predictions

# Metrics Class
class metrics(): 
    def confusion_mat(yhat, y):
        """Computes and returns computes the confusion matrix of a trained model.

        The confusion matrix is a summary and accuracy evaluation of prediction results on a
        classification problem. Provides the count of predicted and actual values - true positive, false
        positive, true negative, and false negative.
        """
        
        return classification_report(y, yhat)
    
    def accuracy(yhat, y):
        """Computes and returns the accuracy score of a trained model.

        The accuracy is the ratio of number of correct predictions to the total number of input
        samples. The best value is 1 and the worst value is 0.
        """
        
        return accuracy_score(y, yhat)       
    
    def precision(yhat, y): 
        """Computes and returns the precision score of a trained model.

        The precision is the ratio tp / (tp + fp) where tp is the number of true positives and
        fp the number of false positives. Precision quantifies the number of positive class predictions
        that actually belong to the positive class. The best value is 1 and the worst value is 0.
        """
        
        return precision_score(y, yhat)
    
    def recall(yhat, y):
        """Computes and returns the recall score of a trained model.

        The recall is the ratio tp / (tp + fn) where tp is the number of true positives and
        fn the number of false negatives. Recall quantifies the number of positive class predictions
        made out of all positive examples in the dataset. The best value is 1 and the worst value is 0.
        """
        
        return recall_score(y, yhat) 
