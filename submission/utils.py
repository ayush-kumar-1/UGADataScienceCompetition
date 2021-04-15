import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf 
import sklearn
import seaborn as sn
from platform import python_version
from tensorflow import keras
from keras import layers
from utils import *
from models import * 
from sklearn.linear_model import LogisticRegression as LogRes
from sklearn.metrics import classification_report as report
from sklearn.metrics import precision_score as precision 
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as f1

if __name__ == '__main__':
    print("python version: \t", python_version())
    print("numpy version: \t\t", np.__version__)
    print("pandas version: \t", pd.__version__)
    print("matplotlib version: \t", matplotlib.__version__)
    print("tensorflow version: \t", tf.__version__)
    print("scikit-learn version: \t", sklearn.__version__)
    print("seaborn version: \t", sn.__version__)

def load_data(path, as_df = False): 
    """
    Loads and preprocesses data into a form that can be taken in by the models 
    that are being used. 
    
    Keyword Arguments: 
    path - the path string to where the data is being loaded 
    as_df - returns a pandas DataFrame if True

    Returns: 
    x - the data matrix 
    y - the response vector 
    """
    #Intake Data
    df = pd.read_csv(path)
    
    #Impute Missing Values with correct val
    df = df.fillna(df.mean())
    #Convert State Types to Numeric Variables
    #No state dummy for alabama to avoid perfect collinearity
    for state_code in df["States"].unique()[1:]:
        df[state_code] = (df["States"] == state_code).astype('int64')
        
    df = df.drop("States", axis = 1)
    #Undergoing similar process for other dummy variables 
    dummys = ["non_mtg_acc_past_due_12_months_num", 
            "non_mtg_acc_past_due_6_months_num", 
            "card_open_36_month_num", 
            "auto_open_ 36_month_num"]

    for var in dummys: 
        for level in df[var].unique()[1:]: 
            name = var + "==" + str(level)
            df[name] = (df[var] == level).astype('int64') 
        
        df = df.drop(var, axis = 1)
        
    if as_df: 
        return df

    X = df.drop("Default_ind", axis = 1).to_numpy()
    y = df["Default_ind"].to_numpy()
    
    return X, y

def decide(y_prob, threshold = 0.5): 
    """
    Returns decisions based on a threshold and given probabilities. 
    
    Keyword Arguments: 
    yhat - the probabiliteis returned by the model 
    threshold - return 1 if >= threshold 0 otherwise
    """
    return (y_prob > threshold).astype(int)

def directional_loss(y, yhat): 
    """
    Returns a negative number if threshold should be lower and a postive 
    number if it should be increased. 
    
    Keyword Arguments: 
    y - the ground truth
    yhat - the predicted probabilites
    """
    #Should be 0 but model returned 1
    return (y-yhat).sum()/y.shape[0]

def tune_threshold(y, y_prob, eta = 0.1, plev = 0.5, max_iter = 100, output = True):
    """
    Tunes the threshold of the decision rule to improve accuracy.
    
    Keyword Arguments: 
    y - theground truth
    y_prob - the model predictions
    eta - learning rate 
    plev - the level of precision we are trying to maintain
    """
    threshold = 0.5
    yhat = decide(y_prob, threshold)
    p = precision(y, yhat)
    r = recall(y, yhat)
    initial_loss = directional_loss(y, yhat)
    if output: 
        print(f"Precision = {p}, Recall = {r}, Threshold = {threshold}")
    
    for i in range(1, max_iter): 
        threshold -= eta/i*threshold
        yhat = decide(y_prob, threshold)
        
        
        p = precision(y, yhat)
        r = recall(y, yhat)
        
        if output: 
            print(f"Precision = {p}, Recall = {r}, Threshold = {threshold}")
        
        if (p <= plev): 
            return threshold
            
    return threshold

def accuracy(y, yhat): 
    return 1 - ((y-yhat)**2).sum()/y.shape[0]