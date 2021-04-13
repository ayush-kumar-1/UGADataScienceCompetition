import numpy as np 
import pandas as pd

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

def tune_threshold(y, y_prob, max_iter = 100, eta = 0.1):
    """
    Tunes the threshold of the decision rule to improve accuracy.
    
    Keyword Arguments: 
    y - theground truth
    y_prob - the model predictions 
    """
    threshold = 0.5
    yhat = decide(y_prob, threshold)
    best_acc = accuracy(y, yhat)
    initial_loss = directional_loss(y, yhat)
    print(f"Accuracy = {best_acc}, Threshold = {threshold}, Loss = {initial_loss}")

    
    for i in range(max_iter): 
        loss = directional_loss(y, yhat)
        new_threshold = threshold + eta*directional_loss(y, yhat)
        yhat = decide(y_prob, new_threshold)
        acc = accuracy(y, yhat)

        print(f"Accuracy = {acc}, Threshold = {new_threshold}, Loss = {loss}")
        
        if (acc <= best_acc): 
            return threshold
        else: 
            threshold = new_threshold
            best_acc = acc
            
    return threshold


def accuracy(y, yhat): 
    return 1 - ((y-yhat)**2).sum()/y.shape[0]