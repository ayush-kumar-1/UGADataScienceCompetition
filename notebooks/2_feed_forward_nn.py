#!/usr/bin/env python
# coding: utf-8

# In[155]:


import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf 
import seaborn as sn
import sklearn

from tensorflow import keras
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression as LogRes
from sklearn.metrics import classification_report as report
from sklearn.metrics import precision_score as precision 
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix

from utils import * 
from models import * 


get_ipython().run_line_magic('matplotlib', 'inline')
print("numpy version: \t\t", np.__version__)
print("pandas version: \t", pd.__version__)
print("matplotlib version: \t", matplotlib.__version__)
print("tensorflow version: \t", tf.__version__)
print("scikit-learn version: \t", sklearn.__version__)
print("seaborn version: \t", sn.__version__)


# In[2]:


###Initial Baseline Models 

x_train, y_train = load_data("../data/Simulated_Data_Train.csv")
x_val, y_val = load_data("../data/Simulated_Data_Validation.csv")
x_test, y_test = load_data("../data/Simulated_Data_Test.csv")


# In[4]:


nn = feed_forward(x_train, y_train, width = 32)
nn.train(20)


print("****** Initial Feed Forward Network *********")
print(report(y_test, nn.predict(x_test)))


# In[36]:


def tune_model_width(build_fn, x_train, y_train, x_val, y_val, max_width = 50): 
    """
    Takes a 3-Layer nueral network and expands width to see if there 
    are tangible benefits to increasing the width of the hidden layer 
    in the model. 
    
    Parameters: 
    build_fn - function that returns a keras nn model with the specified parameters 
    x_train - the data matrix 
    y_train - the response function
    x_val - validation data
    y_val - validation data function
    """
    
    acc = []
    pre = []
    rec = []
        
    for i in range(15, max_width): 
        width = i
        model = feed_forward.build_model(x_train, y_train, width = width, suppress = True)
        model.fit(x_train, y_train, epochs = 100, verbose = 0)
    
        y_val_prob = model.predict(x_val)[:,0]
        y_val_hat = decide(y_val_prob, 0.5)

        acc.append(accuracy(y_val, y_val_hat))
        pre.append(precision(y_val, y_val_hat))
        rec.append(recall(y_val, y_val_hat))
    
    return acc, pre, rec 

acc, pre, rec = tune_model_width(feed_forward.build_model, 
                                 x_train, y_train, x_val, y_val)


# In[37]:


plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = [15, 10]

plt.title("Precision, Recall, Accuracy vs. Model Complexity")
plt.xlabel("Hidden Layer Width")
plt.ylabel("Score")

width = [_ for _ in range(15, 50)]

plt.plot(width, acc)
plt.plot(width, pre)
plt.plot(width, rec)
plt.legend(["Accuracy", "Precision", "Recall"])
plt.savefig("nn_width.png")


# In[12]:


import tqdm

precision_scores = []
recall_scores = []

for i in tqdm.tqdm(range(100)): 
    model = feed_forward(x_train, y_train, width = 32)
    model.model.fit(x_train, y_train, epochs = 100, verbose = 0)
    yhat = model.predict(x_test) 
    precision_scores.append(precision(y_test, yhat))
    recall_scores.append(recall(y_test, yhat))
    


# In[16]:




plt.hist(precision_scores)
plt.hist(recall_scores)
plt.title("Precision and Recall Stability after 100 Epochs")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend(['Precision', 'Recall'])
plt.savefig("StableNetwork.png")


# In[20]:


nn.train(100)


# In[21]:


print(report(y_test, nn.predict(x_test)))


# In[35]:


from sklearn.metrics import precision_recall_curve

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = [10,7.5]

precision_nn, recall_nn, _ = precision_recall_curve(y_val ,nn.predict_proba(x_val))

plt.title("Precision Recall Curve")
plt.plot(precision_nn, recall_nn)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.savefig("pr_curve_nn.png")


# In[25]:


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

y_prob = nn.predict_proba(x_val)

plevels = [_ for _ in np.arange(0, 1, 0.01)]
thresholds = []

for p in plevels: 
    thresholds.append(tune_threshold(y_val, y_prob, plev = p, output = False))
    
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = [10, 7.5]

plt.plot(plevels, thresholds)
plt.xlabel("Precision")
plt.ylabel("Decision Threshold")
plt.title("Precision vs. Decision Threshold")
plt.savefig("precisionvsthreshold_nn.png")


# In[34]:


threshold = tune_threshold(y_val, y_prob, plev=0.55, max_iter = 1000, output = False)
print(threshold)
print(report(y_test, decide(nn.predict_proba(x_test), threshold)))


# In[156]:



def plot_conf_mat(title, conf_mat, save = None): 
    plt.rcParams["figure.figsize"] = [2, 2]
    plt.rcParams["font.size"] = 5
    plt.rcParams["figure.dpi"] = 300
    sn.heatmap(conf_mat, 
                square = True,
                fmt = 'g',
                annot = True,
                cbar = False)
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.title(title)
    if save: 
        plt.savefig(save)
        plt.show()
    else: 
        plt.show()
        
plot_conf_mat("Confusion Matrix for Untuned Nueral Network", 
              confusion_matrix(y_test, nn.predict(x_test)), 
              "untuned_nn_confmat.png")

plot_conf_mat("Confusion Matrix for Tuned Nueral Network", 
             confusion_matrix(y_test, decide(nn.predict_proba(x_test), threshold = 0.2602)), 
              "tuned_nn_confmat.png")


# In[128]:


confusion_matrix(y_test, nn.predict(x_test))


# In[103]:


def explain(x_mean, x_obs,  model): 
    """
    Generates a list of numbers coresonding to how strong its influence is on
    the outcome of the model. Works with comparing probabilities 
    """
    
    mean_prob = model.predict_proba(np.array([x_mean,])) 
    predictive_strength = []
    
    for i in range(len(x_obs)): 
        x_mean_copy = x_mean.copy()
        x_mean_copy[i] = x_obs[i] - x_mean[i]
        
        changed_prob = model.predict_proba(np.array([x_mean_copy,]))
        predictive_strength.append((changed_prob)[0])
        
    return predictive_strength

x_mean = x_test.mean(axis = 0)
x_obs45 = x_test[:][45]

print(nn.predict_proba(np.array([x_mean,])))

explain(x_mean, x_obs45, nn)
credit_data = load_data("../data/Simulated_Data_Train.csv", as_df = True)


explain_df = pd.DataFrame.from_dict({col: val for col, val in zip(credit_data.columns, explain(x_mean, x_obs, nn))}, 
                                      "index", columns = ["Person45"])

x_obs1 = x_test[:][1]
explain_df["Person1"] = explain(x_mean, x_obs1, nn)
explain_df.to_latex("../report/explain.tex")


# In[112]:


average_customer = x_mean.copy()
bank_customer = x_mean.copy()

average_customer[13] = 0
bank_customer[13] = 1

print(average_customer)
print(bank_customer)


print(nn.predict_proba(np.array([average_customer, ])))
print(nn.predict_proba(np.array([bank_customer, ])))

