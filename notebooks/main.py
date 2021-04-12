#!/usr/bin/env python
# coding: utf-8

# UGA Data Science Competition
# Authors: Ayush Kumar, Chole Phelps, Faisal Hossian, Nicholas Sung

# Import Libraries
import numpy as np
import pandas as pd
from model import *


# Read Datasets using Pandas
credit_data = pd.read_csv("../data/Simulated_Data_Train.csv", header = 0)
credit_test_data = pd.read_csv("../data/Simulated_Data_Test.csv", header = 0)

# Data preprocessing
credit_data = credit_data.dropna() #Drop the rows where at least one element is missing

# Prepare Data
X = credit_data.filter(['tot_credit_debt', 'credit_good_age', 'non_mtg_acc_past_due_6_months_num',
                       'inq_12_month_num', 'uti_max_credit_line', 'rep_income'], axis=1)
X = scale(X) # scale X
y = credit_data['Default_ind'] # Response variable

### Logisitc Regression
logreg = log_reg(X,y)

# predictions
predictions = logreg.predict()

print(predictions)

# Confusion Matrix
cfm = metrics.confusion_mat(predictions, y)
print(cfm)

# Accuracy
accuracy_score = metrics.accuracy(predictions, y)
print("Accuracy Score: ", accuracy_score)

# Precision
precision_score = metrics.precision(predictions, y)
print("Precision Score: ", precision_score)

# Recall
recall_score = metrics.recall(predictions, y)
print("Recall Score: ", recall_score)
