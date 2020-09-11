#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter


# In[2]:




class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        # list of classes for this model
        self.classes_ = list(set(list(y)))
        # for calculation of P(y)
        self.P_y = Counter(y)
        # self.P[yj][Xi][xi] = P(xi|yi) where Xi is the feature name and xi is the feature value, yj is a specific class label
        self.P = {}
        all_possible_values = {}
        for key in X:
            all_possible_values[key] = set(X[key])

        for classes in self.classes_:
            self.P[classes] = {}
            # print(y == label)
            for key in X:
                self.P[classes][key] = {}
                count = Counter(X[key].where(y == classes))
                for value in all_possible_values[key]:
                    # print(count[value])
                    l = len(all_possible_values.keys())
                    self.P[classes][key][value] = (count[value] + self.alpha) / (
                                self.P_y[classes] + len(all_possible_values[key]) * self.alpha)
        return

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions


# In[3]:


#  Load training data
data_train = pd.read_csv(r"C:\DSCI-633\assignments\data\audiology_train.csv",header=None)
# Separate independent variables and dependent variables
independent = range(69)
X = data_train[independent]
y = data_train[70]
# Train model
clf = my_NB()
clf.fit(X,y)
# Load testing data
data_test = pd.read_csv(r"C:\DSCI-633\assignments\data\audiology_test.csv",header=None)
X_test = data_test[independent]
# Predict
predictions = clf.predict(X_test)
# Predict probabilities
probs = clf.predict_proba(X_test)
# Print results
for i,pred in enumerate(predictions):
    print("%s\t%f" % (pred, probs[pred][i]))


# In[ ]:




