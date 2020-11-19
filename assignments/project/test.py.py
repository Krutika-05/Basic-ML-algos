#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from project import my_model
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample 
from sklearn.model_selection import train_test_split
import gensim
import time
import sys

sys.path.insert(0, '..')
from assignment8.my_evaluation import my_evaluation


def test(data):
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    split_point = int(0.8 * len(y))
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    eval = my_evaluation(predictions, y_test)
    f1 = eval.f1(target=1)
    return f1


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("F1 score: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print(runtime)

