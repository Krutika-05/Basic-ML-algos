
import numpy as np
#from my_evaluation import my_evaluation
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from pprint import pprint
from copy import deepcopy
from pdb import set_trace

import pandas as pd

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba)!=type(None):
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions)+list(self.actuals)))
    
        print(self.classes_)
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion_matrix = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables


        correct = self.predictions == self.actuals
        wrong = self.predictions != self.actuals
        self.acc = float(Counter(correct)[True])/len(correct)
        self.err = float(Counter(wrong)[True])/len(wrong)
        self.confusion_matrix = {}

        #initializing the values to be zero
        for label in self.classes_:
            tp=0
            fp=0
            tn=0
            fn=0


            for i in range(len(self.predictions)):

                if self.actuals[i]==label and label==self.predictions[i]:

                    tp=tp+1
                    #print(this is the value for tp)

                elif  self.predictions[i]==label and label!=self.actuals[i]:
                    fp=fp+1
                    #print(this is value for fp)
            

                elif self.actuals[i]==label and self.predictions[i]!=label:
                    fn=fn+1

                elif self.actuals[i]!=label and label!=self.predictions[i]:
                    tn=tn+1
                    #print(self.confusion_matrix)
                
                self.confusion_matrix[label] = {"TP":tp, "TN": tn, "FP": fp, "FN": fn}

        print("confusion matrix:")
        print(self.confusion_matrix)

        return   
    
    def accuracy(self):
        if self.confusion_matrix==None:
            self.confusion()
        return self.acc

    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class

        # output: prec = float
        # note: be careful for divided by 0

        if self.confusion_matrix==None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp+fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            #return the accuracy for micro averages
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        precision = 0
                    else:
                        precision = float(tp) / (tp + fp)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    prec += precision * ratio

        #print(prec)
        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        if target==None:
            rec=0
            
            

        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp+fn == 0:
                rec = 0
            else:
                rec = float(tp) / (tp + fn)
        else:
            if average == "micro":
                rec = self.accuracy()
            else:
                rec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        recall = 0
                    else:
                        recall = float(tp) / (tp + fn)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                        rec += recall * ratio

        #print(rec)
        return rec

 
    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float

            
        if target in self.classes_:
            prec=self.precision(target,average)
            rec=self.recall(target,average)

            if prec+rec == 0:
                f1 = 0
            else:
                f1 = 2 * ((prec * rec) / (prec + rec))
        else:
            if average == "micro":
                f1 = self.accuracy()

            else:

                f1=0
                n = len(self.actuals)
                for label in self.classes_:
                    prec=self.precision(label,average)
                    rec=self.recall(label,average)
                    #initial condition
                    if prec + rec == 0:
                        f1Score = 0
                    else:
                        f1Score = 2 * ((prec * rec) / (prec + rec))
                    if average == "macro":

                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    f1 += f1Score * ratio


        return f1


        
    def auc(self, target):
        # compute AUC of ROC curve for each class
        # return auc = {self.classes_[i]: auc_i}, dict
        if type(self.pred_proba) == type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]

                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                tpr = 0
                fpr = 0
                auc_target = 0
                for i in order:

                    if self.actuals[i] == target:
                        tp = tp+1
                        fn = fn-1
                        tpr = tp/(tp+fn)
                    else:
                        fp = fp + 1
                        tn = tn - 1
                        pre_fpr = fpr
                        fpr = fp / (fp + tn)
                        auc_target = auc_target +(tpr*(fpr-pre_fpr))


            else:
                raise Exception("Unknown target class.")

            return auc_target




