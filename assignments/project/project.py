import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
import time
import sys
sys.path.insert(0, '..')
from assignment8.my_evaluation import my_evaluation

class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        self.y_cls = y
        X = self.preprocessing_data(X)
        
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True, smooth_idf=True, ngram_range=(1,4))
        XX = self.preprocessor.fit_transform(X["description"])
        #self.classifier = KNeighborsClassifier(n_neighbors = 5,class_weight='balanced')
        self.classifier = SGDClassifier(class_weight="balanced",random_state=5)
        #self.classifier = svm.SVC(class_weight="balanced")
        self.classifier.fit(XX, y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X = self.preprocessing_data(X)
        XX = self.preprocessor.transform(X["description"])
        predictions = self.classifier.predict(XX)
        return predictions
    
    def preprocessing_data(self, data_frame):
#         
        data_frame['location'] = data_frame.location.fillna('none')
    

        data_frame['requirements'] = data_frame.description.fillna('not specified')
        
        
        data_frame['description'] = data_frame.description.fillna('not specified')
        
        data_frame['has_company_logo'] = data_frame.has_company_logo.map({1 : 'True', 0 : 'False'})
        
        data_frame.drop(['telecommuting','has_questions'],axis = 1, inplace = True)  
        
    
        data_frame['description'] = data_frame.description.str.replace(r'<[^>]*>', '')
        
        data_frame['requirements'] = data_frame.requirements.str.replace(r'<[^>]*>', '')
        
        stopwords = gensim.parsing.preprocessing.STOPWORDS
        
        for column in data_frame.columns:
            data_frame[column] = data_frame[column].str.replace(r'\W', ' ').str.replace(r'\s$','')
            
    
        special_characters_remove = ["!",'"',"$","%","&","'","(",")",
              "*","+",",","-",".","/",":","#","<","=",">","?","@","[","\\","}","^","_","-","`","{","|","]","~","–","•"]
        for char in special_characters_remove:
            data_frame[column] = data_frame[column].str.replace(char, ' ')
            data_frame[column] = data_frame[column].str.split().str.join(" ") 
            
   
        

        return data_frame