import pandas as pd

<<<<<<< HEAD
if __name__ == "__main__":
    #  Load training data
    data_train = pd.read_csv("../data/Iris_train.csv")
    # Separate independent variables and dependent variables
    independent = ["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm",	"PetalWidthCm"]
    X = data_train[independent]
    y = data_train["Species"]
    # Train model
    clf = my_DT()
    clf.fit(X,y)
    # Load testing data
    data_test = pd.read_csv("../data/Iris_test.csv")
    X_test = data_test[independent]
    # Predict
    predictions = clf.predict(X_test)
    # Predict probabilities
    probs = clf.predict_proba(X_test)
    # Print results
    for i,pred in enumerate(predictions):
        print("%s\t%f" %(pred, probs[pred][i]))
=======
   #  Load training data
   data_train = pd.read_csv("../data/Iris_train.csv")
   # Separate independent variables and dependent variables
   independent = ["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm",	"PetalWidthCm"]
   X = data_train[independent]
   y = data_train["Species"]
   # Train model
   clf = my_DT()
   clf.fit(X,y)
   # Load testing data
   data_test = pd.read_csv("../data/Iris_test.csv")
   X_test = data_test[independent]
   # Predict
   predictions = clf.predict(X_test)
   # Predict probabilities
   probs = clf.predict_proba(X_test)
   # Print results
   for i,pred in enumerate(predictions):
    print("%s\t%f" %(pred, probs[pred][i]))
>>>>>>> 0d9e770d79e93c5d2444f46d9a1482dc51c6e962
