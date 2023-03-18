import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D, MaxPooling2D
from sklearn.svm import SVC
from keras.datasets import mnist
import joblib
HEIGHT = WIDTH = 28
#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors 
#X_train: (60000, 28, 28)
#Y_train: (60000,)
#X_test:  (10000, 28, 28)
#Y_test:  (10000,)
print(train_X)
class Preprocessing:
    def __init__(self, train_X , test_X):
        self.train_X = train_X
        self.test_X = test_X

    def normalization(self):
        #convert the datatype
        self.train_X = self.train_X.astype('float32')
        self.test_X = self.test_X.astype('float32')
        #normalization into scale [0,1]
        self.train_X /=255
        self.test_X /= 255
        return self.train_X ,  self.test_X

class  SVM:
    def __init__(self, kernel):
        self.kernel = kernel
  

    def train_model(self):
        svm = SVC(kernel = self.kernel )
        svm.fit(train_X)
        joblib.dump(svm, f"my_svm_{self.kernel}.joblib")
    def predict_model(self,X_test):
        return self.train_model().predict(X_test)

class Random_Forest:
    def __init__(self,criterion):
        self.criterion= criterion
    def define_model(self):
        model = RandomForestClassifier(criterion= self.criterion)
        model.fit(train_X)
        joblib.dump(model, f"my_random_forest_{self.criterion}.joblib")
    def predict_model(self,X_test):
        return self.define_model().predict(X_test)
    
    
if __name__ =="__main__":
    preprocessing = Preprocessing(train_X= train_X , test_X= test_X)
    train_X, test_X = preprocessing.normalization()
    while True:
        print("1.SVM")
        print("2.Random forest")
        choice = int(input("choose model to train"))
        if choice ==1:
            kernel = input('kernel linear or rbf: ')
            
            model = SVM(kernel = kernel )
            model.predict_model(test_X)     
        if choice == 2:
            criterion = input('choos the criterion gini or entropy: ')  
            model = Random_Forest(criterion= criterion)
            model.predict_model(test_X)
                                    
    

    