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
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score
import joblib
HEIGHT = WIDTH = 28
#loading the dataset
(X_train, y_train), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors 
#X_train: (60000, 28, 28)
#Y_train: (60000,)
#X_test:  (10000, 28, 28)
#Y_test:  (10000,)

class Preprocessing:
    def __init__(self, X_train , test_X):
        self.X_train = X_train
        self.test_X = test_X
    def normalization(self):
        #reshape
        self.X_train= self.X_train.reshape(self.X_train.shape[0], -1)
        self.test_X = self.test_X.reshape(self.test_X.shape[0], -1)
        #convert the datatype
        self.X_train = self.X_train.astype('float32')
        self.test_X = self.test_X.astype('float32')
        #normalization into scale [0,1]
        self.X_train /=255
        self.test_X /= 255
        return self.X_train ,  self.test_X

class  SVM:
    def train_model(self):
        svm_param_grid = {'C': [1, 10], 'gamma': [1,0.01], 'kernel': ['rbf', 'linear']}
        svm_collection = []
        for params in ParameterGrid(svm_param_grid):
            svm = SVC(**params)
            filename = f"svm_{params['C']}_{params['gamma']}_{params['kernel']}.joblib"
            #add name of each model into the list
            svm_collection.append(filename)
            joblib.dump(svm,filename)
        return svm_collection
    def predict_model(self,X_test):
        svm_collection = self.train_model()
        y_pred_list = []
        for filename in svm_collection:
            model= joblib.load(filename= filename)
            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
        return y_pred_list

class Random_Forest:
    def train_model(self):
        rf_param_grid = {'n_estimators': [10, 50], 'criterion': ['gini', 'entropy']}
        random_forest_collection = []
        for params in ParameterGrid(rf_param_grid):
            rf = RandomForestClassifier(**params)
            
            filename = f"rf_{params['n_estimators']}_{params['criterion']}.joblib"
            #add name of each model into the list
            random_forest_collection.append(filename)
            # Save model to file
            joblib.dump(rf,filename)
        return random_forest_collection
           

        
        """param_grid = {'n_estimators': [
            10, 50, 100], 'criterion': ['gini', 'entropy']}
        grid = GridSearchCV(RandomForestClassifier(), param_grid)
        print(grid.best_params_)
        grid.fit(X_train, y_train)
        best_param = grid.best_params_
        param_n_estimators = best_param['n_estimators']
        param_criterion = best_param['criterion']
        model = RandomForestClassifier(**grid.best_params_)
        model.fit(X_train,y_train)
        joblib.dump(model, f"my_random_forest_{param_n_estimators}_{param_criterion}.joblib")"""
    def predict_model(self,X_test):
        random_forest_collection = self.train_model()
        y_pred_list = []
        for filename in random_forest_collection:
            model= joblib.load(filename= filename)
            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
        return y_pred_list


class Evaluate:
    def __init__(self,pred_val,test_val,model):
        self.pred_val = pred_val
        self.test_val = test_val
        self.model
    def plot_param(self):
        pass
        
    def confusion_matrix(self):
        cm = confusion_matrix(self.test_val,self.pred_val)
        pass
    

        
if __name__ =="__main__":
    preprocessing = Preprocessing(X_train= X_train , test_X= test_X)
    X_train, test_X = preprocessing.normalization()
    while True:
        print("1.SVM")
        print("2.Random forest")
        choice = int(input("choose model to train"))
        if choice ==1:
            model = SVM()
            y_pred_vals = model.predict_model(X_test=test_X)
            print(y_pred_vals)
        if choice == 2:
            model = Random_Forest()
            y_pred_vals = model.predict_model(X_test= test_X)
            print(y_pred_vals)              
    

"""
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# SVM
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
svm_scores = []

for params in ParameterGrid(svm_param_grid):
    svm = SVC(**params)
    scores = cross_val_score(svm, X_train,y_train,cv=5)
    svm_scores.append(scores.mean())

plt.plot(svm_scores,label='SVM')

# Random Forest
rf_param_grid = {'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}
rf_scores = []

for params in ParameterGrid(rf_param_grid):
    rf = RandomForestClassifier(**params)
    scores = cross_val_score(rf,X_train,y_train,cv=5)
    rf_scores.append(scores.mean())

plt.plot(rf_scores,label='Random Forest')

plt.xlabel('Parameter Set')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.show()

param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
        grid = GridSearchCV(SVC(), param_grid)
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        best_param = grid.best_params_
        param_c = best_param['C']
        param_gamma = best_param['gamma']
        param_kernel = best_param['kernel']
        svm = SVC(**grid.best_params_ )
        svm.fit(X_train,y_train)
        joblib.dump(svm, f"my_svm_{param_c}_{param_gamma}_{param_kernel}.joblib")
"""
