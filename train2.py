import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, plot_confusion_matrix, recall_score, precision_score, accuracy_score
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D, MaxPooling2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.datasets import mnist
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score, train_test_split
import joblib
from sklearn.tree import DecisionTreeClassifier
HEIGHT = WIDTH = 28
#loading the dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

train_size = 20000 / len(x)
test_size = 5000 / len(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                    test_size=test_size,
                                                    random_state=2023)
#printing the shapes of the vectors
#x_train: (60000, 28, 28)
#Y_train: (60000,)
#x_test:  (10000, 28, 28)
#Y_test:  (10000,)


class Preprocessing:
    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test

    def normalization(self):
        #reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        #convert the datatype
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        #normalization into scale [0,1]
        self.x_train /= 255
        self.x_test /= 255
        return self.x_train,  self.x_test


class SVM:
    def __init__(self):
        self.svc = SVC()
        self.gscv = None

    def train_model(self):
        params_grid = [{'kernel': ['rbf', 'poly', 'sigmoid'],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'gamma': [0.0001, 0.001, 0.01, 0.1]},
                       {'kernel': ['linear'],
                        'C': [0.001, 0.01, 0.1, 1],
                        'gamma': [0.0001, 0.001]}]
        #find the best parameter for training set
        self.gscv = GridSearchCV(self.svc, param_grid=params_grid)
        self.gscv.fit(x_train, y_train)
        params = self.get_best_params()
        # create a file name based on the model and parameters
        filename = f"SVM_{params['kernel']}_{params['C']}_{params['gamma']}.joblib"
        # save the model to file
        joblib.dump(self.gscv.best_estimator_, filename)

    def get_best_params(self):
        #return the best parameters found by GridSearchCV
        return self.gscv.best_params_

    def get_best_score(self):
        return self.gscv.best_score_


class Random_Forest:
    def __init__(self) -> None:
        self.rf = RandomForestClassifier()
        self.gscv = None

    def train_model(self):

        #define a grid of parameters to search over
        params_grid = [{'n_estimators': [10, 50, 100, 200],
                        'max_depth': [None, 5, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'criterion': ['gini', 'entropy', 'log_loss']}]
        #find the best parameter for training set using grid search and cross-validation
        #specify the number of folds (e.g. 5) in the cv parameter
        self.gscv = GridSearchCV(self.rf, param_grid=params_grid)
        self.gscv.fit(x_train, y_train)
        params = self.get_best_params()

        #create a file name based on the model and parameters
        filename = f"RandomForest_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}_{params['min_samples_leaf']}.joblib"
        #save the model to file
        joblib.dump(self.gscv.best_estimator_, filename)

    def get_best_params(self):
        #return the best parameters found by GridSearchCV
        return self.gscv.best_params_

    def get_best_score(self):
        return self.gscv.best_score_


class KNN:
    def __init__(self):
        self.knn = KNeighborsClassifier()
        self.gscv = None

    def train_model(self):
        params_grid = params_grid = [{'n_neighbors': [3, 5, 7, 9, 11, 13],
                                      'weights': ['uniform', 'distance'],
                                      'metric': ['euclidean', 'manhattan', 'minkowski']}]
        self.gscv = GridSearchCV(self.knn, param_grid=params_grid)
        self.gscv.fit(x_train, y_train)
        params = self.get_best_params()
        #create a file name based on the model and parameters
        filename = f"KNN_{params['n_neighbors']}_{params['weights']}_{params['metric']}.joblib"
        #save the model to file
        joblib.dump(self.gscv.best_estimator_, filename)

    def get_best_params(self):
        return self.gscv.best_params_

    def get_best_score(self):
        return self.gscv.best_score_


class DecisionTree:
    def __init__(self) -> None:
        self.dt = DecisionTreeClassifier()
        self.gscv = None

    def train_model(self):
        params_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [4, 5, 6, 7, 8, 9, 10],
            'splitter': ['best', 'random'],
            'min_samples_split': [2, 3, 4]
        }
        self.gscv = GridSearchCV(
            estimator=self.dt, param_grid=params_grid)

        self.gscv.fit(x_train, y_train)
        params = self.get_best_params()
        #create a file name based on the model and parameters
        filename = f"DTree_{params['criterion']}_{params['splitter']}_{params['max_depth']}.joblib"
        #save the model to file
        joblib.dump(self.gscv.best_estimator_, filename)

    def get_best_params(self):
        return self.gscv.best_params_

    def get_best_score(self):
        return self.gscv.best_score_


class AlexNet:
    def __init__(self, HEIGHT, WIDTH, n_outputs):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.n_outputs = n_outputs

    def define_model(self):
        input = Input(shape=(self.HEIGHT, self.WIDTH, 1))

        # first layer
        x = Conv2D(filters=96, kernel_size=11, strides=4,
                   name='conv1', activation='relu')(input)
        x = MaxPooling2D(pool_size=3, strides=2, name='pool1')(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D(2)(x)

        # second layer
        x = Conv2D(filters=256, kernel_size=3, strides=1,
                   name="conv2", activation='relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2, name="pool2")(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D(1)(x)

        # third layer
        x = Conv2D(filters=384, kernel_size=3, strides=1,
                   name='conv3', activation='relu')(x)
        x = ZeroPadding2D(1)(x)

        # fourth layer
        x = Conv2D(filters=384, kernel_size=3, strides=1,
                   name='conv4', activation='relu')(x)
        x = ZeroPadding2D(1)(x)

    #fifth layer
        x = Conv2D(filters=256, kernel_size=3, strides=1,
                   name='conv5', activation='relu')(x)

        x = Flatten()(x)

        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(0.5, name='dropout_6')(x)

        x = Dense(4096, activation='relu',  name='fc7')(x)
        x = Dropout(0.5, name='dropout_7')(x)

        x = Dense(self.n_outputs, activation='softmax', name='fc8')(x)

        model = Model(inputs=input, outputs=x)
        return model

    def train_model(self, x_train, y_train, x_test, y_test):
        model = self.define_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                     ModelCheckpoint(filepath='alexnet_model_mnist.h5', monitor='val_loss', save_best_only=True)]
        #train the model
        train = model.fit(x_train, y_train, epochs=20, validation_data=(
            x_test, y_test), callbacks=callbacks)

class LeNet:
    def __init__(self, HEIGHT, WIDTH, n_outputs):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.n_outputs = n_outputs

    def define_model(self):
        input = Input(shape=(self.HEIGHT, self.WIDTH, 1))
        x = Conv2D(20, (5, 5), padding="same")(input)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        pass


class Evaluate:
    def __init__(self, test_val, model):
        self.test_val = test_val  # true labels
        self.model = model  # model name
        self.models = {}  # dictionary to store models
        self.predictions = {}  # dictionary to store predictions

    def load_models(self):
        """Load models from a folder and store them in a dictionary"""
        folder = "models"  # the folder where the models are saved
        for file in os.listdir(folder):  # iterate over the files in the folder
            if file.endswith(".joblib"):  # check if the file is a joblib file
                # get the model name from the file name
                model_name = file.split(".")[0]
                # get the full path of the file
                model_path = os.path.join(folder, file)
                model = joblib.load(model_path)  # load the model from the file
                # store the model in the dictionary
                self.models[model_name] = model

    def predict_models(self, x_test):
        """Predict on x_test using different models and store them in a dictionary"""
        for model_name, model in self.models.items():  # iterate over the models in the dictionary
            y_pred = model.predict(x_test)  # predict using the model on x_test
            self.predictions[model_name] = y_pred  # store the predictions

    def plot_confusion_matrix(self):
        """Plot confusion matrices for different models"""
        fig, axes = plt.subplots(nrows=len(self.models), ncols=1, figsize=(
            10, 10))  # create a figure with subplots for each model
        # set a title for the figure
        fig.suptitle("Confusion matrices for different models")

        # iterate over the predictions in the dictionary
        for i, (model_name, y_pred) in enumerate(self.predictions.items()):

            plot_confusion_matrix(model=self.models[model_name], X=x_test, y_true=self.test_val,
                                  ax=axes[i], cmap="Blues", values_format="d")  # plot confusion matrix using sklearn function on subplot i

            axes[i].set_title(model_name)  # set a title for subplot i

            # generate a classification report
            report = classification_report(y_true=self.test_val, y_pred=y_pred)
            print(f"Classification report for {model_name}:")
            print(report)  # print the report


"""class Evaluate:
    def __init__(self, test_val, model):
        self.test_val = test_val
        self.model = model
    def load_models(self):
        folder = "models" #the folder where the models are saved
        for file in os.listdir(folder): #iterate over the files in the folder
            if file.endswith(".joblib"): #check if the file is a joblib file
                model_name = file.split(".")[0] #get the model name from the file name
                model_path = os.path.join(folder, file) #get the full path of the file
                model = joblib.load(model_path) #load the model from the file
                self.models[model_name] = model #store the model in the dictionary

    def predict_models(self, x_test):
        for model_name, model in self.models.items(): #iterate over the models in the dictionary
            y_pred = model.predict(x_test) #predict using the model on x_test
            self.predictions[model_name] = y_pred #store the predictions in the dictionary
            return self.predictions

    def plot_confusion_matrix(self):
        fig, axes = plt.subplots(nrows=len(self.models), ncols=1, figsize=(10, 10)) #create a figure with subplots for each model
        fig.suptitle("Confusion matrices for different models") #set a title for the figure
        
        for i, (model_name, y_pred) in enumerate(self.predictions.items()): #iterate over the predictions in the dictionary
            
            cm = confusion_matrix(self.test_val, y_pred) #compute confusion matrix using true and predicted labels
            
            axes[i].imshow(cm, cmap="Blues") #plot confusion matrix as an image on subplot i
            
            axes[i].set_title(model_name)"""
if __name__ == "__main__":
    preprocessing = Preprocessing(x_train=x_train, x_test=x_test)
    x_train, x_test = preprocessing.normalization()
    y_preds = {}
    while True:
        print("1.Train and save models")
        choice = int(input("choose your choice: "))
        if choice == 1:
            model_rf = Random_Forest()
            model_rf.train_model()
            # load the best model from the file
            filename_rf = f"RandomForest_{model_rf.get_best_params()['n_estimators']}_{model_rf.get_best_params()['max_depth']}_{model_rf.get_best_params()['min_samples_split']}_{model_rf.get_best_params()['min_samples_leaf']}.joblib"
            trained_rf = joblib.load(filename_rf)
            # predict on the test data using the best model
            y_pred_rf = trained_rf.predict(x_test)
            y_preds['random-forest'] = y_pred_rf

            model_knn = KNN()
            # train the model using GridSearchCV and save it to a file
            model_knn.train_model()
            # load the best model from the file
            filename_knn = f"KNN_{model_knn.get_best_params()['n_neighbors']}_{model_knn.get_best_params()['weights']}_{model_knn.get_best_params()['metric']}.joblib"
            trained_knn = joblib.load(filename_knn)
            # predict on the test data using the best model
            y_pred_knn = trained_knn.predict(x_test)
            y_preds['k-nearest-neighbors'] = y_pred_knn

            model_dt = DecisionTree()
            # train the model using GridSearchCV and save it to a file
            model_dt.train_model()
            # load the best model from the file
            filename_dt = f"DTree_{model_dt.get_best_params()['criterion']}_{model_dt.get_best_params()['splitter']}_{model_dt.get_best_params()['max_depth']}.joblib"
            trained_dt = joblib.load(filename_dt)
            # predict on the test data using the best model
            y_pred_dt = trained_dt.predict(x_test)
            y_preds['decision-tree'] = y_pred_dt

            model_svm = SVM()
            # train the model using GridSearchCV and save it to a file
            model_svm.train_model()
            filename_svm = f"SVM_{model_svm.get_best_params()['kernel']}_{model_svm.get_best_params()['C']}_{model_svm.get_best_params()['gamma']}.joblib"
            trained_svm = joblib.load(filename_svm)
            # predict on the test data using the best model
            y_pred_svm = trained_svm.predict(x_test)
            y_preds['svm'] = y_pred_svm
        if choice == 2:
            x_train = x_train.reshape(x_train.shape[0], HEIGHT, WIDTH, 1)
            x_test = x_test.reshape(x_test.shape[0], HEIGHT, WIDTH, 1)

            uniques_values, counts = np.unique(y_train, return_counts=True)
            no_classes = len(uniques_values)

            y_train = to_categorical(y_train, no_classes)
            y_test = to_categorical(y_test, no_classes)

            model = AlexNet(HEIGHT=HEIGHT, WIDTH=WIDTH, n_outputs=no_classes)
            model.train_model(x_train, y_train, x_test, y_test)
            model_alexnet = load_model('alexnet_model_mnist.h5')
            y_pred = model_alexnet.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(accuracy)
        if choice == 3:
            x_train = x_train.reshape(x_train.shape[0], HEIGHT, WIDTH, 1)
            x_test = x_test.reshape(x_test.shape[0], HEIGHT, WIDTH, 1)

            uniques_values, counts = np.unique(y_train, return_counts=True)
            no_classes = len(uniques_values)

            y_train = to_categorical(y_train, no_classes)
            y_test = to_categorical(y_test, no_classes)
        else:
            break


"""
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# SVM
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
svm_scores = []

for params in ParameterGrid(svm_param_grid):
    svm = SVC(**params)
    scores = cross_val_score(svm, x_train,y_train,cv=5)
    svm_scores.append(scores.mean())

plt.plot(svm_scores,label='SVM')

# Random Forest
rf_param_grid = {'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}
rf_scores = []

for params in ParameterGrid(rf_param_grid):
    rf = RandomForestClassifier(**params)
    scores = cross_val_score(rf,x_train,y_train,cv=5)
    rf_scores.append(scores.mean())

plt.plot(rf_scores,label='Random Forest')

plt.xlabel('Parameter Set')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.show()

param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
        grid = GridSearchCV(SVC(), param_grid)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        best_param = grid.best_params_
        param_c = best_param['C']
        param_gamma = best_param['gamma']
        param_kernel = best_param['kernel']
        svm = SVC(**grid.best_params_ )
        svm.fit(x_train,y_train)
        joblib.dump(svm, f"my_svm_{param_c}_{param_gamma}_{param_kernel}.joblib")


        svm_param_grid = {'C': [1, 10], 'gamma': [1,0.01]}
        kernel = ['rbf', 'linear']
        svm_collection = []
        for params in ParameterGrid(svm_param_grid):
            svm = SVC(**params)
            svm.fit(x_train,y_train)
            filename = f"svm_{params['C']}_{params['gamma']}_{params['kernel']}.joblib"
            #add name of each model into the list
            svm_collection.append(filename)
            joblib.dump(svm,filename)
        return svm_collection

        svm_collection = self.train_model()
        y_pred_list = []
        for filename in svm_collection:
            model= joblib.load(filename= filename)
            y_pred = model.predict(x_test)
            y_pred_list.append(y_pred)
        return y_pred_list


        param_grid = {'n_estimators': [
            10, 50, 100], 'criterion': ['gini', 'entropy']}
        grid = GridSearchCV(RandomForestClassifier(), param_grid)
        print(grid.best_params_)
        grid.fit(x_train, y_train)
        best_param = grid.best_params_
        param_n_estimators = best_param['n_estimators']
        param_criterion = best_param['criterion']
        model = RandomForestClassifier(**grid.best_params_)
        model.fit(x_train,y_train)
        joblib.dump(model, f"my_random_forest_{param_n_estimators}_{param_criterion}.joblib")
"""
