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


# create an instance of the SVM class
model_svm = SVM()
# train the model using GridSearchCV and save it to a file
model_svm.train_model()
# load the best model from the file
filename = f"SVM_{model_svm.get_best_params()['kernel']}_{model_svm.get_best_params()['C']}_{model_svm.get_best_params()['gamma']}.joblib"
best_model = joblib.load(filename)
# predict on the test data using the best model
y_pred_svm = best_model.predict(X_test)
y_preds['svm'] = y_pred_svm
