class Evaluate:
    def __init__(self, pred_val, test_val, model):
        self.pred_val = pred_val
        self.test_val = test_val
        self.model = model

    def plot_param(self):
        if isinstance(self.model, Random_Forest):
            param_name = 'n_estimators'
            param_range = [10, 50, 100]
        elif isinstance(self.model, SVM):
            param_name = 'C'
            param_range = [0.1, 1, 10]
        else:
            raise ValueError('Invalid model type')

        train_scores = []
        test_scores = []

        for param_value in param_range:
            if isinstance(self.model, Random_Forest):
                model = RandomForestClassifier(n_estimators=param_value)
            elif isinstance(self.model, SVM):
                model = SVC(C=param_value)

            train_score = cross_val_score(model, X_train, y_train).mean()
            test_score = cross_val_score(model, X_test, y_test).mean()

            train_scores.append(train_score)
            test_scores.append(test_score)

        plt.plot(param_range, train_scores, label='Training score')
        plt.plot(param_range, test_scores, label='Test score')

        plt.xlabel(param_name)
        plt.ylabel('Score')

        plt.legend()

        plt.show()
