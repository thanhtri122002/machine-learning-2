class Evaluate:
    def __init__(self,pred_val,test_val,model):
        self.pred_val = pred_val
        self.test_val = test_val
        self.model = model
    def plot_param(self):
        if isinstance(self.model, Random_Forest):
            param_names = ['n_estimators', 'criterion']
            param_ranges = [[10, 50, 100], ['gini', 'entropy']]
        elif isinstance(self.model, SVM):
            param_names = ['C', 'gamma', 'kernel']
            param_ranges = [[0.1, 1, 10], [1, 0.1, 0.01], ['rbf', 'linear']]
        else:
            raise ValueError('Invalid model type')
        
        for i in range(len(param_names)):
            param_name = param_names[i]
            param_range = param_ranges[i]
            
            train_scores = []
            test_scores = []

            for param_value in param_range:
                if isinstance(self.model, Random_Forest):
                    if param_name == 'n_estimators':
                        model = RandomForestClassifier(n_estimators=param_value)
                    elif param_name == 'criterion':
                        model = RandomForestClassifier(criterion=param_value)
                elif isinstance(self.model, SVM):
                    if param_name == 'C':
                        model = SVC(C=param_value)
                    elif param_name == 'gamma':
                        model = SVC(gamma=param_value)
                    elif param_name == 'kernel':
                        model = SVC(kernel=param_value)

                train_score = cross_val_score(model,X_train,y_train).mean()
                test_score=cross_val_score(model,X_test,y_test).mean()

                train_scores.append(train_score)
                test_scores.append(test_score)

            plt.figure()
            
            plt.plot(param_range,train_scores,label='Training score')
            plt.plot(param_range,test_scores,label='Test score')

            plt.xlabel(param_name)
            plt.ylabel('Score')

            plt.legend()

        plt.show()