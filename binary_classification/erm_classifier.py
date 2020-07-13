from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from base_classifier import *
from utils import *
from prepare_data import *
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin

NUM_SAMPLES = 100

class erm_classifier(base_binary_classifier, ClassifierMixin, BaseEstimator):
    def fit(self, _X, _Y, classifier_name="logistic"):
        if self.sensitive_train is not None:
            total_train = np.c_[self.train_X, self.sensitive_train]
        else:
            total_train = self.train_X
        
        self.classifier_name = classifier_name
        if classifier_name == "logistic":
            estimator = LogisticRegression(solver='liblinear', fit_intercept=True)
            estimator.fit(total_train, self.train_Y)
        elif classifier_name == "SVM":
            estimator = SVC(gamma="scale", probability=True)
            estimator.fit(total_train, self.train_Y)
        elif classifier_name == "XGBoost":
            D_train = xgb.DMatrix(total_train, label=self.train_Y)
            param = { 'eta':0.3,
                      'max_depth':3,
                      'objective':'binary:logistic'
                    }
            steps = 20
            estimator = xgb.XGBClassifier(**param)
            estimator.fit(total_train, self.train_Y)

        self.model = estimator
        self.trained = True

    def predict(self, x_samples, sensitive_features=None):
        if self.trained == False:
            print("Train the model first!")
            return
        if sensitive_features is not None:
            x_samples = np.c_[x_samples, sensitive_features]
        
        y_samples = self.model.predict(x_samples)
        return y_samples
    
    def predict_proba(self, x_samples, sensitive_features=None):
        if self.trained == False:
            print("Train the model first!")
            return
        if sensitive_features is not None:
            x_samples = np.c_[x_samples, sensitive_features]
        
        y_samples = self.model.predict_proba(x_samples)[:,1]
        return y_samples


def main():
    X_train, X_test, sensitive_features_train, sensitive_features_test, \
            y_train, y_test, sensitive_feature_names = get_data_compas()
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    # 0 should be black, 1 white
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}

    erm_classifier(X_train, y_train, X_test, y_test)

    # uncomment this for fariness throught unawareness
    #sensitive_train_binary, sensitive_test_binary = np.zeros(len(y_train)), np.zeros(len(y_test))
    
    classifier = erm_classifier(X_train, y_train, sensitive_train_binary, sensitive_features_dict)
    classifier.fit(X_train, y_train, classifier_name="logistic")
    total_train, total_test = 0, 0
    for i in range(NUM_SAMPLES):
        total_train += classifier.get_accuracy(X_train, y_train, sensitive_train_binary)
        total_test += classifier.get_accuracy(X_test, y_test, sensitive_test_binary)
    print("Train Acc:", total_train/NUM_SAMPLES) 
    print("Test Acc:", total_test/NUM_SAMPLES) 
        
    _, _ = classifier.get_proportions(X_train, sensitive_train_binary)
    _, _ = classifier.get_proportions(X_test, sensitive_test_binary)
    classifier.get_test_flips(X_test, sensitive_test_binary, True)
    classifier.get_group_confusion_matrix(sensitive_train_binary, X_train, y_train) 
    print("\n")


def xgboost_classifier(X_train, y_train, X_test, y_test):
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)

    param = { 'eta':0.3,
              'max_depth':3,
              'objective':'binary:hinge'
            }
    steps = 20

    model = xgb.train(param, D_train, steps)
    preds = model.predict(D_test)
    acc = 1 - np.sum(np.power(preds - y_test, 2))/len(y_test)
    precision = precision_score(preds, y_test)
    recall = recall_score(preds, y_test)
    print("Test Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)

if __name__ == "__main__":
    main()
