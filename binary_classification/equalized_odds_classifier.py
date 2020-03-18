from sklearn.linear_model import LogisticRegression
from base_classifier import *
from tempeh.configurations import datasets
from erm_classifier import *
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

class erm_classifier_prob(BaseEstimator, ClassifierMixin):
    def __init__(self, logistic_regression_estimator):
        self.logistic_regression_estimator = logistic_regression_estimator
    
    def fit(self, X, y):
        estimator = LogisticRegression()
        estimator.fit(X, y)
        return self

    def predict(self, x_samples):
        y_samples = self.model.predict_proba(x_samples)[:,1]
        return y_samples

class equalized_odds_classifier(base_binary_classifier):
    def train(self, sensitive_features):
        self.sensitive_features = sensitive_features
        estimator = LogisticRegression()
        prob_estimator = erm_classifier_prob(estimator)
        prob_estimator.fit(self.train_X, self.train_Y)
        self.model = ThresholdOptimizer(estimator=prob_estimator, constraints="equalized_odds")
        self.model.fit(self.train_X, self.train_Y, sensitive_features=self.sensitive_features) 

    def predict(self, x_samples, sensitive_features):
        y_samples = self.model.predict(x_samples, sensitive_features=sensitive_features)
        return y_samples
    
    def get_accuracy(self, X, y_true, sensitive_features):
        y_pred = self.predict(X, sensitive_features)
        return 1 - np.sum(np.power(y_pred - y_true, 2))/len(y_true) 

def convert_to_binary(str_features, str1, str2):
    bin_features = np.array([])
    for val in str_features:
        if val == str1:
            bin_features = np.append(bin_features, 0)
        else:
            bin_features = np.append(bin_features, 1)
    return bin_features

def main():
    compas_dataset = datasets['compas']()
    X_train, X_test = compas_dataset.get_X()
    y_train, y_test = compas_dataset.get_y()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Add the sensitive feature
    sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('race')
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            "African-American", "Caucasian")
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            "African-American", "Caucasian")
    X_train_sen = np.c_[X_train, sensitive_train_binary]
    X_test_sen = np.c_[X_test, sensitive_test_binary]

    classifier = equalized_odds_classifier(X_train, y_train, X_test, y_test)
    classifier.train(sensitive_train_binary)
    train_acc = classifier.get_accuracy(X_train, y_train)
    test_acc = classifier.get_accuracy(X_test, y_test)
    print("Training accuracy of classifier: ", train_acc)
    print("Test accuracy of classifier: ", test_acc)
    
    _, _ = classifier.get_proportions(sensitive_features_train, X_train)
    #classifier.get_test_flips(X_test, sensitive_test_binary)

if __name__ == "__main__":
    main()
