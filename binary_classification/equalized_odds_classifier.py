from sklearn.linear_model import LogisticRegression
from base_classifier import *
from tempeh.configurations import datasets
from erm_classifier import *
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

NUM_SAMPLES = 100

class erm_classifier_prob(BaseEstimator, ClassifierMixin, base_binary_classifier):
    def fit(self, X, y):
        estimator = LogisticRegression()
        estimator.fit(X, y)
        self.model = estimator
        self.trained = True
        return self

    def predict(self, x_samples):
        y_samples = self.model.predict_proba(x_samples)[:,1]
        return y_samples

class equalized_odds_classifier(base_binary_classifier):
    def train(self):
        erm_classifier = erm_classifier_prob(self.train_X, self.train_Y, self.sensitive_train)
        erm_classifier.fit(self.train_X, self.train_Y)
        self.model = ThresholdOptimizer(estimator=erm_classifier, constraints="equalized_odds")
        self.model.fit(self.train_X, self.train_Y, sensitive_features=self.sensitive_train) 

    def predict(self, x_samples, sensitive_features):
        y_samples = self.model.predict(x_samples, sensitive_features=sensitive_features)
        return y_samples
    
    def get_accuracy(self, X, y_true, sensitive_features):
        y_pred = self.predict(X, sensitive_features)
        return 1 - np.sum(np.power(y_pred - y_true, 2))/len(y_true) 

    def predict_prob(self, x_samples, sensitive_features):
        y_samples = self.model._pmf_predict(x_samples, sensitive_features=sensitive_features)
        return y_samples

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
    sensitive_features_dict = {0:"African-American", 1:"Caucasian"}

    eo_classifier = equalized_odds_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    eo_classifier.train()
    
    total_train, total_test = 0, 0
    for i in range(NUM_SAMPLES):
        total_train += eo_classifier.get_accuracy(X_train, y_train, sensitive_train_binary)
        total_test += eo_classifier.get_accuracy(X_test, y_test, sensitive_test_binary)
    print("Train Acc:", total_train/NUM_SAMPLES) 
    print("Test Acc:", total_test/NUM_SAMPLES) 
    
    _, _ = eo_classifier.get_proportions(X_train, sensitive_train_binary)
    eo_classifier.get_test_flips(X_test, sensitive_test_binary, False)
    eo_classifier.get_group_confusion_matrix(sensitive_train_binary, X_train, y_train) 
    print("\n")
    eo_classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 

if __name__ == "__main__":
    main()
