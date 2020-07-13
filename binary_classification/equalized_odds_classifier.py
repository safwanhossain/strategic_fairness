from base_classifier import *
from tempeh.configurations import datasets
from erm_classifier import *
from fairlearn.postprocessing import ThresholdOptimizer
from prepare_data import *
from utils import *

NUM_SAMPLES = 100

class equalized_odds_classifier(base_binary_classifier):
    def train(self):
        my_erm_classifier = erm_classifier(self.train_X, self.train_Y)
        my_erm_classifier.fit(self.train_X, self.train_Y, classifier_name="logistic")
        self.model = ThresholdOptimizer(estimator=my_erm_classifier, \
                constraints="equalized_odds", prefit=True)
        self.model.fit(self.train_X, self.train_Y, sensitive_features=self.sensitive_train) 

    def predict(self, x_samples, sensitive_features):
        y_samples = self.model.predict(x_samples, sensitive_features=sensitive_features)
        return y_samples
    
    def get_accuracy(self, X, y_true, sensitive_features):
        y_pred = self.predict(X, sensitive_features)
        return 1 - np.sum(np.power(y_pred - y_true, 2))/len(y_true) 

    def predict_proba(self, x_samples, sensitive_features):
        y_samples = self.model._pmf_predict(x_samples, sensitive_features=sensitive_features)
        return y_samples


def main():
    X_train, X_test, sensitive_features_train, sensitive_features_test, \
            y_train, y_test, sensitive_feature_names = get_data_compas()
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            sensitive_feature_names[1], sensitive_feature_names[0])

    # 0 should be disadvantage/black, 1 advantage/white
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}

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
    eo_classifier.get_test_flips(X_test, sensitive_test_binary, True)
    eo_classifier.get_group_confusion_matrix(sensitive_train_binary, X_train, y_train) 

if __name__ == "__main__":
    main()
