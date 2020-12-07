from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from base_classifier import *
from utils import *
from prepare_data import *
from sklearn.metrics import precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin

NUM_SAMPLES = 100

class erm_classifier(base_binary_classifier, ClassifierMixin, BaseEstimator):
    def fit(self, classifier_name="logistic"):
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

    def predict(self, x_samples, sensitive_features=None, sample=None):
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


def main(to_run_fcn, sens, unaware=False):
    X_train, X_test, sensitive_features_train, sensitive_features_test, \
            y_train, y_test, sensitive_feature_names = to_run_fcn(sens)
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    
    # 0 should be black, 1 white
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}

    # uncomment this for fariness throught unawareness
    if unaware == True:
        sensitive_train_binary, sensitive_test_binary = np.zeros(len(y_train)), np.zeros(len(y_test))
    
    classifier = erm_classifier(X_train, y_train, sensitive_train_binary, sensitive_features_dict)
    classifier.fit(classifier_name="logistic")
    total_train, total_test = 0, 0
    for i in range(NUM_SAMPLES):
        total_train += classifier.get_accuracy(X_train, y_train, sensitive_train_binary)
        total_test += classifier.get_accuracy(X_test, y_test, sensitive_test_binary)
    print("================= Getting Accuracy ==================")
    print("Train Acc:", total_train/NUM_SAMPLES) 
    print("Test Acc:", total_test/NUM_SAMPLES) 
    print("\n")
    
    print("================ Getting positive classification rate by group ================")
    classifier.get_proportions(X_train, y_train, sensitive_train_binary)
    classifier.get_proportions(X_test, y_test, sensitive_test_binary)
    print("\n")
    
    print("================ Getting strategic results =================")
    classifier.get_test_flips_expectation(X_test, sensitive_test_binary, percent=True, to_print=True)
    print("\n")

    print("================ Getting Confusion Matrix ==================")
    classifier.get_group_confusion_matrix(sensitive_train_binary, X_train, y_train, to_print=True) 
    print("\n")


if __name__ == "__main__":
    import sys
    to_run = sys.argv[1]
    sens = sys.argv[2]
    
    unaware = False
    if len(sys.argv) == 4:
        unaware = sys.argv[3]
        if unaware == "true":
            unaware = True

    run_dict = {
        'compas': get_data_compas,
        'lawschool': get_data_lawschool,
        'income': get_data_income,
        'student': get_data_student
    }

    main(run_dict[to_run], sens, unaware)







