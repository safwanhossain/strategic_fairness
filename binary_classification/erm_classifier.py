from sklearn.linear_model import LogisticRegression
from base_classifier import *

class erm_classifier(base_binary_classifier):
    def train(self):
        estimator = LogisticRegression()
        estimator.fit(self.train_X, self.train_Y)
        self.model = estimator
        self.trained = True

    def predict(self, x_samples):
        if self.trained == False:
            print("Train the model first!")
            return
        y_samples = self.model.predict(x_samples)
        return y_samples
    
    def predict_prob(self, x_samples):
        if self.trained == False:
            print("Train the model first!")
            return
        y_samples = self.model.predict_proba(x_samples)[0]
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
    from tempeh.configurations import datasets
    compas_dataset = datasets['compas']()
    X_train, X_test = compas_dataset.get_X()
    y_train, y_test = compas_dataset.get_y()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('race')
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            "African-American", "Caucasian")
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            "African-American", "Caucasian")
    sensitive_features_dict = {0:"African-American", 1:"Caucasian"}
    #sensitive_train_binary = np.random.randint(0,2, len(y_train))
    #sensitive_test_binary = np.random.randint(0,1, len(y_test))
    #sensitive_test_binary[0] = 1
    #sensitive_train_binary = np.zeros(len(y_train))
    #sensitive_test_binary = np.zeros(len(y_test))
    X_train = np.c_[X_train, sensitive_train_binary]
    X_test = np.c_[X_test, sensitive_test_binary]

    classifier = erm_classifier(X_train, y_train, sensitive_features_dict)
    classifier.train()
    #train_acc = classifier.get_accuracy(X_train, y_train)
    #test_acc = classifier.get_accuracy(X_test, y_test)
    #print(train_acc)
    #print(test_acc)
    
    #_, _ = classifier.get_proportions(sensitive_features_train, X_train)
    classifier.get_test_flips(X_test, sensitive_test_binary, False)
    classifier.get_group_confusion_matrix(sensitive_train_binary, X_train, y_train) 
    print("\n")
    classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 

if __name__ == "__main__":
    main()
