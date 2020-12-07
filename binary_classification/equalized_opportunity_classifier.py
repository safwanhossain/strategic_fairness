import cvxpy as cvx
from base_classifier import *
from tempeh.configurations import datasets
from erm_classifier import *
from prepare_data import *
from utils import *

NUM_SAMPLES = 100

class equalized_opportunity_classifier(base_binary_classifier):
    def fit(self, _classifier_name="logistic",
            _fairness="hard", _lambda=0.5,
            verbose=False, use_group_in_base_classifier=True):

        # First, create the base classifier, and get its predictions
        self.use_group_in_base_classifier = use_group_in_base_classifier
        if self.use_group_in_base_classifier:
            self.base_erm_classifier = erm_classifier(self.train_X, self.train_Y, self.sensitive_train)
            self.base_erm_classifier.fit(classifier_name=_classifier_name)
            y_pred_train = self.base_erm_classifier.predict(self.train_X, self.sensitive_train)
        y_true_train = self.train_Y
        group_train = self.sensitive_train
        
        assert np.array_equal(np.unique(y_true_train),np.array([0,1])), 'y_true_train has to contain -1 and 1 and only these'
        assert np.array_equal(np.unique(y_pred_train),np.array([0,1])), 'y_pred_train has to contain -1 and 1 and only these'
        
        # If this is set True, _lambda is ignored and EQ is implmented hard
        self._fairness = _fairness        

        assert np.array_equal(np.unique(group_train),np.array([0,1])), 'group_train has to contain 0 and 1 and only these'

        tp0=np.sum(np.logical_and(y_pred_train==1,np.logical_and(y_true_train == 1, group_train == 0))) / float(
            np.sum(np.logical_and(y_true_train == 1, group_train == 0)))
        tp1 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == 1, group_train == 1))) / float(
            np.sum(np.logical_and(y_true_train == 1, group_train == 1)))
        fp0 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == 0, group_train == 0))) / float(
            np.sum(np.logical_and(y_true_train == 0, group_train == 0)))
        fp1 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == 0, group_train == 1))) / float(
            np.sum(np.logical_and(y_true_train == 0, group_train == 1)))
        fn0 = 1 - tp0
        fn1 = 1 - tp1
        tn0 = 1 - fp0
        tn1 = 1 - fp1
    
        p2p0 = cvx.Variable(1)
        p2n0 = cvx.Variable(1)
        n2p0 = cvx.Variable(1)
        n2n0 = cvx.Variable(1)
        p2p1 = cvx.Variable(1)
        p2n1 = cvx.Variable(1)
        n2p1 = cvx.Variable(1)
        n2n1 = cvx.Variable(1)
        
        fpr0 = fp0 * p2p0 + tn0 * n2p0
        fnr0 = fn0 * n2n0 + tp0 * p2n0
        fpr1 = fp1 * p2p1 + tn1 * n2p1
        fnr1 = fn1 * n2n1 + tp1 * p2n1
        tpr0 = 1 - fnr0
        tpr1 = 1 - fnr1
        tnr0 = 1 - fpr0
        tnr1 = 1 - fpr1
        
        error = fpr0 + fnr0 + fpr1 + fnr1
        constraints = [
            p2p0 == 1 - p2n0,
            n2p0 == 1 - n2n0,
            p2p1 == 1 - p2n1,
            n2p1 == 1 - n2n1,
            p2p0 <= 1,
            p2p0 >= 0,
            n2p0 <= 1,
            n2p0 >= 0,
            p2p1 <= 1,
            p2p1 >= 0,
            n2p1 <= 1,
            n2p1 >= 0
        ]
        if self._fairness == "hard":
            constraints.append(tpr0 == tpr1)
            prob = cvx.Problem(cvx.Minimize(error), constraints)
        else:
            penalty = cvx.abs(tpr0 - tpr1) 
            prob = cvx.Problem(cvx.Minimize(error + _lambda*penalty), constraints)
        
        try:
            prob.solve()
        except:
            print("You done goofed up")
            return False

        self.p2p0, self.n2p0, self.p2p1, self.n2p1 = min(max(0, p2p0.value[0]), 1), min(max(0, n2p0.value[0]), 1), min(max(p2p1.value[0], 0), 1), min(max(n2p1.value[0],0), 1)
        self.n2n0, self.p2n0, self.n2n1, self.p2n1 = min(max(0, n2n0.value[0]), 1), min(max(0, p2n0.value[0]), 1), min(max(n2n1.value[0], 0), 1), min(max(p2n1.value[0],0), 1)
        if verbose:
            print(self.p2p0, self.n2p0, self.p2p1, self.n2p1)
        self.trained = True

        fpr0 = fp0 * self.p2p0 + tn0 * self.n2p0
        fnr0 = fn0 * self.n2n0 + tp0 * self.p2n0
        fpr1 = fp1 * self.p2p1 + tn1 * self.n2p1
        fnr1 = fn1 * self.n2n1 + tp1 * self.p2n1
        tpr0 = 1 - fnr0
        tpr1 = 1 - fnr1
        tnr0 = 1 - fpr0
        tnr1 = 1 - fpr1
        
        if verbose:
            print("Lambda: ", _lambda, " The E[group specific rates] are (TPR0, TRP1, FPR0, FPR1):", tpr0, tpr1, fpr0, fpr1)
        return True

    def _predict_sample(self, X_vals, group_test):
        """ In this prediction, we sample a biased coin wp equal to the flipping rates and stochastically decide if we want to flip
        a candidate or not. NOTE: If you use this, individual predictions ARE meaningful
        """
        # get the base predictions from the base classifier
        assert(self.trained == True)
        y_pred_test = self.base_erm_classifier.predict(X_vals, group_test)
        eq_odd_pred_test=np.copy(y_pred_test)
       
        test_ind_y1_g0=np.logical_and(y_pred_test == 1, group_test == 0)
        to_flip=np.random.choice(np.array([0,1]),size=np.sum(test_ind_y1_g0),p=np.array([self.p2p0,1-self.p2p0]))
        eq_odd_pred_test[np.where(test_ind_y1_g0)[0][to_flip==1]]=0

        test_ind_y1_g1=np.logical_and(y_pred_test == 1, group_test == 1)
        to_flip=np.random.choice(np.array([0,1]),size=np.sum(test_ind_y1_g1),p=np.array([self.p2p1,1-self.p2p1]))
        eq_odd_pred_test[np.where(test_ind_y1_g1)[0][to_flip==1]]=0
        
        test_ind_y0_g0=np.logical_and(y_pred_test == 0, group_test == 0)
        to_flip=np.random.choice(np.array([0,1]),size=np.sum(test_ind_y0_g0),p=np.array([1-self.n2p0,self.n2p0]))
        eq_odd_pred_test[np.where(test_ind_y0_g0)[0][to_flip==1]]=1
        
        test_ind_y0_g1=np.logical_and(y_pred_test == 0, group_test == 1)
        to_flip=np.random.choice(np.array([0,1]),size=np.sum(test_ind_y0_g1),p=np.array([1-self.n2p1,self.n2p1]))
        eq_odd_pred_test[np.where(test_ind_y0_g1)[0][to_flip==1]]=1

        return eq_odd_pred_test

    def _predict_expectation(self, X_vals, group_test):
        """ In this prediction, we decide to flip the outcome of a candidate based on the expected flips for their group.
        NOTE: This function only works in aggregate and if using this, any individual prediction is totally meaningless.
        """
        # get the base predictions from the base classifier
        assert(self.trained == True)
        y_pred_test = self.base_erm_classifier.predict(X_vals, group_test)
        eq_odd_pred_test=np.copy(y_pred_test)

        test_ind_y1_g0=np.logical_and(y_pred_test == 1, group_test == 0)
        to_flip = np.zeros(np.sum(test_ind_y1_g0))
        to_flip[[i for i in range(int((1-self.p2p0)*np.sum(test_ind_y1_g0)))]] = 1
        np.random.shuffle(to_flip)
        eq_odd_pred_test[np.where(test_ind_y1_g0)[0][to_flip==1]]=0

        test_ind_y1_g1=np.logical_and(y_pred_test == 1, group_test == 1)
        to_flip = np.zeros(np.sum(test_ind_y1_g1))
        to_flip[[i for i in range(int((1-self.p2p1)*np.sum(test_ind_y1_g1)))]] = 1
        np.random.shuffle(to_flip)
        eq_odd_pred_test[np.where(test_ind_y1_g1)[0][to_flip==1]]=0

        test_ind_y0_g0=np.logical_and(y_pred_test == 0, group_test == 0)
        to_flip = np.zeros(np.sum(test_ind_y0_g0))
        to_flip[[i for i in range(int((self.n2p0)*np.sum(test_ind_y0_g0)))]] = 1
        np.random.shuffle(to_flip)
        eq_odd_pred_test[np.where(test_ind_y0_g0)[0][to_flip==1]]=1
        
        test_ind_y0_g1=np.logical_and(y_pred_test == 0, group_test == 1)
        to_flip = np.zeros(np.sum(test_ind_y0_g1))
        to_flip[[i for i in range(int((self.n2p1)*np.sum(test_ind_y0_g1)))]] = 1
        np.random.shuffle(to_flip)
        eq_odd_pred_test[np.where(test_ind_y0_g1)[0][to_flip==1]]=1

        return eq_odd_pred_test

    def predict(self, X_vals, group_test, sample=False):
        if sample:
            return self._predict_sample(X_vals, group_test)
        else:
            return self._predict_expectation(X_vals, group_test)


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
    
    classifier = equalized_opportunity_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    classifier.fit(_fairness="hard")
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





