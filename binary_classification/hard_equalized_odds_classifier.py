import numpy as np
import cvxpy as cvx
from base_classifier import *
from tempeh.configurations import datasets
from erm_classifier import *
from fairlearn.postprocessing import ThresholdOptimizer
from prepare_data import *
from utils import *

NUM_SAMPLES = 100

class hard_equalized_odds_classifier(base_binary_classifier):
    def fit(self, train_X, train_Y, _classifier_name="logistic", _predictor="hard"):
        # First, create the base classifier, and get its predictions
        self.base_erm_classifier = erm_classifier(self.train_X, self.train_Y)
        self.base_erm_classifier.fit(self.train_X, self.train_Y, classifier_name=_classifier_name)
        y_pred_train = self.base_erm_classifier.predict(train_X)
        y_true_train = train_Y
        group_train = self.sensitive_train
        
        assert np.array_equal(np.unique(y_true_train),np.array([0,1])), 'y_true_train has to contain -1 and 1 and only these'
        assert np.array_equal(np.unique(y_pred_train),np.array([0,1])), 'y_pred_train has to contain -1 and 1 and only these'
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
            n2p1 >= 0,
            tpr0 == tpr1,
            fpr0 == fpr1
        ]
        
        prob = cvx.Problem(cvx.Minimize(error), constraints)
        try:
            prob.solve()
        except:
            print("You done goofed up")

        self.p2p0, self.n2p0, self.p2p1, self.n2p1 = max(0, p2p0.value[0]), max(0, n2p0.value[0]), max(p2p1.value[0], 0), max(n2p1.value[0],0)
        self.n2n0, self.p2n0, self.n2n1, self.p2n1 = max(0, n2n0.value[0]), max(0, p2n0.value[0]), max(n2n1.value[0], 0), max(p2n1.value[0],0)
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
        
        print("The expected rates group specific rates are (TPR0, TRP1, FPR0, FPR1):", tpr0, tpr1, fpr0, fpr1)
        return 

    def predict(self, X_vals, group_test):
        # get the base predictions from the base classifier
        print("Running")
        assert(self.trained == True)
        y_pred_test = self.base_erm_classifier.predict(X_vals)
        eq_odd_pred_test=np.copy(y_pred_test)
       
        test_ind_y1_g0=np.logical_and(y_pred_test == 1, group_test == 0)
        to_flip=np.random.choice(np.array([0,1]),size=np.sum(test_ind_y1_g0),p=np.array([self.p2p0,1-self.p2p0]))
        print(to_flip.mean())
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







