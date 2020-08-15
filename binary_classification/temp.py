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
    def eq_odds_optimal_mix_rates(self, train_X, train_Y, _classifier_name="logistic", _predictor="hard"):
        # First, create the base classifier, and get its predictions
        self.base_erm_classifier = erm_classifier(self.train_X, self.train_Y)
        self.base_erm_classifier.fit(self.train_X, self.train_Y, classifier_name=_classifier_name)
        y_true_train = train_Y
        y_pred_train = self.base_erm_classifier.predict(train_X)
        group_train = self.sensitive_train
        
        tp0=np.sum(np.logical_and(y_pred_train==1,np.logical_and(y_true_train == 1, group_train == 0))) / float(
            np.sum(np.logical_and(y_true_train == 1, group_train == 0)))
        tp1 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == 1, group_train == 1))) / float(
            np.sum(np.logical_and(y_true_train == 1, group_train == 1)))
        fp0 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == -1, group_train == 0))) / float(
            np.sum(np.logical_and(y_true_train == -1, group_train == 0)))
        fp1 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == -1, group_train == 1))) / float(
            np.sum(np.logical_and(y_true_train == -1, group_train == 1)))
        fn0 = 1 - tp0
        fn1 = 1 - tp1
        tn0 = 1 - fp0
        tn1 = 1 - fp1

        sp2p = cvx.Variable(1)
        sp2n = cvx.Variable(1)
        sn2p = cvx.Variable(1)
        sn2n = cvx.Variable(1)

        op2p = cvx.Variable(1)
        op2n = cvx.Variable(1)
        on2p = cvx.Variable(1)
        on2n = cvx.Variable(1)

        sfpr = fp0 * sp2p + tn0 * sn2p
        sfnr = fn0 * sn2n + tp0 * sp2n
        ofpr = fp1 * op2p + tn1 * on2p
        ofnr = fn1 * on2n + tp1 * op2n
        error = sfpr + sfnr + ofpr + ofnr

        pred0 = y_pred_train[np.argwhere(group_train == 0)]
        true0 = train_Y[np.argwhere(group_train == 0)]
        pred1 = y_pred_train[np.argwhere(group_train == 1)]
        true1 = train_Y[np.argwhere(group_train == 1)]
        br0 = np.mean(true0)
        br1 = np.mean(true1)

        sflip = 1 - pred0
        sconst = pred0
        oflip = 1 - pred1
        oconst = pred1

        # group specific ground truth rates - probability that a person from group 1 is truly +ve
        sm_tn = np.logical_and(pred0 == 0, true0 == 0)
        sm_fn = np.logical_and(pred0 == 0, true0 == 1)
        sm_tp = np.logical_and(pred0 == 1, true0 == 1)
        sm_fp = np.logical_and(pred0 == 1, true0 == 0)
        
        om_tn = np.logical_and(pred1 == 0, true1 == 0)
        om_fn = np.logical_and(pred1 == 0, true1 == 1)
        om_tp = np.logical_and(pred1 == 1, true1 == 1)
        om_fp = np.logical_and(pred1 == 1, true1 == 0)
          
        spn_given_p = (sn2p * (sflip * sm_fn).mean() + sn2n * (sconst * sm_fn).mean()) / br0 + \
                      (sp2p * (sconst * sm_tp).mean() + sp2n * (sflip * sm_tp).mean()) / br0

        spp_given_n = (sp2n * (sflip * sm_fp).mean() + sp2p * (sconst * sm_fp).mean()) / (1 - br0) + \
                      (sn2p * (sflip * sm_tn).mean() + sn2n * (sconst * sm_tn).mean()) / (1 - br0)

        opn_given_p = (on2p * (oflip * om_fn).mean() + on2n * (oconst * om_fn).mean()) / br1 + \
                      (op2p * (oconst * om_tp).mean() + op2n * (oflip * om_tp).mean()) / br1

        opp_given_n = (op2n * (oflip * om_fp).mean() + op2p * (oconst * om_fp).mean()) / (1 - br1) + \
                      (on2p * (oflip * om_tn).mean() + on2n * (oconst * om_tn).mean()) / (1 - br1)

        constraints = [
            sp2p == 1 - sp2n,
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p,
        ]

        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        res = np.array([sp2p.value, sn2p.value, op2p.value, on2p.value])
        print(res)
        return res
    
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

        pred_g0 = y_pred_train[np.argwhere(group_train == 0)]
        true_g0 = train_Y[np.argwhere(group_train == 0)]
        pred_g1 = y_pred_train[np.argwhere(group_train == 1)]
        true_g1 = train_Y[np.argwhere(group_train == 1)]
        gt0 = np.mean(true_g0)
        gt1 = np.mean(true_g1)

        sflip = 1 - pred_g0
        sconst = pred_g0
        oflip = 1 - pred_g1
        oconst = pred_g1

        # group specific ground truth rates - probability that a person from group 1 is truly +ve
        ground_tn_0 = np.logical_and(pred_g0 == 0, true_g0 == 0)
        ground_fn_0 = np.logical_and(pred_g0 == 0, true_g0 == 1)
        ground_tp_0 = np.logical_and(pred_g0 == 1, true_g0 == 1)
        ground_fp_0 = np.logical_and(pred_g0 == 1, true_g0 == 0)
        
        ground_tn_1 = np.logical_and(pred_g1 == 0, true_g1 == 0)
        ground_fn_1 = np.logical_and(pred_g1 == 0, true_g1 == 1)
        ground_tp_1 = np.logical_and(pred_g1 == 1, true_g1 == 1)
        ground_fp_1 = np.logical_and(pred_g1 == 1, true_g1 == 0)
        
        spn_given_p = (n2p0 * (sflip * fn0).mean() + n2n0 * (sconst * ground_fn_0).mean()) / gt0 + \
                      (p2p0 * (sconst * tp0).mean() + p2n0 * (sflip * ground_tp_0).mean()) / gt0

        spp_given_n = (p2n0 * (sflip * fp0).mean() + p2p0 * (sconst * ground_fp_0).mean()) / (1 - gt0) + \
                      (n2p0 * (sflip * tn0).mean() + n2n0 * (sconst * ground_tn_0).mean()) / (1 - gt0)
        
        opn_given_p = (n2p1 * (oflip * fn1).mean() + n2n1 * (oconst * ground_fn_1).mean()) / gt1 + \
                      (p2p1 * (oconst * tp1).mean() + p2n1 * (oflip * ground_tp_1).mean()) / gt1

        opp_given_n = (p2n1 * (oflip * fp1).mean() + p2p1 * (oconst * ground_fp_1).mean()) / (1 - gt1) + \
                      (n2p1 * (oflip * tn1).mean() + n2n1 * (oconst * ground_tn_1).mean()) / (1 - gt1)
        
        
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
        prob.solve(verbose=True)

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
        
        print("The expected rates group specific rates are (TPR0, TRP1, FPR0, FPR1):" tpr0, tpr1, fpr0, fpr1)
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








