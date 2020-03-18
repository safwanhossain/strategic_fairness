import numpy as np
import scipy as sp

class base_binary_classifier:
    def __init__(self, train_X, train_y, sensitive_features_dict=None):
        self.train_X = train_X
        self.train_Y = train_y

        self.sensitive_features_dict = sensitive_features_dict
        self.train_size = len(self.train_X)
        self.trained = False
        self.groups = None

    def train(self):
        # train the classifer on the training data according to whatever
        # fairness metric you have in mind        
        # return the training loss
        print("Not Implemented!")
        pass

    def predict(self, samples=None):
        # predict the outcome of the model on the given samples. If samples
        # is none, then the self.test_data will be used
        # return the loss as well as the predictions
        print("Not Implemented!")
        pass

    def get_accuracy(self, X, y_true):
        if self.trained == False:
            print("Train the model first!")
            return
        y_pred = self.predict(X)
        return 1 - np.sum(np.power(y_pred - y_true, 2))/len(y_true) 

    def get_test_flips(self, test_X, sensitive_features_bin, percent=False):
        # This is for strategic behaviour in fairness.
        # For each sample in the test set, obtain the percentage of people who would
        # benefit by flipping and those that would not.
        
        groups = np.unique(sensitive_features_bin)
        indicies = {}
        for index, group in enumerate(groups):
            indicies[group] = np.where(sensitive_features_bin==group)[0]

        test_Y = self.predict(test_X)
        keys = []
        for i in groups:
            for j in groups:
                i, j = int(i), int(j)
                if i == j:
                    continue
                if self.sensitive_features_dict == None:
                    keys.append((i,j))
                else:
                    keys.append((self.sensitive_features_dict[i],self.sensitive_features_dict[j]))

        gain_flips = {z:0 for z in keys}
        loss_flips = {z:0 for z in keys}
        gain_probs = {z:[] for z in keys}
        loss_probs = {z:[] for z in keys}
        for x, y in zip(test_X, test_Y):
            curr_group = x[-1]
            for group in groups:
                if group == curr_group:
                    continue
                pred_prob = self.predict_prob(x.reshape(1,-1))[0]
                curr_group, group = int(curr_group), int(group)
                new_x = np.append(x[:-1], group)
                tup = (self.sensitive_features_dict[curr_group], self.sensitive_features_dict[group])
                new_pred = self.predict(new_x.reshape(1,-1))[0]
                new_pred_prob = self.predict_prob(new_x.reshape(1,-1))[0]
                
                div = 1
                if percent == True:
                    div = len(indicies[curr_group]) 
                # gaining by flipping
                if y == 1 and self.predict(new_x.reshape(1,-1))[0] == 0:
                    gain_probs[tup].append([pred_prob, new_pred_prob])
                    gain_flips[tup] += 1/div
                if y == 0 and self.predict(new_x.reshape(1,-1))[0] == 1:
                    loss_probs[tup].append([pred_prob, new_pred_prob])
                    loss_flips[tup] += 1/div
       
        print("Those gaining by flipping attributes: ", gain_flips)
        print("Those losing by flipping attributes: ", loss_flips)
        return gain_flips, gain_probs, loss_flips, loss_probs

    def get_proportions(self, sensitive_features, X):
        # For a trained classifier, get the proportion of samples (in test and train) that 
        # receive positive classification based on groups (currently only works for binary)
        # sensitive_index is the index of the sensitive attribute (need not be binary)
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        positive_count = {}
        positive_pt = {}
        indicies = {}
        y_pred = self.predict(X)

        for index, group in enumerate(groups):
            indicies[group] = np.where(sensitive_features==group)[0]
            positive_count[group] = sum(y_pred[indicies[group]])
            positive_pt[group] = positive_count[group]/len(indicies[group])
            print("Num people", len(indicies[group]))
            print("Group ", group, " recidivism rate:", positive_pt[group])

        return positive_count, positive_pt
        
    def get_group_confusion_matrix(self, sensitive_features, X, true_Y):
        # For a trained classifier, get the true positive and true negative rates based on
        # group identity. Dobased on groups (currently only works for binary)
        # sensitive_index is the index of the sensitive attribute.
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        tp_rate = {}
        fp_rate = {}
        tn_rate = {}
        fn_rate = {}
        y_pred = self.predict(X)

        for index, group in enumerate(groups):
            indicies = np.where(sensitive_features==group)[0]
            true_class = true_Y[indicies]
            pred_class = y_pred[indicies]
            
            true_pos_index = np.where(true_class==1)[0]
            true_neg_index = np.where(true_class==0)[0]
            tp = len(np.where(pred_class[true_pos_index]==1)[0])/len(true_pos_index)
            tn = len(np.where(pred_class[true_neg_index]==0)[0])/len(true_neg_index)
            fp = len(np.where(pred_class[true_neg_index]==1)[0])/len(true_neg_index)
            fn = len(np.where(pred_class[true_pos_index]==0)[0])/len(true_pos_index)
            tp_rate[group] = tp 
            tn_rate[group] = tn
            fp_rate[group] = fp 
            fn_rate[group] = fn 
        
            accuracy = 1 - np.sum(np.power(true_class - pred_class, 2))/len(true_class) 
            print(self.sensitive_features_dict[group], "confusion matrix (Positive is re-offending)")
            print("\t Group Accuracy: ", accuracy)
            print("\t True positive rate:", tp)
            print("\t True negative rate:", tn)
            print("\t False positive rate:", fp)
            print("\t False negative rate:", fn)
        
        return tp_rate, fp_rate, tn_rate, fn_rate












