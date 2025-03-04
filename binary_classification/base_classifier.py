import numpy as np
import scipy as sp

class base_binary_classifier:
    def __init__(self, train_X, train_y, sensitive_train=None, sensitive_features_dict=None):
        self.train_X = train_X
        self.train_Y = train_y
        self.sensitive_train = sensitive_train

        self.sensitive_features_dict = sensitive_features_dict
        self.train_size = len(self.train_X)
        self.trained = False
        self.groups = None

    def fit(self):
        # train the classifer on the training data according to whatever
        # fairness metric you have in mind        
        # return the training loss
        print("Not Implemented!")
        pass

    def predict(self, samples, sensitive_features=None, sample=None):
        # predict the outcome of the model on the given samples. If samples
        # is none, then the self.test_data will be used
        # return the loss as well as the predictions
        print("Not Implemented!")
        pass

    def predict_proba(self, samples, sensitive_features=None):
        # predict the outcome of the model on the given samples. If samples
        # is none, then the self.test_data will be used
        # return the loss as well as the predictions
        print("Not Implemented!")
        pass

    def get_accuracy(self, X, y_true, sensitive_features=None):
        if self.trained == False:
            print("Train the model first!")
            return
        y_pred = self.predict(X, sensitive_features)
        return 1 - np.sum(np.power(y_pred - y_true, 2))/len(y_true) 

    def get_test_flips_expectation(self, test_X, sensitive_features_bin, percent=False, to_print=False):
        """ Given a bunch of test samples and sensitive features, return the proportion of 
        individuals who would benefit from flipping their attributes. This is meant to be used
        for a soft/probabilistic classifier 
        """
        print("WARNING: Only use this if the underlying classifer is a soft equalized odds classifier")
        groups = np.unique(sensitive_features_bin)
        assert(len(groups) == 2)
        indicies = {}
        for index, group in enumerate(groups):
            indicies[group] = np.where(sensitive_features_bin==group)[0]

        test_Y = self.predict(test_X, sensitive_features_bin)
        gain_dict, loss_dict = {}, {}
        
        # First lets look at strategic behaviour for group 0
        for group in range(len(groups)):
            # notice that group will be a number in [0,1]
            index_g = np.where(sensitive_features_bin == group)[0]
            honest_results = self.predict(test_X[index_g], np.ones(len(index_g))*group)
            strategic_results = self.predict(test_X[index_g], np.ones(len(index_g))*(1-group))
            # those who before were negative but now positive have gained
            num_gained = np.sum(np.logical_and(honest_results == 0, strategic_results == 1))
            # those who before were positive but now negative have lost
            num_lost = np.sum(np.logical_and(honest_results == 1, strategic_results == 0))
            if percent:
                gain_dict[self.sensitive_features_dict[group]] = num_gained/len(index_g.flatten())
                loss_dict[self.sensitive_features_dict[group]] = num_lost/len(index_g.flatten())
            else:
                gain_dict[self.sensitive_features_dict[group]]= num_gained
                loss_dict[self.sensitive_features_dict[group]] = num_lost
            
            if to_print and percent == False:
                print("For group ", self.sensitive_features_dict[group], ": ", num_gained, " gained and ", num_lost, " lost")
            if to_print and percent == True:
                print("For groups ", self.sensitive_features_dict[group], ": ", 100*(num_gained/len(index_g)), " gained and ", 100*(num_lost/len(index_g)), " lost")

        return gain_dict, loss_dict

    def get_test_flips(self, test_X, sensitive_features_bin, percent=False, to_print=False):
        # This is for strategic behaviour in fairness.
        # For each sample in the test set, obtain the percentage of people who would
        # benefit by flipping and those that would not.
        print("================ Please use get_test_flips_expectation(). \
                This should only be for debugging ===============")   
        groups = np.unique(sensitive_features_bin)
        indicies = {}
        for index, group in enumerate(groups):
            indicies[group] = np.where(sensitive_features_bin==group)[0]

        test_Y = self.predict(test_X, sensitive_features_bin, sample=True)
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
        gain_dict, loss_dict = {}, {}

        for group in groups:
            gain_dict[self.sensitive_features_dict[group]] = 0
            loss_dict[self.sensitive_features_dict[group]] = 0

        for x, y, s in zip(test_X, test_Y, sensitive_features_bin):
            curr_group = s
            for group in groups:
                if group == curr_group:
                    continue
                curr_group, group = int(curr_group), int(group)
                new_s = np.array(group).reshape(1,-1)
                tup = (self.sensitive_features_dict[curr_group], self.sensitive_features_dict[group])
                new_pred = self.predict(x.reshape(1,-1), new_s, sample=True)[0]
                
                div, mult = 1, 1
                if percent == True:
                    div, mult = len(indicies[curr_group]), 100
                # gaining by flipping
                if y == 0 and new_pred == 1:
                    gain_flips[tup] += (1/div)*mult
                    gain_dict[self.sensitive_features_dict[curr_group]] += (1/div)*mult

                # lose by flipping
                if y == 1 and new_pred == 0:
                    loss_flips[tup] += (1/div)*mult
                    loss_dict[self.sensitive_features_dict[curr_group]] += (1/div)*mult
      
        if to_print:
            print("Those gaining by flipping attributes: ", gain_flips)
            print("Those losing by flipping attributes: ", loss_flips)
        return gain_dict, loss_dict

    def get_proportions(self, X, y, sensitive_features, to_print=True):
        # For a trained classifier, get the proportion of samples (in test and train) that 
        # receive positive classification based on groups (currently only works for binary)
        # sensitive_index is the index of the sensitive attribute (need not be binary)
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        positive_count = {}
        positive_pt = {}
        indicies = {}
        y_pred = self.predict(X, sensitive_features)
        print("Total samples", len(sensitive_features), \
                "total true positive", sum(y), \
                "total +ve pred", sum(y_pred))

        for index, group in enumerate(groups):
            indicies[group] = np.where(sensitive_features==group)[0]
            positive_count[group] = sum(y_pred[indicies[group]])
            positive_pt[group] = positive_count[group]/len(indicies[group])
            if self.sensitive_features_dict is not None:
                print("Group ", self.sensitive_features_dict[group], \
                        "num people: ", len(indicies[group]), \
                        " +ve classification rate:", \
                        positive_pt[group])
            else:
                print("Group ", group, " +ve classification rate:", positive_pt[group])
        
        print("\n")
        return positive_count, positive_pt
        
    def get_group_confusion_matrix(self, sensitive_features, X, true_Y, to_print=False):
        # For a trained classifier, get the true positive and true negative rates based on
        # group identity. Do based on groups (currently only works for binary)
        # sensitive_index is the index of the sensitive attribute.
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        tp_rate = {}
        fp_rate = {}
        tn_rate = {}
        fn_rate = {}
        
        y_pred = self.predict(X, sensitive_features)
        accuracy = 1 - np.sum(np.power(true_Y - y_pred, 2))/len(true_Y) 

        out_dict = {}   # The format is: {group:[tp, fp, tn, fn]}
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
        
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2*precision*recall/(precision+recall)
            accuracy = 1 - np.sum(np.power(true_class - pred_class, 2))/len(true_class) 
            out_dict[self.sensitive_features_dict[group]] = [tp, tn, fp, fn, accuracy, f1]
           
            if to_print:
                print(self.sensitive_features_dict[group], "confusion matrix (Positive is good outcome)")
                print("\t F1 score: ", f1)
                print("\t Group Accuracy: ", accuracy)
                print("\t True positive rate:", tp)
                print("\t True negative rate:", tn)
                print("\t False positive rate:", fp)
                print("\t False negative rate:", fn)
        
        return out_dict





