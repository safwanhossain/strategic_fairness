from base_classifier import *
from tempeh.configurations import datasets
from erm_classifier import *
from fairlearn.postprocessing import ThresholdOptimizer
from prepare_data import *
from utils import *
from soft_equalized_odds_classifier import *
import csv, itertools
from sweep_soft_eq import main as sweep_main

# We will randomly drop features here and then compute the impact it has on the 
# underlying unfairness and the the propensity to misreport and behave strategically

MAX_NUM_DROPS = 4

def generate_combinations(curr_list):
    all_combinations = []
    for r in range(len(curr_list)):
        combinations_object = itertools.combinations(curr_list, r)

def sweep_drop_features_pd(X, y, sensitive, sensitive_feature_names, filename, rebalance=None):
    feature_list = [i for i in range(X.shape[1])]
    print("There are a total of ", len(feature_list), " features")
    csv_file = open(filename, mode='w+')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    for num_drop in range(MAX_NUM_DROPS):
        to_drop_features = list(itertools.combinations(feature_list, num_drop))
        to_drop_features = [[5,18]]
        for to_drop in to_drop_features:
            if len(to_drop) > 1:
                to_drop = list(to_drop)
                new_X = X.drop(X.columns[to_drop], axis=1)
            else:
                new_X = X
            
            new_X = oneHotCatVars(new_X, new_X.select_dtypes('object').columns).to_numpy()
            sensitive, y = sensitive.reshape(-1,1), y.reshape(-1,1)
            new_X_train, sensitive_train, y_train, \
                    new_X_test, sensitive_test, y_test = get_train_test_split(new_X, y, sensitive)

            if rebalance != None:
                new_X_train, y_train, sensitive_train, new_X_test, y_test, sensitive_test = rebalance_data(new_X_train, y_train, sensitive_train, \
                    new_X_test, y_test, sensitive_test, "labels")
                new_X_train, y_train, sensitive_train, new_X_test, y_test, sensitive_test = rebalance_data(new_X_train, y_train, sensitive_train, \
                    new_X_test, y_test, sensitive_test, "groups")
            
            y_train, y_test = y_train.astype("int"), y_test.astype("int")
            #unfairness, gain_dict, loss_dict = train_dropped(new_X_train, y_train, sensitive_train, new_X_test, \
            #        y_test, sensitive_test, sensitive_feature_names)
            sweep_main(new_X_train, y_train, sensitive_train, new_X_test, y_test, sensitive_test, sensitive_feature_names) 


            if gain_dict is None:
                print(to_drop, " Was not successful")
                continue

            gain_keys, loss_keys = gain_dict.keys(), loss_dict.keys()
            row = [str(to_drop)] + [str(unfairness[0])] + [str(unfairness[1])] + [str(gain_dict)] + [str(loss_dict)]
            csv_writer.writerow(row)
            print("Done: ", to_drop)

def sweep_drop_feature(X_train, y_train, sensitive_train, X_test, y_test, \
        sensitive_test, sensitive_feature_names, filename):
    
    feature_list = [i for i in range(X_train.shape[1])]

    csv_file = open(filename, mode='w+')
    csv_writer = csv.writer(csv_file, delimiter=',')

    for num_drop in range(MAX_NUM_DROPS):
        to_drop_features = list(itertools.combinations(feature_list, num_drop))
        for to_drop in to_drop_features:
            new_X_train = np.delete(X_train, to_drop, axis=1)
            new_X_test = np.delete(X_test, to_drop, axis=1)
            unfairness, gain_dict, loss_dict = train_dropped(new_X_train, y_train, sensitive_train, new_X_test, \
                    y_test, sensitive_test, sensitive_feature_names)
            
            if gain_dict is None:
                print(to_drop, " Was not successful")
                continue

            gain_keys, loss_keys = gain_dict.keys(), loss_dict.keys()
            row = [str(to_drop)] + [str(unfairness[0])] + [str(unfairness[1])] + [str(gain_dict)] + [str(loss_dict)]
            csv_writer.writerow(row)
            print("Done: ", to_drop)

def train_dropped(X_train, y_train, sensitive_train, X_test, y_test, \
        sensitive_test, sensitive_feature_names):
   
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}
    sensitive_train_binary = convert_to_binary(sensitive_train, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    sensitive_test_binary = convert_to_binary(sensitive_test, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    
    # First train an ERM classifier and compute the unfairness (TP delta and FP delta)
    erm = erm_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    y_train = y_train.astype('int')
    erm.fit(X_train, y_train)
    try:
        out_dict = erm.get_group_confusion_matrix(sensitive_train_binary, X_train, y_train)
    except:
        return (None, None), None, None

    tp_delta = abs(out_dict[max(out_dict.keys())][0] - out_dict[min(out_dict.keys())][0])
    fp_delta = abs(out_dict[max(out_dict.keys())][1] - out_dict[min(out_dict.keys())][1])

    # Now create a fair classifier and compute the liklihood of misreporting
    eo_classifier = soft_equalized_odds_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    success = eo_classifier.fit(X_train, y_train, _lambda=1)
    if success is False:
        return (None, None), None, None
        
    gain_dict, loss_dict = eo_classifier.get_test_flips_expectation(X_test, sensitive_test_binary, percent=True)
    return (tp_delta, fp_delta), gain_dict, loss_dict

def main():
    sensitive = "race"
    """
    X, sensitive, y, sensitive_features_names = get_data_income(sensitive, raw=True)
    filename = "income_race_no_change.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename)
    
    X, sensitive, y, sensitive_features_names = get_data_income(sensitive, raw=True)
    filename = "income_race_rebalance_both.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename, rebalance="both")

    sensitive = "sex"
    X, sensitive, y, sensitive_features_names = get_data_income(sensitive, raw=True)
    filename = "income_sex_no_change.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename)
    
    X, sensitive, y, sensitive_features_names = get_data_income(sensitive, raw=True)
    filename = "income_sex_rebalance_both.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename, rebalance="both")

    sensitive = "Pstatus"
    X, sensitive, y, sensitive_features_names = get_data_student(sensitive, raw=True)
    filename = "student_math_Pstatus.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename)
    """

    sensitive = "famsup"
    X, sensitive, y, sensitive_features_names = get_data_student(sensitive, raw=True)
    filename = "student_math_famsup.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename)
    """
    sensitive = "Pstatus"
    X, sensitive, y, sensitive_features_names = get_data_student(sensitive, subject="por", raw=True)
    filename = "student_por_Pstatus.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename)
    
    sensitive = "famsup"
    X, sensitive, y, sensitive_features_names = get_data_student(sensitive, subject="por", raw=True)
    filename = "student_math_famsup.csv"
    sweep_drop_features_pd(X, y, sensitive, sensitive_features_names, filename)
    """

if __name__ == "__main__":
    main()


