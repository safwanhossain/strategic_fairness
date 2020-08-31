from base_classifier import *
from tempeh.configurations import datasets
from erm_classifier import *
from fairlearn.postprocessing import ThresholdOptimizer
from prepare_data import *
from utils import *
from soft_equalized_odds_classifier import *
import matplotlib.pyplot as plt

def main(X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test, sensitive):
    classifer = "logistic"
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}

    num_steps = 10
    lambda_vals = [0 + i*1.0/num_steps for i in range(num_steps + 1)]
    tp_vals_delta = [0 for i in range(num_steps + 1)]
    fp_vals_delta = [0 for i in range(num_steps + 1)]
    gain_0 = [0 for i in range(num_steps + 1)]
    gain_1 = [0 for i in range(num_steps + 1)]
    loss_0 = [0 for i in range(num_steps + 1)]
    loss_1 = [0 for i in range(num_steps + 1)]

    for i, lam in enumerate(lambda_vals):
        eo_classifier = soft_equalized_odds_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
        eo_classifier.fit(X_train, y_train, _lambda=lam)
        train_out = eo_classifier.get_group_confusion_matrix(sensitive_train_binary, X_train, y_train) 
        test_out = eo_classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 
        gain_dict, loss_dict = eo_classifier.get_test_flips_expectation(X_test, sensitive_test_binary, percent=True)
        print(gain_dict, loss_dict)
        tp_vals_delta[i] = np.abs(train_out[sensitive_features_dict[0]][0] - train_out[sensitive_features_dict[1]][0])
        fp_vals_delta[i] = np.abs(train_out[sensitive_features_dict[0]][1] - train_out[sensitive_features_dict[1]][1])
        gain_0[i] = gain_dict[sensitive_features_dict[0]]
        gain_1[i] = gain_dict[sensitive_features_dict[1]]
        loss_0[i] = loss_dict[sensitive_features_dict[0]]
        loss_1[i] = loss_dict[sensitive_features_dict[1]]

    plt.figure()
    plt.plot(lambda_vals, tp_vals_delta, color='r', linestyle='-', label="TP Delta")
    plt.plot(lambda_vals, fp_vals_delta, color='b', linestyle='-', label="FP Delta")
    plt.legend()
    plt.savefig("error_plots.png")
    
    plt.figure()
    delta_gain_0 = np.array(gain_0) - np.array(loss_0)
    delta_gain_1 = np.array(gain_1) - np.array(loss_1)
    plt.plot(lambda_vals, delta_gain_0, color='r', linestyle='-', label=sensitive_features_dict[0]+" Delta Gain")
    plt.plot(lambda_vals, delta_gain_1, color='b', linestyle='-', label=sensitive_features_dict[1]+" Delta Gain")
    plt.legend()
    plt.savefig("strategic_gain.png")
    
    #plt.figure()
    #plt.plot(lambda_vals, gain_1, color='r', linestyle='-', label=sensitive_features_dict[1]+"Gain")
    #plt.plot(lambda_vals, loss_1, color='b', linestyle='-', label=sensitive_features_dict[1]+"Loss")
    #plt.legend()
    #plt.savefig("group_1_sp.png")

if __name__ == "__main__":
    sensitive = "famsup"
    X_train, X_test, sensitive_features_train, sensitive_features_test, \
            y_train, y_test, sensitive_feature_names = get_data_student(sensitive)
    print(sensitive_features_train) 
    y_train, y_test = y_train.astype("int"), y_test.astype("int")
    print(len(y_test), sum(y_train))
    print(len(y_test), sum(y_test))
    main(X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test, sensitive)


