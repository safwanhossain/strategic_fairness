from plotting import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from base_classifier import *
from erm_classifier import *
from equalized_odds_classifier import *
from demographic_parity_classifier import *
from utils import *
from prepare_data import *
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin
from hard_equalized_odds_classifier import *

def main_compas(sensitive, _classifier, _predictor="hard"):
    X_train, X_test, sensitive_features_train, sensitive_features_test, \
            y_train, y_test, sensitive_feature_names = get_data_compas(sensitive)
    if sensitive == "race":
        sensitive_train_binary = convert_to_binary(sensitive_features_train, \
                sensitive_feature_names[1], sensitive_feature_names[0])
        sensitive_test_binary = convert_to_binary(sensitive_features_test, \
                sensitive_feature_names[1], sensitive_feature_names[0])
    else:
        sensitive_train_binary, sensitive_test_binary = sensitive_features_train, \
                sensitive_features_test
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}
    
    #=============================== ERM ================================================
    classifier = erm_classifier(X_train, y_train, sensitive_train_binary, sensitive_features_dict)
    classifier.fit(X_train, y_train, classifier_name=_classifier)
    classifier.get_proportions(X_train, y_train, sensitive_train_binary)
    erm_gain_dict, erm_loss_dict = classifier.get_test_flips(X_test, sensitive_test_binary, True)
    erm_confusion_mat = classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 
    
    #================================ Equalized Odds ====================================
    eo_classifier = hard_equalized_odds_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    eo_classifier.fit(X_train, y_train, _classifier_name=_classifier, _predictor=_predictor)
    eo_gain_dict, eo_loss_dict = eo_classifier.get_test_flips(X_test, sensitive_test_binary, True)
    eo_confusion_mat = eo_classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 

    #================================ Demographic Parity ====================================
    dp_classifier = demographic_parity_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    dp_classifier.fit(X_train, y_train, _classifier_name=_classifier, _predictor=_predictor)
    dp_gain_dict, dp_loss_dict = dp_classifier.get_test_flips(X_test, sensitive_test_binary, True)
    dp_confusion_mat = dp_classifier.get_group_confusion_matrix(sensitive_test_binary, \
            X_test, y_test) 

    plot_vertical(erm_confusion_mat, erm_gain_dict, erm_loss_dict, \
                            eo_confusion_mat, eo_gain_dict, eo_loss_dict, \
                            dp_confusion_mat, dp_gain_dict, dp_loss_dict, \
                            "compas_"+sensitive+"_"+_classifier+"_"+_predictor+".png", \
                            "COMPAS (" + sensitive + " sensitive)")

def main_lawschool(sensitive, _classifier, _predictor="hard"):
    X_train, X_test, sensitive_features_train, sensitive_features_test, \
            y_train, y_test, sensitive_feature_names = get_data_lawschool(sensitive)
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    
    # 0 should be black, 1 white
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}

    #=============================== ERM ================================================
    classifier = erm_classifier(X_train, y_train, sensitive_train_binary, sensitive_features_dict)
    classifier.fit(X_train, y_train, classifier_name=_classifier)
    classifier.get_proportions(X_train, y_train, sensitive_train_binary)
    erm_gain_dict, erm_loss_dict = classifier.get_test_flips(X_test, sensitive_test_binary, True)
    erm_confusion_mat = classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 
    
    #================================ Equalized Odds ====================================
    eo_classifier = equalized_odds_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    eo_classifier.fit(X_train, y_train, _classifier_name=_classifier, _predictor=_predictor)
    eo_gain_dict, eo_loss_dict = eo_classifier.get_test_flips(X_test, sensitive_test_binary, True)
    eo_confusion_mat = eo_classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 

    #================================ Demographic Parity ====================================
    dp_classifier = demographic_parity_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    dp_classifier.fit(X_train, y_train, _classifier_name=_classifier, _predictor=_predictor)
    dp_gain_dict, dp_loss_dict = dp_classifier.get_test_flips(X_test, sensitive_test_binary, True)
    dp_confusion_mat = dp_classifier.get_group_confusion_matrix(sensitive_test_binary, \
            X_test, y_test) 

    plot_vertical(erm_confusion_mat, erm_gain_dict, erm_loss_dict, \
                            eo_confusion_mat, eo_gain_dict, eo_loss_dict, \
                            dp_confusion_mat, dp_gain_dict, dp_loss_dict, \
                            "lawschool_"+sensitive+"_"+_classifier+"_"+_predictor+".png", "Lawschool (" + sensitive + " sensitive)")

def main_income(sensitive, _classifier, _predictor="hard"):
    X_train, X_test, sensitive_features_train, sensitive_features_test, \
            y_train, y_test, sensitive_feature_names = get_income_data(sensitive)
    sensitive_train_binary = convert_to_binary(sensitive_features_train, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    sensitive_test_binary = convert_to_binary(sensitive_features_test, \
            sensitive_feature_names[1], sensitive_feature_names[0])
    
    # 0 should be black, 1 white
    sensitive_features_dict = {0:sensitive_feature_names[0], 1:sensitive_feature_names[1]}

    #=============================== ERM ================================================
    classifier = erm_classifier(X_train, y_train, sensitive_train_binary, sensitive_features_dict)
    classifier.fit(X_train, y_train, classifier_name=_classifier)
    classifier.get_proportions(X_train, y_train, sensitive_train_binary)
    erm_gain_dict, erm_loss_dict = classifier.get_test_flips(X_test, sensitive_test_binary, True)
    erm_confusion_mat = classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 
    
    #================================ Equalized Odds ====================================
    eo_classifier = equalized_odds_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    eo_classifier.fit(X_train, y_train, _classifier_name=_classifier, _predictor=_predictor)
    eo_gain_dict, eo_loss_dict = eo_classifier.get_test_flips(X_test, sensitive_test_binary, True)
    eo_confusion_mat = eo_classifier.get_group_confusion_matrix(sensitive_test_binary, X_test, y_test) 

    #================================ Demographic Parity ====================================
    dp_classifier = demographic_parity_classifier(X_train, y_train, sensitive_train_binary, \
            sensitive_features_dict)
    dp_classifier.fit(X_train, y_train, _classifier_name=_classifier, _predictor=_predictor)
    dp_gain_dict, dp_loss_dict = dp_classifier.get_test_flips(X_test, sensitive_test_binary, True)
    dp_confusion_mat = dp_classifier.get_group_confusion_matrix(sensitive_test_binary, \
            X_test, y_test) 

    plot_vertical(erm_confusion_mat, erm_gain_dict, erm_loss_dict, \
                            eo_confusion_mat, eo_gain_dict, eo_loss_dict, \
                            dp_confusion_mat, dp_gain_dict, dp_loss_dict, \
                            "income_"+sensitive+"_"+_classifier+"_"+_predictor+".png", "income (" + sensitive + " sensitive)")

if __name__ == "__main__":
    main_compas("race", "logistic", "hard")
    #main_compas("sex", "logistic", "hard")
    #main_lawschool("race", "logistic", "hard")
    #main_lawschool("sex", "logistic", "hard")
    #main_income("race", "logistic", "hard")
    #main_income("sex", "logistic", "hard")

    #main_compas("race", "logistic", "soft")
    #main_compas("sex", "logistic", "soft")
    #main_lawschool("race", "logistic", "soft")
    #main_lawschool("sex", "logistic", "soft")
    #main_income("race", "logistic", "soft")
    #main_income("sex", "logistic", "hard")



