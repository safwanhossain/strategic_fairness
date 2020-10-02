import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import *

# In all datasets, let "1" denote the good/desirable outcome and "0"
# denote he poor outcome

def get_data_compas(sensitive, get_stats=False):
    """ Here the desirable outcome is 0 - not going to re-offend. So gonna flip it
    """
    assert sensitive == "race" or sensitive == "sex"
    from tempeh.configurations import datasets
    compas_dataset = datasets['compas']()
    X_train, X_test = compas_dataset.get_X()
    y_train, y_test = compas_dataset.get_y()
    y_train, y_test = 1 - y_train.flatten(), 1 - y_test.flatten()
    
    if sensitive == "race":
        sensitive_feature_names = ["African-American", "Caucasian"]
        sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('race')
    else:
        sensitive_feature_names = ["Female", "Male"]
        sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('sex')
    
    if get_stats == True:
        get_dataset_stats(X_train, y_train, sensitive, sensitive_features_train)

    return X_train, X_test, sensitive_features_train, sensitive_features_test, y_train, y_test, \
            sensitive_feature_names
    
def get_data_lawschool(sensitive, get_stats=False, rebalance=None):
    assert sensitive == "race" or sensitive == "sex"
    from tempeh.configurations import datasets
    compas_dataset = datasets['lawschool_passbar']()
    X_train, X_test = compas_dataset.get_X()
    y_train, y_test = compas_dataset.get_y()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    if sensitive == "race":
        sensitive_feature_names = ["black", "white"]
        sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('race')
    else:
        sensitive_feature_names = ["female", "male"]
        sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('gender')
    
    if rebalance == None:
        pass
    elif rebalance == "labels":
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_data(X_train, y_train, sensitive_features_train, \
                X_test, y_test, sensitive_features_test, "labels")
    elif rebalance == "race":
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_data(X_train, y_train, sensitive_features_train, \
            X_test, y_test, sensitive_features_test, "groups")
    elif rebalance == "both" and sensitive == "race":
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_data(X_train, y_train, sensitive_features_train, \
            X_test, y_test, sensitive_features_test, "labels")
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_data(X_train, y_train, sensitive_features_train, \
                X_test, y_test, sensitive_features_test, "groups")
    else:
        print("--------------ERROR---------------")
        return

    if get_stats == True:
        get_dataset_stats(X_train, y_train, sensitive, sensitive_features_train)

    return X_train, X_test, sensitive_features_train, sensitive_features_test, y_train, y_test, \
            sensitive_feature_names

def oneHotCatVars(df, df_cols):
    df_1 = adult_data = df.drop(columns = df_cols, axis = 1)
    df_2 = pd.get_dummies(df[df_cols])
    return (pd.concat([df_1, df_2], axis=1, join='inner'))

def get_train_test_split(X, y, sensitive, ratio=0.7):
    all_f = np.concatenate([X, sensitive, y], axis=1)
    np.random.shuffle(all_f)

    train_len = int(ratio*len(y))
    X_train, sen_train, y_train = all_f[:train_len, :-2], all_f[:train_len, -2], all_f[:train_len, -1].flatten()
    X_test, sen_test, y_test = all_f[train_len:, :-2], all_f[train_len:, -2], all_f[train_len:, -1].flatten()
    return X_train, sen_train ,y_train, X_test, sen_test, y_test

def get_data_income(s_attr, get_stats=False, rebalance=None, raw=False, drop=[]):
    s_dict = {'race': 8, 'sex':9}
    filename = 'adult-all.csv'
    dataframe = pd.read_csv(filename, header=None, na_values='?')
    dataframe = dataframe.dropna()

    last_ix = len(dataframe.columns) - 1
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    y = LabelEncoder().fit_transform(y).reshape(-1,1)
    X, sensitive = X.drop(s_dict[s_attr], axis=1), X[s_dict[s_attr]].to_numpy().reshape(-1,1)
    
    if s_attr == "race":
        sensitive_feature_names = ["Not White", "White"]
        sensitive[np.where(sensitive!="White")[0]] = "Not White"
    if s_attr == "sex":
        sensitive_feature_names = ["Female", "Male"]
    if raw == True:
        return X, sensitive, y, sensitive_feature_names

    if len(drop) != 0:
        print("Before shape: ", X.shape)
        X = X.drop(X.columns[drop], axis=1)
        print("After shape: ", X.shape)


    X = oneHotCatVars(X, X.select_dtypes('object').columns).to_numpy()
    X_train, sen_train, y_train, X_test, sen_test, y_test = get_train_test_split(X, y, sensitive)
    
    if rebalance == None:
        pass
    elif rebalance == "labels":
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_data(X_train, y_train, sen_train, \
                X_test, y_test, sen_test, "labels")
    elif rebalance == "race":
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_data(X_train, y_train, sen_train, \
            X_test, y_test, sen_test, "groups")
    elif rebalance == "both" and s_attr == "race":
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_data(X_train, y_train, sen_train, \
            X_test, y_test, sen_test, "labels")
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_data(X_train, y_train, sen_train, \
                X_test, y_test, sen_test, "groups")
    else:
        print("--------------ERROR---------------")
        return
    
    if get_stats == True:
        get_dataset_stats(X_train, y_train, s_attr, sen_train)
    
    return X_train, X_test, sen_train, sen_test, y_train.astype("int"), y_test.astype("int"), \
            sensitive_feature_names

def get_data_student(s_attr, get_stats=False, subject="math", rebalance=None, raw=False, drop=[]):
    s_dict = {'sex': 1, 'Pstatus':5, 'famsup':16}
    if subject == "math":
        filename = 'student_dataset/student-mat.csv'
    else:
        filename = 'student_dataset/student-por.csv'
    dataframe = pd.read_csv(filename, header=None, sep=";", na_values='?')
    dataframe = dataframe.dropna()

    last_ix = len(dataframe.columns) - 1
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    y = LabelEncoder().fit_transform(y).reshape(-1, 1)
    X, sensitive = X.drop(s_dict[s_attr], axis=1), X[s_dict[s_attr]].to_numpy().reshape(-1,1)
    
    if s_attr == "sex":
        sensitive_feature_names = ["F", "M"]
    if s_attr == "Pstatus":
        sensitive_feature_names = ["A", "T"]
    if s_attr == "famsup":
        sensitive_feature_names = ["no", "yes"]
   
    new_y = np.zeros(len(y))
    new_y[np.where(y >= 10)[0]] = 1
    new_y = new_y.reshape(-1,1)

    if raw == True:
        return X, sensitive, new_y, sensitive_feature_names
    if len(drop) != 0:
        print("Before shape: ", X.shape)
        X = X.drop(X.columns[drop], axis=1)
        print("After shape: ", X.shape)
    
    X = oneHotCatVars(X, X.select_dtypes('object').columns).to_numpy()
    X_train, sen_train, y_train, X_test, sen_test, y_test = get_train_test_split(X, new_y, sensitive)
    
    if s_attr == "sex":
        sensitive_feature_names = ["F", "M"]
    if s_attr == "Pstatus":
        sensitive_feature_names = ["A", "T"]
    if s_attr == "famsup":
        sensitive_feature_names = ["no", "yes"]

    return X_train, X_test, sen_train, sen_test, y_train, y_test, \
            sensitive_feature_names

def test_balancing():
    print("------------------------- COMPAS DATASET ------------------------------")
    get_data_compas("race", get_stats=True)
    print("\n")
    get_data_compas("sex", get_stats=True)
    
    print("\n\n------------------------- LAW SCHOOL DATASET ------------------------------")
    get_data_lawschool("race", get_stats=True, rebalance=None)
    print("\n")
    get_data_lawschool("sex", get_stats=True, rebalance=None)
    print("\n")
    get_data_lawschool("race", get_stats=True, rebalance="both")
    print("\n")
    get_data_lawschool("sex", get_stats=True, rebalance="labels")
    
    print("\n\n------------------------- INCOME DATASET ------------------------------")
    get_data_income("race", get_stats=True, rebalance=None)
    print("\n")
    get_data_income("sex", get_stats=True, rebalance=None)
    print("\n")
    get_data_income("race", get_stats=True, rebalance="both")
    print("\n")
    get_data_income("sex", get_stats=True, rebalance="labels")

if __name__ == "__main__":
    X_train, X_test, sensitive_features_train, sensitive_features_test, y_train, y_test, \
            sensitive_feature_names = get_data_student("sex", get_stats=True)
    
    sorted_features = rank_features_by_IG(X_train, y_train)

