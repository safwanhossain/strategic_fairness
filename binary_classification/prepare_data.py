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
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_data_labels(X_train, y_train, sensitive_features_train, \
                X_test, y_test, sensitive_features_test)
    elif rebalance == "race":
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_groups(X_train, y_train, sensitive_features_train, \
            X_test, y_test, sensitive_features_test)
    elif rebalance == "both" and sensitive == "race":
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_groups(X_train, y_train, sensitive_features_train, \
            X_test, y_test, sensitive_features_test)
        X_train, y_train, sensitive_features_train, X_test, y_test, sensitive_features_test = rebalance_data_labels(X_train, y_train, sensitive_features_train, \
                X_test, y_test, sensitive_features_test)
    else:
        print("--------------ERROR---------------")
        return

    if get_stats == True:
        get_dataset_stats(X_train, y_train, sensitive, sensitive_features_train)

    return X_train, X_test, sensitive_features_train, sensitive_features_test, y_train, y_test, \
            sensitive_feature_names


def get_data_income(s_attr, get_stats=False, rebalance=None):
    def oneHotCatVars(df, df_cols):
        df_1 = adult_data = df.drop(columns = df_cols, axis = 1)
        df_2 = pd.get_dummies(df[df_cols])
        return (pd.concat([df_1, df_2], axis=1, join='inner'))
    
    s_dict = {'race': 8, 'sex':9}
    filename = 'adult-all.csv'
    dataframe = pd.read_csv(filename, header=None, na_values='?')
    dataframe = dataframe.dropna()

    last_ix = len(dataframe.columns) - 1
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    y = LabelEncoder().fit_transform(y)

    X, sensitive = X.drop(s_dict[s_attr], axis=1), X[s_dict[s_attr]]
    X = oneHotCatVars(X, X.select_dtypes('object').columns)
    
    X, sen, y = X.to_numpy(), sensitive.to_numpy().reshape(-1,1), y.reshape(-1,1)
    all_f = np.concatenate([X,sen, y], axis=1)
    np.random.shuffle(all_f)

    train_len = int(0.7*len(y))
    X_train, sen_train, y_train = all_f[:train_len, :-2], all_f[:train_len, -2], all_f[:train_len, -1].flatten()
    X_test, sen_test, y_test = all_f[train_len:, :-2], all_f[train_len:, -2], all_f[train_len:, -1].flatten()
    
    if s_attr == "race":
        sensitive_feature_names = ["Not White", "White"]
        sen_train[np.where(sen_train!="White")[0]] = "Not White"
        sen_test[np.where(sen_test!="White")[0]] = "Not White"
    if s_attr == "sex":
        sensitive_feature_names = ["Female", "Male"]
    
    if rebalance == None:
        pass
    elif rebalance == "labels":
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_data_labels(X_train, y_train, sen_train, \
                X_test, y_test, sen_test)
    elif rebalance == "race":
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_groups(X_train, y_train, sen_train, \
            X_test, y_test, sen_test)
    elif rebalance == "both" and s_attr == "race":
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_groups(X_train, y_train, sen_train, \
            X_test, y_test, sen_test)
        X_train, y_train, sen_train, X_test, y_test, sen_test = rebalance_data_labels(X_train, y_train, sen_train, \
                X_test, y_test, sen_test)
    else:
        print("--------------ERROR---------------")
        return
    
    if get_stats == True:
        get_dataset_stats(X_train, y_train, s_attr, sen_train)
    
    return X_train, X_test, sen_train, sen_test, y_train.astype("int"), y_test.astype("int"), \
            sensitive_feature_names

if __name__ == "__main__":
    print("------------------------- COMPAS DATASET ------------------------------")
    get_data_compas("race", get_stats=True)
    print("\n")
    get_data_compas("sex", get_stats=True)
    
    print("\n\n------------------------- LAW SCHOOL DATASET ------------------------------")
    get_data_lawschool("race", get_stats=True, rebalance="both")
    print("\n")
    get_data_lawschool("sex", get_stats=True, rebalance="labels")
    
    print("\n\n------------------------- INCOME DATASET ------------------------------")
    get_data_income("race", get_stats=True, rebalance="both")
    print("\n")
    get_data_income("sex", get_stats=True, rebalance="labels")



