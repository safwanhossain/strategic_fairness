import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_data_compas():
    from tempeh.configurations import datasets
    compas_dataset = datasets['compas']()
    X_train, X_test = compas_dataset.get_X()
    y_train, y_test = compas_dataset.get_y()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    sensitive_feature_names = ["African-American", "Caucasian"]
    sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('race')
    return X_train, X_test, sensitive_features_train, sensitive_features_test, y_train, y_test, \
            sensitive_feature_names
    
def get_data_lawschool():
    from tempeh.configurations import datasets
    compas_dataset = datasets['lawschool_passbar']()
    X_train, X_test = compas_dataset.get_X()
    y_train, y_test = compas_dataset.get_y()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    sensitive_feature_names = ["Black", "White"]
    sensitive_features_train, sensitive_features_test = \
            compas_dataset.get_sensitive_features('race')
    return X_train, X_test, sensitive_features_train, sensitive_features_test, y_train, y_test, \
            sensitive_feature_names


def get_income_data(s_attr):
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

    if s_attr == "race":
        sensitive_feature_names = ["Not White", "White"]
    if s_attr == "sex":
        sensitive_feature_names = ["Female", "Male"]

    train_len = int(0.7*len(y))
    X_train, sen_train, y_train = all_f[:train_len, :-2], all_f[:train_len, -2], all_f[:train_len, -1].flatten()
    X_test, sen_test, y_test = all_f[train_len:, :-2], all_f[train_len:, -2], all_f[train_len:, -1].flatten()
    return X_train, X_test, sen_train, sen_test, y_train.astype("int"), y_test.astype("int"), \
            sensitive_feature_names


