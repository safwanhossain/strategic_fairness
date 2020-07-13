import numpy as np

def convert_to_binary(str_features, str1, str2):
    bin_features = np.array([])
    for val in str_features:
        if val == str1:
            bin_features = np.append(bin_features, 1)
        else:
            bin_features = np.append(bin_features, 0)
    return bin_features


