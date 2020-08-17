import numpy as np

def convert_to_binary(str_features, str1, str2):
    bin_features = np.array([])
    for val in str_features:
        if val == str1:
            bin_features = np.append(bin_features, 1)
        else:
            bin_features = np.append(bin_features, 0)
    return bin_features


def get_dataset_stats(X_vals, y_vals, sensitive, sensitive_vals):
    print("---------- Printing Label Data ------------")
    unique = np.unique(y_vals)
    for u in unique:
        size = len(np.where(y_vals == u)[0])
        print("Label: ", u, " has ", size, " training samples")
    
    print("---------- Printing Group Data (", sensitive, ") ------------")
    unique = np.unique(sensitive_vals)
    for u in unique:
        size = len(np.where(sensitive_vals == u)[0])
        print("Group: ", u, " has ", size, " training samples")


def rebalance_data_labels(X_train, y_train, sensitive_train, X_test, y_test, sensitive_test):
    print("Rebalancing Labels")
    
    unique = np.unique(y_train)
    ratio_dict = {}
    for u in unique:
        size = len(np.where(y_train == u)[0])
        ratio_dict[u] = size
    
    max_label = max(ratio_dict, key=ratio_dict.get)
    min_label = min(ratio_dict, key=ratio_dict.get)
    max_min_ratio = int((ratio_dict[max_label]//2) / ratio_dict[min_label])
    max_label_indicies = np.where(y_train == max_label)[0]
    new_X_train, new_y_train, new_sensitive_train = X_train[max_label_indicies[:len(max_label_indicies)//2]], \
            y_train[max_label_indicies[:len(max_label_indicies)//2]], sensitive_train[max_label_indicies[:len(max_label_indicies)//2]]
    
    min_label_indicies = np.where(y_train != max_label)[0]
    new_X_train = np.append(new_X_train, np.tile(X_train[min_label_indicies], (max_min_ratio, 1)), axis=0)
    new_y_train = np.append(new_y_train, np.tile(y_train[min_label_indicies], (max_min_ratio, )), axis=0)
    new_sensitive_train = np.append(new_sensitive_train, np.tile(sensitive_train[min_label_indicies], (max_min_ratio, )), axis=0)
    random_perm = np.random.shuffle(np.arange(len(new_y_train)))
    new_X_train, new_y_train, new_sensitive_train = new_X_train[random_perm], new_y_train[random_perm], new_sensitive_train[random_perm]

    # Rebalance Test set
    unique = np.unique(y_test)
    ratio_dict = {}
    for u in unique:
        size = len(np.where(y_test == u)[0])
        ratio_dict[u] = size

    max_label = max(ratio_dict, key=ratio_dict.get)
    min_label = min(ratio_dict, key=ratio_dict.get)
    max_min_ratio = int((ratio_dict[max_label]//2) / ratio_dict[min_label])
    
    max_label_indicies = np.where(y_test == max_label)[0]
    new_X_test, new_y_test, new_sensitive_test = X_test[max_label_indicies[:len(max_label_indicies)//2]], \
            y_test[max_label_indicies[:len(max_label_indicies)//2]], sensitive_test[max_label_indicies[:len(max_label_indicies)//2]]
    min_label_indicies = np.where(y_test != max_label)[0]
    
    new_X_test = np.append(new_X_test, np.tile(X_test[min_label_indicies], (max_min_ratio, 1)), axis=0)
    new_y_test = np.append(new_y_test, np.tile(y_test[min_label_indicies], (max_min_ratio, )), axis=0)
    new_sensitive_test = np.append(new_sensitive_test, np.tile(sensitive_test[min_label_indicies], (max_min_ratio, )), axis=0)
    random_perm = np.random.shuffle(np.arange(len(new_y_test)))
    new_X_test, new_y_test, new_sensitive_test = new_X_test[random_perm], new_y_test[random_perm], new_sensitive_test[random_perm]

    return new_X_train[0], new_y_train[0], new_sensitive_train[0], new_X_test[0], new_y_test[0], new_sensitive_test[0]

def rebalance_groups(X_train, y_train, sensitive_train, X_test, y_test, sensitive_test):
    print("Rebalancing Groups")
    # Rebalance Training set
    unique = np.unique(sensitive_train)
    ratio_dict = {}
    for u in unique:
        size = len(np.where(sensitive_train == u)[0])
        ratio_dict[u] = size

    max_label = max(ratio_dict, key=ratio_dict.get)
    min_label = min(ratio_dict, key=ratio_dict.get)
    max_min_ratio = int((ratio_dict[max_label]//2) / ratio_dict[min_label])

    max_label_indicies = np.where(sensitive_train == max_label)[0]
    new_X_train = X_train[max_label_indicies[:len(max_label_indicies)//2]] 
    new_y_train = y_train[max_label_indicies[:len(max_label_indicies)//2]] 
    new_sensitive_train = sensitive_train[max_label_indicies[:len(max_label_indicies)//2]]
    min_label_indicies = np.where(sensitive_train != max_label)[0]
  
    new_X_train = np.append(new_X_train, np.tile(X_train[min_label_indicies], (max_min_ratio, 1)), axis=0)
    new_y_train = np.append(new_y_train, np.tile(y_train[min_label_indicies], (max_min_ratio, )), axis=0)
    new_sensitive_train = np.append(new_sensitive_train, np.tile(sensitive_train[min_label_indicies], (max_min_ratio, )), axis=0)
    random_perm = np.random.shuffle(np.arange(len(new_sensitive_train)))
    new_X_train, new_y_train, new_sensitive_train = new_X_train[random_perm], new_y_train[random_perm], new_sensitive_train[random_perm]

    # Rebalance Test set
    unique = np.unique(sensitive_test)
    ratio_dict = {}
    for u in unique:
        size = len(np.where(sensitive_test == u)[0])
        ratio_dict[u] = size

    max_label = max(ratio_dict, key=ratio_dict.get)
    min_label = min(ratio_dict, key=ratio_dict.get)
    max_min_ratio = int((ratio_dict[max_label]//2) / ratio_dict[min_label])
    
    max_label_indicies = np.where(sensitive_test == max_label)[0]
    new_X_test, new_y_test, new_sensitive_test = X_test[max_label_indicies[:len(max_label_indicies)//2]], \
            y_test[max_label_indicies[:len(max_label_indicies)//2]], sensitive_test[max_label_indicies[:len(max_label_indicies)//2]]
    min_label_indicies = np.where(sensitive_test != max_label)[0]
    
    new_X_test = np.append(new_X_test, np.tile(X_test[min_label_indicies], (max_min_ratio, 1)), axis=0)
    new_y_test = np.append(new_y_test, np.tile(y_test[min_label_indicies], (max_min_ratio, )), axis=0)
    new_sensitive_test = np.append(new_sensitive_test, np.tile(sensitive_test[min_label_indicies], (max_min_ratio, )), axis=0)
    random_perm = np.random.shuffle(np.arange(len(new_y_test)))
    new_X_test, new_y_test, new_sensitive_test = new_X_test[random_perm], new_y_test[random_perm], new_sensitive_test[random_perm]
    
    return new_X_train[0], new_y_train[0], new_sensitive_train[0], new_X_test[0], new_y_test[0], new_sensitive_test[0]

