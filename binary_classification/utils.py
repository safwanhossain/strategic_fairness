import numpy as np

def convert_to_binary(str_features, str1, str2):
    if np.array_equal(np.unique(str_features), np.array([0,1])):
        return str_features

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

def rebalance_data(X_train, y_train, sensitive_train, X_test, y_test, sensitive_test, wrt):
    assert(wrt == "labels" or wrt == "groups")
    print("Rebalancing ", wrt)
   
    # Rebalance the train set
    if wrt == "labels":
        unique = np.unique(y_train)
    else:
        unique = np.unique(sensitive_train)
    
    ratio_dict = {}
    for u in unique:
        if wrt == "labels":
            size = len(np.where(y_train == u)[0])
        else:
            size = len(np.where(sensitive_train == u)[0])
        ratio_dict[u] = size
    
    max_label = max(ratio_dict, key=ratio_dict.get)
    min_label = min(ratio_dict, key=ratio_dict.get)
    if wrt == "labels":
        max_label_indicies = np.where(y_train == max_label)[0]
        min_label_indicies = np.where(y_train != max_label)[0]
    else:
        max_label_indicies = np.where(sensitive_train == max_label)[0]
        min_label_indicies = np.where(sensitive_train != max_label)[0]
    
    oversampled_min_labels = np.random.choice(min_label_indicies, ratio_dict[max_label])
    new_X_train, new_y_train, new_sensitive_train = X_train[max_label_indicies], \
            y_train[max_label_indicies], sensitive_train[max_label_indicies]
    new_X_train, new_y_train, new_sensitive_train = np.append(new_X_train, X_train[oversampled_min_labels]), \
            np.append(new_y_train, y_train[oversampled_min_labels]), np.append(new_sensitive_train, sensitive_train[oversampled_min_labels])
    random_perm = np.random.shuffle(np.arange(len(new_y_train)))
    new_X_train, new_y_train, new_sensitive_train = new_X_train[random_perm], new_y_train[random_perm], new_sensitive_train[random_perm]
    
    # Rebalance the test set
    if wrt == "labels":
        unique = np.unique(y_test)
    else:
        unique = np.unique(sensitive_test)
    
    ratio_dict = {}
    for u in unique:
        if wrt == "labels":
            size = len(np.where(y_test == u)[0])
        else:
            size = len(np.where(sensitive_test == u)[0])
        ratio_dict[u] = size
    
    max_label = max(ratio_dict, key=ratio_dict.get)
    min_label = min(ratio_dict, key=ratio_dict.get)
    if wrt == "labels":
        max_label_indicies = np.where(y_test == max_label)[0]
        min_label_indicies = np.where(y_test != max_label)[0]
    else:
        max_label_indicies = np.where(sensitive_test == max_label)[0]
        min_label_indicies = np.where(sensitive_test != max_label)[0]
    
    oversampled_min_labels = np.random.choice(min_label_indicies, ratio_dict[max_label])
    new_X_test, new_y_test, new_sensitive_test = X_test[max_label_indicies], \
            y_test[max_label_indicies], sensitive_test[max_label_indicies]
    new_X_test, new_y_test, new_sensitive_test = np.append(new_X_test, X_test[oversampled_min_labels]), \
            np.append(new_y_test, y_test[oversampled_min_labels]), np.append(new_sensitive_test, sensitive_test[oversampled_min_labels])
    random_perm = np.random.shuffle(np.arange(len(new_y_test)))
    new_X_test, new_y_test, new_sensitive_test = new_X_test[random_perm], new_y_test[random_perm], new_sensitive_test[random_perm]
    
    return new_X_train[0], new_y_train[0], new_sensitive_train[0], new_X_test[0], new_y_test[0], new_sensitive_test[0]
   

def information_gain(X_train, y_train, feature_index):
    from scipy.stats import entropy
    value,counts = np.unique(y_train, return_counts=True)
    pure_entropy = entropy(counts)

    unique_vals = np.unique(X_train[:,feature_index])
    total_conditional = 0
    for a in unique_vals:
        indicies = np.where(X_train[feature_index] == a)
        y_train_conditional = y_train[indicies]
        value,counts = np.unique(y_train_conditional, return_counts=True)
        conditional_entropy = entropy(counts)
        prob = len(indicies) / len(X_train)
        total_conditional += prob*conditional_entropy
    
    ig = pure_entropy - total_conditional
    return ig


def rank_features_by_IG(X_train, y_train):
    num_features = X_train.shape[1]
    
    ig_gains = []
    for feature in range(num_features):
        ig_gain = information_gain(X_train, y_train, feature)
        ig_gains.append(ig_gain)

    feature_list = np.array([i for i in range(num_features)])
    sorted_feature_list = feature_list[np.argsort([i for i in range(num_features)])]
    return sorted_feature_list

def remove_features(X_train, X_test, to_remove):
    X_train = np.delete(X_train, to_remove, axis=1)
    X_test = np.delete(X_test, to_remove, axis=1)
    
