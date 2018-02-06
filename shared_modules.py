import numpy as np

def preprocess(file):
    d = np.loadtxt(file, dtype = 'str', delimiter = '\t')
    label = d[:, -1]
    
    result = np.ones([d.shape[0], d.shape[1]], dtype=float)

    for column in range(d.shape[1]):
        try:
            _ = float(d[0, column])
            result[:, column] = [float(val) for val in d[:, column]] # "= d[:,column]" also works for some reason.
        except ValueError:
            unique_in_column = np.unique(d[:, column])
            unique_dict = dict(zip(unique_in_column, range(len(unique_in_column))))

            result[:, column] = [unique_dict[a] for a in d[:, column]]
    return result

def partition_train_test(cv_iter, cv_folds, data_with_labels):
    testing_set, training_set = None, None
    cv_set_size = int(data_with_labels.shape[0] / cv_folds)
    
    if cv_iter == cv_folds - 1:
        testing_set = data_with_labels[(cv_iter * cv_set_size): ]
        training_set = data_with_labels[:(cv_iter * cv_set_size)]
    else:
        testing_set = data_with_labels[(cv_iter * cv_set_size): ((cv_iter + 1) * cv_set_size - 1)]
        training_set = data_with_labels[:(cv_iter * cv_set_size)]
        temp = ((cv_iter + 1) * cv_set_size - 1)
        training_set = np.concatenate((training_set, data_with_labels[((cv_iter + 1) * cv_set_size - 1):]), axis = 0)
    return training_set, testing_set

def perf_evaluation(a, b, c, d):
    precision = a / (a + c)
    recall = a / (a + b)
    f_measure = (2 * recall * precision) / (recall + precision)
    accuracy = (a + d) / (a + b + c + d)
    
    return precision, recall, f_measure, accuracy