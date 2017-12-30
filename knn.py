import argparse
import queue

import numpy as np

from shared_modules import *


def calc_euclidean(lista, listb):
    if len(lista) != len(listb):
        raise ValueError('The lengths are dissimilar')
    distance = 0
    for i in range(len(lista)):
        distance += (lista[i] - listb[i])**2
    return np.sqrt(distance)



def normalise_zscore(dataset, means, stddevs):
    for col in range(dataset.shape[1]-1): # this should be minus 1
        for row in range(dataset.shape[0]):
            dataset[row, col] = (dataset[row, col] - means[col]) / (stddevs[col])
    return dataset



def knn2(data_with_labels):
    k = 5
    cv_folds = 10
    
    eval_result = []
    
    for cv_iter in range(cv_folds):
        training_set, testing_set = partition_train_test(cv_iter, cv_folds, data_with_labels)

        means = np.mean(training_set, axis=0)
        stds = np.std(training_set, axis=0)
        
        training_set = normalise_zscore(training_set, means, stds)

        testing_set = normalise_zscore(testing_set, means, stds)
        
        a, b, c, d = 0, 0, 0, 0

        for tst_row in range(testing_set.shape[0]):
            que = []
            for trn_row in range(training_set.shape[0]):
                distance = 0.0
                for col in range(training_set.shape[1]-1):
                    distance += (testing_set[tst_row, col] - training_set[trn_row, col])**2
                distance = distance**0.5
                que.append([distance, trn_row, training_set[trn_row, -1]]) # only append two things
            que.sort()
            top_k = [elem[2] for elem in que[:k]]
            counts = np.bincount(top_k)
            best = np.argmax(counts)

            a, b, c, d = update_eval_attrs(
                best, testing_set[tst_row, -1], 
                a, b, 
                c, d)
        
        eval_result.append(perf_evaluation(a, b, c, d))
    
    '''Printing Results'''
    print_result(eval_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="enter filename like 1 for dataset 1 or 2 for dataset 2")
    args = parser.parse_args()
    if int(args.filename) not in [1,2]:
        print('Can only perform algorithm on dataset 1 or 2')
        return None

    filename = 'project3_dataset' + args.filename + '.txt'
    print('Performing knn on '+filename)
    # "project3_dataset1.txt"
    data_with_labels = preprocess(filename)
    knn2(data_with_labels)

if __name__ == '__main__':
    main()
