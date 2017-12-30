import argparse

import numpy as np
import scipy.stats as stats

from shared_modules import *


def naivebayes(data_with_labels):
    cv_folds = 10
    
    eval_result = []
    
    for cv_iter in range(cv_folds):
        training_set, testing_set = partition_train_test(cv_iter, cv_folds, data_with_labels)
    
        training_labels = training_set[:, -1]
        training_data = training_set[:, :-1]

        '''Training'''
        prob_dict = {key: {} for key in np.unique(training_set[:,-1])}

        unique, counts = np.unique(training_set[:,-1], return_counts=True)
        priors = dict(zip(unique, counts))

        for row in range(training_set.shape[0]):
            label = training_set[row, -1]
            for col in range(training_set.shape[1]-1):
                if col not in prob_dict[label]:
                    prob_dict[label][col] = [training_set[row, col]]
                else:
                    prob_dict[label][col].append(training_set[row, col])

        trained_dict = {key: {} for key in np.unique(training_set[:, -1])}
        for label in prob_dict:
            for col in prob_dict[label]:
                val_list = prob_dict[label][col]
                trained_dict[label][col] = [np.mean(val_list), np.std(val_list)]
        
        '''Testing'''
        a, b, c, d = 0, 0, 0, 0
        
        for row in range(testing_set.shape[0]):
            predictions = []
            for label in trained_dict:
                cur_prob = priors[label] / sum(priors.values())
                for col in range(testing_set.shape[1]-1):
                    col_mean, col_std = trained_dict[label][col]
                    pdf_val = stats.norm.pdf(testing_set[row, col], col_mean, col_std)
                    cur_prob *= pdf_val
                predictions.append((cur_prob, label))
            predictions.sort(reverse=True)
            a, b, c, d = update_eval_attrs(
                predictions[0][1], testing_set[row, -1], 
                a, b, 
                c, d)
        eval_result.append(perf_evaluation(a, b, c, d))
    
    '''Printing Results'''
    print_result(eval_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename_last_digit", help="enter filename like 1 for dataset 1 or 2 for dataset 2")
    args = parser.parse_args()
    if int(args.filename_last_digit) not in [1,2]:
        print('Can only perform algorithm on dataset 1 or 2')
        return None

    filename = 'project3_dataset' + args.filename_last_digit + '.txt'
    print('Performing Naive Bayes on ' + filename)
    # "project3_dataset1.txt"
    data_with_labels = preprocess(filename)
    naivebayes(data_with_labels)

if __name__ == '__main__':
    main()