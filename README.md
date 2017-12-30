# DataMining

**Classification**

Performed binary classification on genetic data using the K-nearest neighbors and Naive Bayes algorithms. Predicted presence of health defect given the presence of genetic markers.

*Performance*
K-nearest Neighbors:
Dataset 1:
1. Accuracy:	0.93972027972
2. Precision:	0.96466507177 
3. Recall:		0.876586820587
4. F-measure:	0.91351705142

Dataset 2:
1. Accuracy:	0.679583333333
2. Precision:	0.54471028971
3. Recall: 	 	0.471616015563
4. F-measure:	0.482067715559

Naive Bayes:
Dataset 1:
1. Accuracy:	0.94013986014
2. Precision:	0.928340270967
3. Recall:		0.908994227994
4. F-measure:	0.91716995148

Dataset 2:
1. Accuracy: 	0.704305555556
2. Precision: 	0.572205373917
3. Recall: 		0.631216035427
3. F-measure:	0.590502955655

*Run instructions*

K-Nearest Neighbors:

To run Naïve Bayes:

“python naivebayes.py <dataset-name-last-digit>”
… where 0 <  <dataset-name-last-digit>  < 3

example: “python naivebayes.py 1” for running the naïve bayes algorithm on Project3_dataset1.txt.
To run on dataset2, use: “python naivebayes.py 1”

Hyperparameters: 
There are none in the Naïve Bayes algorithm.

Environment: 
Python 3

Requirement:
The datasets should be in the same folder as the naivebayes.py file


Naive Bayes:

To run KNN:

“python knn.py <dataset-name-last-digit>”
… where 0 <  <dataset-name-last-digit>  < 3

example: “python knn.py 1” for running the K nearest neighbors algorithm on Project3_dataset1.txt.
To run on dataset2, use: “python knn.py 1”

Hyperparameters: 
The value of k can be changed from the first line of the knn function in the knn.py file

Environment: 
Python 3

Requirement:
The datasets should be in the same folder as the naivebayes.py file
