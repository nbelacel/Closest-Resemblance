The Closest Resemblance Classifier: CR-General

=====================================================================================================================================
Usage for K-CR.py
=====================================================================================================================================
----------------
•••usage: K-CR.py [-h]  [-k CV_K] [-t T] [-cls CLASSIFIERS] [-file FILE_NAME]


•example usage: python K-CR.py -k 5 -t 1 -cls 0 -file Geobia.csv




----------------
•••optional arguments:
  -h, --help            show this help message and exit.

  -k CV_K               Cross validation parameter. CV_K = 0 indicates leave
                        one out, while any other positive integer
                        bigger than 1 indicates the number of folds for k fold
                        validation.

  -t T                  Used when generating prototypes with
                        single interval algorithm using mean and standard
                        deviation.

  -cls CLASSIFIERS      An integer as indicator. 0 means executing all the
                        other baseline classifiers on the current dataset. 1
                        only CR classifier.

  -file FILE_NAME       A string indicates the name of the file. It could be
                        the single file including both training and testing
                        data, or the training file when a test file name is
                        provided.



----------------
•••Datasets:
This program has the ability to take in both txt and csv files.

In either case, the first line of the file should be dataset information (the header names of each column or other
information with your description) and continue with the data.

In addition, if the dataset contains both data and label, the label
should be placed at the last column.

    For example:
    V1  V2  V3  V4
    0.2 0.3 0.4 1
    ...
    0.4 0.5 0.6 0

In txt file, the data should be separated by '\t', and in csv file, the data should be separated by ','.

=====================================================================================================================================
Usage for Statistical testing and for testing several data sets using 5-fold cross validation.
=====================================================================================================================================
•••usage: StatisticalAnalysis.py [-h] [-t T] [-files FILE_NAMES]
          - files FILE_NAMES space seperated file names to perform statistical testing.

example usage :

StatisticalAnalysis.py -t 1.1 -files Dermatology.csv HeartDisease.csv thyroid.csv pageblocks.csv BreastCancer.csv shuttle.csv newthyroid.csv Geobia.csv Glass.csv leaf.csv Sonar.csv SPECTF.csv
=====================================================================================================================================
----------------
•••Environment:
This program is developed and tested under following dependencies and packages:
Python		3.7
Pandas 1.3.5
pip 23.0.1
scikit-learn	1.0.2
numpy		1.19.5
