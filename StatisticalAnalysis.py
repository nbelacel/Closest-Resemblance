import warnings

#from FuzzyKClosestResembler import FuzzyKClosestResembler

warnings.filterwarnings("ignore")
import math
import numpy as np
import time
import argparse
from datetime import date
from sklearn import datasets
import data_handling
import KNN, RF, MLP, SVM
from sklearn.preprocessing import Normalizer
import Random_forest_feature
import KClosestResembler, KClosestResemblerSD
import cross_validation
import KCR_auc, KCR_SD_auc
import csv
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
#import KClosestResembler
import KClosestResemblerSD
import KClosestResemblerPercentile
import KCR_percentile_auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare




def handlingMissingValues(X,y,classes, num_attris, data_clean):


    option = data_clean

    if(option == 1):
        rows,cols = X.shape
        meansArray = []
        for i in range(len(np.unique(y))):
            meansArray.append([])
            for j in range(X.shape[1]):
                meansArray[i].append(np.nanmean(X[tuple(list(np.where(y == i)))][:, j]))
        for i in range(rows):
            for j in range(cols):
                if(math.isnan(X[i][j])):
                    if(math.isnan(meansArray[y[i]][j])):
                        X[i][j] = 0
                    else:
                        X[i][j] = meansArray[y[i]][j]
        return X

    elif(option == 2):
        X = X[~np.isnan(X).any(axis=1)]
        return X
    elif(option ==3):
        X = X[:,~np.isnan(X).any(axis=0)]
        return X
    else:
        print("Invalid data cleaning option\n")


def get_parser():
    '''
    Get command line input arguments
    :return: parser
    '''
    # Get parser for command line arguments.
    parser = argparse.ArgumentParser(description="Statistical Analysis Report",
                                     usage="\n * example usage:     python StatisticalAnalysis.py -t 1 -files iris.csv Pima.csv GEOBIA.csv \n")
    #parser.add_argument("-beta",
    #                    dest="beta",
    #                    default="0.5",
    #                    help="A float number as threshold for interval filtering.")
    #parser.add_argument("-n",
    #                    dest="num_intervals",
    #                    default="2",
    #                    help="An integer indicates the number of intervals required for each feature.")
    parser.add_argument("-t",
                        dest="t",
                        default="1",
                        help="A float number indicates for normal distribution.")
    parser.add_argument("-files",nargs='+',
                        dest="file_names",
                        help="A string indicates the name of the file. It could be the single file including both training and testing data, or the training file when a test file name is provided.")
    #parser.add_argument("-opt",
    #                    dest="optimization",
    #                    default="1",
    #                    help="An integer indicates the number of iteration processed for weight optimization. 0 means there is no optimization,all features have equal weights. 1 means optimazation with correaltion(spearman) and  Any integer bigger than 1 indicates the number of iterations for weights optimization.")


    return parser

def get_auc(y_actual, y_pred, classes):
    AUCs = []

    if(len(np.unique(y_pred)) == len(np.unique(y_actual)) and len(np.unique(y_pred)) == 1):
        return 1

    for i in classes:
        y_actual_temp = []
        y_pred_temp = []

        for y in y_actual:
            if y == i:
                y_actual_temp.append(0)
            else:
                y_actual_temp.append(1)

        for y in y_pred:
            if y == i:
                y_pred_temp.append(0)
            else:
                y_pred_temp.append(1)

        auc = roc_auc_score(y_actual_temp, y_pred_temp)
        AUCs.append(auc)
    return np.mean(np.array(AUCs))

def folds_auc(cls, X, y):
    accuracy = []
    training_times = []
    testing_times = []
    kf = KFold(n_splits=5, random_state=1119, shuffle=True)
    for train_index, test_index in kf.split(X):

        # Splitting out training and testing data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Measure training time
        start_train = time.time()
        cls.fit(X_train,y_train)
        end_train = time.time()
        start_test = time.time()
        y_pred = cls.predict(X_test)
        end_test = time.time()

        accuracy.append(get_auc(y_test, y_pred, np.unique(y_test)))
        training_times.append(end_train - start_train)
        testing_times.append(end_test - start_test)
        avg_training_time = np.mean(training_times)
        avg_testing_time = np.mean(testing_times)

    return accuracy, avg_training_time, avg_testing_time

def PairTTest(data):

    print("\n===========================================\nPair T Test\n===========================================\n")

    for file in data:
        for algo in data[file]:
            if(algo == 'kcr'):
                continue

            print("\n----------------------\npair t test for dataset "+ file+" for algorithms KCR and "+algo+"\n----------------------")
            stat, p = ttest_rel(data[file]['kcr'], data[file][algo])
            print('stat=%.6f, p value=%.6f' % (stat, p))
            if p > 0.05:
                print('Algorithms are classifying the data similarly with 95% confidence')
            else:
                print('Algorithms are classifying the data differently (Inconsistent) with 95% confidence')
                if(p>=0.001):
                    print('but the Algorithms are classifying the data similarly with 99% confidence')

def WilcoxonTest(data):

    print("\n===========================================\nWilcoxon Test\n===========================================\n")

    for algo in data:
        if(algo == 'kcr'):
            continue

        print("\n----------------------\nWilcoxon test for dataset for algorithms KCR and " + algo + "\n----------------------")
        stat, p = wilcoxon(data['kcr'], data[algo])
        print('stat=%.6f, p=%.6f' % (stat, p))
        if p > 0.05:
            print('Algorithms are classifying the data similarly with 95% confidence')
        else:
            print('Algorithms are classifying the data differently (Inconsistent) with 95% confidence')
            if (p >= 0.001):
                print('but the Algorithms are classifying the data similarly with 99% confidence')

def FriedmanTest(data):

    print("\n===========================================\nFriedman Test\n===========================================\n")
    d1 = data['kcr']
    d2 = data['kcr1']
    d3 = data['knn']
    d4 = data['svm']
    d5= data['rf']
    d6= data['mlp']
    d7= data['nb']
    d8= data['nc']


    stat, p = friedmanchisquare(d1,d2,d3,d4, d5, d6, d7, d8)
    print('stat=%.6f, p=%.6f' % (stat, p))
    if p > 0.05:
        print('Algorithms are classifying the data similarly with 95% confidence')
    else:
        print('Algorithms are classifying the data differently (Inconsistent) with 95% confidence')
        if (p >= 0.001):
            print('but the Algorithms are classifying the data similarly with 99% confidence')


if __name__ == '__main__':

    print("Reading and Analysing Command Line Arguments\n")
    try:
        parser = get_parser()
        args = parser.parse_args()
    except:
        print("\nTry *** python K-CR.py -h or --help *** for more details.")
        exit(0)
    print("Commands line arguments mentioned are \n-t := "+args.t+"\n-files := "+str(args.file_names)+"\n")


    d = {}
    ts= {}
    training_times = {}
    testing_times = {}
    results = {}
    for file in args.file_names:
        print("Calculating AUC's using k fold validations for KCR Percentile KCR SD KNN RF SVM MLP NB NC with the data set "+file)
        results[file] = {}
        training_times[file] = {}
        testing_times[file] = {}
        d[file],ts[file] = data_handling.loading(file)
        d[file] = handlingMissingValues(d[file], ts[file], np.unique(ts[file]), d[file].shape[1], 1)

        kcr = KClosestResemblerPercentile.KClosestResemblerPercentile()
        kcr1 = KClosestResemblerSD.KClosestResemblerSD(float(args.t))

        results[file]["kcr"], train_time, test_time = folds_auc(kcr, d[file], ts[file])
        training_times[file]["kcr"] = train_time
        testing_times[file]["kcr"] = test_time

        results[file]["kcr1"], train_time, test_time = folds_auc(kcr1, d[file], ts[file])
        training_times[file]["kcr1"] = train_time
        testing_times[file]["kcr1"] = test_time

        knn = KNeighborsClassifier(n_neighbors=3)
        results[file]["knn"], train_time, test_time = folds_auc(knn, d[file], ts[file])
        training_times[file]["knn"] = train_time
        testing_times[file]["knn"] = test_time

        clf = svm.SVC(kernel='linear', C=1, probability=True)
        results[file]["svm"], train_time, test_time = folds_auc(clf, d[file], ts[file])
        training_times[file]["svm"] = train_time
        testing_times[file]["svm"] = test_time

        rf = RandomForestClassifier(n_estimators=300, random_state=0)
        results[file]["rf"], train_time, test_time = folds_auc(rf, d[file], ts[file])
        training_times[file]["rf"] = train_time
        testing_times[file]["rf"] = test_time

        mlp = MLPClassifier()
        results[file]["mlp"], train_time, test_time = folds_auc(mlp, d[file], ts[file])
        training_times[file]["mlp"] = train_time
        testing_times[file]["mlp"] = test_time

        nb = GaussianNB()
        results[file]["nb"], train_time, test_time = folds_auc(nb, d[file], ts[file])
        training_times[file]["nb"] = train_time
        testing_times[file]["nb"] = test_time

        nc = NearestCentroid()
        results[file]["nc"], train_time, test_time = folds_auc(nc, d[file], ts[file])
        training_times[file]["nc"] = train_time
        testing_times[file]["nc"] = test_time

    print("\n===========================K fold AUC results for data sets===========================\n")
    for file in results:
        print("\n------------------"+file+"------------------\n")
        print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format('Fold', 'KCR Percentile','KCR SD' 'KNN','RF','SVM', 'MLP','NB','NC'))
        for i in range(len(results[file]['kcr'])):
            print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format(i, results[file]['kcr'][i], results[file]['kcr1'][i], results[file]['knn'][i], results[file]['rf'][i],results[file]['svm'][i], results[file]['mlp'][i], results[file]['nb'][i], results[file]['nc'][i]))

    dataset_results = {}
    dataset_results1 = {}
    dataset_results2 = {}
    dataset_results3 = {}
    for file in results:
        for algo in results[file]:
            if(algo in dataset_results):
                dataset_results[algo].append(np.mean(results[file][algo]))
                dataset_results1[algo].append(np.std(results[file][algo]))
                dataset_results2[algo].append(np.mean(training_times[file][algo]))
                dataset_results3[algo].append(np.mean(testing_times[file][algo]))

            else:
                dataset_results[algo] = []
                dataset_results[algo].append(np.mean(results[file][algo]))
                dataset_results1[algo] = []
                dataset_results1[algo].append(np.std(results[file][algo]))
                dataset_results2[algo] = []
                dataset_results2[algo].append(training_times[file][algo])
                dataset_results3[algo] = []
                dataset_results3[algo].append(testing_times[file][algo])
    print("\n=======================Average Accuracies for datasets=======================\n")
    print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format('dataset', 'KCR Percentile', 'KCR SD', 'KNN', 'RF', 'SVM', 'MLP', 'NB', 'NC'))
    for (i, j) in zip(range(len(dataset_results['kcr'])),args.file_names):
        print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format(j, dataset_results['kcr'][i], dataset_results['kcr1'][i], dataset_results['knn'][i], dataset_results['rf'][i] , dataset_results['svm'][i], dataset_results['mlp'][i], dataset_results['nb'][i], dataset_results['nc'][i]))
    print("\n=======================Average Training Time for datasets=======================\n")
    print(
        "{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format('dataset', 'KCR Percentile', 'KCR SD',
                                                                                 'KNN', 'RF', 'SVM', 'MLP', 'NB', 'NC'))
    for (i, j) in zip(range(len(dataset_results['kcr'])), args.file_names):
        print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format(j, dataset_results2['kcr'][i],dataset_results2['kcr1'][i],dataset_results2['knn'][i],dataset_results2['rf'][i],dataset_results2['svm'][i],dataset_results2['mlp'][i],dataset_results2['nb'][i],dataset_results2['nc'][i]))
    print("\n=======================Average Testing Time for datasets=======================\n")
    print(
        "{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format('dataset', 'KCR Percentile', 'KCR SD',
                                                                                 'KNN', 'RF', 'SVM', 'MLP', 'NB', 'NC'))
    for (i, j) in zip(range(len(dataset_results['kcr'])), args.file_names):
        print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} ".format(j, dataset_results3['kcr'][i],dataset_results3['kcr1'][i],dataset_results3['knn'][i],dataset_results3['rf'][i],dataset_results3['svm'][i],dataset_results3['mlp'][i],dataset_results3['nb'][i],dataset_results3['nc'][i]))

    PairTTest(results)
    WilcoxonTest(dataset_results)
    FriedmanTest(dataset_results)
