import warnings

warnings.filterwarnings("ignore")

import numpy as np
import time
import argparse
from datetime import date
from sklearn import datasets
import data_handling
import KNN, RF, MLP, SVM, Naive, NC, RGLVQ_Classifier, MGLVQ_Classifier
from sklearn.preprocessing import Normalizer
import Random_forest_feature
import KClosestResembler, KClosestResemblerSD
import cross_validation
import KCR_auc,KCR_SD_auc,KCR_percentile_auc
import csv
from  data_handling import loading_test_file_without_label,loading


# function used for parsing command line arguments
def get_parser():
    '''
    Get command line input arguments
    :return: parser
    '''
    # Get parser for command line arguments.
    parser = argparse.ArgumentParser(description="K-CR-General",
                                     usage="\n * example usage: python K-CR.py -k 10 -t 1 -cls 0 -file iris.csv \n * (better) optional usage:   python K-CR.py -beta 0.2 -n 2 -k 10 -t 1 -cls 0 -file iris.csv > file_of_your_name.txt \n By using the second command, instead of print out all different outage in terminal, they will be stored in a file named file_of_your_name.txt.")


    parser.add_argument("-k",
                        dest="cv_k",
                        help="A integer as validation indicator. 0 indicates leave one out validation, while any other positive integer bigger than 1 indiactes the number of folds for k fold validation.")
    parser.add_argument("-t",
                        dest="t",
                        help="A integer. Only used when generating prototypes with single interval algorithm using mean and standard deviation.")
    parser.add_argument("-cls",
                        dest="classifiers",
                        help="A integer as indicator. 0 means executing all the other baseline calssifiers on the current dataset. 1 otherwise.")
    parser.add_argument("-file",
                        dest="file_name",
                        help="A string indicates the name of the file. It could be the single file including both training and testing data, or the training file when a test file name is provided.")
    parser.add_argument("-testfile",  nargs='+',
                        default="None",
                        dest="test_file_name",
                        help="A string indicates the name of the file followed by integer whether test file has labels. 0 for no labels and 1 for labels. Defaults as None if there is no separate training and testing file.")
    parser.add_argument("-feature",default=0,
                        dest="feature_selection",
                        help="A integer indicates whether to select best features using Feature  Selection and 0 indicates no feature selelction needed and 1 is to select feature selection using Random forest and 2 is using best weights for K-CR's (2 is applicable only for 4 kcr models and without seperate test files.")
    parser.add_argument("-dataclean",
                        default="1",
                        dest="data_clean",
                        help="An integer indicates what to do when it sees missing values or garbage in the dataset, 1 indicates to replace with mean of feature of respective class, 2 is to remove that record, 3 to remove the feature (* applicable for only KCR and KCR SD)")
    parser.add_argument("-validation",
                        default="1",
                        dest="validation",
                        help="An integer indicates whether to have validation set, 0 means no validation and 1 means need validation set and default is 1")

    return parser


# K-CR project starts here
if __name__ == '__main__':

    print("Reading and Analysing Command Line Arguments\n")
    try:
        # Parsing Command Line Arguments using Argparse Package
        parser = get_parser()
        args = parser.parse_args()
    except:
        # help description is printed to console if any exception occured during parsing cmd parameters
        print("\nTry *** python K-CR.py -h or --help *** for more details.")
        exit(0)


    # K in Kfold must be between 0 and 10, 0 or 1 for LeaveOneOut and >1 is Strtified KFold
    if(int(args.cv_k) < 0 or int(args.cv_k) > 10):
        print("'K' values must be in between 0 to 10 ")



    print("Loading Data File\n")
    # Parsing input file as data and target
    d, ts = data_handling.loading(args.file_name)


    # option for Random Forest Feature Selection
    if(int(args.feature_selection) == 1 and args.test_file_name == "None"):
        print("Analysing Best Features using Random Forest Algorithm")
        d = Random_forest_feature.feature_selection(d, ts)

    # Algorithms to execute when no seperate test files
    if args.test_file_name == "None":

        # Optimizations for K-CR must be greater than or equal to 0
        args.optimization =0
        if int(args.optimization) < 0:
            print("option for optimization 'opt' should be greater or equal to 0")
            exit(0)



        print("\nEvaluating KCR with (10 & 90 percentile)\n")
        KCR_percentile_auc.KCR_percentile(d, ts, args.file_name, int(args.cv_k), int(args.optimization),data_clean=int(args.data_clean),validation = int(args.validation),feature_selection = int(args.feature_selection))

        # Evaluating K-CR with standard deviation using respective parameters
        print("\nEvaluating KCR with standard deviation\n")
        KCR_SD_auc.KCR_SD(d, ts, args.file_name, int(args.cv_k), int(args.optimization), t=float(args.t),data_clean=int(args.data_clean),validation = int(args.validation),feature_selection = int(args.feature_selection))


        # if classifers parameter is 0, execute other classifiers for comparsion
        if int(args.classifiers) == 0:

            # Evaluating K Nearest Neighbors Algorithm
            print("\nEvaluating KNN\n")
            KNN.KNN(d, ts, args.file_name, int(args.cv_k))

            # Evaluating Nearest Centroid Algorithm
            print("\nEvaluating Nearest Centroid\n")
            NC.NC(d, ts, args.file_name, int(args.cv_k))

            # Evaluating Random Forests Algorithm
            print("\nEvaluating Random Forest\n")
            RF.RF(d, ts, args.file_name, int(args.cv_k))

            # Evaluating Support Vector Machine
            print("\nEvaluating SVM\n")
            SVM.SVM(d, ts, args.file_name, int(args.cv_k))

            # valuating MLP
            print("\nEvaluating MLP Classifier\n")
            MLP.MLP(d, ts, args.file_name, int(args.cv_k))

            # Evaluating Bayesian Algorithm
            print("\nEvaluating Naive Bayes Classifier\n")
            Naive.Naive(d, ts, args.file_name, int(args.cv_k))



    # Else case is executed when we have seperate Test files
    else:
        print("Loading Test File\n")
        test_file_name = args.test_file_name[0]
        label_status = int(args.test_file_name[1])
        args.test_file_name = test_file_name

        # if label_status is 1 i.e, Test files contains output labels
        if(label_status == 1):
            d_test, ts_test = loading(args.test_file_name)
            test_classes = np.unique(np.array(ts_test))



            # Evaluating K-CR with seperate test file with output class labels in test
            print("\nK Closest Resembler\n")
            #KCR_auc.KCR(d, ts, args.file_name, int(args.cv_k), float(args.beta), int(args.num_intervals),int(args.optimization), d_test,ts_test, args.test_file_name,data_clean=int(args.data_clean),validation = int(args.validation))

            # Evaluating Fuzzy K-CR with seperate test file with output class labels in test
            print("\nEvaluating Fuzzy KCR \n")
            #FuzzyKCR_auc.FuzzyKCR(d, ts, args.file_name, int(args.cv_k), float(args.beta), int(args.num_intervals),int(args.optimization), int(args.t), d_test, ts_test,args.test_file_name,data_clean=int(args.data_clean),validation = int(args.validation))

            # Evaluating K-CR with 25 and 75 percentile using seperate test file with output class labels in test
            print("\nEvaluating KCR with (25 & 75 percentile)\n")
            KCR_percentile_auc.KCR_percentile(d, ts, args.file_name, int(args.cv_k), int(args.optimization),d_test, ts_test,args.test_file_name, data_clean=int(args.data_clean),validation = int(args.validation))

            # Evaluating K-CR with standard deviation seperate test file with output class labels in test
            print("\nEvaluating KCR with standard deviation\n")
            KCR_SD_auc.KCR_SD(d, ts, args.file_name, int(args.cv_k), int(args.optimization), int(args.t),d_test, ts_test,args.test_file_name,data_clean=int(args.data_clean),validation = int(args.validation))

            # if classifier is 0 then execute all other classifiers for comparison
            if int(args.classifiers) == 0:

                # Evaluating K Nearest Neighbors Algorithm
                print("\nEvaluating KNN\n")
                KNN.KNN(d, ts, args.file_name, int(args.cv_k), d_test, ts_test, args.test_file_name)

                # Evaluating Nearest Centroid Algorithm
                print("\nEvaluating Nearest Centriod\n")
                NC.NC(d, ts, args.file_name, int(args.cv_k), d_test, ts_test, args.test_file_name)

                # Evaluating Random Forest Algorithm
                print("\nEvaluating Random Forest\n")
                transformer = Normalizer().fit(d)
                d_rf = transformer.transform(d)
                transformer = Normalizer().fit(d_test)
                d_test_rf = transformer.transform(d_test)
                RF.RF(d_rf, ts, args.file_name, int(args.cv_k), d_test_rf, ts_test, args.test_file_name)

                # Evaluating Support Vector machine Algorithm
                print("\nEvaluating SVM\n")
                SVM.SVM(d, ts, args.file_name, int(args.cv_k), d_test, ts_test, args.test_file_name)

                # Evaluating MLP Algorithm
                print("\nEvaluating MLP\n")
                MLP.MLP(d, ts, args.file_name, int(args.cv_k), d_test, ts_test, args.test_file_name)

                # Evaluating  Bayesian Algorithm
                print("\nEvaluating Naive Bayes\n")
                Naive.Naive(d, ts, args.file_name, int(args.cv_k), d_test, ts_test, args.test_file_name)

                # Evaluating Relational Generalized Learning Vector Quantization Classifier
                #print("\nEvaluating Relational Generalized Learning Vector Quantization Classifier\n")
                #RGLVQ_Classifier.RGLVQ_Classifier(d, ts, args.file_name, int(args.cv_k), d_test, ts_test, args.test_file_name)

                # Evaluating Median Generalized Learning Vector Quantization
                #print("\nMedian Generalized Learning Vector Quantization\n")
                #MGLVQ_Classifier.MGLVQ_Classifier(d, ts, args.file_name, int(args.cv_k), d_test, ts_test, args.test_file_name)

        # else is executed when test file doesn't contain class labels
        else:
            print("\nEvaluating KCR\n")
            d_test = loading_test_file_without_label(test_file_name)
            #kcr = KClosestResembler.KClosestResembler(float(args.beta), int(args.num_intervals))
            #kcr.fit(d,ts)
            #prediction = kcr.predict(d_test)
            #print(prediction)
