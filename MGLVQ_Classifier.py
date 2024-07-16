from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from proto_dist_ml.mglvq import MGLVQ
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

import cross_validation

from datetime import date
import time
from sklearn.model_selection import StratifiedKFold


# define the levenshtein distance
def levenshtein(x, y):
    m = len(x)
    n = len(y)
    Delta = np.zeros((m+1, n+1), dtype=int)
    for i in range(m):
        Delta[i+1, 0] = Delta[i, 0] + 1
    for j in range(n):
        Delta[0, j+1] = Delta[0, j] + 1
    for i in range(m):
        for j in range(n):
            delta_ij = 0 if x[i] == y[j] else 1
            Delta[i+1, j+1] = np.min([delta_ij + Delta[i,j], 1 + Delta[i+1, j], 1 + Delta[i, j+1]])
    return Delta[m][n]



def MGLVQ_Classifier(X, y, filename, K, d_test=None, ts_test=None, testfilename=None):
	# loo = LeaveOneOut()
    if (K <= 1):
        kf = LeaveOneOut()
    else:
        kf = StratifiedKFold(n_splits=K)

    accuracies = []

    eva_actual = []
    eva_pred = []
    saved_model = None
    number_of_prototypes = 2
    accuracy = 0



    if d_test is None:
		# for train_index, test_index in loo.split(X):
		#   # Splitting out training and testing data
		#   X_train, X_test = X[train_index], X[test_index]
		#   y_train, y_test = y[train_index], y[test_index]

        train_time = []
        predict_time = []

        for train_index, test_index in kf.split(X,y):
            a = time.time()
			# Splitting out training and testing data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # compute the pairwise Levenshtein distance between all strings
            D = np.zeros((len(X_train), len(X_train)))
            for k in range(len(X_train)):
                k_x = X_train[k]
                for l in range(k + 1, len(X_train)):
                    k_y = X_train[l]
                    # compute the Levenshtein distance between x and y
                    # and store it symmetrically
                    D[k, l] = levenshtein(k_x, k_y)
                    D[l, k] = D[k, l]

            D_test = np.zeros((len(X_test), len(X_train)))
            for k in range(len(X_test)):
                k_x = X_test[k]
                for l in range(len(X_train)):
                    k_y = X_train[l]
                    D_test[k, l] = levenshtein(k_x, k_y)



            clf = MGLVQ(number_of_prototypes)

            clf.fit(D, y_train)
            b = time.time()
            train_time.append(b - a)
            c = time.time()
            y_pred = clf.predict(D_test)
            d = time.time()
            predict_time.append(d - c)

            accuracies.append(metrics.accuracy_score(y_test, y_pred))
            eva_actual = eva_actual + list(y_test)
            eva_pred = eva_pred + list(y_pred)


        cross_validation.evaluation(np.array(eva_actual), np.array(eva_pred), np.unique(y),
									date.today().strftime("%d%m%Y") + '_' + filename + '_MGLVQ_')
        print("Total training time: %f ; Total predicting time: %f" % (
			np.sum(np.array(train_time)), np.sum(np.array(predict_time))))

		# print("\n***** Results *****")
        print("Average accuracy of using MGLVQ on %s: %f" % (filename, np.mean(np.array(accuracies))))
        print("----------------------------------------------------")
    else:
        # compute the pairwise Levenshtein distance between all strings
        D = np.zeros((len(X), len(X)))
        for k in range(len(X)):
            k_x = X[k]
            for l in range(k + 1, len(X)):
                k_y = X[l]
                # compute the Levenshtein distance between x and y
                # and store it symmetrically
                D[k, l] = levenshtein(k_x, k_y)
                D[l, k] = D[k, l]

        D_test = np.zeros((len(d_test), len(X)))
        for k in range(len(d_test)):
            k_x = d_test[k]
            for l in range(len(X)):
                k_y = X[l]
                D_test[k, l] = levenshtein(k_x, k_y)

        clf = MGLVQ(number_of_prototypes)


        train_start = time.time()
        clf.fit(D, y)
        train_stop = time.time()
        predict_start = time.time()
        pred_test = clf.predict(D_test)
        predict_stop = time.time()

        if (len(ts_test) == 0):
            print("the predicted values are :\n")
            for i in range(d_test):
                print(d_test[i] + ":=" + pred_test[i])
            return

        cross_validation.evaluation(ts_test, pred_test, np.unique(y),
									date.today().strftime("%d%m%Y") + '_' + filename + '_' + testfilename + '_MGLVQ_' )
        print("Total training time: %f ; Total predicting time: %f" % ((train_stop - train_start), (predict_stop - predict_start)))
        print("Average accuracy of using MGLVQ on %s: %f" % (filename, metrics.accuracy_score(ts_test, pred_test)))
        print("----------------------------------------------------")
