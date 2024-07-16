

from sklearn.model_selection import train_test_split
import KClosestResembler
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import cross_validation
from scipy import stats
from datetime import date
import time
import random
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def handlingMissingValues(X,y,classes, num_attris, data_clean):



    option = data_clean

    if(option == 1):
        rows,cols = X.shape
        meansArray = []
        class_label_mapper = {}
        index =0
        for i in np.unique(y):
            class_label_mapper[i] = index
            index+=1

        for i in range(len(np.unique(y))):
            meansArray.append([])
            for j in range(X.shape[1]):
                meansArray[i].append(np.nanmean(X[tuple(list(np.where(y == i)))][:, j]))
        for i in range(rows):
            for j in range(cols):
                if(math.isnan(X[i][j])):
                    if(math.isnan(meansArray[class_label_mapper[y[i]]][j])):
                        X[i][j] = 0
                    else:
                        X[i][j] = meansArray[class_label_mapper[y[i]]][j]
        return X

    elif(option == 2):
        X = X[~np.isnan(X).any(axis=1)]
        return X
    elif(option ==3):
        X = X[:,~np.isnan(X).any(axis=0)]
        return X
    else:
        print("Invalid data cleaning option\n")


def auc_score(y_actual,y_pred,classes):
    AUCs = []
    classes = np.unique(y_actual)

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

    return np.mean(AUCs)


def get_best_weights(X, y, num_attris, beta,num_intervals, optimization ):
    classes = np.unique(y)

    aa = time.time()
    kcr = KClosestResembler.KClosestResembler(beta, num_intervals)
    if(optimization == 0):
        return kcr.get_weights(X, y, classes, num_attris, 0)

    kcr.fit(X, y)
    bb = time.time()


    auc =0
    best_weights = []
    for itr in range(optimization + 1):
        print("----------------------------- Optimization " + str(itr) + " -----------------------------------")

        #auc = 0
        eva_actual = []
        eva_pred = []

        weights = []
        #best_weights = []
        weights = kcr.get_weights(X, y, classes, num_attris, itr)
        kcr.set_weights(weights)

        print("Weights:\n" + str(weights))


        loo = LeaveOneOut()

        cc= time.time()
        for train_index, test_index in loo.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_pred = kcr.predict(X_test)


            eva_actual = eva_actual + list(y_test)
            eva_pred = eva_pred + list(y_pred)

        dd = time.time()



        aucs = auc_score(eva_actual, eva_pred, classes)

        if aucs > auc:
            auc = aucs
            best_weights = weights

    return best_weights

def featureSelectionOnWeights(best_weights, X, y):
    beta_threshold = 1/len(best_weights)

    index = []
    for i in range(len(best_weights)):
        if(best_weights[i] < beta_threshold):
            index.append(i)


    return np.delete(X,index,1),np.delete(best_weights,index,0)


def KCR(X, y, filename, K, beta, num_intervals, optimization, d_test=None, ts_test=None, testfilename=None, data_clean = 1,validation =1,feature_selection = 0):

    # loo = LeaveOneOut()
    if (K <= 1):
        kf = LeaveOneOut()
    else:
        kf = StratifiedKFold(n_splits=K)



    num_attris = X.shape[1]
    classes = np.unique(y)

    X = handlingMissingValues(X, y, classes, num_attris , data_clean)
    num_attris = X.shape[1]


    best_weights = get_best_weights(X, y, num_attris, beta,num_intervals, optimization)

    if (feature_selection == 2 and d_test is None):
        X, best_weights = featureSelectionOnWeights(best_weights, X, y)
        print(best_weights)
        num_attris = X.shape[1]


    if d_test is None:
        train_time = []
        validation_time = []
        predict_time = []
        best_eva_actual = []
        best_eva_pred = []
        kcr2 = KClosestResembler.KClosestResembler(beta, num_intervals)
        kcr2.set_weights(best_weights)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            a = time.time()
            kcr2.fit(X_train, y_train)
            b= time.time()

            train_time.append(b-a)

            c = time.time()



            if(validation == 1):
                val_train = np.array([], dtype=np.int64).reshape(0,num_attris)
                val_test = np.array([], dtype=np.int64).reshape(0,1)
                for cls in classes:
                # extract samples of class c
                    indices = np.where(y_train == cls)
                    v_train = X_train[tuple(list(indices))]
                    v_test =  y_train[tuple(list(indices))]

                    if(len(v_train) == 1):
                        val_train = np.vstack([val_train, v_train])
                        val_test = np.append(val_test, v_test)
                    else:
                        subset_data = np.arange(int((v_train.shape[0])/10))
                        if(len(subset_data) <= 0):
                            val_train = np.vstack([val_train, v_train])
                            val_test = np.append(val_test, v_test)
                        else:
                            val_train = np.vstack([val_train,v_train[subset_data]])
                            val_test = np.append(val_test, v_test[subset_data])


                kcr2.predict(val_train, val_test,test='validate')
            d = time.time()
            validation_time.append(d - c)




            e = time.time()
            y_pred = kcr2.predict(X_test)
            f = time.time()
            predict_time.append(f-e)


            best_eva_actual += list(y_test)
            best_eva_pred += list(y_pred)



        print("\n\n---K Closest Resembler--\n\n")
        print("Best Weights:\n"+str(best_weights)+"\n")
        cross_validation.evaluation(np.array(best_eva_actual), np.array(best_eva_pred), classes,date.today().strftime("%d%m%Y") + '_' + filename + '_KCR_')
        print("Total training time: %f ;Total validation time: %f ; Total predicting time: %f" % (np.sum(np.array(train_time)), np.sum(np.array(validation_time)), np.sum(np.array(predict_time))))
                    # print("\n***** Results *****")
        print("Average accuracy of using KCR  on %s: %f" % ( filename, accuracy_score(np.array(best_eva_actual), np.array(best_eva_pred))))
        print("----------------------------------------------------")




    else:
        kcr2 = KClosestResembler.KClosestResembler(beta, num_intervals)
        kcr2.set_weights(best_weights)




        train_start = time.time()
        kcr2.fit(X, y)
        train_stop = time.time()
        predict_start = time.time()
        pred_test = kcr2.predict(d_test)
        predict_stop = time.time()

        if (len(ts_test) == 0):
            print("\n\n---K Closest Resembler--\n\n")
            print("the predicted values are :\n")
            for i in range(d_test):
                print(d_test[i] + ":=" + pred_test[i])
            return


        print("\n\n---K Closest Resembler--\n\n")
        print("Best Weights:\n" + str(best_weights) + "\n")
        cross_validation.evaluation(ts_test, pred_test, np.unique(y), date.today().strftime("%d%m%Y") + '_' + filename + '_' + testfilename + '_KCR_')
        print("Total training time: %f ; Total predicting time: %f" % ((train_stop - train_start), (predict_stop - predict_start)))
        print("Average accuracy of using KCR on %s: %f" % (filename, accuracy_score(ts_test, pred_test)))
        print("----------------------------------------------------")













































