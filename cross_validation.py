import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import scikitplot as skplt
import matplotlib.pyplot as plt

from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr

from sklearn.ensemble import ExtraTreesClassifier



# Evaluation is used in every classifier to calculate different metrics of classifiers like AUC, Accuracy, ROC etc.
def evaluation(y_actual, y_pred, classes, clf_name):
    '''
    Evaluation method. Calculating TP, FP, TN, FN and outputing a report consists confusino matrix, accuracy, presion, recall and f1-measure.
    :param y_actual: a list of the actual label of each sample
    :param y_pred: a list of the predicted label of each sample
    :param classes: unique labels within targets
    '''

    print("\n***** Confusion Matrix *****")
    cnf_matrix = confusion_matrix(y_actual, y_pred)
    print(cnf_matrix)

    dict = {}
    # for i in range(len(classes)):
    #     dict[i] = "Class "+str(i+1)
    for i in classes:
        dict[i] = "Class "+str(i)

    #dict = {0:'asphalt', 1:'building', 2:'car', 3:'concrete', 4:'grass', 5:'pool', 6:'shadow', 7:'soil', 8:'tree'}
    #dict = {0:'class 1', 1:'class 2', 2:'class 3',3:'class 4', 4:'class 5', 5:'class 6',6:'class 7', 7:'class 8', 8:'class 9'}
    plot_confusion_matrix(y_actual, y_pred, clf_name, classes, ymap=dict, figsize=(10, 10))

    print("\n***** TP, FP, TN, FN *****")
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    for i in range(len(classes)):
        print("class " + str(i) + ": " + "TP: %d, FP: %d, TN: %d, FN: %d" % (TP[i], FP[i], TN[i], FN[i]))
        # print("TP: %d, FP: %d, TN: %d, FN: %d" % (TP[i], FP[i], TN[i], FN[i]))

    print("\n***** Precision, Recall, F1-score *****")
    print(classification_report(y_actual, y_pred, labels=classes))

    print("***** AUC, ROC *****")
    new_y_pred = []

    dict2 = {}
    k = 0
    for i in classes:
        dict2[i] = k
        k += 1

    for pred in y_pred:
        temp = [0] * len(classes)
        temp[dict2[int(pred)]] = 1
        new_y_pred.append(temp)

    skplt.metrics.plot_roc_curve(np.array(y_actual), np.array(new_y_pred))





    directory = './ROC/' + clf_name + '.png'
    plt.savefig(directory)

    y_pred.reshape(len(y_pred), 1)

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
        print('AUC: %.2f' % auc)
    print('Average AUC: %.3f' % np.mean(np.array(AUCs)))


        # fpr, tpr, thresholds = roc_curve(y_actual_temp, y_pred_temp)
        # plot_roc_curve(fpr, tpr, i, clf_name)


# def plot_roc_curve(fpr, tpr, i, clf_name):
#     fig = plt.figure()
#     plt.plot(fpr, tpr, color='orange', label='ROC')
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     fig.savefig('./ROC_figs/' + clf_name + '_' + str(i) + '.png', dpi=fig.dpi)
#     fig.clf()


# calculate 25% percentile value
def percentile_value(arr):
    temp_arr = arr.copy()
    temp_arr.sort()

    left_per_index = round(0.1 * temp_arr.shape[0]) - 1
    right_per_index = round(0.9 * temp_arr.shape[0]) - 1

    return float(temp_arr[left_per_index]), float(temp_arr[right_per_index])


# plot_confusion_matrix constructs confusio matrix for classifiers
def plot_confusion_matrix(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap=plt.cm.Oranges)
    directory = './CM/' + filename + '.png'
    plt.savefig(directory)
    # plt.show()
    plt.clf()