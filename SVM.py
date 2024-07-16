from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

import cross_validation

from datetime import date
import time
from sklearn.model_selection import StratifiedKFold


def SVM(X, y, filename, K, d_test=None, ts_test=None, testfilename=None):
	# loo = LeaveOneOut()
	if (K <= 1):
		kf = LeaveOneOut()
	else:
		kf = StratifiedKFold(n_splits=K)

	accuracies = []

	eva_actual = []
	eva_pred = []

	accuracy = 0

	kernel = 'linear'

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

			clf = svm.SVC(kernel=kernel, C=1).fit(X_train, y_train)
			b = time.time()
			train_time.append(b - a)
			c = time.time()

			y_pred = clf.predict(X_test)
			d = time.time()
			predict_time.append(d - c)

			if metrics.accuracy_score(y_test, y_pred) > accuracy:
				accuracy = metrics.accuracy_score(y_test, y_pred)

			accuracies.append(metrics.accuracy_score(y_test, y_pred))
			eva_actual = eva_actual + list(y_test)
			eva_pred = eva_pred + list(y_pred)

		print("\n\n---SVM with kernel used: "+kernel)
		cross_validation.evaluation(np.array(eva_actual), np.array(eva_pred), np.unique(y),
									date.today().strftime("%d%m%Y") + '_' + filename + '_SVM_' + kernel)
		print("Total training time: %f ; Total predicting time: %f" % (
			np.sum(np.array(train_time)), np.sum(np.array(predict_time))))

		# print("\n***** Results *****")
		print("Average accuracy of using SVM on %s: %f" % (filename, np.mean(np.array(accuracies))))
		print("----------------------------------------------------")
	else:
		clf = svm.SVC(kernel=kernel, C=1).fit(X, y)
		train_start = time.time()
		train_stop = time.time()
		predict_start = time.time()
		y_pred = clf.predict(d_test)
		predict_stop = time.time()
		print("\n\n---SVM with kernel used: "+kernel)
		if (len(ts_test) == 0):
			print("the predicted values are :\n")
			for i in range(d_test):
				print(d_test[i] + ":=" + y_pred[i])
			return

		cross_validation.evaluation(ts_test, y_pred, np.unique(y),
									date.today().strftime(
										"%d%m%Y") + '_' + filename + '_' + testfilename + '_SVM_' + kernel)
		print("Total training time: %f ; Total predicting time: %f" % ((train_stop - train_start), (predict_stop - predict_start)))
		print("Average accuracy of using SVM on %s: %f" % (filename, metrics.accuracy_score(ts_test, y_pred)))
		print("----------------------------------------------------")