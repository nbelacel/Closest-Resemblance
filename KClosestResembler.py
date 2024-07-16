import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import random
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score

class KClosestResembler:

    # constructor with class variables
    def __init__(self, BetaThreshold = 0.1, NoOfIntervals = 2 , weights = []):
        self.BetaThreshold = BetaThreshold
        self.NoOfIntervals = NoOfIntervals
        self.weights = weights
        self.prototypes = []
        self.fitted_prototypes = []

        self.proto = []
        self.final_proto = []
        self.nprototypes = 0
        self.nprototypes = []
        self.final_proto = []

    # function to create intervals for a specific class.
    def _discretization(self, train, n_clusters):
        '''
        Performing discretization on the training data
        :param train: The training data with shape (x,)
        :param n_clusters: Number of specified clusters for discretization, int
        :return: Intervals for each attribute of each class, list with shape [[], []..., []]
        '''
        if len(train) == 0:
            print("No training data for a class, Ensure alteast one record for a class in training\n")
            exit(0)

        if len(train) < n_clusters:
            mini = min(train)
            maxi = max(train)
            intervals = []
            intervals.append([mini, maxi])
            # for i in range(n_clusters):
            #     intervals.append([mini, maxi])
            return intervals

        featuremin = min(train)
        featuremax = max(train)

        kmeans = KMeans(n_clusters=n_clusters).fit(train.reshape(-1, 1))

        # exit(0)

        intervals = []

        for c in np.unique(np.array(kmeans.labels_)):
            indices = [i for i, j in enumerate(kmeans.labels_) if j == c]
            c_train = np.array(train)[indices]
            # print("values inside the cluster",c_train)
            # print("the stdev of values within the clusters", statistics.stdev(c_train))
            fmin = min(c_train)
            fmax = max(c_train)
            intervals.append([fmin, fmax])

        if len(intervals) != n_clusters:
            for i in range((n_clusters - len(intervals))):
                intervals.append(intervals[-1])

        return intervals


    # setter to set weight
    def set_weights(self,weights):
        self.weights = weights

    # getter to get weight based on optimization value
    def get_weights(self, data, targets, classes, num_attris, optimization=0 ):

        best_weights = [0] * len(classes)
        accuracy = 0

        best_eva_actual = []
        best_eva_pred = []

        accuracy_class = [0] * len(classes)

        train_time = []
        predict_time = []

        weights = []
        if (optimization == 0):
            weight = []
            for i in range(num_attris):
                weight.append(1 / num_attris)

            weights = np.array(weight)

        elif (optimization >= 1):
            weight = []
            for i in range(num_attris):
                weight.append(random.randint(1, num_attris))
            value = sum(weight)
            weightt = []
            for i in range(num_attris):
                weightt.append(weight[i] / value)

            weights = np.array(weightt)

        else:
            print('enter valid number iterations for weights')

        return weights



    def _fdistance(self, left, right, value):
        if left <= value <= right:
            fd = 0
        elif value < left:
            fd = left - value
        else:
            fd = value - right
        return fd



    def _predict_buffer(self,prototypes, sample, attr_weights, num_attris):

        performancematrix=[]
        l = len(sample)
        nproto=len(prototypes)
        for i in prototypes:
            for o in range(l):
                k, j = i[o]
                m = sample[o]
                fd = self._fdistance(k, j, m)
                performancematrix.append(fd)
        #print("performance matrix for ",nproto," prototypes",performancematrix)
        similaritymatrix=[0]*(nproto*nproto)


        for i in range(l):
            for j in range(nproto):
                for k in range(nproto):
                    if k > j:
                        if performancematrix[(j*l)+i] < performancematrix[(k*l)+i]:
                            similaritymatrix[(j*nproto)+k]+= attr_weights[i]*1
                            similaritymatrix[(k * nproto) + j] += 0
                        elif performancematrix[(j*l)+i] == performancematrix[(k*l)+i]:
                            pass
                        else:
                            similaritymatrix[(j * nproto) + k] += 0
                            similaritymatrix[(k * nproto) + j] += attr_weights[i] * 1
                    elif k == j:
                        similaritymatrix[(j * nproto) + k] = 1

        #print(similaritymatrix)

        pfFlux = [0]*nproto

        for i in range(nproto):
            fFluxOut = 0
            fFluxIn = 0
            for j in range(nproto):
                fFluxOut += similaritymatrix[(i * nproto) + j]
            for j in range(nproto):
                fFluxIn += similaritymatrix[(j * nproto) + i]

            pfFlux[i] = fFluxOut - fFluxIn

        #print(pfFlux)
        max_index = pfFlux.index(max(pfFlux))
        #print("index",str(max_index))
        #print(prototypes[max_index])
        return prototypes[max_index][-1]

    def _validate_buffer(self,prototypes, sample, attr_weights, num_attris):

        performancematrix=[]
        l = sample.shape[0]
        nproto=len(prototypes)
        for i in prototypes:
            for o in range(l):
                k, j = i[o]
                m = sample[o]
                fd = self._fdistance(k, j, m)
                performancematrix.append(fd)
        #print("performance matrix for ",nproto," prototypes",performancematrix)
        similaritymatrix=[0]*(nproto*nproto)


        for i in range(l):
            for j in range(nproto):
                for k in range(nproto):
                    if k > j:
                        if performancematrix[(j*l)+i] < performancematrix[(k*l)+i]:
                            similaritymatrix[(j*nproto)+k]+= attr_weights[i]*1
                            similaritymatrix[(k * nproto) + j] += 0
                        elif performancematrix[(j*l)+i] == performancematrix[(k*l)+i]:
                            pass
                        else:
                            similaritymatrix[(j * nproto) + k] += 0
                            similaritymatrix[(k * nproto) + j] += attr_weights[i] * 1
                    elif k == j:
                        similaritymatrix[(j * nproto) + k] = 1

        #print(similaritymatrix)

        pfFlux = [0]*nproto

        for i in range(nproto):
            fFluxOut = 0
            fFluxIn = 0
            for j in range(nproto):
                fFluxOut += similaritymatrix[(i * nproto) + j]
            for j in range(nproto):
                fFluxIn += similaritymatrix[(j * nproto) + i]

            pfFlux[i] = fFluxOut - fFluxIn

        #print(pfFlux)
        max_index = pfFlux.index(max(pfFlux))
        #print("index",str(max_index))
        #print(prototypes[max_index])
        pfFlux = np.array(pfFlux)

        return prototypes[max_index][-1], pfFlux


    def _filterBestIntervals(self,records, prototypes):

        thres = 0.1
        prototypes.sort(key = lambda x: x[0])

        number_of_intervals = len(prototypes)
        count = dict.fromkeys(range(number_of_intervals),0)

        for i in records:
            for j in range(len(prototypes)):
                if(prototypes[j][0]<=i and prototypes[j][-1]>=i):
                    count[j]+=1


        count = {k: v / total for total in (sum(count.values()),) for k, v in count.items()}

        new_prototypes=[]

        new_prototypes.append(prototypes[0])
        for i in range(1,len(prototypes)):
            if(count[i]>=thres):
                new_prototypes.append(prototypes[i])
            else:
                new_prototypes[len(new_prototypes)-1][-1] = prototypes[i][-1]

        prototypes = new_prototypes

        return prototypes

    # _creating_intervals creates intervals for all classes
    def _creating_intervals(self,c, sample_of_c, ts, num_attris, num_intervals=2):
        prototype_c = []

        # learning the interval for each attribute a of class c
        bestIntervalStatus = 1
        for a in range(num_attris):
            # extract data of attribute a of class c
            attr_a_of_c = sample_of_c[:, a]
            preferred_intervals = self._discretization(attr_a_of_c, num_intervals)

            if(bestIntervalStatus == 1):
                preferred_intervals = self._filterBestIntervals(attr_a_of_c, preferred_intervals)
                num_intervals = len(preferred_intervals)
                bestIntervalStatus = 0

            prototype_c.append(preferred_intervals)


        return np.array(prototype_c)


    # _creating_prototypes creates prototypes
    def _creating_prototypes(self,pfcurrentdata,ncurrentcluster):
        intervals_c = self.prototypes
        threshold = self.BetaThreshold

        self.final_proto = []
        self.proto = []
        nsize = len(pfcurrentdata)
        nvaluesperrecord = pfcurrentdata.shape[1]
        nDivisor = nsize
        pbpopulation = np.array([1] * nsize)
        nkmeanscluster = intervals_c.shape[1]
        pbglobalhistory = np.array([0] * (nkmeanscluster * nvaluesperrecord))
        npriorprototypes = self.nprototypes

        for i in range(nkmeanscluster):
            self._newcreateprototypeinternal(pbpopulation, pbglobalhistory, pfcurrentdata, 0, i, threshold, ncurrentcluster,
                                       nsize, nDivisor, intervals_c)

        pbpopulation = []
        pbglobalhistory = []

        if len(self.final_proto) == 0:
            print("try with different threshold value")
            exit(0)
        return self.final_proto

    def _newcreateprototypeinternal(self, pbpopulation, pbglobalhistory, pfcurrentdata, nlevel, npos, threshold,
                                   ncurrentcluster, nsize, ndivisor, intervals):
        ncount = 0;
        nkmeanscluster = intervals.shape[1]
        nvaluesperrecord = pfcurrentdata.shape[1]
        pbinternalpopulation = np.copy(pbpopulation)
        pbglobalhistory[(nlevel * nkmeanscluster) + npos] = 1
        for i in range(nsize):
            if pbinternalpopulation[i] == 1:
                linterval, rinterval = intervals[nlevel][npos][:2]
                value = pfcurrentdata[i][nlevel]
                if value >= linterval and value <= rinterval:
                    ncount += 1
                else:
                    pbinternalpopulation[i] = 0

        if float(ncount) / float(ndivisor) >= float(threshold):
            if (nlevel == (nvaluesperrecord - 1)):
                for j in range(nvaluesperrecord):
                    for k in range(nkmeanscluster):
                        if pbglobalhistory[(j * nkmeanscluster) + k] == 1:
                            self.proto.append(list(intervals[j][k]))
                self.proto.append(ncurrentcluster)
                self.final_proto.append(list(self.proto))
                self.proto = []
            else:
                for j in range(nkmeanscluster):
                    self._newcreateprototypeinternal(pbinternalpopulation, pbglobalhistory, pfcurrentdata, nlevel + 1, j,
                                               threshold, ncurrentcluster, nsize, ncount, intervals)

        for i in range(nkmeanscluster):
            pbglobalhistory[(nlevel * nkmeanscluster) + i] = 0
        pbinternalpopulation = []

    # fit function trains the model
    def fit(self, X , y):

        fitted_prototypes = []
        num_attris = X.shape[1]
        classes = np.unique(y)
        self.classes = classes
        if (self.weights == []):
            self.weights = self.get_weights(X, y, classes, num_attris, 0)

        for i in classes:
            indices = np.where(y == i)
            sample_of_c = X[tuple(list(indices))]
            targets = [i] * len(sample_of_c)
            self.prototypes = self._creating_intervals(i, sample_of_c, targets, num_attris,
                                                       num_intervals=int(self.NoOfIntervals))
            fitted_prototypes.append(self._creating_prototypes(sample_of_c, i))

        proto = []

        for i in fitted_prototypes:
            for j in i:
                proto.append(j)

        self.fitted_prototypes = np.array(proto)

    def _auc_score(self, y_actual, y_pred, classes):
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



    # predict will tests/validate the data
    def predict(self, X, y=None, test='test'):
        '''
                Performing prediction
                :param data: Data currently working on, 2 dimensional numpy array with shape (x, y)
                :param attr_weights: weights of each attribute, []
                :param prototypes: prototypes in list of lists of lists, has shape
                                 [
                                  [
                                   [],[],...,[], x
                                  ],
                                  [
                                   [],[],...,[], y
                                  ],
                                  ...,
                                  [
                                   [],[],...,[], z
                                  ],
                                 ]
                :param num_attris: number of attributes, int
                :return: list of predictions, []
                '''

        if(test == 'test'):
            predicts = np.zeros(X.shape[0])

            for i in range(X.shape[0]):
                predict = self._predict_buffer(self.fitted_prototypes, X[i], self.weights, X.shape[1])

                if predict != 0:
                    predicts[i] = predict

            # if mode is 0:
            #     if predict == targets[i]:
            #         print("The sample was correctly classified to class %d." % predict)
            #     else:
            #         print("The sample was classified to class %d while it should belongs to class %d" % (predict, targets[i]))

            return predicts
        else:
            prototype_threshold = 0.3
            predicts = np.zeros(X.shape[0])
            temp_fitted_prototypes = self.fitted_prototypes

            for i in range(X.shape[0]):
                predict, flows = self._validate_buffer(temp_fitted_prototypes, X[i], self.weights, X.shape[1])
                number_of_classes = len(self.classes)

                y_actual = y[i]
                temp_fitted_prototypes_classes = [sub[-1] for sub in temp_fitted_prototypes]
                temp_flows = flows[np.where(temp_fitted_prototypes_classes == y_actual)]

                norm  = np.linalg.norm(temp_flows)
                temp_flows = temp_flows/norm

                temp_flows_with_threshold = temp_flows[np.where(temp_flows>= prototype_threshold)]


                if(temp_flows_with_threshold.shape[0] >= 1):
                    new_temp_fitted_prototypes = []
                    good_proto_index=0
                    for proto in range(temp_fitted_prototypes.shape[0]):
                        if(temp_fitted_prototypes[proto][-1] == y_actual):
                            if(temp_flows[good_proto_index]>= prototype_threshold):
                                new_temp_fitted_prototypes.append(temp_fitted_prototypes[proto])
                            good_proto_index+=1
                        else:
                            new_temp_fitted_prototypes.append(temp_fitted_prototypes[proto])



                    if (len(set([item[-1] for item in new_temp_fitted_prototypes])) != number_of_classes):
                        continue

                    temp_fitted_prototypes = np.array(new_temp_fitted_prototypes)


                if predict != 0:
                    predicts[i] = predict



            standard_fitted_prototype = self.fitted_prototypes
            predicts_standard_prototype = self.predict(X)
            self.fitted_prototypes = temp_fitted_prototypes
            predicts_new_prototype = self.predict(X)

            standard_auc = self._auc_score(y,predicts_standard_prototype,np.unique(y))
            new_auc = self._auc_score(y,predicts_new_prototype,np.unique(y))

            if(standard_auc > new_auc):
                self.fitted_prototypes = standard_fitted_prototype





