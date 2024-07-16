import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import random
import numpy as np
import itertools

class KClosestResemblerPercentile:

    def __init__(self, weights = []):
        self.prototypes = []
        self.weights = weights


    # calculate 25% percentile value
    def _percentile_value(self,arr):
        temp_arr = arr.copy()
        temp_arr.sort()

        left_per_index = round(0.1 * temp_arr.shape[0]) - 1
        right_per_index = round(0.9 * temp_arr.shape[0]) - 1

        return float(temp_arr[left_per_index]), float(temp_arr[right_per_index])

    def set_prototypes(self, prototypes):
        self.prototypes = prototypes
    def creating_prototypes(self, classes, d, ts, num_attris):
        prototypes = []
        for c in classes:
            # extract samples of class c
            indices = np.where(ts == c)
            sample_of_c = d[list(indices)]

            prototype_c = []

            for a in range(num_attris):
                # extract data of attribute a of class c
                attr_a_of_c = sample_of_c[:, a]
                interval_l, interval_r = self._percentile_value(attr_a_of_c)
                prototype_c.append([round(interval_l, 6), round(interval_r, 6)])
            prototypes.append(prototype_c)
        return prototypes
    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self, data, targets, classes, num_attris, optimization=0):
        best_weights = [0] * len(classes)
        accuracy_class = [0] * len(classes)
        best_prototype = self.prototypes
        accuracy = 0

        best_eva_actual = []
        best_eva_pred = []


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

    def _absolute_distance(self, interval_l, interval_r, active_item):
        return max([0, interval_l - active_item, active_item - interval_r])

    def _resemblance_degree(self,distance_p1, distance_p2):
        if distance_p1 > distance_p2:
            return 0
        else:
            return 1

    def _outranking_index(self,attri_weights, resemblance_degrees):
        return np.sum(attri_weights * resemblance_degrees)

    # Scoring Function & Assignment Decision
    def _scoring(self, prototype_score, classes):
        max_proto_score_index = [i for i, j in enumerate(prototype_score) if j == max(prototype_score)]

        if len(max_proto_score_index) == 1:
            return classes[max_proto_score_index[0]]
        else:
            non_discrim = []

            for i in max_proto_score_index:
                non_discrim.append(classes[i])
            return non_discrim

    def _testing_model(self, X):
        predicts = []

        for sample in X:
            outrank_index_matrix = []
            prototype_score = []

            for prototype1 in self.prototypes:
                outrank_indices = []
                for prototype2 in self.prototypes:
                    if np.array_equal(prototype1, prototype2):
                        outrank_indices.append(0)
                    else:
                        res_degs = []
                        for m in range(self.num_attris):
                            abs_dis_one = self._absolute_distance(prototype1[m][0], prototype1[m][1], sample[m])
                            abs_dis_two = self._absolute_distance(prototype2[m][0], prototype2[m][1], sample[m])

                            res_deg = self._resemblance_degree(abs_dis_one, abs_dis_two)
                            res_degs.append(res_deg)
                        outrank_index = self._outranking_index(self.weights, np.array(res_degs))
                        outrank_indices.append(outrank_index)

                outrank_index_matrix.append(outrank_indices)

            for i in range(len(self.prototypes)):
                score = 0
                for j in range(len(self.prototypes)):
                    if i == j:
                        continue
                    else:
                        score += outrank_index_matrix[i][j] - outrank_index_matrix[j][i]
                prototype_score.append(score)

            predicts.append(self._scoring(prototype_score, self.classes))

        for i in range(len(predicts)):
            if type(predicts[i]) is list:
                predicts[i] = self.classes[-1]

        return np.array(predicts)


    def fit(self,X,y):


        num_attris = X.shape[1]
        classes = np.unique(y)

        self.classes = classes
        self.num_attris = num_attris

        if (self.prototypes == []):
            prototypes = self.creating_prototypes(classes, X, y, num_attris)
            self.prototypes = prototypes
        if (self.weights == []):
            weights = self.get_weights(X,y,classes, num_attris)
            self.weights = weights


    def predict(self, X):
        return self._testing_model(X)
