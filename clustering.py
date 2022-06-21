import copy
import math
import sys
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix

def euclidean_distance(v1, v2):
    temp = 0
    for i in range(len(v1)):
        temp += (v1[i] - v2[i])**2

    return math.sqrt(temp)


class HierarchicalClustering:

    def __init__(self):
        self.clusters = []

    def fit(self, X, y, clusters_nr):
        self.clusters = [[X[i].tolist() + [f"{y[i]}"]] for i in range(len(X))]
        while len(self.clusters) > clusters_nr:
            self.clusters = self._fit_iter(self.clusters)

    def _fit_iter(self, clusters):
        avg = [self._cluster_avrg(c) for c in clusters]
        conf_matr = distance_matrix(avg, avg)
        min_ = sys.maxsize
        min_i = -1
        min_j = -1

        for i in range(len(conf_matr)):
            for j in range(len(conf_matr[i])):
                if i != j and i > j:
                    dist = conf_matr[i][j]
                    if dist < min_:
                        min_ = dist
                        min_i = i
                        min_j = j
                else:
                    break

        clusters.append(clusters[min_i] + clusters[min_j])
        clusters.pop(min_i)
        clusters.pop(min_j)
#
        return clusters

    @staticmethod
    def _cluster_avrg(cluster):
        res = [0 for i in range(len(cluster[0]) - 1)]

        for features in cluster:
            for i in range(len(features)):
                if features[i] != features[-1]:
                    res[i] += features[i]
        return [s/len(cluster) for s in res]

    @staticmethod
    def count_classes(cluster):
        count = {}
        for feaures in cluster:
            if feaures[-1] not in count:
                count[feaures[-1]] = 1
            else:
                count[feaures[-1]] += 1
        return count

    def predict(self, X):
        predictions = []
        for x in X:
            min_ = sys.maxsize
            min_i = -1
            for i in range(len(self.clusters)):
                dist = euclidean_distance(x, self._cluster_avrg(self.clusters[i]))
                if dist < min_:
                    min_i = i
                    min_ = dist
            predictions.append(min_i)
        return predictions


if __name__ == '__main__':
    data = load_breast_cancer()

    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = HierarchicalClustering()
    model.fit(X_train, y_train, 2)
    print("\n")
    for i in range(len(model.clusters)):
        print(f"number of class instances in cluster {i} = {model.count_classes(model.clusters[i])}")

    print(f"\ntest set instances cluster affiliation:\n{model.predict(X_test)}")
    print(f"test set instances groundtruth:\n{y_test.tolist()}")
