
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, ClusterMixin

class SNN(BaseEstimator, ClusterMixin):
    """
    Sheared Nearest Neighbor clustering algorithm implementation.

    Source: Ertöz, L., Steinbach, M., & Kumar, V. (2003, May). Finding clusters of different sizes, shapes,
    and densities in noisy, high dimensional data. In Proceedings of the 2003 SIAM international conference
    on data mining (pp. 47-58). Society for Industrial and Applied Mathematics.
    """
    def __init__(self, K, Eps, MinPts_fraction):
        """
        Initialize parameters
        :param K: Number of nearest neighbors
        :param Eps: DBSCAN theshold
        :param MinPts_fraction: fraction of K value (minimum number of links to be core point)
        """
        self.K = K
        self.Eps = Eps
        self.MinPts = int(self.K * MinPts_fraction)  # Parametrization

    def similarity_matrix(self, data):
        """
        Function to compute the similarity matrix: data points are nodes connected by edges, where the edges weight
        belongs to the spatial similarity which is the opposite  to the spatial distance.

        :param data: array of data points (x,y)
        :return: indices of the data points that belong to the similarity matrix
        """
        knn_graph = kneighbors_graph(data, n_neighbors=self.K, include_self=False)
        knn_index = [knn_graph[i].nonzero()[1] for i in range(len(data))]

        return knn_index

    def snn_similarity(self, nn_p1, nn_p2):
        """
        Compute SNN similarity score which correspond to the weight between two data points

        :param nn_p1: data point 1 from the similiarity matrix
        :param nn_p2: data point 2 from the similarity matrix
        :return: weight between the two points
        """

        intersection = set(nn_p1).intersection(nn_p2)
        size_link = len(intersection)
        weight_link = size_link / len(nn_p1)

        return weight_link

    def snn_graph(self, knn_index):
        """
        Function to build the SNN graph which is a matrix that contain the weight between each data point
        within the similarity matrix.

        :param knn_index: indices of the data points that belong to the similarity matrix
        :return: SNN graph, matirx array of weights
        """
        matrix = np.zeros((len(knn_index), len(knn_index)))
        for i in range(len(knn_index)):
            for j in range(len(knn_index)):
                sim = self.snn_similarity(knn_index[i], knn_index[j])
                matrix[i][j] = 1 - sim
        return matrix

    def find_core_points(self, snn_graph):
        """
        DBSCAN searches the core points within the SNN graph.

        :param snn_graph: SNN graph, matirx array of weights
        :return: indices of the core point, cluster label of each data point
        """

        # DBASCAN algorithm
        dbscan = DBSCAN(eps=self.Eps, min_samples=self.MinPts, metric="precomputed")
        dbscan = dbscan.fit(snn_graph)

        core_points = dbscan.components_
        labels = dbscan.labels_  # cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.

        return core_points, labels

    def fit(self, data, y=None, sample_weight=None):
        """
        Complete SNN clustering algorithm

        :param data: array of data points

        :return: self
        """

        # snn clustering algorithm
        sim_matrix_index = self.similarity_matrix(data)
        snn_graph = self.snn_graph(sim_matrix_index)
        self.core_points_, self.labels_ = self.find_core_points(snn_graph)

        return self

    def fit_predict(self, data, y=None, sample_weight=None):
        """
        Prediction of the labels for each data point

        :param data: array of data points

        :return: clustering labels (ints)
        """
        self.fit(data)
        return self.labels_