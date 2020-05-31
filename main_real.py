
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score, fowlkes_mallows_score, silhouette_score
from pyclustering.cluster.cure import cure;
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from SNN_clustering_algorithm import SNN
from sklearn.metrics import classification_report

def main():

    """Comaprison between K-means, Spectral Clustering, CURE, DBSCAN, OPTICS and SNN clustering algorithms
    in small and medium real world data sets. Comparison with K-means and SNN clustering algorithm in large
    real world data set.

          """
    # ===============================
    # SMALL AND MEDIUM REAL DATA SETS
    # ===============================
    from sklearn import datasets
    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'eps': .5,
                    'MinPts_fraction': 0.5,
                    'n_neighbors': 20,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1,
                    'width': 2.5,
                    'height': 2.5}

    # Small and medium world real datasets
    iris = datasets.load_iris(return_X_y=True)
    breast_cancer = datasets.load_breast_cancer(return_X_y=True)

    datasets = [
        (iris, {'name': 'iris', 'n_clusters': 3, 'd_eps': 0.8, 'coord_x': 2, 'coord_y': 1,
                'n_neighbors': 30, 'eps': 0.35, 'MinPts_fraction': 0.5}),

        (breast_cancer, {'name': 'breast_cancer', 'n_clusters': 2, 'd_eps': 2, 'coord_x': 2, 'coord_y': 3,
                         'n_neighbors': 60, 'eps': 0.5, 'MinPts_fraction': 0.5}),
    ]

    snn_parameters = []
    results = []
    total_ypred = []
    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        snn_parameters.append([params['name'], params['n_neighbors'], params['eps'], params['MinPts_fraction']])

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster algorithms
        # ============

        k_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['d_eps'])
        optics = cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'],
                                min_cluster_size=params['min_cluster_size'])
        snn = SNN(K=params['n_neighbors'], Eps=params['eps'], MinPts_fraction=0.5)

        clustering_algorithms = (
            ('Original', None),
            ('K_means', k_means),
            ('SpectralClustering', spectral),
            ('CURE', cure),
            ('DBSCAN', dbscan),
            ('OPTICS', optics),
            ('SNN', snn)

        )

        for name, algorithm in clustering_algorithms:

            if name == 'CURE':
                cure_inst = algorithm(X, params['n_clusters']);
                cure_inst.process();
                clusters = cure_inst.get_clusters();
                y_pred = [0] * len(X)
                for i in range(len(clusters)):
                    cluster_cure = clusters[i]
                    for index in cluster_cure:
                        y_pred[index] = i
            elif name == 'Original':
                y_pred = y

            else:
                algorithm.fit(X)
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

            total_ypred.append(y_pred)

            mutual_info = None
            rand_index = None
            fowlkes_mallows = None
            calinski_score = None
            davies_bouldin = None
            silhouette = None

            if len(np.unique(y_pred)) > 1 and len(np.unique(y)) > 1:
                # External indices:
                mutual_info = round(adjusted_mutual_info_score(y, y_pred, average_method='arithmetic'), 3)
                rand_index = round(adjusted_rand_score(y, y_pred), 3)
                fowlkes_mallows = round(fowlkes_mallows_score(y, y_pred), 3)

                # Internal indexes
                calinski_score = round(calinski_harabaz_score(X, y_pred), 3)
                davies_bouldin = round(davies_bouldin_score(X, y_pred), 3)
                silhouette = round(silhouette_score(X, y_pred), 3)

            results.append([params['name'], name, mutual_info, rand_index, fowlkes_mallows, calinski_score, davies_bouldin,
                            silhouette])

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, params['coord_x']], X[:, params['coord_y']], s=10, color=colors[y_pred])

            plt.xlim(-params['width'], params['width'])
            plt.ylim(-params['height'], params['height'])
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

    outputfile = "./results/real_datasets_sklearn_metrics"
    results_df = pd.DataFrame(results, columns=['Dataset', 'Algorithm', 'AMI', 'ARI', 'FM',
                                                'CHI', 'DBI', 'Silhouette'])
    results_df.to_csv(outputfile + '.csv', index=False, header=True)
    results_df.to_excel(outputfile + '.xlsx', index=False, header=True)

    plt.savefig('./results/real_datasets_sklearn.png')
    plt.show()

    # ===============================
    # LARGE REAL DATA SET
    # ===============================

    def correct_detections(y_pred):
        """Print correct de"""
        dos_cor, normal_cor, probe_cor, r2l_cor, u2r_cor = 0, 0, 0, 0, 0

        for val in y_pred[0:999]:
            if val == 0:
                dos_cor += 1
        for val in y_pred[1000:1999]:
            if val == 1:
                normal_cor += 1
        for val in y_pred[2000:2999]:
            if val == 2:
                probe_cor += 1
        for val in y_pred[3000:3999]:
            if val == 3:
                r2l_cor += 1
        for val in y_pred[4000:4999]:
            if val == 4:
                u2r_cor += 1

        print(dos_cor, normal_cor, probe_cor, r2l_cor, u2r_cor)


    # ===== K means clustering ======

    pd_dataset = pd.read_csv('./csv_files/KDD.csv')
    X = pd_dataset.iloc[:, :-1].to_numpy()
    y = pd_dataset.iloc[:, -1].to_numpy()


    k_means = cluster.MiniBatchKMeans(n_clusters=5,random_state=42)
    k_means.fit(X)
    if hasattr(k_means, 'labels_'):
        y_pred_kmeans =k_means.labels_.astype(np.int)
    else:
        y_pred_kmeans = k_means.predict(X)

    # Count detections per cluster
    unique, counts = np.unique(y_pred_kmeans, return_counts=True)
    print(dict(zip(unique, counts)))

    # Evaluation
    print(classification_report(y, y_pred_kmeans))
    correct_detections(y_pred_kmeans)

    # ===== SNN clustering ======

    snn = SNN(K=300,Eps=0.4,MinPts_fraction=0.5)
    snn.fit(X)
    if hasattr(snn, 'labels_'):
        y_pred = snn.labels_.astype(np.int)
    else:
        y_pred = snn.predict(X)

    # Count detections per cluster
    unique, counts = np.unique(y_pred, return_counts=True)
    print(dict(zip(unique, counts)))

    # Evaluation
    print(classification_report(y, y_pred))
    correct_detections(y_pred)


    # External and Internal indices evaluation :
    results = []
    total_ypred = [('Original', y), ('k-means', y_pred_kmeans), ('SNN', y_pred)]
    for name, y_pred in total_ypred:
        silhouette = round(silhouette_score(X, y_pred), 3)

        results.append(
            ['KDD CUP 99', name,silhouette])

        outputfile = "./results/real_datasets_KDD_CUP_metrics"
        results_df = pd.DataFrame(results, columns=['Dataset', 'Algorithm', 'Silhouette'])
        results_df.to_csv(outputfile + '.csv', index=False, header=True)
        results_df.to_excel(outputfile + '.xlsx', index=False, header=True)

if __name__ == "__main__":
    main()