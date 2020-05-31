
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score, fowlkes_mallows_score, silhouette_score
from pyclustering.cluster.cure import cure;
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from SNN_clustering_algorithm import SNN

def main():

    """Comaprison between K-means, Spectral Clustering, CURE, DBSCAN, OPTICS and SNN clustering algorithms
    in artificial data sets. Choose between trhee options of data sets (comment or undo the comment for the
      desired set of data sets):

      1) Small artificial datasets
      2) Complex artificial datasets
      3) Varying densities artificial datasets

      """

    # ===========================
    # === ARTIFICIAL DATASETS ===
    # ===========================
    from sklearn import cluster, datasets
    np.random.seed(0)
    # ============

    # Generate datasets (taken from SKLEARN example)
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    # ============
    # Set up cluster parameters
    # ============
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
                    'min_cluster_size': 0.1}

    # ========= Small artificial datasets ===============
    datasets = [
        (noisy_circles,
         {'name': 'noisy_circles', 'quantile': .2, 'n_clusters': 2, 'min_samples': 20, 'xi': 0.25, 'eps': 0.5,
          'd_eps': 0.15}),
        (noisy_moons, {'name': 'noisy_moons', 'n_clusters': 2, 'd_eps': 0.3}),
        (varied, {'name': 'varied', 'eps': .5, 'd_eps': 0.18, 'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'name': 'aniso', 'eps': .5, 'd_eps': 0.15, 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {'name': 'blobs', 'd_eps': 0.3}),
        (no_structure, {'name': 'no_structure', 'd_eps': 0.15})]

    # ========= Complex shape artificial data sets =========

    # datasets = [
    #     (None, {'name': 'complex9', 'n_clusters': 9, 'n_neighbors': 40, 'eps': 0.45, 'd_eps': 0.15, 'MinPts_fraction': 0.5}),
    #     (None,{'name': 'cure-t0-2000n-2D', 'n_clusters': 3, 'n_neighbors': 35, 'eps': 0.45, 'd_eps': 0.15,
    #      'MinPts_fraction': 0.4}),
    #     (None, {'name': 'cure-t1-2000n-2D', 'n_clusters': 6, 'n_neighbors': 35, 'eps': 0.40, 'd_eps': 0.15, 'xi': 0.035,
    #      'MinPts_fraction': 0.4}),
    #     (None,{'name': '3-spiral', 'n_clusters': 3, 'n_neighbors': 10, 'eps': 0.45, 'd_eps': 0.15, 'MinPts_fraction': 0.35})]

    # ======== Varying densities artificial data sets======

    # datasets = [
    #     (None,{'name': 'triangle1', 'n_clusters': 4, 'n_neighbors': 50, 'eps': 0.5, 'd_eps': 0.15, 'MinPts_fraction': 0.5}),
    #     (None,{'name': 'triangle2', 'n_clusters': 4, 'n_neighbors': 50, 'eps': 0.5, 'd_eps': 0.15, 'MinPts_fraction': 0.5}),
    #     (None,{'name': 'st900', 'n_clusters': 9, 'n_neighbors': 50, 'eps': 0.4, 'd_eps': 0.15, 'MinPts_fraction': 0.5}),
    #     (None,{'name': 'compound', 'n_clusters': 6, 'n_neighbors': 25, 'eps': 0.4, 'd_eps': 0.15, 'MinPts_fraction': 0.5})]



    results = []
    for i_dataset, (dataset, algo_params) in enumerate(datasets):

        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        if dataset == None:
            name = params['name']
            pd_dataset = pd.read_csv('./csv_files/' + name + '.csv')
            X = pd_dataset.iloc[:, :-1].to_numpy()
            y = pd_dataset.iloc[:, -1].to_numpy()
        else:

            X, y = dataset


        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster objects
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
            ('k_means', k_means),
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
            else:
                algorithm.fit(X)
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

            # EVALUATION
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

            # Plot the results
            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))

            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

    outputfile = "artificial_datasets_sklearn"
    #outputfile = "artificial_datasets"
    #outputfile = "artificial_datasets_densities"

    # Save evaluation metrics:
    results_df = pd.DataFrame(results, columns=['Dataset', 'Algorithm', 'AMI', 'ARI', 'FM',
                                                'CHI', 'DBI', 'Silhouette'])
    results_df.to_csv('./results/'+outputfile + '_metrics.csv', index=False, header=True)
    results_df.to_excel('./results/'+outputfile + '_metrics.xlsx', index=False, header=True)

    # Save plots:
    plt.savefig('./results/' + outputfile +'.png')
    plt.show()



if __name__ == "__main__":
    main()