from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score


def read_csv():
    fname = "pulsar.csv"
    names = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]
    dataset = pd.read_csv(fname, names=names)

    x = dataset.loc[:, names[:-1]].values
    y = dataset.loc[:, ['V9']].values
    return x, y


def outliers(x, y, ignore_pulsars=False):
    mean = np.mean(x, axis=0)
    stdev = np.std(x, axis=0)
    ind = np.unique(np.where(np.abs(x-mean) > 3 * stdev)[0])
    #  >these three lines are to preserve all pulsars, because it deletes half of them
    if ignore_pulsars:
        pind = np.unique(np.where(y == 1)[0])
        iind = np.unique(np.where(np.isin(ind, pind)))
        ind = np.delete(ind, iind)
    #  >end
    print("outliers: %d" % len(ind))
    x = np.delete(x, ind, axis=0)
    y = np.delete(y, ind, axis=0)
    return x, y


def reduce_dim(x, dim=3):
    pca = PCA(n_components=dim)
    components = pca.fit_transform(x)
    reduced = pd.DataFrame(data=components, columns=['dim' + str(i+1) for i in range(dim)])
    return reduced.values


def clusterization(X, n_clusters, type="agglomerative", ommit_last_class=False):
    cluster = None
    if type == "agglomerative":
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='minkowski', linkage='complete')
    elif type == "kmeans":
        cluster = KMeans(n_clusters)
    elif type == "DBSCAN":
        cluster = DBSCAN(eps=0.45)
    elif type == "HDBSCAN":
        cluster = HDBSCAN(min_cluster_size=450, min_samples=1, metric='manhattan')

    cluster.fit_predict(X)
    cluster.labels_ -= np.min(cluster.labels_)
    if ommit_last_class:
        cluster.labels_[np.where(cluster.labels_ == 3)] = 2
    num = len(np.unique(cluster.labels_))
    return num, cluster.labels_


def classification(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    cl = RandomForestClassifier()
    cl.fit(X_train, y_train)
    return cl


def plot3D(x, y, size=0.2, categories=2, labels=('not pulsars', 'pulsars'), minc=0):
    fig = plt.figure()
    ax = Axes3D(fig)
    names = ['x', 'y', 'z']

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])

    for i in range(categories):
        d = x[np.where(y == (i+minc))[0], :]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=size, label=labels[i])
    plt.legend(loc='best')
    plt.show()


def plot2D(x, y, size=0.2, categories=2, labels=('not pulsars', 'pulsars'), minc=0):
    plt.xlabel('x')
    plt.ylabel('y')

    for i in range(categories):
        d = x[np.where(y == (i+minc))[0], :]
        plt.scatter(d[:, 0], d[:, 1], s=size, label=labels[i])

    plt.legend(loc='best')
    plt.show()


def run():
    X, Y = read_csv()
    print("number of records: ", len(Y))
    print("pulsars before outlier detection: ", len(np.where(Y == 1)[0]))

    X, Y = outliers(X, Y, False)
    print("reduced number of records: ", len(Y))
    print("pulsars after outlier detection: ", len(np.where(Y == 1)[0]))

    category = 0  # 1 pulsars, 0 not pulsars
    XX = X[np.where(Y == category)[0], :]
    y = Y[np.where(Y == category)[0], :]

    x = XX[:, [0, 1]]  # [0, 1], [2, 5]
    # [0,4], [3, 6], [0, 5]
    #xx = reduce_dim(x, 1)
    #x = reduce_dim(x, 3)
    num_of_clusters = 2

    num_of_clusters, y = clusterization(x, num_of_clusters, "HDBSCAN", True)
    colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']

    if len(x[0]) == 3:
        plot3D(x, y, 0.5, num_of_clusters, ['c' + str(i+1) for i in range(num_of_clusters)], np.min(y))
    else:
        plot2D(x, y, 0.5, num_of_clusters, ['c' + str(i+1) for i in range(num_of_clusters)], np.min(y))


if __name__ == "__main__":
    run()
