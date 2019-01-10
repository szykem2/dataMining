import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hdbscan import HDBSCAN
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def read_csv():
    fname = "pulsar_stars.csv"
    dataset = pd.read_csv(fname)
    names = list(dataset.columns.values)

    x = dataset.loc[:, names[:-1]].values
    y = dataset.iloc[:, -1].values
    return x, y, dataset


def outliers(x, y, ignore_pulsars=False):  # TODO: other methods
    mean = np.mean(x, axis=0)
    stdev = np.std(x, axis=0)
    ind = np.unique(np.where(np.abs(x - mean) > 3 * stdev)[0])
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
    reduced = pd.DataFrame(data=components, columns=['dim' + str(i + 1) for i in range(dim)])
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


def plot_data_summary(data):
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(data.describe()[1:].transpose(),
                annot=True, linewidth=2,
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75))
    plt.title("Data summary")
    fig.tight_layout()
    plt.show()
    return


def plot_data_correlation(data):
    correlation = data.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation,
                annot=True, linewidth=2,
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75))
    plt.title("Correlation between variables")
    fig.tight_layout()
    plt.show()
    return


def plot_data_proportion(Y):
    fig = plt.figure(figsize=(12, 6))
    ax = sns.countplot(x=Y)
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(100. * y / len(Y)),
                    (x.mean(), y), ha="center", va="bottom")
    plt.title("Data distribution (pulsars, others)")
    plt.xlabel("target_class")
    fig.tight_layout()
    plt.show()
    return


def plot_variable_comparision(data):
    compare = data.groupby("target_class").mean().reset_index()
    compare = compare.drop("target_class", axis=1)
    compare.plot(kind="bar", cmap="tab20c")
    plt.title("Comparision of mean values between target classes")
    plt.xlabel("target_class")
    plt.ylabel("mean value")

    compare1 = data.groupby("target_class").std().reset_index()
    compare1 = compare1.drop("target_class", axis=1)
    compare1.plot(kind="bar", cmap="tab20c")
    plt.title("Comparision of standard deviation values between target classes")
    plt.xlabel("target_class")
    plt.ylabel("mean value")

    # compare_mean = compare.transpose().reset_index()
    # compare_mean = compare_mean.rename(columns={'index': "features", 0: "not_star", 1: "star"})
    # plt.figure(figsize=(13, 14))
    # plt.subplot(211)
    # sns.pointplot(x="features", y="not_star", data=compare_mean, color="r")
    # sns.pointplot(x="features", y="star", data=compare_mean, color="g")
    # plt.xticks(rotation=60)
    # plt.xlabel("")
    # plt.grid(True, alpha=.3)
    # plt.title("COMPARING MEAN OF ATTRIBUTES FOR TARGET CLASSES")
    #
    # compare_std = compare1.transpose().reset_index()
    # compare_std = compare_std.rename(columns={'index': "features", 0: "not_star", 1: "star"})
    # plt.subplot(212)
    # sns.pointplot(x="features", y="not_star", data=compare_std, color="r")
    # sns.pointplot(x="features", y="star", data=compare_std, color="g")
    # plt.xticks(rotation=60)
    # plt.grid(True, alpha=.3)
    # plt.title("COMPARING STANDARD DEVIATION OF ATTRIBUTES FOR TARGET CLASSES")
    # plt.subplots_adjust(hspace=.4)
    # print ("[GREEN == STAR , RED == NOTSTAR]")

    columns = [x for x in data.columns if x not in ["target_class"]]
    length = len(columns)
    plt.figure(figsize=(13, 20))
    for i, j in itertools.izip_longest(columns, range(length)):
        plt.subplot(4, 2, j + 1)
        sns.lvplot(x=data["target_class"], y=data[i])
        plt.title(i)
        plt.subplots_adjust(hspace=1.)
        plt.axhline(data[i].mean(), linestyle="dashed", color="k", label="Mean value for data")
        plt.legend(loc="best")

    plt.show()
    return


def plot_variable_pair(data):
    sns.pairplot(data)
    plt.title("Pair plot for variables")
    plt.show()
    return


def plot_3d(x, y, size=0.2, categories=2, labels=('not pulsars', 'pulsars'), minc=0):
    fig = plt.figure()
    ax = Axes3D(fig)
    names = ['x', 'y', 'z']

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])

    for i in range(categories):
        d = x[np.where(y == (i + minc))[0], :]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=size, label=labels[i])
    plt.legend(loc='best')
    plt.show()
    return


def plot_2d(x, y, size=0.2, categories=2, labels=('not pulsars', 'pulsars'), minc=0):
    plt.xlabel('x')
    plt.ylabel('y')

    for i in range(categories):
        d = x[np.where(y == (i + minc))[0], :]
        plt.scatter(d[:, 0], d[:, 1], s=size, label=labels[i])

    plt.legend(loc='best')
    plt.show()
    return


def run():
    X, Y, dataset = read_csv()
    print("number of records: ", Y.shape[0])
    print("number of columns: ", X.shape[1])
    print("pulsars before outlier detection: ", len(np.where(Y == 1)[0]))

    plot_data_summary(dataset)
    plot_data_correlation(dataset)
    plot_data_proportion(Y)
    plot_variable_comparision(dataset)
    plot_variable_pair(dataset)

    X, Y = outliers(X, Y, False)
    print("reduced number of records: ", len(Y))
    print("pulsars after outlier detection: ", len(np.where(Y == 1)[0]))

    category = 0  # 1 pulsars, 0 not pulsars
    XX = X[np.where(Y == category)[0], :]
    y = Y[np.where(Y == category)[0]]

    x = XX[:, [0, 1]]  # [0, 1], [2, 5]
    # [0,4], [3, 6], [0, 5]
    # xx = reduce_dim(x, 1)
    # x = reduce_dim(x, 3)
    num_of_clusters = 2

    num_of_clusters, y = clusterization(x, num_of_clusters, "HDBSCAN", True)
    # colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']

    if len(x[0]) == 3:
        plot_3d(x, y, 0.5, num_of_clusters, ['c' + str(i + 1) for i in range(num_of_clusters)], np.min(y))
    else:
        plot_2d(x, y, 0.5, num_of_clusters, ['c' + str(i + 1) for i in range(num_of_clusters)], np.min(y))


if __name__ == "__main__":
    run()
