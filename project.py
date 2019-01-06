from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score


def read_csv():
    fname = "pulsar.csv"
    names = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]
    dataset = pd.read_csv(fname, names=names)

    x = dataset.loc[:, names[:-1]].values
    y = dataset.loc[:, ['V9']].values
    return x, y


def outliers(x, y):
    mean = np.mean(x, axis=0)
    stdev = np.std(x, axis=0)
    ind = np.unique(np.where(np.abs(x-mean) > 3 * stdev)[0])
    #  >these three lines are to preserve all pulsars, because it deletes half of them
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


def clusterization(X):
    pass


def classification(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    cl = RandomForestClassifier()
    cl.fit(X_train, y_train)
    return cl


def plot3D(x, y):
    fig = plt.figure()
    ax = Axes3D(fig)
    names = ['x', 'y', 'z']

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])

    d1 = x[np.where(y == 0)[0], :]
    d2 = x[np.where(y == 1)[0], :]

    ax.scatter(d1[:, 0], d1[:, 1], d1[:, 2])
    ax.scatter(d2[:, 0], d2[:, 1], d2[:, 2])
    plt.show()


def plot2D(x, y):
    plt.xlabel('x')
    plt.ylabel('y')

    d1 = x[np.where(y == 0)[0], :]
    d2 = x[np.where(y == 1)[0], :]

    plt.scatter(d1[:, 0], d1[:, 1])
    plt.scatter(d2[:, 0], d2[:, 1])
    plt.show()


def run():
    X, Y = read_csv()
    print("number of records: ", len(Y))
    print("pulsars before outlier detection: ", len(np.where(Y == 1)[0]))
    X, Y = outliers(X, Y)
    print("reduced number of records: ", len(Y))
    print("pulsars after outlier detection: ", len(np.where(Y == 1)[0]))

    x_reduced = reduce_dim(X, 2)
    #plot3D(X[:, :3], Y)
    plot2D(x_reduced, Y)


if __name__ == "__main__":
    run()
