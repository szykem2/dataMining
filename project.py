import collections
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def read_csv():
    fname = "pulsar_stars.csv"
    dataset = pd.read_csv(fname)
    names = list(dataset.columns.values)

    x = dataset.loc[:, names[:-1]].values
    y = dataset.iloc[:, -1].values
    return x, y, dataset


def outliers(x, y, type, ignore_pulsars=False):  # TODO: other methods
    ind = None
    if type == "distance":
        mean = np.mean(x, axis=0)
        stdev = np.std(x, axis=0)
        ind = np.where(np.abs(x - mean) > 3 * stdev)[0]
    elif type == "density":
        clusterer = HDBSCAN(min_cluster_size=15)
        clusterer.fit(x)
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
        sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
        plt.show()
        ind = np.where(clusterer.outlier_scores_ > threshold)[0]
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
        cluster = DBSCAN(eps=0.7, metric='manhattan')
    elif type == "HDBSCAN":
        cluster = HDBSCAN(min_cluster_size=450, min_samples=1, metric='manhattan')

    cluster.fit_predict(X)
    cluster.labels_ -= np.min(cluster.labels_)
    if ommit_last_class:
        cluster.labels_[np.where(cluster.labels_ == 3)] = 2
    num = len(np.unique(cluster.labels_))
    return num, cluster.labels_


def classification(x, y, labels):
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier()]

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA", "LogisticRegression", "ExtraTree", "Gradient Boosting"]

    predictions = []
    results = {}

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        results[name] = accuracy_score(y_test, prediction)
        predictions.append(prediction)

    print(results)
    keys = list(results.keys())
    values = list(results.values())
    best = keys[values.index(max(values))]
    print(best, max(values))

    prediction = predictions[names.index(best)]
    algorithm = classifiers[names.index(best)]

    print("\nclassification report :\n", (classification_report(y_test, prediction)))

    plt.figure(figsize=(13, 10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt="d", linecolor="k", linewidths=3)
    plt.title("CONFUSION MATRIX", fontsize=20)

    predicting_probabilites = algorithm.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, predicting_probabilites)
    plt.subplot(222)
    plt.plot(fpr, tpr, label=("Area_under the curve :", auc(fpr, tpr)), color="r")
    plt.plot([1, 0], [1, 0], linestyle="dashed", color="k")
    plt.legend(loc="best")
    plt.title("ROC - CURVE & AREA UNDER CURVE", fontsize=20)

    try:
        dataframe = pd.DataFrame(algorithm.feature_importances_, labels).reset_index()
    except AttributeError as e:
        dataframe = pd.DataFrame(algorithm.coef_.ravel(), labels).reset_index()

    dataframe = dataframe.rename(columns={"index": "features", 0: "coefficients"})
    dataframe = dataframe.sort_values(by="coefficients", ascending=False)
    plt.subplot(223)
    ax = sns.barplot(x="coefficients", y="features", data=dataframe, palette="husl")
    plt.title("FEATURE IMPORTANCES", fontsize=20)
    for i, j in enumerate(dataframe["coefficients"]):
        ax.text(.011, i, j, weight="bold")
    plt.show()


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


def plot_data_proportion(data):
    fig = plt.figure(figsize=(12, 6))
    ax = sns.countplot(x=data)
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(100. * y / len(data)),
                    (x.mean(), y), ha="center", va="bottom")
    plt.title("Data distribution (pulsars, others)")
    plt.xlabel("target_class")
    fig.tight_layout()
    plt.show()
    return


def plot_splitted_data_proportion(y_train, y_test):
    train_count = collections.Counter(y_train)
    test_count = collections.Counter(y_test)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.pie(list(train_count.values()),
            autopct="%1.0f%%", labels=["not start", "star"])
    plt.title("Proportion of target class in train data")
    plt.subplot(122)
    plt.pie(list(test_count.values()),
            autopct="%1.0f%%", labels=["not start", "star"])
    plt.title("Proportion of target class in test data")
    plt.show()
    return


def plot_variable_comparision(data):
    compare = data.groupby("target_class").mean().reset_index()
    compare = compare.drop("target_class", axis=1)
    compare.plot(kind="bar", cmap="tab20c")
    plt.title("Comparision of mean values between target classes")
    plt.xlabel("target_class")
    plt.ylabel("mean value")
    plt.show()

    compare1 = data.groupby("target_class").std().reset_index()
    compare1 = compare1.drop("target_class", axis=1)
    compare1.plot(kind="bar", cmap="tab20c")
    plt.title("Comparision of standard deviation values between target classes")
    plt.xlabel("target_class")
    plt.ylabel("mean value")
    plt.show()

    compare_mean = compare.transpose().reset_index()
    compare_mean = compare_mean.rename(columns={'index': "features", 0: "not_star", 1: "star"})
    plt.figure(figsize=(13, 14))
    ax = plt.subplot(211)
    sns.pointplot(ax=ax, x="features", y="not_star", data=compare_mean, color="r")
    sns.pointplot(ax=ax, x="features", y="star", data=compare_mean, color="g")
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("mean value")
    plt.grid(True, alpha=.3)
    ax.legend(handles=ax.lines[::len(compare_mean) + 1], labels=["Not star", "Star"])
    ax.set_xticklabels([t.get_text().split("T")[0] for t in ax.get_xticklabels()])
    plt.title("COMPARING MEAN OF ATTRIBUTES FOR TARGET CLASSES")
    plt.show()

    compare_std = compare1.transpose().reset_index()
    compare_std = compare_std.rename(columns={'index': "features", 0: "not_star", 1: "star"})
    ax = plt.subplot(212)
    sns.pointplot(ax=ax, x="features", y="not_star", data=compare_std, color="r")
    sns.pointplot(ax=ax, x="features", y="star", data=compare_std, color="g")
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("standard deviation value")
    plt.grid(True, alpha=.3)
    ax.legend(handles=ax.lines[::len(compare_std) + 1], labels=["Not star", "Star"])
    ax.set_xticklabels([t.get_text().split("T")[0] for t in ax.get_xticklabels()])
    plt.gcf().autofmt_xdate()
    plt.title("COMPARING STANDARD DEVIATION OF ATTRIBUTES FOR TARGET CLASSES")
    plt.subplots_adjust(hspace=.4)
    plt.tight_layout()
    plt.show()

    columns = [x for x in data.columns if x not in ["target_class"]]
    length = len(columns)

    # next two are basically the same
    plt.figure(figsize=(13, 20))
    for i, j in itertools.zip_longest(columns, range(length)):
        plt.subplot(4, 2, j + 1)
        sns.boxenplot(x=data["target_class"], y=data[i])
        plt.title(i)
        plt.axhline(data[i].mean(), linestyle="dashed", color="k", label="Mean value for data")
        plt.legend(loc="best")
    plt.subplots_adjust(hspace=1.)
    plt.show()

    plt.figure(figsize=(13, 25))
    for i, j in itertools.zip_longest(columns, range(length)):
        plt.subplot(length / 2, length / 4, j + 1)
        sns.violinplot(x=data["target_class"], y=data[i])
        plt.title(i)
    plt.subplots_adjust(hspace=1.)
    plt.show()
    return


def plot_variable_distribution(data):
    columns = [x for x in data.columns.values if x not in ["target_class"]]
    length = len(columns)
    colors = ["r", "g", "b", "m", "y", "c", "k", "orange"]

    plt.figure(figsize=(13, 20))
    for i, j, k in itertools.zip_longest(columns, range(length), colors):
        plt.subplot(length / 2, length / 4, j + 1)
        sns.distplot(data[i], color=k)
        plt.axvline(data[i].mean(), color="k", linestyle="dashed", label="mean")
        plt.axvline(data[i].std(), color="b", linestyle="dotted", label="stdev")
        plt.title(i)
        plt.subplots_adjust(hspace=.3)
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    return


def plot_important_variable_comparision(data):
    plt.figure(figsize=(14, 7))

    plt.subplot(121)
    plt.scatter(x=data.columns.values[2], y=data.columns.values[3],
                data=data[data["target_class"] == 1], alpha=.7,
                label="pulsar stars", s=30, color="g", linewidths=.4, edgecolors="black")
    plt.scatter(x=data.columns.values[2], y=data.columns.values[3],
                data=data[data["target_class"] == 0], alpha=.6,
                label="not pulsar stars", s=30, color="r", linewidths=.4, edgecolors="black")
    plt.axvline(data[data["target_class"] == 1].ix[:, 2].mean(),
                color="g", linestyle="dashed", label="mean pulsar star")
    plt.axvline(data[data["target_class"] == 0].ix[:, 2].mean(),
                color="r", linestyle="dashed", label="mean non pulsar star")
    plt.axhline(data[data["target_class"] == 1].ix[:, 3].mean(),
                color="g", linestyle="dashed")
    plt.axhline(data[data["target_class"] == 0].ix[:, 3].mean(),
                color="r", linestyle="dashed")
    plt.legend(loc="best")
    plt.xlabel(data.columns.values[2])
    plt.ylabel(data.columns.values[3])
    plt.title("Comparision between skewness and kurtosis for target classes")

    plt.subplot(122)
    plt.scatter(x=data.columns.values[7], y=data.columns.values[6],
                data=data[data["target_class"] == 0], alpha=.7,
                label="not pulsar stars", s=30, color="r", linewidths=.4, edgecolors="black")
    plt.scatter(x=data.columns.values[7], y=data.columns.values[6],
                data=data[data["target_class"] == 1], alpha=.7,
                label="pulsar stars", s=30, color="g", linewidths=.4, edgecolors="black")
    plt.axvline(data[data["target_class"] == 1].ix[:, 6].mean(),
                color="g", linestyle="dashed", label="mean pulsar star")
    plt.axvline(data[data["target_class"] == 0].ix[:, 6].mean(),
                color="r", linestyle="dashed", label="mean non pulsar star")
    plt.axhline(data[data["target_class"] == 1].ix[:, 7].mean(),
                color="g", linestyle="dashed")
    plt.axhline(data[data["target_class"] == 0].ix[:, 7].mean(),
                color="r", linestyle="dashed")
    plt.legend(loc="best")
    plt.xlabel(data.columns.values[7])
    plt.ylabel(data.columns.values[6])
    plt.title("Comparision between skewness and kurtosis of dmsnr_curve for target classes")

    plt.subplots_adjust(wspace=.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.scatter(x=data.columns.values[0], y=data.columns.values[1],
                data=data[data["target_class"] == 0], alpha=.7,
                label="not pulsar stars", s=30, color="r", linewidths=.4, edgecolors="black")
    plt.scatter(x=data.columns.values[0], y=data.columns.values[1],
                data=data[data["target_class"] == 1], alpha=.7,
                label="pulsar stars", s=30, color="g", linewidths=.4, edgecolors="black")
    plt.legend(loc="best")
    plt.xlabel(data.columns.values[0])
    plt.ylabel(data.columns.values[1])
    plt.title("Comparision between mean and stdev profile for target classes")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.scatter(x=data.columns.values[4], y=data.columns.values[5],
                data=data[data["target_class"] == 0], alpha=.7,
                label="not pulsar stars", s=30, color="r", linewidths=.4, edgecolors="black")
    plt.scatter(x=data.columns.values[4], y=data.columns.values[5],
                data=data[data["target_class"] == 1], alpha=.7,
                label="pulsar stars", s=30, color="g", linewidths=.4, edgecolors="black")
    plt.legend(loc="best")
    plt.xlabel(data.columns.values[4])
    plt.ylabel(data.columns.values[5])
    plt.title("Comparision between mean and stdev profile for target classes")
    plt.tight_layout()
    plt.show()

    return


def plot_variable_pair(data):
    sns.pairplot(data)
    plt.title("Pair plot for variables")
    plt.show()
    return


def plot_3d(x, y, axis_labels, size=0.2, categories=2, labels=('not pulsars', 'pulsars'), minc=0):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    for i in range(categories):
        d = x[np.where(y == (i + minc))[0], :]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=size, label=labels[i])
    plt.legend(loc='best')
    plt.show()
    return


def plot_2d(x, y, axis_labels, size=0.2, categories=2, labels=('not pulsars', 'pulsars'), minc=0):
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])

    for i in range(categories):
        d = x[np.where(y == (i + minc))[0], :]
        plt.scatter(d[:, 0], d[:, 1], s=size, label=labels[i])

    plt.legend(loc='best')
    plt.show()
    return


def make_cluster(X, Y, labels, idxs, num_of_clusters, algorithm, ommit):
    x = X[:, idxs]
    num_of_clusters, y = clusterization(x, num_of_clusters, algorithm, ommit)

    if len(x[0]) == 3:
        plot_3d(x, y, labels[idxs], 0.5, num_of_clusters, ['c' + str(i + 1) for i in range(num_of_clusters)], np.min(y))
    else:
        plot_2d(x, y, labels[idxs], 0.5, num_of_clusters, ['c' + str(i + 1) for i in range(num_of_clusters)], np.min(y))


def run():
    X, Y, dataset = read_csv()
    print("number of records: ", Y.shape[0])
    print("number of columns: ", X.shape[1])
    print("pulsars before outlier detection: ", len(np.where(Y == 1)[0]))

    # plot_data_summary(dataset)
    # plot_data_correlation(dataset)
    # plot_data_proportion(Y)
    # plot_variable_comparision(dataset)  # to bym wywalil
    # plot_variable_distribution(dataset)
    # plot_variable_pair(dataset)
    # plot_important_variable_comparision(dataset)

    headers = np.array(dataset.columns.values.tolist())
    # make_cluster(X, Y, headers, [0, 1], 2, "agglomerative", False)
    X, Y = outliers(X, Y, "distance", True)  # FIXME: opcja density nie działa
    print("reduced number of records: ", len(Y))
    print("pulsars after outlier detection: ", len(np.where(Y == 1)[0]))

    pulsars_x = X[np.where(Y == 1)[0], :]
    pulsars_y = Y[np.where(Y == 1)[0]]
    npulsars_x = X[np.where(Y == 0)[0], :]
    npulsars_y = Y[np.where(Y == 0)[0]]

    make_cluster(npulsars_x, npulsars_y, headers, [0, 1], 2, "HDBSCAN", True)
    make_cluster(npulsars_x, npulsars_y, headers, [2, 5], 2, "DBSCAN", False)
    make_cluster(pulsars_x, pulsars_y, headers, [3, 6], 4, "agglomerative", False)

    print("Starting classification: ")
    classification(X, Y, headers[0:8])


if __name__ == "__main__":
    run()
