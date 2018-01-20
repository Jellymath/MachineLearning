import random
import statistics

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data_set = datasets.load_breast_cancer()
X = data_set.data

feature_names = data_set.feature_names
print(feature_names)
# feature in this data set looks like (mean % ... % error ... worst %).
# Maybe need to do more sophisticated random if we don't wanna observe 2 interpretations of same parameter
feature_indexes = random.sample(range(feature_names.size), 2)
feature_indexes.sort()
print('Selected feature names:')
for index in feature_indexes:
    print(feature_names[index])
print(feature_indexes)

X_plot = data_set.data[:, feature_indexes]
y = data_set.target

x_min, x_max = X_plot[:, 0].min() - .5, X_plot[:, 0].max() + .5
y_min, y_max = X_plot[:, 1].min() - .5, X_plot[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()


# Plot the training points
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel(feature_names[feature_indexes[0]])
plt.ylabel(feature_names[feature_indexes[1]])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()


X = StandardScaler().fit_transform(data_set.data)

print('Covariation matrix of data set:')
cov_X = np.cov(X.T)

print('Eigenvalues of covariation matrix:')
eig_vals = np.linalg.eigvals(cov_X)
print(eig_vals)

number_of_components = 3
pca = PCA(n_components=number_of_components)
X_reduced = pca.fit_transform(X)

print('Covariation matrix of reduced data set:')
cov_X_reduced = np.cov(X_reduced.T)
print(cov_X_reduced)

print('Eigenvalues of covariation matrix of reduced data set:')
eig_vals_reduced, eig_vecs_reduced = np.linalg.eig(cov_X_reduced)
print(eig_vals_reduced)

print('Variances of reduced data set')
variances_reduced = X_reduced.var(axis=0)
print(variances_reduced)

print('Traces')
trace = np.trace(cov_X)
trace_reduced = np.trace(cov_X_reduced)
print(trace)
print(trace_reduced)

explained_variance = eig_vals / eig_vals.sum()
print(explained_variance)
explained_variance_reduced = eig_vals_reduced / eig_vals_reduced.sum()
print(explained_variance_reduced)

plt.scatter(range(feature_names.size), y=explained_variance)
plt.show()

plt.scatter(range(number_of_components), y=explained_variance_reduced)
plt.show()

# X_std = StandardScaler().fit_transform(X)
#
# cov_X_std = np.cov(X_std.T)
# print(cov_X_std)
#
# eig_vals_std, eig_vecs_std = np.linalg.eig(cov_X_std)
# print(eig_vals_std)

# part 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
k = random.randint(2, 5)
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
target_names = ['0', '1']
print(classification_report(y_test, y_pred, target_names=target_names))

kf = KFold(n_splits=10)
scores = [neigh.score(X[test], y[test]) for _, test in kf.split(X)]
print(scores)

print(statistics.variance(scores))

print("Cycle begins")
for score_function in [metrics.recall_score, metrics.accuracy_score, metrics.f1_score]:
    for k_fold in [2, 5, 8, 10]:
        means = []
        variances = []
        possible_k_neigh = range(2, 21)
        for k_neigh in possible_k_neigh:
            print("k_neigh is " + str(k_neigh) + ", k_fold is " + str(k_fold))
            neigh = KNeighborsClassifier(n_neighbors=k_neigh)
            neigh.fit(X_train, y_train)
            kf = KFold(n_splits=k_fold)
            scores = [score_function(y[test], neigh.predict(X[test])) for _, test in kf.split(X)]
            print(scores)
            print(statistics.mean(scores))
            print(statistics.variance(scores))
            means.append(statistics.mean(scores))
            variances.append(statistics.variance(scores))
        plt.scatter(possible_k_neigh, means)
        plt.title(score_function.__name__)
        plt.xlabel('k_neigh')
        plt.ylabel('Mean score')
    plt.show()

# part 3
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.1)
k = random.randint(2, 5)
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
target_names = ['0', '1']
print(classification_report(y_test, y_pred, target_names=target_names))

kf = KFold(n_splits=10)
scores = [neigh.score(X_reduced[test], y[test]) for _, test in kf.split(X_reduced)]
print(scores)

print(statistics.variance(scores))

print("Cycle begins")
for score_function in [metrics.recall_score, metrics.accuracy_score, metrics.f1_score]:
    for k_fold in [2, 5, 8, 10]:
        means = []
        variances = []
        possible_k_neigh = range(2, 21)
        for k_neigh in possible_k_neigh:
            print("k_neigh is " + str(k_neigh) + ", k_fold is " + str(k_fold))
            neigh = KNeighborsClassifier(n_neighbors=k_neigh)
            neigh.fit(X_train, y_train)
            kf = KFold(n_splits=k_fold)
            scores = [score_function(y[test], neigh.predict(X_reduced[test])) for _, test in kf.split(X)]
            print(scores)
            print(statistics.mean(scores))
            print(statistics.variance(scores))
            means.append(statistics.mean(scores))
            variances.append(statistics.variance(scores))
        plt.scatter(possible_k_neigh, means)
        plt.title(score_function.__name__)
        plt.xlabel('k_neigh')
        plt.ylabel('Mean score')
    plt.legend()
    plt.show()