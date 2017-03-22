#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import floor
import numpy as np
from numpy import trapz
from numpy.random.mtrand import permutation, rand


def holdout_method(features, target):
    N = features.shape[0]
    N_train = floor(N * 0.7)
    idx = permutation(N)
    idx_train = idx[:N_train]
    idx_test = idx[N_train:]

    features_train, target_train = features.ix[idx_train], target[idx_train]
    features_test, target_test = features.ix[idx_test], target[idx_test]

    return features_train, target_train, features_test, target_test

def kfold_cross_validation(features, target, k=10):
    N = features.shape[0]

    predictions = np.empty(N)
    folds = np.random.randint(0, k, size=N)

    for idx in np.arange(k):
        features_train, target_train = features[folds != idx, :], target[folds != idx]
        features_test = features[folds == idx, :]

        model = predict(features_train, target_train)
        predictions[folds == idx] = predict = (model, features_test)

    accuracy = evaluate_acc(predictions, target)
    return accuracy


def roc_curve(true_labels, predicted_labels, n_points=100, pos_class=1):
    thr = np.linspace(0, 1, n_points)
    tpr = np.zeros(n_points)
    fpr = np.zeros(n_points)

    pos = true_labels == pos_class
    neg = np.logical_not(pos)
    pos_count = np.count_nonzero(pos)
    neg_count = np.count_nonzero(neg)

    for i, t in enumerate(thr):
        tpr[i] = np.count_nonzero(np.logical_and(predicted_labels >= t, pos)) / pos_count # часть от действительно положительных
        fpr[i] = np.count_nonzero(np.logical_and(predicted_labels >= t, neg)) / neg_count # часть от действительно отрицательных
    return fpr, tpr, thr


def auc(true_labels, predicted_labels, pos_class=1):
    fpr, tpr, thr = roc_curve(true_labels, predicted_labels, pos_class=pos_class)
    area = -trapz(tpr, fpr)
    return area


def rmse(true_values, predictive_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predictive_values[i]) ** 2
    return np.sqrt(residuals / n)


def r2(true_values, predicted_values):
    n = len(true_values)
    mean = np.mean(true_values)
    residuals = 0
    total = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i]) ** 2
        total += (true_values[i] - mean) ** 2
    return 1.0 - residuals / total


def residual_plot(target_test, predictions):
    fig_size = (15, 7)
    f, ax = plt.subplots(figsize=fig_size)
    sns.regplot(x=target_test, y=predictions - target_test, ax=ax, fit_reg=False, scatter_kws={"s": 50})
    plt.plot([0, target_test.max()], [0, 0])


def grid_search_cv(X, y):
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC

    gam_vec, cost_vec = np.meshgrid(np.linspace(0.01, 10., 11), np.linspace(1., 10., 11))

    AUC_all = []
    N = len(y)
    K = 10
    folds = np.random.randint(0, K, N)

    for param_ind in np.arange(len(gam_vec.ravel())):
        y_cv_pred = np.empty(N)

        for ii in np.arange(K):
            X_train = X.ix[folds != ii]
            y_train = y[folds != ii]
            X_test = X.ix[folds == ii]

            model = SVC(gamma=gam_vec.ravel()[param_ind], C=cost_vec.ravel()[param_ind])
            model.fit(X_train, y_train)
            y_cv_pred[folds == ii] = model.predict(X_test)

        AUC_all.append(roc_auc_score(y, y_cv_pred))
    indmax = np.argmax(AUC_all)
    print 'Maximum = %.3f' % (np.max(AUC_all))
    print 'Tuning parameters: (gamma = %f, C = %f)' % (gam_vec.ravel()[indmax], cost_vec.ravel()[indmax])



def main():
    model = train(features_train, target_train)
    predictions = predict(model, features_test)
    accuracy = evaluate(target_test, predictions)

# the same results !
print model.score(features.ix[test_idx], target[test_idx]), np.mean(target[test_idx] == predictions)