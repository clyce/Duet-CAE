import sklearn.metrics as metrics
import numpy as np


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


nmi = metrics.normalized_mutual_info_score
ari = metrics.adjusted_rand_score


def evaluate_(pred_labels, y):
    return (metrics.completeness_score(y, pred_labels),
            metrics.homogeneity_score(y, pred_labels),
            metrics.v_measure_score(y, pred_labels),
            nmi(y, pred_labels),
            ari(y, pred_labels),
            acc(y, pred_labels))


def evaluate(model, X, y):
    return evaluate_(model.fit_predict(X), y)
