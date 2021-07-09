import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, average_precision_score


def simple_entropy(x):  # for 1 sample
    return (-x * np.log2(x)).sum(axis=0)  # SHANNON


def entropy(x):  # for multiple sample; not log2!
    return np.sum((-x * np.log(np.clip(x, 1e-12, 1))), axis=-1)


def exp_entropy(mc_preds):
    return np.mean(entropy(mc_preds), axis=0)  # N


def pred_entropy(mc_preds):
    return entropy(np.mean(mc_preds, axis=0))  # N


def mutual_info(mc_preds):
    return pred_entropy(mc_preds) - exp_entropy(mc_preds)  # N


def entropy_tf(X: tf.Tensor) -> tf.Tensor:
    return K.sum(-X * K.log(K.clip(X, 1e-6, 1)), axis=-1)


def expected_entropy(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    mean entropy of the predictive distribution across the MC samples.
    """

    return K.mean(entropy_tf(mc_preds), axis=0)  # batch_size


def predictive_entropy(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    entropy of the mean predictive distribution across the MC samples.
    """
    return entropy_tf(K.mean(mc_preds, axis=0))


def BALD(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Calculate the BALD (Bayesian Active Learning by Disagreement) of a model;
    the difference between the mean of the entropy and the entropy of the mean
    of the predicted distribution on the n_mc x batch_size x n_classes tensor
    mc_preds. In the paper, this is referred to simply as the MI.
    """
    BALD = predictive_entropy(mc_preds) - expected_entropy(mc_preds)
    return BALD


def AUROC_PR(pred_known, pred_unknown):
    neg = list(np.max(pred_known, axis=-1))
    pos = list(np.max(pred_unknown, axis=-1))
    auroc, aupr = compute_auc_aupr(neg, pos, pos_label=0)
    return auroc, aupr


def compute_auc_aupr(neg, pos, pos_label=1):
    ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
    neg = np.array(neg)[np.logical_not(np.isnan(neg))]
    pos = np.array(pos)[np.logical_not(np.isnan(pos))]
    scores = np.concatenate((neg, pos), axis=0)
    auc = roc_auc_score(ys, scores)  # AUROC
    aupr = average_precision_score(ys, scores)  # AUPR
    if pos_label == 1:
        return auc, aupr
    else:
        return 1 - auc, 1 - aupr


def simple_entropy(x):
    return (-x * np.log2(np.clip(x, 1e-12, 1))).sum(axis=0)


def samples_entropy(confidences):
    return np.round(np.mean([simple_entropy(x) for x in confidences]), 4)


def FPRatRecall(pred_known, pred_unknown, recall=0.95):  # pos_label=1
    pos = list(np.max(pred_known, axis=-1))
    neg = list(np.max(pred_unknown, axis=-1))
    y_test = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
    neg = np.array(neg)[np.logical_not(np.isnan(neg))]
    pos = np.array(pos)[np.logical_not(np.isnan(pos))]
    y_pred = np.concatenate((neg, pos), axis=0)
    m = tf.keras.metrics.SpecificityAtSensitivity(recall)
    m.update_state(y_test, y_pred)
    return 1 - m.result().numpy()


def test_metrics():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]])  # 1 2 2
    pred = np.array([[0, 0.96, 0.04], [0, 0.04, 0.96], [0, 0.06, 0.94], [0, 0.98, 0.02]])
    pred2 = np.array([[0, 0.92, 0.08], [0.6, 0.3, 0.1], [0.1, 0.16, 0.74], [0, 0.90, 0.1]])
    mc_preds = np.array([pred, pred2])  # 2 x 4 x 3 ; T x B/N x K

    for metric in [entropy_tf, expected_entropy, predictive_entropy, BALD]:
        print(metric.__name__, metric(tf.convert_to_tensor(mc_preds)))

    for metric in [entropy, exp_entropy, pred_entropy, mutual_info]:
        print(metric.__name__, metric(mc_preds))

    print(AUPR.__name__, AUPR(pred, pred2))

    print(samples_entropy(pred))
    print(np.mean(entropy(pred)))
    print(np.mean(exp_entropy(pred)))
