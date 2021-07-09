import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.metrics import MeanMetricWrapper
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# from seqeval.metrics import f1_score, classification_report

import resource


class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        usage = str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(f"epoch: {epoch}; usage: {usage}")


class ParamsCallback(Callback):
    def __init__(self, _config, idx2label, idx2voc, command):
        super(ParamsCallback, self).__init__()
        self.config = _config
        self.l = idx2label
        self.v = idx2voc
        self.command = command

    def on_epoch_end(self, epoch, log={}):
        from arkham.utils.model_utils import config_to_json

        if epoch == 0:
            # os.makedirs(self.config["out_folder"])
            config_to_json(self.config, self.l, self.v, self.command)


class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model


class LR_Callback(Callback):
    def on_batch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        decayed_lr = self.model.optimizer._decayed_lr(tf.float32)  # lr {lr}
        iterations = self.model.optimizer.iterations
        print(f"It {iterations.numpy()} dlr {decayed_lr}")
        """
        decay = self.model.optimizer.decay
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))
        """


class ChunkF1(tf.keras.metrics.Metric):
    # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score

    def __init__(self, idx2label, name='chunk_f1', pad_value=0, average="micro", **kwargs):
        super(ChunkF1, self).__init__(name=name, **kwargs)

        self.idx2label = idx2label
        self.num_classes = len(self.idx2label)
        self.pad_value = pad_value
        self.average = average  # micro macro weighted

        self.true_positives_col = self.add_weight(
            'TP-class', shape=[self.num_classes], initializer='zeros', dtype=tf.float32
        )
        self.false_positives_col = self.add_weight(
            'FP-class', shape=[self.num_classes], initializer='zeros', dtype=tf.float32
        )
        self.false_negatives_col = self.add_weight(
            'FN-class', shape=[self.num_classes], initializer='zeros', dtype=tf.float32
        )
        self.weights_intermediate = self.add_weight(
            'weights-int-f1', shape=[self.num_classes], initializer='zeros', dtype=tf.float32
        )
        # tf.lookup.StaticHashTable(
        # tf.lookup.KeyValueTensorInitializer(tf.constant(list(idx2label.keys()), dtype=tf.int32), tf.constant(list(idx2label.values()))), -1)
        # self.score = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        def calc_f1(y_true, y_pred):

            tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
            tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
            fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
            fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

            p = tp / (tp + fp + tf.keras.backend.epsilon())
            r = tp / (tp + fn + tf.keras.backend.epsilon())

            f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
            f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
            return tf.reduce_mean(f1)
        """

        # reduce dimension [if categorical].
        if len(y_true.shape) == 3:
            y_true = tf.argmax(y_true, -1)
        y_pred = tf.argmax(y_pred, -1)

        mask = tf.greater(y_true, self.pad_value)

        y_true = tf.cast(tf.boolean_mask(y_true, mask), 'int32')
        y_pred = tf.cast(tf.boolean_mask(y_pred, mask), 'int32')

        # # tf.lookup.StaticHashMap for chunk-based evaluation
        # true = tf.map_fn(lambda x: self.idx2label.lookup(x), y_true)
        # pred = tf.map_fn(lambda x: self.idx2label.lookup(x), y_pred)

        y_true = tf.one_hot(y_true, len(self.idx2label), axis=-1)
        y_pred = tf.one_hot(y_pred, len(self.idx2label), axis=-1)

        # true positive across column
        self.true_positives_col.assign_add(tf.cast(tf.math.count_nonzero(y_pred * y_true, axis=0), tf.float32))
        # false positive across column
        self.false_positives_col.assign_add(tf.cast(tf.math.count_nonzero(y_pred * (y_true - 1), axis=0), tf.float32))
        # false negative across column
        self.false_negatives_col.assign_add(tf.cast(tf.math.count_nonzero((y_pred - 1) * y_true, axis=0), tf.float32))

        self.weights_intermediate.assign_add(tf.cast(tf.reduce_sum(y_true, axis=0), tf.float32))
        # weights = weights_intermediate / tf.reduce_sum(weights_intermediate)
        # score = calc_f1(true, pred)
        # self.score.assign_add(score)

    def result(self):
        p_sum = tf.cast(self.true_positives_col + self.false_positives_col, tf.float32)
        precision_macro = tf.cast(tf.math.divide_no_nan(self.true_positives_col, p_sum), tf.float32)

        r_sum = tf.cast(self.true_positives_col + self.false_negatives_col, tf.float32)
        recall_macro = tf.cast(tf.math.divide_no_nan(self.true_positives_col, r_sum), tf.float32)

        mul_value = 2 * precision_macro * recall_macro
        add_value = precision_macro + recall_macro
        f1_macro_int = tf.cast(tf.math.divide_no_nan(mul_value, add_value), tf.float32)

        f1_score = tf.reduce_mean(f1_macro_int)

        if self.average == 'weighted':

            f1_int_weights = tf.cast(
                tf.math.divide_no_nan(self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)), tf.float32
            )
            f1_score = tf.reduce_sum(f1_macro_int * f1_int_weights)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {}
        base_config = super(ChunkF1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.true_positives_col.assign(np.zeros(self.num_classes), np.float32)
        self.false_positives_col.assign(np.zeros(self.num_classes), np.float32)
        self.false_negatives_col.assign(np.zeros(self.num_classes), np.float32)
        self.weights_intermediate.assign(np.zeros(self.num_classes), np.float32)


# def heteroscedastic_mse(y_true, y_pred, T=10):
#     sampled = y_pred.sample(T)
#     score = tf.reduce_mean([tf.reduce_mean(tf.math.squared_difference(sampled[t], y_true)) for t in range(T)])
#     return score


class Heteroscedastic_Acc(tf.keras.metrics.Metric):
    def __init__(self, name='heteroscedastic_acc', T=10, **kwargs):
        super(Heteroscedastic_Acc, self).__init__(name=name, **kwargs)
        self.T = T
        self.score = self.add_weight('score', initializer='zeros', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sampled = y_pred.sample(self.T)
        score = tf.reduce_mean(
            [tf.keras.metrics.categorical_accuracy(sampled[t], tf.cast(y_true, tf.float32)) for t in range(self.T)]
        )
        self.score.assign_add(score)
        self.total.assign_add(1)

    def fast_compute(self, y_true, sampled):
        score = tf.reduce_mean(
            [tf.keras.metrics.categorical_accuracy(sampled[t], tf.cast(y_true, tf.float32)) for t in range(self.T)]
        )
        return score

    def result(self):
        return self.score / self.total

    def get_config(self):
        # Returns the serializable config of the metric.
        config = {}
        base_config = super(Heteroscedastic_Acc, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.score.assign(0.0, np.float32)
        self.total.assign(0.0, np.float32)


class Heteroscedastic_MSE(tf.keras.metrics.Metric):
    def __init__(self, name='heteroscedastic_mse', T=10, **kwargs):
        super(Heteroscedastic_MSE, self).__init__(name=name, **kwargs)
        self.T = T
        self.score = self.add_weight('score', initializer='zeros', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        raise Exception("This does  not work properly YET; super frustrating")
        sampled = y_pred.sample(self.T)
        score = tf.reduce_mean(
            [tf.keras.metrics.mean_squared_error(sampled[t], tf.cast(y_true, tf.float32)) for t in range(self.T)]
        )
        self.score.assign_add(score)
        self.total.assign_add(1)

    def fast_compute(self, y_true, sampled):
        score = tf.reduce_mean(
            [tf.reduce_mean(tf.math.squared_difference(sampled[t], y_true), axis=-1) for t in range(self.T)]
        )
        return score

    def result(self):
        return self.score  # / self.total

    def get_config(self):
        # Returns the serializable config of the metric.
        config = {}
        base_config = super(Heteroscedastic_MSE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.score.assign(0.0, np.float32)
        self.total.assign(0.0, np.float32)


def mean_squared_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)


def macro_soft_f1(y, y_hat):
    # https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    # https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


# class F1Metrics(Callback):

#     def __init__(self, id2xlabel, pad_value=0, validation_data=None, digits=4):
#         """
#         Args:
#             id2xlabel (dict): id to label mapping.
#             (e.g. {1: 'B-LOC', 2: 'I-LOC'})
#             pad_value (int): padding value.
#             digits (int or None): number of digits in printed classification report
#               (use None to print only F1 score without a report).
#         """
#         super(F1Metrics, self).__init__()
#         self.id2xlabel = id2xlabel
#         self.pad_value = pad_value
#         self.validation_data = validation_data
#         self.digits = digits
#         self.is_fit = validation_data is None

#     def convert_idx_to_name(self, y, array_indexes):
#         """Convert label index to name.

#         Args:
#             y (np.ndarray): label index 2d array.
#             array_indexes (list): list of valid index arrays for each row.

#         Returns:
#             y: label name list.
#         """
#         y = [[self.id2xlabel[idx] for idx in row[row_indexes]] for
#              row, row_indexes in zip(y, array_indexes)]
#         return y

#     def predict(self, X, y):
#         """Predict sequences.

#         Args:
#             X (np.ndarray): input data.
#             y (np.ndarray): tags.

#         Returns:
#             y_true: true sequences.
#             y_pred: predicted sequences.
#         """
#         y_pred = self.model.predict_on_batch(X)

#         # reduce dimension.
#         y_true = np.argmax(y, -1)
#         y_pred = np.argmax(y_pred, -1)

#         non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

#         y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
#         y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

#         return y_true, y_pred

#     def score(self, y_true, y_pred):
#         """Calculate f1 score.

#         Args:
#             y_true (list): true sequences.
#             y_pred (list): predicted sequences.

#         Returns:
#             score: f1 score.
#         """
#         score = f1_score(y_true, y_pred)
#         print(' - f1: {:04.2f}'.format(score * 100))
#         if self.digits:
#             print(classification_report(y_true, y_pred, digits=self.digits))
#         return score

#     def on_epoch_end(self, epoch, logs={}):
#         if self.is_fit:
#             self.on_epoch_end_fit(epoch, logs)
#         else:
#             self.on_epoch_end_fit_generator(epoch, logs)

#     def on_epoch_end_fit(self, epoch, logs={}):
#         X = self.validation_data[0]
#         y = self.validation_data[1]
#         y_true, y_pred = self.predict(X, y)
#         score = self.score(y_true, y_pred)
#         logs['f1'] = score

#     def on_epoch_end_fit_generator(self, epoch, logs={}):
#         y_true = []
#         y_pred = []
#         for X, y in self.validation_data:
#             y_true_batch, y_pred_batch = self.predict(X, y)
#             y_true.extend(y_true_batch)
#             y_pred.extend(y_pred_batch)
#         score = self.score(y_true, y_pred)
#         logs['f1'] = score


def my_metrics(_run, logs):
    for metric in ["loss", "acc", "val_loss", "val_acc", "lr", 'val_loss']:
        if metric in logs:
            _run.log_scalar(metric, float(logs.get(metric)))
    # _run.result = float(logs.get('val_loss'))


class LogEpochMetrics(Callback):
    def on_epoch_end(self, _, logs={}):
        my_metrics(logs=logs)


class UnopenedFieldCallback(Callback):
    def __init__(self, validation_data=None, max_fp=0.05):
        """Summary

        Args:
            validation_data (None, optional): Description
            max_fp (float, optional): Description
        """
        super(Callback, self).__init__()
        self.max_fp = max_fp
        self.validation_data = validation_data
        self.y_val = None
        if self.y_val is None:
            _, labels = list(zip(*iter(self.validation_data.unbatch())))
            self.y_val = np.array(labels)
        # self.X_val, self.y_val = validation_data

    def uof(y_true, y_pred, max_fp, precalc=False):
        def best_threshold_at(thresholds, max_fp=0.05):
            """
            Formula to give best threshold closest to 5% FP (maximizing UOF)
            """
            formula = lambda fp, uof: max_fp / max(max_fp, fp) * uof
            best = 0, 0
            for thresh in thresholds:
                uof, fp, uof_pure = thresholds[
                    thresh
                ]  # ["uof"], thresholds[thresh]["uof"], thresholds[thresh]["uof_pure"]
                score = formula(fp, uof)
                if score > best[1]:
                    best = thresh, score
            thresholds[best[0]] = (*thresholds[best[0]], best[0])
            return thresholds[best[0]]

        """
        y_true = tf.cast(tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1)), tf.int32)
        y_score = tf.cast(tf.reduce_max(y_pred, -1), tf.float32)
        """
        if not precalc:
            y_correct = np.equal(np.argmax(y_pred, -1), np.argmax(y_true, -1))
            y_score = np.max(y_pred, -1) * 100
        else:
            y_correct = y_true
            y_score = y_pred * 100

        stepsize = -0.5
        threshes = {}
        # best_threshold = (0, 100, 0, 100)  # keep diff
        for thresh in np.arange(100, 0, stepsize):
            above = y_correct[np.where((y_score > thresh))]
            if not len(above) > 0:
                continue
            tp = np.count_nonzero(above)  # np.sum == equal
            fp = (len(above) - tp) / len(above)  # rate
            uof = len(above) / y_correct.shape[0]
            uof_pure = tp / y_correct.shape[0]
            threshes[thresh] = uof, fp, uof_pure
            """
            print("\n UoF: %.4f FP: %.4f T>: %.4f A>T: %.4f \n" % (uof, fp, thresh, uof_pure))
            if round(fp, 2) <= max_fp:
                best_threshold = (uof, fp, uof_pure, thresh)  # (round(uof, 4), round(fp, 4), round(uof_pure, 4), thresh)
            else:
                break
            """
        best_threshold = best_threshold_at(threshes, max_fp=max_fp)
        return best_threshold

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data, verbose=0)
        uof, fp, uof_pure, thresh = UnopenedFieldCallback.uof(self.y_val, y_pred, self.max_fp)
        # if round(fp, 2) <= self.max_fp:
        print("\n UoF - epoch: %d - UoF: %.4f FP: %.3f T>: %.1f A>T: %.4f \n" % (epoch + 1, uof, fp, thresh, uof_pure))


class PerformanceVisualizationCallback(Callback):
    def __init__(self, validation_data, image_dir="./images"):
        super().__init__()
        self.validation_data = validation_data
        self.y_val = None
        self.sequence_labels = False
        if self.y_val is None:
            _, labels = list(zip(*iter(validation_data.unbatch())))
            if any(labelled.shape[0] > 1 for labelled in labels):
                self.sequence_labels = True
                labels = np.hstack(labels)
            self.y_val = np.array(labels)

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):

        from pdb import set_trace

        set_trace()

        y_pred = np.asarray(self.model.predict(self.validation_data))
        y_pred_class = np.argmax(y_pred, axis=1)
        if self.sequence_labels:
            y_pred_class = np.hstack(y_pred_class)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(self.y_val, y_pred_class, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(self.y_val, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))


# def f1(y_true, y_pred):
#     y_pred = tf.round(y_pred)
#     tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
#     tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
#     fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
#     fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

#     p = tp / (tp + fp + tf.keras.backend.epsilon())
#     r = tp / (tp + fn + tf.keras.backend.epsilon())

#     f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     return tf.reduce_mean(f1)


# def f1_loss(y_true, y_pred):

#     tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
#     tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
#     fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
#     fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

#     p = tp / (tp + fp + tf.keras.backend.epsilon())
#     r = tp / (tp + fn + tf.keras.backend.epsilon())

#     f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     return 1 - tf.reduce_mean(f1)


class UoF(tf.keras.metrics.Metric):
    # tf.config.experimental_run_functions_eagerly(True)

    def __init__(self, max_fp=0.05, **kwargs):
        super(UoF, self).__init__(name="uof_" + str(max_fp), **kwargs)
        self.max_fp = max_fp
        self.confidences = None
        self.correctness = None

    @tf.function
    def uof(self):
        def best_threshold_at(thresholds, max_fp=0.05):
            """
            Formula to give best threshold closest to 5% FP (maximizing UOF)
            """
            formula = lambda fp, uof: max_fp / tf.math.maximum(max_fp, fp) * uof
            best = thresholds[next(iter(thresholds.keys()))]  # random tmp #reversed?
            for thresh in thresholds:
                uof, fp, uof_pure, _ = thresholds[thresh]
                score = formula(fp, uof)
                best = tf.cond(score > best[1], lambda: thresholds[thresh], lambda: best)
            return best

        y_correct, y_score, max_fp = self.correctness, self.confidences * 100, self.max_fp
        N = y_correct.shape[0]

        stepsize = -0.5
        threshes = {}
        for thresh in np.arange(100, 0, stepsize):
            mask = tf.greater(y_score, tf.constant(thresh, "float32"))
            above = tf.cast(tf.boolean_mask(y_correct, mask), 'int32')
            count_above = tf.math.minimum(tf.size(above), tf.constant(1))
            tp = tf.reduce_sum(above)
            fp = (count_above - tp) / count_above
            uof = count_above / N
            uof_pure = tp / N
            threshes[thresh] = (
                tf.cast(uof, 'float32'),
                tf.cast(fp, 'float32'),
                tf.cast(uof_pure, 'float32'),
                tf.cast(thresh, 'float32'),
            )

        best_threshold = best_threshold_at(threshes, max_fp=0.05)
        return best_threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct = tf.cast(tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1)), tf.int32)
        confidence = tf.cast(tf.math.reduce_max(y_pred, -1), tf.float32)
        if self.confidences is None or self.correctness is None:
            self.confidences = confidence
            self.correctness = correct
        else:
            self.confidences = tf.concat((self.confidences, confidence), axis=0)
            self.correctness = tf.concat((self.correctness, correct), axis=0)

    def result(self):
        if self.confidences.shape[0] is not None and self.correctness.shape[0] is not None:
            uof, fp, uof_pure, thresh = self.uof()
        else:
            uof, fp, uof_pure, thresh = 0, 0, 0, 0
        return tf.cast(uof, 'float32'), tf.cast(fp, 'float32'), tf.cast(thresh, 'float32')
        '''
        print(
            "UoF: %.4f FP: %.3f T>: %.1f A>T: %.4f \n" % (uof, fp, thresh, uof_pure)
        )
        '''

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {}
        base_config = super(UoF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.confidences = None
        self.correctness = None
