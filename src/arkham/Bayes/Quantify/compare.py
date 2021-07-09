#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import regex as re
import pandas as pd
from collections import OrderedDict
from prettytable import PrettyTable
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    matthews_corrcoef,
    mean_squared_error,
    hamming_loss,
)
from netcal.metrics import ECE
import scipy
from seqeval.metrics.sequence_labeling import get_entities, classification_report

from arkham.utils.utils import pickle_loader
from arkham.utils.model_utils import load_model_config
from arkham.utils.custom_metrics import entropy, exp_entropy, pred_entropy, mutual_info

from arkham.Bayes.Quantify.evaluate import (
    TouristLeMC,
    multilabel_prediction,
    stats_to_entities,
    equality,
    reshape_to_sequencelen,
)
from arkham.Bayes.Quantify.automate import plot_df_automation


metric_names = [
    "Acc",
    "MSE(↓)",
    "F1(m)",
    "F1(M)",
    "NLL(↓)",
    "ECE(↓)",
    "Brier(↓)",
    "AUC",
    "Softmax(μ)",
    "Entropy(μ)",
    "WS-D",
    "MCE",
    # ACE um.ace
    "MU(μ)",
    "DU(μ)",
    "KL(µ)",
    "AUB",  # area under bluedots
    "UoF@5",
    "FP@5(↓)",
    "T>",
]
# DEV: other defaults can be wrapped with:
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter; contain arrow
# sklearn.metrics.make_scorer(score_func, *, greater_is_better=True, needs_proba=False, needs_threshold=False)


def calculate_ECE(ground_truth, confidences, n_bins=100):
    ece = ECE(n_bins)
    return ece.measure(confidences, ground_truth)


def plot_reliability(ground_truth, confidences, n_bins=100):
    from netcal.presentation import ReliabilityDiagram

    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(confidences, ground_truth)  # visualize miscalibration of uncalibrated
    plt.show()


def multilabel_encode(target_names, indices):
    empty = np.zeros((indices.shape[0], len(target_names)), dtype=int)
    for i, index in enumerate(indices):
        empty[i, np.array(index, dtype=int)] = 1
    return empty


def multilabel_measures(gold, predicted):
    collector = {}

    sample_size = gold.shape[0]

    flat_gold = np.hstack(gold.ravel())
    flat_predicted = np.hstack(predicted.ravel())

    target_names = sorted(set(list(np.unique(flat_gold)) + list(np.unique(flat_predicted))))

    # labelset_size = np.max([np.sum(x) for x in gold]) #onehot
    collector["cardinality"] = len(flat_predicted) / sample_size
    collector["density"] = np.sum([len(x) / len(target_names) for x in predicted]) / sample_size
    print("Multilabel nature of data and predictions:\n")
    print("{}\t{}".format("Label cardinality :", collector["cardinality"]))
    print("{}\t{}".format("Label density:", collector["density"]))

    onehot_predicted = multilabel_encode(target_names, predicted)
    onehot_gold = multilabel_encode(target_names, gold)
    # need to onehot encode again
    report = classification_report(onehot_gold, onehot_predicted)  # , target_names=target_names
    repdict = classification_report(onehot_gold, onehot_predicted, output_dict=True)  # , target_names=target_names
    print("Classification report:\n", report)

    matrix = multilabel_confusion_matrix(onehot_gold, onehot_predicted)
    # print("Confusion matrix:\n", matrix)

    return repdict, matrix


def brier_multi(targets, probs, label2idx):  # targets need to be one-hot encoded as well
    if targets.shape != probs.shape: 
        targets = tf.keras.utils.to_categorical(targets, num_classes=len(label2idx))
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


def average_confidence(confidences):
    return np.round(np.mean(np.max(confidences, axis=-1)), 4)


def search_rate(version):
    try:
        rate = re.search("(0\.)?\d\d*", version).group(0)
    except:
        rate = 1
    return rate


def calculate_diversity(stats, idx2label, multilabel):
    import tensorflow_probability as tfp

    def kl(x, y):
        x = np.where(x == 0, float(1e-32), x)
        y = np.where(y == 0, float(1e-32), y)

        X = tfp.distributions.Categorical(probs=x)
        Y = tfp.distributions.Categorical(probs=y)
        d = tfp.distributions.kl_divergence(X, Y)
        d = d[~pd.isnull(d)]
        return np.mean(d)

    if not "mean_mc_array" in stats and not "raw" in stats:
        return 0, []  
    samples = np.array(stats["mean_mc_array"])
    shaped = samples.shape

    if len(samples.shape) < 3:
        if not "raw" in stats:
            print("need to evaluate RAW for measuring diversity")
            return 0, []
        else:
            samples = np.transpose(np.array(stats["raw"]), (1, 0, 2))
            shaped = samples.shape
    accuracy = []
    diversity = []
    if shaped[0] == 50:
        samples = np.reshape(samples, (5, 10, *shaped[1:]))
        shaped = samples.shape
    for model in range(shaped[0]):
        predictions = samples[model]
        if len(predictions.shape) > 2:
            predictions = np.mean(predictions, axis=0)
        predictions2 = stats["softmax"]
        if multilabel:
            p, c = multilabel_prediction(predictions, idx2label)
            acc = accuracy_score(
                multilabel_encode(idx2label.keys(), np.array(stats["gold"])),
                multilabel_encode(idx2label.keys(), np.array(p)),
            )
        else:
            p = np.vectorize(idx2label.get)(np.argmax(predictions, -1))
            acc = accuracy_score(np.array(stats["gold"]), p)
        accuracy.append(acc)
        diversity.append(kl(predictions, predictions2))

    return diversity, accuracy


def ensemble_predictions(stats):
    sampled_predictions = stats["mean_mc_array"]  # T/M x N x K
    sample_variance = np.var(sampled_predictions, axis=0)  # N x K
    sample_mean = np.mean(sampled_predictions, axis=0)  # N x K
    ensemble_mean = np.mean(sample_mean, axis=-1)  # N
    ensemble_variance = np.mean(sample_variance, axis=-1)  # N
    # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
    # https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
    bilaj = (
        np.mean([(np.var(s, axis=-1) + np.mean(s, axis=-1) ** 2) for s in sampled_predictions], axis=0)
        - ensemble_mean ** 2
    )
    stats["epistemics"] = bilaj
    stats["var_mc_array"] = sample_variance
    stats["entropy_mc_overall"] = exp_entropy(
        sampled_predictions
    )  
    return stats


def merge_stats(pickles, modeldirectory, M=5, identifier=""):
    """
    From different model pickles, collect logits, average predictions, reassemble/
    """
    proto = pickles[0]
    new = TouristLeMC(None, None, proto.idx2voc, proto.idx2label, proto.use_aleatorics, proto.posterior_sampling)

    shared = ["document", "gold"]
    redo = ["predicted", "confidence", "softmax"]  # confidence is a lame one
    drop = ['filename', 'field', 'evaluation', 'evaluated']  # just skip :)
    possible = ['mean_mc_array', 'var_mc_array', 'var_mc_overall', 'entropy_mc_overall', 'epistemics', 'aleatorics']
    if hasattr(proto, "multilabel") and proto.multilabel:
        new.multilabel = True
    else:
        new.multilabel = False

    if hasattr(proto, "sequence_labels") and proto.sequence_labels:
        new.sequence_labels = True
    else:
        new.sequence_labels = False

    for sampling, stats in proto.stats.items():
        new.create_stats_collector(sampling)
        # some items are shared over stats or between models
        for col in shared:
            if col != "gold":
                try:
                    assert proto.stats[sampling][col] == pickles[1].stats[sampling][col]
                except AssertionError as e:
                    print("NEED TO reevaluate each model!")
                    raise e
            new.stats[sampling][col] = proto.stats[sampling][col]

        if sampling == "mc":
            # source: https://github.com/OATML/bdl-benchmarks/blob/alpha/baselines/diabetic_retinopathy_diagnosis/ensemble_mc_dropout/model.py
            N = len(proto.stats[sampling]["gold"])
            samples = np.array([evaluator.stats[sampling]["raw"] for evaluator in pickles])  # M x N x T x K
            samples = np.transpose(samples, (0, 2, 1, 3))  # M x T x N x K
            model_logits = samples.reshape(-1, *samples.shape[-2:])  # T*M x N x K
        else:
            model_logits = np.array([evaluator.stats[sampling]["softmax"] for evaluator in pickles])

        new.stats[sampling]["mean_mc_array"] = model_logits
        new.stats[sampling]["softmax"] = np.mean(model_logits, axis=0)
        if new.multilabel:
            new.stats[sampling]["predicted"], new.stats[sampling]["confidence"] = multilabel_prediction(
                new.stats[sampling]["softmax"], new.idx2label
            )
            new.stats[sampling]["confidence"] = new.stats[sampling]["confidence"] * 100
        else:
            new.stats[sampling]["confidence"] = np.max(new.stats[sampling]["softmax"], -1) * 100
            new.stats[sampling]["predicted"] = np.vectorize(new.idx2label.get)(
                np.argmax(new.stats[sampling]["softmax"], -1)
            )
        new.stats[sampling] = ensemble_predictions(new.stats[sampling])
        if proto.use_aleatorics:
            new.stats[sampling]["aleatorics"] = np.mean(
                np.array([evaluator.stats[sampling]["aleatorics"] for evaluator in pickles]), axis=0
            )

    new.compute_stats(out_folder=None)
    identity = "" if M == 5 else str(M)
    identity += identifier
    new.dump(modeldirectory, identifier=identity)
    return new


def combine_ensemble(modeldirectory, M=5, identifier=""):
    """
    combine all existing eval.pickles into 1 eval.pickle
    mainly "reduce_mean" on "confidences"
    compounding in stats of 1
    """

    def uniques(modeldirectory):
        uniques = set()
        for x in os.listdir(modeldirectory):
            m = re.match("M\d+_", x)
            if m:
                uniques.add(m.group(0))
        return sorted(uniques)

    unique_identifiers = uniques(modeldirectory)

    pickles = [pickle_loader(os.path.join(modeldirectory, i + identifier + "eval.pickle")) for i in unique_identifiers]

    if pickles:
        if M != len(pickles):
            pickles = pickles[:M]
        merge_stats(pickles, modeldirectory, M=M, identifier=identifier)


def area_under_bluedots(stats):
    def surface_area(x, y):
        curve_auc = auc(x, y)
        return curve_auc

    threshes = list(range(100, -1, -1))  # list(range(0, 101))

    x = sorted([stats[t]["fp"] for t in threshes])
    y = sorted([stats[t]["uof"] for t in threshes], reverse=True)  # inverse relation

    curve_auc = surface_area(x, y)  # non-equidistant
    return round(100 * curve_auc, 4)


def plot_entity_confidence(confidences, boolean, title=""):
    def jensen_shannon_distance(p, q):
        """
        method to compute the Jenson-Shannon Distance
        between two probability distributions
        """
        # calculate m
        m = (p + q) / 2

        # compute Jensen Shannon Divergence
        divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

        # compute the Jensen Shannon Distance
        distance = np.sqrt(divergence)

        return distance

    mask = np.nonzero(boolean)[0]
    print("Correct mean ", np.mean(confidences[mask]))
    correct_mean = np.mean(confidences[mask])
    plt.scatter(np.array(mask), np.array(confidences[mask]), alpha=0.5, color="green", marker="+")
    mask = np.nonzero(~np.array(boolean))[0]
    print("Incorrect mean ", np.mean(confidences[mask]))
    incorrect_mean = np.mean(confidences[mask])
    plt.scatter(np.array(mask), np.array(confidences[mask]), alpha=0.5, color="red", marker="+")
    plt.title(f"{title}")
    plt.show()

    wasserstein_distance = np.round(
        scipy.stats.wasserstein_distance(confidences[np.nonzero(boolean)[0]], confidences[mask]), 6
    )
    # js_divergence = np.round(jensen_shannon_distance(confidences[np.nonzero(boolean)[0]], confidences[mask]), 6)
    plt.title(f"WS distance: {wasserstein_distance} ; {title}")

    sns.distplot(
        confidences[mask],
        hist=True,
        kde=False,
        color="red",
        kde_kws={'shade': True, 'linewidth': 3},
        label="Incorrect " + str(round(incorrect_mean, 4)),
    )
    sns.distplot(
        confidences[np.nonzero(boolean)[0]],
        hist=True,
        kde=False,
        color="green",
        kde_kws={'shade': True, 'linewidth': 3},
        label="Correct " + str(round(correct_mean, 4)),
    )
    plt.legend()
    plt.show()


def plot_over_properties(y_true, y_true_idx, y_pred, confidences, mask):
    """
    source: https://towardsdatascience.com/a-pathbreaking-evaluation-technique-for-named-entity-recognition-ner-93da4406930c
    Sequence length
    Entity density
    avg entity lenght
    """

    ECEs = [calculate_ECE(np.array(y_true_idx[i]), confidences[i], n_bins=10) for i in range(len(y_true))]
    Edensities = [1 - (y_true[i].count('O') / len(y_true[i])) for i in range(len(y_true))]
    Elens = [len(y_true[i]) for i in range(len(y_true))]

    sns.regplot(Edensities, ECEs, scatter_kws={"color": "blue"}, line_kws={"color": "orange"}, marker=".")

    r2 = np.corrcoef(Edensities, ECEs)[0, 1] ** 2

    plt.title("Entity density R²:" + str(round(r2, 4)))
    plt.xlabel("density")
    plt.ylabel("ECE(B=10)")
    plt.show()

    r2 = np.corrcoef(Elens, ECEs)[0, 1] ** 2

    sns.regplot(Elens, ECEs, scatter_kws={"color": "blue"}, line_kws={"color": "orange"}, marker=".", ci=68)
    plt.title("Sequence length R²:" + str(round(r2, 4)))
    plt.xlabel("seqlen")
    plt.ylabel("ECE(B=10)")
    plt.show()


def confusion_marginals(y_true, y_pred, labels, title=""):
    cm = confusion_matrix(y_true, y_pred)  # , normalize="")
    mask = np.zeros(cm.shape, dtype=bool)
    np.fill_diagonal(mask, 1)
    cm = np.ma.masked_array(cm, mask)
    cm_df = pd.DataFrame(cm, labels, labels)
    sns.heatmap(cm_df, annot=True)
    plt.title(title)
    plt.show()


def confusion_entities(y_true, y_pred, labels):
    from collections import Counter

    gold_spans = get_entities(y_true.tolist())
    pred_spans = get_entities(y_pred.tolist())

    # DEV: imperfect, yet suits purpose
    negs = [
        (label, ".".join(y_true[start : end + 1].tolist()), ".".join(y_pred[start : end + 1].tolist()))
        for label, start, end in gold_spans
        if (label, start, end) not in pred_spans
    ]
    fps = [
        (label, ".".join(y_pred[start : end + 1].tolist()), ".".join(y_true[start : end + 1].tolist()))
        for label, start, end in pred_spans
        if (label, start, end) not in gold_spans
    ]

    mispredictions = negs + fps
    entity_labels = ['O'] + sorted(set([x[0] for x in mispredictions]))
    for label in entity_labels:
        if label == "O":
            continue
        label_mispredictions = [(x[1], x[2]) for x in mispredictions if x[0] == label]
        gold_sort = Counter([x[0] for x in label_mispredictions])
        label_mispredictions = sorted(label_mispredictions, key=lambda pair: gold_sort[pair[0]], reverse=True)
        label_golds, label_preds = list(zip(*label_mispredictions))
        all_combinations = sorted(set(label_golds + label_preds))
        error_counter = Counter(label_mispredictions)
        print(label, error_counter.most_common(10))
        confusion_marginals(label_golds, label_preds, all_combinations, title=label)


def calculate_MCE(ground_truth, confidences, n_bins=100):
    import calibration as cal

    calibration_error = cal.get_ece(confidences, ground_truth, debias=False, mode='marginal')
    return calibration_error


def evaluate_model(model, identifier, stats, evaluator, entity_level=False, plot=False, debug=False):
    k = OrderedDict()
    version = model + "_" + str(identifier)

    k["version"] = version

    predicted = np.array(stats["predicted"])
    groundtruth = np.array(stats["gold"])

    label2idx = {v: k for k, v in evaluator.idx2label.items()}

    if "ood" in label2idx and "amazon_reviews" in args.identifier:
        print("POPPING ood")
        ood = label2idx["ood"]
        label2idx.pop("ood")
        evaluator.idx2label.pop(ood)

    evaluator.multilabel = True if hasattr(evaluator, "multilabel") and evaluator.multilabel else False
    if evaluator.multilabel:
        indices_groundtruth = np.array([np.vectorize(label2idx.get)(g) for g in groundtruth], dtype="object")
        indices_predicted = np.array([np.vectorize(label2idx.get)(p) for p in predicted], dtype="object")
    else:
        indices_groundtruth = np.vectorize(label2idx.get)(groundtruth)
        indices_predicted = np.vectorize(label2idx.get)(predicted)

    labels = sorted([str(x) for x in list(evaluator.idx2label.values())])
    confidences = np.array(stats["softmax"])  # , dtype="object"

    auc = tf.keras.metrics.AUC(
        num_thresholds=200,
        curve='ROC',
        summation_method='interpolation',
        dtype=None,
        thresholds=None,
        multi_label=True if evaluator.multilabel else False,
        label_weights=None,
    )
    max_idx = np.max(list(evaluator.idx2label.keys())) + 1  # should be equal to #labels

    if evaluator.sequence_labels:
        from datasets import load_metric

        metric = load_metric("seqeval")
        seqmetrics = metric.compute(predictions=[predicted.tolist()], references=[groundtruth.tolist()])
        accuracy = seqmetrics["overall_accuracy"]
        print(classification_report([groundtruth.tolist()], [predicted.tolist()]))

        stats = evaluator.get_logits(identifier, as_frame=False, entity_level=entity_level)

        if plot:
            original_seq_lengths = np.array([conf_array.shape[0] for conf_array in confidences])
            predicted_sents = np.array(reshape_to_sequencelen(predicted, original_seq_lengths))
            groundtruth_sents = np.array(reshape_to_sequencelen(groundtruth, original_seq_lengths))
            indices_groundtruth_sents = np.array(
                reshape_to_sequencelen(np.vectorize(label2idx.get)(groundtruth), original_seq_lengths)
            )
            plot_over_properties(
                groundtruth_sents, indices_groundtruth_sents, predicted_sents, confidences, mask=label2idx.get("O")
            )

        if plot and debug:
            confusion_marginals(predicted, groundtruth, labels=labels, title="Marginals confusion")
            confusion_entities(predicted, groundtruth, labels=labels)

        if entity_level:
            if plot:
                plot_entity_confidence(stats["confidence"], stats["y"], title=identifier)
            accuracy = sum(stats["y"]) / len(stats["y"])
            print(f"Entity-level accuracy: {accuracy}")
            k["version"] += "_entity"
            k["WS-D"] = np.round(
                scipy.stats.wasserstein_distance(
                    stats["confidence"][np.nonzero(stats["y"])[0]], stats["confidence"][np.nonzero(~stats["y"])[0]]
                ),
                6,
            )
            indices_groundtruth = stats["gold"] = stats["y"].astype(int)  # 0111101
            indices_predicted = stats["predicted"] = np.ones(len(stats["y"]))  # 111111
            confidences = stats["softmax"] = np.array(
                [np.array([1 - (p / 100), p / 100], dtype=np.float32) for p in stats["confidence"]]
            )
            max_idx = 2  # binary
            label2idx = dict(enumerate(range(0, 2)))
        else:
            # all what is needed to evaluate with metrics
            indices_groundtruth = np.vectorize(label2idx.get)(stats["gold"])
            indices_predicted = np.vectorize(label2idx.get)(stats["predicted"])
            confidences = stats["softmax"]

            k["MCE"] = round(calculate_MCE(indices_groundtruth, confidences), 4)
        """
        from arkham.utils.callbacks import ChunkF1
        metric = ChunkF1(evaluator.idx2label, average="weighted")        
        metric.update_state(indices_groundtruth, confidences)
        result = metric.result().numpy()
        print(result)
        """

        auc.update_state(np.eye(max_idx)[indices_groundtruth], confidences)

        metrics = [
            round(x, 4)
            for x in [
                accuracy,
                mean_squared_error(indices_groundtruth, indices_predicted),
                seqmetrics["overall_f1"],
                f1_score(indices_groundtruth, indices_predicted, average="macro"),
                log_loss(indices_groundtruth, confidences),
                calculate_ECE(indices_groundtruth, confidences),
                brier_multi(indices_groundtruth, confidences, label2idx),
                auc.result().numpy(),
                average_confidence(confidences),
                exp_entropy(confidences),
            ]
        ]

    else:

        if evaluator.multilabel:
            onehot_predicted = multilabel_encode(labels, predicted)
            onehot_gold = multilabel_encode(labels, groundtruth)

            auc.update_state(onehot_gold, confidences)

            metrics = [
                round(x, 4)
                for x in [
                    accuracy_score(onehot_gold, onehot_predicted),
                    mean_squared_error(onehot_gold, onehot_predicted),
                    f1_score(onehot_gold, onehot_predicted, average="weighted"),
                    f1_score(onehot_gold, onehot_predicted, average="macro"),
                    log_loss(onehot_gold, confidences),
                    calculate_ECE(onehot_gold, confidences),
                    brier_multi(onehot_gold, confidences, label2idx),
                    auc.result().numpy(),
                    average_confidence(confidences),
                    exp_entropy(confidences),
                ]
            ]
            k["hamming"] = hamming_loss(onehot_gold, onehot_predicted)
        else:
            auc.update_state(np.eye(max_idx)[indices_groundtruth], confidences)

            if plot:
                # confusion_marginals(predicted, groundtruth, labels=labels, title="Marginals confusion")
                plot_entity_confidence(
                    np.array(stats["confidence"]), np.array(equality(predicted, groundtruth)), title=identifier
                )

            metrics = [
                round(x, 4)
                for x in [
                    accuracy_score(groundtruth, predicted),
                    mean_squared_error(indices_groundtruth, indices_predicted),
                    f1_score(groundtruth, predicted, average="weighted"),
                    f1_score(groundtruth, predicted, average="macro"),
                    log_loss(indices_groundtruth, confidences, labels=list(evaluator.idx2label.keys())),
                    calculate_ECE(indices_groundtruth, confidences),
                    brier_multi(indices_groundtruth, confidences, label2idx),
                    auc.result().numpy(),
                    average_confidence(confidences),
                    exp_entropy(confidences),
                ]
            ]

    if "WS-D" in k:
        metrics.append(k["WS-D"])
    else:
        metrics.append(0)

    if "MCE" in k:
        metrics.append(k["MCE"])
    else:
        metrics.append(0)

    if "aleatorics" in stats or "epistemics" in stats:
        metrics.extend(
            [np.round(np.mean(stats.get("epistemics", 0)), 4), np.round(np.mean(stats.get("aleatorics", 0)), 4)]
        )
    else:
        metrics.extend([np.round(np.mean(stats.get("epistemics", 0)), 4), 0])

    # try:
    if not evaluator.sequence_labels:
        diversities, accuracies = calculate_diversity(stats, evaluator.idx2label, multilabel=evaluator.multilabel)
        KL = np.mean(diversities)
        metrics.append(np.round(KL, 6))
        k["accuracies"] = accuracies
        k["diversities"] = diversities
    else:
        metrics.append(0)

    if plot:
        plot_reliability(indices_groundtruth, confidences)

    k["unopened_field"] = {}
    metrics.extend([0, 0, 0, 0])


    for i in range(len(metrics)):
        k[metric_names[i]] = metrics[i]
    return metrics, k


def make_style(df, absolute=False):
    def highlight_max(s):
        """
        highlight the maximum in a Series.
        """
        is_max = s == s.max()
        return [
            ";".join(["color: yellow", "font-weight: bold", "text-decoration: underline"]) if v and s[i] != 0 else ''
            for i, v in enumerate(is_max)
        ]

    def highlight_min(s):
        """
        highlight the minimum in a Series.
        """
        is_min = s == s.min()  # 'background-color: green',
        return [
            ";".join(["color: yellow", "font-weight: bold", "text-decoration: underline"]) if v and s[i] != 0 else ''
            for i, v in enumerate(is_min)
        ]

    # softmax-> difference colouring?
    lower_better_cols = [col for col in df.columns if df.dtypes[col] not in ["object"] and "↓" in col]
    higher_better_cols = [col for col in df.columns if df.dtypes[col] not in ["object"] and not "↓" in col]
    if absolute:
        for col in higher_better_cols:
            df[col] = df[col].apply(lambda x: abs(x))
    palette = "RdYlGn"
    style = df.style.background_gradient(cmap=palette, subset=higher_better_cols)
    style = (
        style.background_gradient("RdYlGn" + "_r", subset=lower_better_cols)
        .apply(highlight_max, subset=higher_better_cols)
        .apply(highlight_min, subset=lower_better_cols)
        # .set_na_rep("-")
    )
    style = style.format("{:.4%}")
    return style


def deduce_methods(df, modelpaths=None, modelroot=None):  # data_set_models
    def refine_method(df):
        methods = [
            "Unregularized",  # 1
            "Regularized",  # 1
            "MC Dropout",  # 1
            "Heteroscedastic",  # 1
            "MC Heteroscedastic",  # 2
            "Concrete Dropout",  # 1
            "Heteroscedastic Concrete Dropout",  # 2
            "MC Concrete Dropout",  # 2
            "MC Heteroscedastic Concrete Dropout",  # 3
            "Deep Ensemble",  # 1
            "Deep Ensemble Regularized",  # 1
            "MC Dropout Ensemble",  # 2
            "Concrete Dropout Ensemble",  # 2
            "MC Concrete Dropout Ensemble",  # 3
            "Heteroscedastic Concrete Dropout Ensemble",  # 3
            "MC Heteroscedastic Concrete Dropout Ensemble",  # 4
        ]
        refs = []
        for i, row in df.iterrows():
            ref = []
            if row["sampling"] == "mc":
                ref.append("MC")
            if row["use_aleatorics"]:
                ref.append("Heteroscedastic")
            if row["concrete_dropout"]:
                ref.append("Concrete Dropout")
            if float(row["ensemble"]) > 1:
                if len(ref) == 0:
                    ref.append("Deep Ensemble")
                    if "baseline" in row["version"]:
                        ref.append("Regularized")
                else:
                    ref.append("Ensemble")
                if not "M5" in row["version"]:
                    print("****** MIGHT NEED TO RESET PARAMS *****", row["version"])
            if " ".join(ref) == "MC":
                ref = "MC Dropout"

            elif len(ref) == 0:
                if "nodropout" in row["version"]:
                    ref = "Unregularized"
                else:
                    ref = "Regularized"
            else:
                ref = " ".join(ref)
            refs.append(ref)
        df["ref"] = refs
        return df

    def refine_aleatorics(df):
        defs = {
            "num_layers": lambda x: 2 if not "1layer" in x else 1,
            "distribution": lambda x: "normal" if not "multivar" in x else "multivar",
            "event_shape": lambda x: "independent" if "independent" in x else "",
            "sigma_scaler": lambda x: "exp0.5"
            if not "nofunc" in x and not "softplus" in x
            else "nofunc"
            if "nofunc" in x
            else "softplus"
            if "softplus" in x
            else "",
        }

        for d in defs:  # if False?
            df[d] = df["use_aleatorics"].apply(defs[d])
            df.loc[df["use_aleatorics"] == str(False), d] = ""
        return df

    def search_base(version):
        try:
            base = re.search("nodropout|baseline|aleatoric", version).group(0)
        except:
            base = ""
        return base

    # new_key: old_key OR function to apply to version
    identifiers = {
        "dataset": "identifier",
        "base": lambda x: search_base(x),
        "sampling": lambda x: re.search("nonbayesian|mc", x).group(0),
        "use_aleatorics": "use_aleatorics",
        "concrete_dropout": "dropout_concrete",  # might not work with older models :/
        "ensemble": "ensemble",  # might not work with older models :/
        "ood": "ood",  # might not work with older models :/
        "arx": "model",
    }
    backup = {
        "ensemble": lambda x: True if re.search("M\d+_", x) else 1,
        "ood": lambda x: True if "ood" in x else False,
        "use_aleatorics": lambda x: False if "baseline" in x else "",
        "concrete_dropout": lambda x: True if "_True_" in x else False,
    }

    if not modelpaths:
        modelpaths = [
            os.path.join(modelroot, p)
            for p in df["version"].apply(lambda x: re.sub("M[01234]_", "", re.split("_(mc|nonbayesian)", x)[0]))
        ]

    params = [load_model_config(modelpath) for modelpath in modelpaths]

    for new_key, identifier in identifiers.items():
        if isinstance(identifier, str):
            df[new_key] = [param.get(identifier) for param in params]
            if identifier == "ensemble":
                df["ensemble"] = df.apply(
                    lambda x: x["ensemble"] if not re.search("M[01234]_", x["version"]) else 1, axis=1
                )
            if not any(df[new_key]):
                df[new_key] = df["version"].apply(backup[new_key])
        else:
            df[new_key] = df["version"].apply(identifier)

    df = refine_method(df)

    for new_key in identifiers:
        df[new_key] = df[new_key].astype(str)

    if any(df["use_aleatorics"]):
        df = refine_aleatorics(df)

    return df


def benchmark_filter(dataset, models, ood=False):
    def combine(c):
        r = dataset + "_" + c
        if ood:
            r += "_ood"
        return r

    configurations = list(
        reversed(
            [
                "aleatorics_M5_concrete",
                "aleatorics_M5",
                "baseline_M5_concrete",
                "baseline_M5",
                "nodropout_M5",
                "aleatorics_concrete",
                "baseline_concrete",
                "aleatorics",
                "baseline",
                "nodropout",
            ]
        )
    )
    configurations = [combine(c) for c in configurations]  # +_ood
    keep = []
    bases = [os.path.basename(model) for model in models]
    for c in configurations:
        if c in bases:
            keep.append(bases.index(c))
    return [models[i] for i in keep]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import argparse

    parser = argparse.ArgumentParser("""Stats and Metrics""")
    parser.add_argument("modeldirectory", type=str, default="/mnt/lerna/models")
    parser.add_argument(
        "-d",
        dest="datasets",
        type=str,
        default=list(reversed(["conll03", "yelp2013", "yelp2014", "yelp2015", "imdb", "twitter", "wos", "Reuters"])),
    )
    parser.add_argument(
        "-i", dest="identifier", type=str, default="", help="identifier to add to eval.pickle [ensemble-member]"
    )
    parser.add_argument("-f", dest="filter", type=str, default="", help="to specify even more per dataset")
    parser.add_argument(
        "-r", dest="blacklist", type=str, default="focaccia", help="to specify even more per dataset what not"
    )
    parser.add_argument(
        "-b", dest="benchmark", action="store_true", default=False, help="to specify even more per dataset what not"
    )
    parser.add_argument("-p", dest="plot", action="store_true", default=False, help="plot automation")
    parser.add_argument("-e", dest="ensembler", action="store_true", default=False, help="average over ensemble models")
    parser.add_argument("-m", dest="ensemblesize", type=int, default=5, help="average over M ensemble models")
    parser.add_argument(
        "--el", dest="entity_level", action="store_true", default=False, help="activate entity-level evaluation"
    )

    args = parser.parse_args()

    modeldirectory = args.modeldirectory
    datasets = args.datasets.split("|") if not isinstance(args.datasets, list) else args.datasets

    if not os.path.exists(os.path.join(modeldirectory, str(args.identifier) + "eval.pickle")) or args.ensemblesize != 5:
        if not os.path.exists(os.path.join("/mnt/lerna/models", modeldirectory, str(args.identifier) + "eval.pickle")):
            combine_ensemble(modeldirectory, M=args.ensemblesize, identifier=str(args.identifier))
        else:
            modeldirectory = os.path.join("/mnt/lerna/models", modeldirectory)

    eval_pickle = os.path.join(modeldirectory, str(args.identifier) + "eval.pickle")

    if os.path.exists(eval_pickle) and datasets not in [["0"], ["voc"]]:  # simplex model to be evaluated!
        keep = []
        t = PrettyTable(["version"] + metric_names)
        evaluator = pickle_loader(eval_pickle)
        model = os.path.basename(modeldirectory) + str(args.identifier)
        for identifier, stats in evaluator.stats.items():
            # print(f"** {identifier} **")
            metrics, k = evaluate_model(
                model, identifier, stats, evaluator, entity_level=args.entity_level, plot=args.plot
            )
            t.add_row([k["version"]] + metrics)
            keep.append(k)
        if "M5" in modeldirectory and args.ensembler:
            for m in ["M0_", "M1_", "M2_", "M3_", "M4_"]:
                if args.identifier == "raw":
                    identity = ""
                else:
                    identity = str(args.identifier)
                evaluator = pickle_loader(os.path.join(modeldirectory, identity + str(m) + "eval.pickle"))
                model = os.path.basename(modeldirectory) + str(args.identifier) + str(m)
                for sampling, stats in reversed(evaluator.stats.items()):
                    if sampling == "mc" and "nodropout" in model:
                        continue
                    print(f"** {sampling} **")
                    metrics, k = evaluate_model(
                        model, sampling, stats, evaluator, entity_level=args.entity_level, plot=args.plot
                    )
                    t.add_row([k["version"]] + metrics)
                    keep.append(k)
        t.sortby = "NLL(↓)"
        # t.reversesort = True #only with acc
        print(t)
        df = pd.DataFrame(keep)
        df = deduce_methods(df, modelpaths=[modeldirectory] * len(keep), modelroot="")
        style = make_style(df)
        style.to_excel(os.path.join(modeldirectory, model + '.xlsx'), index=False)
        if args.plot:
            plot_df_automation(df)
        # print(df.to_latex(index=False))

        del os.environ['CUDA_VISIBLE_DEVICES']
        sys.exit(1)

    models = sorted(
        [
            os.path.join(modeldirectory, x)
            for x in os.listdir(modeldirectory)
            if not ".out" in x
            and not ".log" in x
            and not "ood" in x
            and not "binary" in x
            and not "xlsx" in x
            and not "ovadia" in x
        ]
    )

    for dataset in datasets:
        keep = []
        data_set_models = sorted(
            [
                model
                for model in models
                if dataset.lower() in model.lower() and args.filter in model and not args.blacklist in model
            ]
        )
        if args.benchmark:
            data_set_models = benchmark_filter(dataset, data_set_models, ood=False)
        if not data_set_models:
            continue
        t = PrettyTable(["version"] + metric_names)
        for modelpath in data_set_models:
            if not os.path.exists(os.path.join(modelpath, "eval.pickle")) or not os.path.exists(
                os.path.join(modelpath, "params.json")
            ):
                print(f"{modelpath} has not been evaluated properly")
                continue
            print(modelpath)
            try:
                if str(args.identifier) == "raw":
                    assert not "M5" in modelpath
                evaluator = pickle_loader(os.path.join(modelpath, str(args.identifier) + "eval.pickle"))
            except Exception as e:
                print("Fallback to regular eval.pickle", e)
                evaluator = pickle_loader(os.path.join(modelpath, "eval.pickle"))

            model = os.path.basename(modelpath)
            for identifier, stats in reversed(evaluator.stats.items()):
                if identifier == "mc" and "nodropout" in model:
                    continue
                print(f"** {identifier} **")
                metrics, k = evaluate_model(
                    model, identifier, stats, evaluator, entity_level=args.entity_level, plot=args.plot
                )
                t.add_row([k["version"]] + metrics)
                keep.append(k)
            if "M5" in modelpath and args.ensembler:
                for m in ["M0_", "M1_", "M2_", "M3_", "M4_"]:
                    if args.identifier == "raw":
                        identity = ""
                    else:
                        identity = str(args.identifier)
                    evaluator = pickle_loader(os.path.join(modelpath, str(identity) + str(m) + "eval.pickle"))
                    model = os.path.basename(modelpath) + str(identity) + str(m)
                    for sampling, stats in reversed(evaluator.stats.items()):
                        if sampling == "mc" and "nodropout" in model:
                            continue
                        print(f"** {sampling} **")
                        metrics, k = evaluate_model(
                            model, sampling, stats, evaluator, entity_level=args.entity_level, plot=args.plot
                        )
                        t.add_row([k["version"]] + metrics)
                        keep.append(k)

        if keep:
            df = pd.DataFrame(keep)
            df = deduce_methods(df, modelpaths=None, modelroot=modeldirectory)
            style = make_style(df)
            out = dataset
            if args.filter:
                out += "_" + args.filter
            if args.benchmark:
                out += "_benchmark"
            if args.ensembler:
                out += "_M"
            if args.identifier == "raw":
                out += "raw_Mlevel"
            style.to_excel(os.path.join(modeldirectory, out + '.xlsx'), index=False)
            t.sortby = "NLL(↓)"
            # t.reversesort = True #only with acc
            print(t)
            # plot_multiples(keep, title=dataset)
            print()
            # print(df.to_latex(index=False))
            if args.plot:
                plot_df_automation(df)
        #     calculate_diversity(evaluator)

    if datasets in [["0"], ["voc"]]:
        df = pd.DataFrame.from_dict(keep)
        df["version"] = df["version"].apply(lambda x: os.path.basename(modeldirectory) + "/" + str(x))
        df["base"] = df["version"].apply(lambda x: re.search("nodropout|baseline|aleatoric", x).group(0))
        df["method"] = df["version"].apply(lambda x: re.search("nonbayesian|mc", x).group(0))
        df["rate"] = df["version"].apply(lambda x: search_rate(x))
        df = df[~df["rate"].isin(["1", "2", "5", "10", "20", "50", "100"])].reset_index(drop=True)

        df.to_csv(os.path.join(modeldirectory, "rates_" + datasets[0] + ".csv"), index=False)

    del os.environ['CUDA_VISIBLE_DEVICES']
