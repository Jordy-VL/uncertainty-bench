import os
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

from collections import Iterable
from tqdm import tqdm
from arkham.utils.model_utils import MODELROOT

import numpy as np
from netcal.scaling import TemperatureScaling
from arkham.utils.model_utils import MODELROOT, _get_weights, load_model_path


def surface_area(x, y):
    from sklearn.metrics import auc

    curve_auc = auc(x, y)
    return curve_auc


def tempscaler(confidences, ground_truth):
    temperature = TemperatureScaling()
    temperature.fit(confidences, ground_truth)
    calibrated = temperature.transform(confidences)
    return temperature, calibrated


def evaluation_measures(gold, predicted):
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
        auc,
        roc_auc_score,
    )

    report = classification_report(gold, predicted)
    print('Classification report:')
    print(report)
    matrix = confusion_matrix(gold, predicted)
    print('Confusion matrix:')
    print(matrix)
    return classification_report(gold, predicted, output_dict=True), matrix


def compare_predict_gold(gold, predicted):
    checker = []
    assert len(gold) == len(predicted)
    for index, x in enumerate(predicted):
        # if
        # print("predicted: ",x,"\tgold: ",gold[index])
        # wait()
        if x == gold[index]:
            checker.append("correct")
        else:
            checker.append("incorrect")
    return checker


def plot_probabilities(label_predicted, label_gold, label_probs, thresh, title="label", verbose=False):
    from matplotlib import pyplot as plt
    import seaborn as sns

    check_label = compare_predict_gold(label_gold, label_predicted)
    len_list = list(range(0, len(check_label)))
    fig = plt.figure()
    plot = sns.stripplot(x=len_list, y=label_probs, hue=check_label, jitter=True, palette=['g', 'r'])
    plot.set_title("Prediction  -  Ground truth:\n " + title, fontsize=12, fontweight='bold')
    sup_title = 'Prediction  -  Ground truth Flagged'
    plt_title = "document type"
    plot.axes.get_xaxis().set_ticks([])
    plt.xlabel("Document instance")
    plt.ylabel("Confidence level (%)")
    plot.yaxis.label.set_rotation(0)
    plot.yaxis.labelpad = 40
    plot.xaxis.labelpad = 10
    plt.plot(len_list, [thresh for x in len_list], color='blue', linewidth=2.0, linestyle=":")
    plt.show()
    # plt.savefig(title + ".png")


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=130_107, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def multiclass_argmaxprobs(output_probabilities):
    a_argmax = np.expand_dims(np.argmax(output_probabilities, axis=1), axis=1)
    return np.ravel(np.take_along_axis(output_probabilities, a_argmax, 1))


def prob_to_predict(output_probabilities, classes):
    argmaxprobs = multiclass_argmaxprobs(output_probabilities)
    indices = np.ravel(np.argmax(output_probabilities, axis=1))
    return [classes[i] for i in indices], [100 * prob for prob in argmaxprobs.tolist()]


def get_sample_weights(y):
    counter = np.unique(y, return_counts=True, axis=0)  # returns tuple of len2 (1 unique items array, 2 counts array)
    counterlist = dict([(arr.tobytes(), counter[1][index]) for index, arr in enumerate(counter[0])])
    majorityindex = np.argmax(counter[1])
    majority = counter[0][majorityindex]
    majoritycount = counter[1][majorityindex]
    sample_weights = np.array([float(majoritycount / counterlist[x.tobytes()]) for x in y])
    return sample_weights


def prec_recall(y_test, y_score, n_classes):
    from sklearn.metrics import precision_recall_curve

    sample_weight = (
        None  # np.vectorize(lambda x: 1 if not x else 0.1)(y_test) ; should make it balanced? | acts like a cost
    )
    precision, recall, thresholds = precision_recall_curve(y_test, y_score, sample_weight=sample_weight)

    thresh_dict = {int(t): (p, r) for p, r, t in zip(precision, recall, thresholds)}
    # make unique by integer -> dictionary and round

    for t, (p, r) in sorted(thresh_dict.items(), key=lambda x: (2 * x[1][0] + x[1][1]) / 2):
        print(f"precision: {p}, recall: {r}, thresh: {t}")
    plt.plot(recall, precision, lw=2, label="")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()


def make_feature_vectors(d, feature_cols):
    # WEIRD BEHAVIOUR FROM SOFTMAX ARRAY IN PD
    vectors = []
    for i in tqdm(range(len(d))):
        vector = []
        for col in feature_cols:
            value = d[col].iloc[i]
            if isinstance(value, Iterable):
                vector.extend(value)
            else:
                vector.append(value)
        vectors.append(np.array(vector))
    vectors = np.array(vectors)
    print(vectors.shape)
    return vectors


def plotprob(y, y_score, thresh, ax=None, plot=True):
    stats = {}
    stats["pos_over"], stats["pos_under"], stats["neg_over"], stats["neg_under"] = [], [], [], []

    def estimate(status, value, index):
        if status:
            if value >= thresh:
                stats["pos_over"].append(index)
            else:
                stats["pos_under"].append(index)
        else:
            if value >= thresh:
                stats["neg_over"].append(index)
            else:
                stats["neg_under"].append(index)

    if max(y_score) < 1.01:
        y_score = np.array(y_score * 100)
    for index, (status, value) in enumerate(zip(y, y_score)):
        estimate(status, value, index)

    uof, fp, uof_pure = (
        round((len(stats["pos_over"]) + len(stats["neg_over"])) / len(y), 4),
        round(len(stats["neg_over"]) / max(1, ((len(stats["pos_over"]) + len(stats["neg_over"])))), 2),
        round((len(stats["pos_over"])) / len(y), 4),
    )

    title = f"thresh: {thresh} uof: {uof} fp: {fp} uofp: {uof_pure}"
    print(title)
    if plot:
        if not ax:
            ax = plt.subplots()[1]
        ax.plot(thresh)
        ax.scatter(stats["pos_over"], np.array(y_score)[stats["pos_over"]], alpha=0.5, color="green", marker="+")
        ax.scatter(stats["neg_over"], np.array(y_score)[stats["neg_over"]], alpha=0.5, color="red", marker=".")
        ax.scatter(stats["pos_under"], np.array(y_score)[stats["pos_under"]], alpha=0.5, color="orange", marker="+")
        ax.scatter(stats["neg_under"], np.array(y_score)[stats["neg_under"]], alpha=0.5, color="blue", marker=".")
        ax.legend()
        ax.set_title(title)
    return uof, fp, uof_pure


def plot_multiples(list_of_stats, title="", labels=""):
    import seaborn as sns

    sns.set(color_codes=True)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, stats in enumerate(list_of_stats):
        threshes = sorted(stats.keys())
        x = [stats[t]["fp"] for t in threshes]
        y = [stats[t]["uof"] for t in threshes]
        # curve_auc = surface_area(x, y)  # non-equidistant
        # print("computed AUC: {}".format(curve_auc))
        plt.xlabel('FP')
        plt.ylabel('Unopened Field')
        plt.plot(x, y, color=cycle[i], label=labels[i])
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_df_automation(df):
    import matplotlib._color_data as mcd
    from cycler import cycler
    import seaborn as sns

    def plot_multiples(list_of_stats, title="", labels=""):
        sns.set(color_codes=True)
        cycle = sns.color_palette("Paired", len(list_of_stats))
        ls_cycle = cycler(linestyle=['-', '--', '-.', ':'])
        cc = ls_cycle()
        for i, stats in enumerate(list_of_stats):
            threshes = list(range(0, 101))
            x = sorted([stats[t]["fp"] for t in threshes], reverse=True)
            y = sorted([stats[t]["uof"] for t in threshes], reverse=True)
            plt.xlabel('FP')
            plt.ylabel('Unopened Field')
            plt.plot(x, y, label=labels[i], **next(cc), color=cycle[i])
        if title:
            plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, frameon=False)
        plt.show()

    df.replace(str(np.NaN), np.NaN, inplace=True)
    df.replace(np.NaN, "", inplace=True)
    df = df.sort_values(by=["AUB"], ascending=True)

    labels = ["-".join([str(y) for y in x]) for x in zip(df["version"], df["AUB"])]
    plot_multiples(df["unopened_field"], labels=labels)


def calibrate_model(version):
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    model, _config = load_model_path(modelpath)
    _config["out_folder"] = modelpath

    df = pd.read_pickle(os.path.join(modelpath, "logits.pkl"))

    df["y"] = df["gold"].apply(lambda x: _config["label2idx"][x])

    dev = df.loc[df["dataset"] == "dev"].reset_index(drop=True)
    test = df.loc[df["dataset"] == "test"].reset_index(drop=True)

    X_train, X_test, y_train, y_test = (
        make_feature_vectors(dev, ["softmax"]),
        make_feature_vectors(test, ["softmax"]),
        np.ravel(dev["y"]),
        np.ravel(test["y"]),
    )

    # X_train_cal = temperature.transform(X_train)

    temperature = TemperatureScaling()
    temperature.fit(X_train, y_train)
    X_test_cal = temperature.transform(X_test)
    preds, argmax_probs = prob_to_predict(X_test_cal, list(_config["label2idx"].keys()))
    boolean = [1 if _config["label2idx"][preds[i]] == y_test[i] else 0 for i in range(len(preds))]

    """
    threshes = [80, 82, 84, 85, 86, 87]
    f, ax = plt.subplots(len(threshes))
    for i, thresh in enumerate(threshes):
        plotprob(boolean, argmax_probs, thresh, ax[i])
    plt.show()
    """

    before = {}
    prepreds, preprobs = prob_to_predict(X_test, list(_config["label2idx"].keys()))
    before_boolean = [1 if _config["label2idx"][prepreds[i]] == y_test[i] else 0 for i in range(len(prepreds))]
    print("*** BEFORE ****")
    threshes = np.arange(0, 100, 1)
    for i, thresh in enumerate(threshes):
        before[thresh] = {}
        before[thresh]["uof"], before[thresh]["fp"], before[thresh]["uof_pure"] = plotprob(
            before_boolean, preprobs, thresh, plot=False
        )

    after = {}
    threshes = np.arange(0, 100)
    for i, thresh in enumerate(threshes):
        after[thresh] = {}
        after[thresh]["uof"], after[thresh]["fp"], after[thresh]["uof_pure"] = plotprob(
            boolean, argmax_probs, thresh, plot=False
        )

    plot_multiples([before, after], title="temp scaling", labels=["before", "after"])

    sys.exit(1)


def main(version):
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    df = pd.read_pickle(os.path.join(modelpath, "logits.pkl"))

    df["y"] = df["y"].apply(lambda x: int(x))

    # anything type integer
    feature_cols = ["softmax", "aleatorics", "epistemics", "entropy"]

    dev = df.loc[df["dataset"] == "dev"].reset_index(drop=True)
    test = df.loc[df["dataset"] == "test"].reset_index(drop=True)

    X_train, X_test, y_train, y_test = (
        make_feature_vectors(dev, feature_cols),
        make_feature_vectors(test, feature_cols),
        np.ravel(dev["y"]),
        np.ravel(test["y"]),
    )

    before = {}
    preprobs = 100 * np.max(np.array([np.array(x) for x in test["softmax"].values]), axis=-1)
    print("*** BEFORE ****")
    threshes = np.arange(0, 100, 1)
    for i, thresh in enumerate(threshes):
        before[thresh] = {}
        before[thresh]["uof"], before[thresh]["fp"], before[thresh]["uof_pure"] = plotprob(
            y_test, preprobs, thresh, plot=False
        )

    label = "logreg"
    print("** ", label, " **")
    clf = LogisticRegression(
        random_state=0
    )  # SVC(kernel="rbf", probability=True)  # RandomForestClassifier(n_estimators=300, random_state=1)  #   #
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    preds, argmax_probs = prob_to_predict(probs, clf.classes_)

    prec_recall(y_test, argmax_probs, 2)

    evaluation_measures(y_test, preds)
    print("log loss before: ", log_loss(y_test, probs))

    print("*** AFTER ****")
    threshes = [80, 82, 84, 85, 86, 87]
    f, ax = plt.subplots(len(threshes))
    for i, thresh in enumerate(threshes):
        plotprob(y_test, argmax_probs, thresh, ax[i])
    plt.show()

    after = {}
    threshes = np.arange(0, 100, 1)
    for i, thresh in enumerate(threshes):
        after[thresh] = {}
        after[thresh]["uof"], after[thresh]["fp"], after[thresh]["uof_pure"] = plotprob(
            y_test, argmax_probs, thresh, plot=False
        )

    plot_multiples([before, after], title=label, labels=["before", "after"])
    # SO, to deploy this one; we need -> previous model; predict -> translate predict to feature vectors -> predict with calibrator -> prediction of model with confidence of calibrator
    # plot_probabilities(y_test, preds, argmax_probs, 50, title="probs-" + label, verbose=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("""Get logits for a model""")
    parser.add_argument("version", type=str, default="")
    parser.add_argument("identifier", nargs="?", type=str, default="nonbayesian")
    parser.add_argument("-c", dest="calibrator", default=False, action="store_true", help="test temperature scaling")
    args = parser.parse_args()
    if args.calibrator:
        calibrate_model(args.version)
    main(args.version)
