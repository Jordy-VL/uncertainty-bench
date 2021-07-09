#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

import os
import sys
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import regex as re
import arviz as az

az.style.use("arviz-darkgrid")

import pandas as pd
import seaborn as sns
import matplotlib
import sklearn

from netcal.metrics import ECE
from scipy.stats import pointbiserialr, pearsonr

from arkham.utils.model_utils import MODELROOT, multiclass_argmaxprobs, load_model_config
from arkham.utils.utils import pickle_loader, num_plot
from arkham.utils.custom_metrics import entropy, exp_entropy, pred_entropy, mutual_info, AUROC_PR, FPRatRecall

from arkham.Bayes.Quantify.compare import deduce_methods, make_style, evaluate_model, benchmark_filter
from arkham.Bayes.Quantify.evaluate import TouristLeMC, determine_ood_classes
from arkham.Bayes.Quantify.evaluate import main as generate_evaluation_data
import warnings

warnings.filterwarnings("ignore")


def plot_reliability(ground_truth, confidences, n_bins=10):
    from netcal.presentation import ReliabilityDiagram

    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(confidences, ground_truth)  # visualize miscalibration of uncalibrated


def rank_by_metric(ranker, greater_is_better=True):
    reverse = True if greater_is_better else False
    return [i for i, x in sorted(enumerate(ranker), key=lambda pair: pair[1], reverse=reverse)]


def plot_pdf(pdf):
    az.plot_dist(pdf, color="C2", label="softmax")
    plt.show()


def get_bounds(means, aleatorics, mode="upper", sample=0):
    if mode == "lower":
        bounds = np.array([max(0, mu - sigma) for mu, sigma in list(zip(means, aleatorics))])
    elif mode == "upper":
        bounds = np.array([min(1, mu + sigma) for mu, sigma in list(zip(means, aleatorics))])

    if sample:
        mask = np.random.choice(list(range(len(bounds))), sample)
        bounds = bounds[mask]
    return bounds


def barplotter(boundz):
    for correctness in ["correct", "incorrect"]:
        sns.barplot(
            x=list(range(len(bounds[correctness][0]))),
            y=[0.866_625_454_397_026_2, 0.576_800_958_400_334_7],
            hue=["green", "red"],
            orient='v',
            errcolor="darkred",
            label=correctness,
        )
        plt.title("confidence bounds")
        plt.show()


def upper_confidence_bound(confs, boolean, aleatorics=None, epistemics=None):
    if not any(aleatorics) and not any(epistemics):
        aleatorics = np.array([0 for _ in range(len(confs))])
        epistemics = np.array([0 for _ in range(len(confs))])
    else:
        sns.kdeplot(
            aleatorics,
            data2=epistemics,
            shade=False,
            vertical=False,
            kernel='gau',
            bw='scott',
            gridsize=100,
            cut=3,
            clip=None,
            legend=True,
            cumulative=False,
            shade_lowest=True,
            cbar=plt.get_cmap('hot'),
        )
        plt.title("Scatterplot")
        plt.show()
    boundz = {}
    for correctness in ["correct", "incorrect"]:
        if correctness == "correct":
            mask = np.nonzero(boolean)
            color = "green"
        else:
            mask = boolean == False
            color = "red"

        c_confs = confs[mask]
        c_variance = aleatorics[mask]
        c_epistemics = epistemics[mask]
        print(
            correctness,
            ": mean ",
            np.mean(c_confs),
            " aleatorics: ",
            np.mean(c_variance),
            " epistemics: ",
            np.mean(c_epistemics),
        )

        boundz[correctness] = (c_confs, c_variance)
        # bounds = get_bounds(c_confs, c_variance, mode="lower", sample=0)
        """
        boundz[correctness] = bounds
        indices = np.arange(len(bounds))
        """
        mask1 = np.random.choice(list(range(len(c_variance))), 100)

        # sns.stripplot(x=np.arange(len(mask1)), y=c_variance[mask1], jitter=False, color=color)
        # sns.stripplot(x=indices, y=bounds, jitter=False, color=color)
        # plt.xticks([])

    # barplotter(boundz)

    plt.title("confidence bounds")
    plt.show()


def plot_buffers(data):
    def plot_to_buf(data, height=2800, width=2800, inc=0.3):
        xlims = (data[:, 0].min(), data[:, 0].max())
        ylims = (data[:, 1].min(), data[:, 1].max())
        dxl = xlims[1] - xlims[0]
        dyl = ylims[1] - ylims[0]

        print('xlims: (%f, %f)' % xlims)
        print('ylims: (%f, %f)' % ylims)

        buffer = np.zeros((height + 1, width + 1))
        for i, p in enumerate(data):
            print('\rloading: %03d' % (float(i) / data.shape[0] * 100), end=' ')
            x0 = int(round(((p[0] - xlims[0]) / dxl) * width))
            y0 = int(round((1 - (p[1] - ylims[0]) / dyl) * height))
            buffer[y0, x0] += inc
            if buffer[y0, x0] > 1.0:
                buffer[y0, x0] = 1.0
        return xlims, ylims, buffer

    # data.shape = (310216, 2) <<< your data here
    xlims, ylims, I = plot_to_buf(data, height=h, width=w, inc=0.3)
    ax_extent = list(xlims) + list(ylims)
    plt.imshow(I, vmin=0, vmax=1, cmap=plt.get_cmap('hot'), interpolation='lanczos', aspect='auto', extent=ax_extent)
    plt.grid(alpha=0.2)
    plt.title('Latent space')
    plt.colorbar()


def scatterplots(d, x, y, color=None):
    plt.scatter(np.array(d[x]), np.array(d[y]), alpha=0.5, color=color, marker="+")
    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    if color:
        import matplotlib.patches as mpatches

        green_patch = mpatches.Patch(color='green', label='correct')
        red_patch = mpatches.Patch(color='red', label='error')
        plt.legend(handles=[green_patch, red_patch], loc="best")
    # plt.show()


# TODO(yovadia): Write unit-tests.
def compute_accuracies_at_confidences(labels, probs, thresholds):
    # Accuracy vs Confidence curves:
    """Compute accuracy of samples above each confidence threshold.
    Args:
      labels: Array of integer categorical labels.
      probs: Array of categorical probabilities.
      thresholds: Array of floating point probability thresholds in [0, 1).
    Returns:
      accuracies: Array of accuracies over examples with confidence > T for each T
          in thresholds.
      counts: Count of examples with confidence > T for each T in thresholds.
    """
    assert probs.shape[:-1] == labels.shape

    predict_class = probs.argmax(-1)
    predict_confidence = probs.max(-1)

    shape = (len(thresholds),) + probs.shape[:-2]
    accuracies = np.zeros(shape)
    counts = np.zeros(shape)

    eq = np.equal(predict_class, labels)
    for i, thresh in enumerate(thresholds):
        mask = predict_confidence >= thresh
        counts[i] = mask.sum(-1)
        accuracies[i] = np.ma.masked_array(eq, mask=~mask).mean(-1)
    return accuracies, counts


def dist_attribute_ood_bool(df, attribute, identifier=None, path="", plot=False):
    fig = plt.figure()
    for correctness in [True, False]:
        if correctness:
            label1 = "correct_IID"
            label2 = "correct_OOD"
        else:
            label1 = "incorrect_IID"
            label2 = "incorrect_OOD"

        sns.distplot(
            df[(df["known"] == 1) & (df["y"] == correctness)][attribute],
            hist=False,
            kde=True,
            color="green" if correctness else "red",
            kde_kws={'shade': True, 'linewidth': 3},
            label=label1,
        )
        sns.distplot(
            df[(df["known"] == 0) & (df["y"] == correctness)][attribute],
            hist=False,
            kde=True,
            color="purple",
            kde_kws={'shade': True, 'linewidth': 3},
            label=label2,
        )

        if args.crossdomain:
            sns.distplot(
                df[(df["y"] == "ood") & (df["y"] == correctness)][attribute],
                hist=False,
                kde=True,
                color="purple",
                kde_kws={'shade': True, 'linewidth': 3},
                label="incorrect_CROSS",
            )
        # correctood = df[(df["known"] == 0) & (df["y"] == True)]
        # if len(correctood) > 0:

    plt.legend()
    # plt.xlim(0,1.1)
    # plt.title(attribute + "_density " + identifier)
    out = os.path.join(path, attribute + "_" + identifier)
    if args.crossdomain:
        out += args.identifier
    if os.path.exists(out + ".png"):
        os.remove(out + ".png")
    plt.savefig(out)
    # if plot:
    #    plt.show()


def threed(df, uncertainty, identifier=None, path="", plot=False):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(30, 20))
    fig.suptitle(str(uncertainty) + " projection")
    ax = fig.add_subplot(111, projection='3d')

    df = df.sort_values(by=["known"])

    from sklearn.preprocessing import MinMaxScaler

    df[uncertainty] = MinMaxScaler().fit_transform(df[[uncertainty]])

    for correctness in [True, False]:
        for known in [0, 1]:
            label = ""
            color = "black"
            if correctness and known:
                color = "green"
                label = "correct_IID"
            # if correctness and not known:
            #     color = "blue"
            #     label = "correct_unknown"
            if not correctness and known:
                color = "red"
                label = "incorrect_IID"
            if not correctness and not known:
                color = "purple"
                label = "incorrect_OOD"

            data = df[(df["known"] == known) & (df["y"] == correctness)]

            xs = data.index.values
            ys = data["confidence"]
            zs = data[uncertainty]
            marker = "+" if correctness else "*"

            ax.scatter(xs, ys, zs, marker=marker, label=label, color=color)

    ax.set_xlabel('index')
    ax.set_ylabel('confidence')
    ax.set_zlabel(uncertainty)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(path, uncertainty + "_" + identifier + "_3D")
    if os.path.exists(out + ".png"):
        os.remove(out + ".png")
    # plt.savefig(out)
    # if plot:
    #     plt.show()


def plot_cm(gold, predicted, name=""):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(gold, predicted)
    print("Confusion matrix:")
    print(cm)

    # Show confusion matrix
    plt.matshow(cm)
    plt.title('Confusion matrix of the %s classifier' % name)
    plt.colorbar()
    plt.show()


def confidence_ood_plot(labels, probs):
    thresholds = np.linspace(0, 1, 10, endpoint=False)
    accuracies, counts = compute_accuracies_at_confidences(labels, probs, thresholds)


def ranker(df):
    metrics = [x + "_R" for x in ["confidence", "aleatorics", "epistemics", "entropy", "mutual_information"]]
    ranks = OrderedDict()
    for i, row in df.iterrows():
        for metric in metrics:
            v = abs(row[metric])
            ranks[row["ref"] + "_" + metric] = v
    sorting = sorted(list(ranks.values()), reverse=True)
    """
    new_df = pd.DataFrame(list(ranks.values()), columns=["ref",""])
    new_df["rank"] = [s + 1 for s in sorting]
    l = new_df.to_latex()
    print(l)
    """


def merge_crossdomain(modelpath, identifier):
    evaluator = pickle_loader(os.path.join(modelpath, "eval.pickle"))
    OOD_evaluator = pickle_loader(os.path.join(modelpath, str(identifier) + "eval.pickle"))
    merged = deepcopy(OOD_evaluator)

    if args.outofdomain:
        evaluation_data = generate_evaluation_data(
            modelpath,
            test_data=None,
            evaluation_data=None,
            downsampling=0,
            dump=False,
            raw_data=True,
            identifier="",
            data_identifier=identifier,
            get_evaluation_data=True,
        )  # raw
        ood_groundtruth = np.array([x[0] for x in list(zip(*evaluation_data))[1]])

    for sampling, stats in reversed(OOD_evaluator.stats.items()):
        for key, value in stats.items():
            if key == "evaluated":
                continue
            iid = evaluator.stats[sampling][key]
            if not len(iid):
                continue
            if key == "gold" and args.outofdomain:
                prevalue = value
                value = ood_groundtruth[: len(prevalue)]  # IMPORTANT since batch_size drops samples at the end
                merged.known = [1] * len(iid) + [0] * len(prevalue)
            try:
                merged.stats[sampling][key] = np.concatenate((iid, value))
            except Exception as e:
                print(f"{key} not mergeable, dropping", e)
                # drop key
    return merged


def equality(predicted, gold):
    equals = []
    for pred, g in zip(predicted, gold):
        if isinstance(g, np.ndarray):
            equal = set(pred) == set(g)
        else:
            equal = pred == g
        equals.append(True if equal else False)
    return equals


def IID_check(gold, unknown):
    IIDs = []
    for g in gold:
        if isinstance(g, np.ndarray):
            OOD = any(x in g for x in unknown)
        else:
            OOD = g in unknown
        IIDs.append(int(not OOD))
    return IIDs


def select_quality(abl, uncertainty, quality="low", s=5):
    from scipy.stats import rankdata

    higher_is_better = False
    if uncertainty == "confidence":
        higher_is_better = True
    # cherrypick
    abl["sorted"] = rankdata(abl[uncertainty])
    abl_sort = abl.sort_values(by=["sorted"], ascending=False if quality == "low" else True)
    for i in range(s):
        print(abl_sort["document"].iloc[i], abl_sort[uncertainty].iloc[i])
    print()


def main(version, modelpath, evaluator, plot=False):
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
        "MU(μ)",
        "DU(μ)",
        "KL(µ)",
        "AUB",  # area under bluedots
        "UoF@5",
        "FP@5(↓)",
        "T>",
    ]
    """
    1. Determine stats for Unknown/OOD classes
    """

    labels = list(evaluator.idx2label.values())
    if "ood" in labels:
        unknown = ["ood"]
    else:
        unknown = determine_ood_classes(modelpath)
        if not isinstance(unknown, list):
            unknown = [unknown]
    in_domain = [l for l in labels if l not in unknown]

    nlpstats = False
    if nlpstats:
        if "CLINC" in modelpath:
            data_identifier = "CLINC_ood"
            m = modelpath.replace("_ood", "")
        else:
            data_identifier = ""
            m = modelpath
        evaluation_data = generate_evaluation_data(
            m,
            test_data=None,
            evaluation_data=None,
            downsampling=0,
            dump=False,
            identifier="",
            data_identifier=data_identifier,
            get_evaluation_data=True,
        )
        # unique noun density

        unbatched = [x[: np.where(x)[0][-1] + 1] if any(x) else x for x, y in iter(evaluation_data.unbatch())]
        doclens = [x.shape[0] for x in unbatched]  # remove padding?
        oov_rates = [len(np.where(x == 0)[0]) / x.shape[0] for x in unbatched]

    corrs = []
    t = PrettyTable(["version"] + metric_names)
    for sampling, stats in reversed(evaluator.stats.items()):
        if sampling == "mc" and "nodropout" in modelpath:
            continue
        correlations = OrderedDict()

        try:
            base = re.search("aleatoric|baseline|nodropout", version).group(0) + "_" + sampling
        except Exception as e:
            base = "baseline"
        print("** ", version + "_" + sampling, " **")

        predicted = np.array(stats["predicted"])
        groundtruth = np.array(stats["gold"])

        if predicted.dtype != groundtruth.dtype:
            predicted = predicted.astype(str)
            groundtruth = groundtruth.astype(str)
            unknown = [str(x) for x in unknown]
            labels = [str(x) for x in labels]
            in_domain = [str(x) for x in in_domain]

        confidences = np.array(stats["softmax"])
        argmax_confidence = np.array(stats["confidence"]) / 100
        if hasattr(evaluator, "multilabel"):
            if evaluator.multilabel:
                argmax_confidence = np.array([np.mean(x) for x in argmax_confidence])

        boolean = np.array(equality(predicted, groundtruth))

        if args.outofdomain:
            known = np.array(evaluator.known)
        else:
            known = np.array(IID_check(groundtruth, unknown))
        unk = np.where(known == 0)[0]
        knw = np.where(known)[0]

        if sum(boolean[unk]) > 0 and not args.outofdomain:
            print("correct ood impossible")

        if sum(known) == len(known) or len(unk) == 0:
            raise Exception("This model has not been properly OOD evaluated or trained")

        # print("IN domain: ", sklearn.metrics.accuracy_score(predicted[knw], groundtruth[knw]))
        if not args.crossdomain:
            try:
                iid_stats = {k: np.array(v)[knw] if len(v) == len(stats["predicted"]) else v for k, v in stats.items()}
                iid_evaluator = deepcopy(evaluator)
                iid_evaluator.idx2label = {k: v for k, v in evaluator.idx2label.items() if v in in_domain}
                metrics, k = evaluate_model(version, sampling, iid_stats, iid_evaluator)
                t.add_row([k["version"] + "-IID"] + metrics)
            except Exception as e:
                print(e)

        """
        Out-of-domain generalization
        """
        if args.outofdomain:
            # would want to evaluate "accuracy"; yet first need to remap labels to same space
            ood_stats = {k: np.array(v)[unk] if len(v) == len(stats["predicted"]) else v for k, v in stats.items()}
            ood_evaluator = deepcopy(evaluator)
            ood_evaluator.idx2label = dict(enumerate(sorted(np.unique(ood_stats["gold"]))))

            if "amazon_reviews" in args.identifier:
                remap_predictions = ood_stats["predicted"]
                unsure_indices = np.array([])
            else:
                if len(ood_evaluator.idx2label) == 2:
                    f = lambda p: 0 if p < 5 else 1 if p > 6 else p
                    remap_predictions = np.vectorize(f)(ood_stats["predicted"])
                    unsure_indices = np.where(np.isin(remap_predictions, [5, 6]))[0]

                elif len(ood_evaluator.idx2label) == 5:
                    f = lambda x: int(x / 2 + 0.5)
                    remap_predictions = np.vectorize(f)(ood_stats["predicted"])
                    unsure_indices = np.array([])

            ood_boolean = np.array(equality(remap_predictions, ood_stats["gold"]))
            accuracy = sum(ood_boolean) / len(ood_boolean)
            correlations["Acc_OOD"] = accuracy
            print("OOD generalization", accuracy)

            if any(unsure_indices):
                ood_boolean = ood_boolean[~unsure_indices]
                sure_accuracy = sum(ood_boolean) / len(ood_boolean)
                print("OOD generalization without unsure", sure_accuracy)
                correlations["Acc_OOD_sure"] = sure_accuracy
            # metrics, k = evaluate_model(version, sampling, ood_stats, ood_evaluator)

        """
        2. Determine stats for Unknown/OOD classes
        """

        stats["aleatorics"] = (
            stats["aleatorics"] if "aleatorics" in stats else [0 for _ in range(len(stats["predicted"]))]
        )
        stats["epistemics"] = (
            stats["epistemics"] if "epistemics" in stats else [0 for _ in range(len(stats["predicted"]))]
        )
        stats["entropy_mc_overall"] = (
            stats["entropy_mc_overall"] if "entropy_mc_overall" in stats else entropy(confidences)
        )

        if not any(stats.get("mutual_information", [])):
            if len(stats.get("mean_mc_array", np.zeros(1)).shape) == 3:
                stats["mutual_information"] = mutual_info(stats["mean_mc_array"])
            else:
                stats["mutual_information"] = [0 for _ in range(len(stats["predicted"]))]

        if "baseline" in base and "mc" in base:
            if np.mean(stats["epistemics"]) == stats["epistemics"][0]:
                stats["epistemics"] = stats.pop("aleatorics")
                stats["aleatorics"] = [0 for _ in range(len(stats["predicted"]))]

        if False:
            new_s = {k: np.array(v)[unk] if len(v) == len(stats["predicted"]) else v for k, v in stats.items()}

            fig, ax = plt.subplots(len(labels), figsize=(20, 10))
            fig.suptitle('predictions for unknown')

            if len(labels) < 40:
                for i, label in enumerate(labels):
                    class_samples = new_s["confidence"][np.where(new_s["predicted"] == label)]
                    ax[i].title.set_text("Class " + str(label))
                    ax[i].set_xlim(0, 101)
                    sns.boxplot(y=class_samples, orient='h', ax=ax[i])
                plt.show()

            plot_cm(new_s["gold"], new_s["predicted"], name=base)

        """
        3. Collect uncertainty and y metrics for OOD and regular
        """
        df = pd.DataFrame(
            list(
                zip(
                    known,
                    boolean,
                    argmax_confidence,
                    stats["aleatorics"],
                    stats["epistemics"],
                    stats["entropy_mc_overall"],
                    stats["mutual_information"],
                )
            ),
            columns=["known", "y", "confidence", "aleatorics", "epistemics", "entropy", "mutual_information"],
        )
        print(df.head())

        metric_names = ["confidence", "aleatorics", "epistemics", "entropy", "mutual_information"]
        if nlpstats:
            df["document"] = np.array(stats["document"])
            df["predicted"] = predicted
            df["gold"] = groundtruth
            df["oov"] = oov_rates
            df["doclen"] = doclens
            if hasattr(evaluator, "multilabel"):
                if evaluator.multilabel:
                    df["density"] = [len(x) for x in predicted]
                    df["cardinality"] = [len(x) for x in groundtruth]

            df.to_csv(modelpath + "/" + sampling + "oov-doclen" + ".csv", index=False)

        correlations["version"] = version + "_" + str(sampling)
        """
        Additional OOD metrics! 
        """
        auroc, aupr = AUROC_PR(confidences[~unk], confidences[unk])
        print('== eval in and ood using max(Py|x) == \nAUPR:', aupr, "\n AUROC:", auroc)
        FPRathigh = FPRatRecall(confidences[~unk], confidences[unk])
        correlations["AUPR"] = aupr
        correlations["AUROC"] = auroc
        correlations["FPRat95(↓)"] = FPRathigh
        correlations["Acc_IID"] = len(df[(df["known"] == 1) & (df["y"] == True)]) / len(df[df["known"] == 1])
        # correlations["F1(m)_IID"] =

        if plot:
            for column in metric_names:
                dist_attribute_ood_bool(df, column, base, path=modelpath)
                if column != "confidence":
                    threed(df, column, base, path=modelpath)

        for column in metric_names:
            if not "aleatoric" in base and column == "aleatoric":
                continue
            if not "mc" in base and column == "epistemic":
                continue
            # correlation, p = pearsonr(df["known"], df[column])
            correlation, p = pointbiserialr(df["known"], df[column])
            correlations[column + "_R"] = correlation
            correlations[column + "_p"] = (
                lambda p: "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
            )(p)
            print(f"{column} {correlation} {p}")

        """
        thresholds = np.linspace(0, 1, 10, endpoint=False)
        accuracies, counts = compute_accuracies_at_confidences(
            labels, probs, thresholds)
        # pass
        """
        corrs.append(correlations)
    print(t)
    return corrs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import argparse

    parser = argparse.ArgumentParser("""OOD stats and metrics""")
    parser.add_argument("modeldirectory", type=str, nargs="?", default="/mnt/lerna/models")
    parser.add_argument("-d", dest="datasets", type=str, default="")
    parser.add_argument(
        "-i", dest="identifier", type=str, default="", help="identifier to add to eval.pickle [ensemble-member]"
    )
    parser.add_argument(
        "-b", dest="benchmark", action="store_true", default=False, help="to specify even more per dataset what not"
    )
    parser.add_argument(
        "-c",
        dest="crossdomain",
        action="store_true",
        default=False,
        help="merge evals; then create cross-domain comparison",
    )
    parser.add_argument(
        "-o",
        dest="outofdomain",
        action="store_true",
        default=False,
        help="merge evals; then create out-of-domain comparison",
    )
    parser.add_argument("-f", dest="filter", type=str, default="", help="to specify even more per dataset")
    parser.add_argument("-e", dest="ensembler", action="store_true", default=False, help="average over ensemble models")
    parser.add_argument("-p", dest="plot", default=False, action="store_true", help="plot instead of save to dir")
    parser.add_argument(
        "-m", dest="M_ablation", default=False, action="store_true", help="recurse all ensembles; collect and line plot"
    )

    args = parser.parse_args()

    if args.datasets:
        datasets = args.datasets.split("|")
        models = sorted(
            [
                os.path.join(args.modeldirectory, x)
                for x in os.listdir(args.modeldirectory)
                if not ".out" in x and not ".log" in x and re.search(args.filter, x)
            ]
        )
        if not args.outofdomain:
            models = [x for x in models if "ood" in x]
        for dataset in datasets:
            keep = []
            data_set_models = sorted([model for model in models if dataset.lower() in model.lower()])
            if args.benchmark:
                data_set_models = benchmark_filter(
                    dataset, data_set_models, ood=True if not args.outofdomain else False
                )
            if not data_set_models:
                print(f"{dataset} no models to be found")
                continue
            for modelpath in data_set_models:
                if not os.path.exists(os.path.join(modelpath, "eval.pickle")) or not os.path.exists(
                    os.path.join(modelpath, "params.json")
                ):
                    print(f"{modelpath} has not been evaluated properly")
                    continue
                if args.crossdomain:
                    evaluator = merge_crossdomain(modelpath, args.identifier)
                else:
                    evaluator = pickle_loader(os.path.join(modelpath, "eval.pickle"))
                model = os.path.basename(modelpath) + str(args.identifier)
                try:
                    correlations = main(model, modelpath, evaluator, plot=args.plot)
                    keep.extend(correlations)
                except Exception as e:
                    print(f"{modelpath} has not been able to be run: {e}")
                if "M5" in modelpath and args.ensembler:
                    for m in ["M0_", "M1_", "M2_", "M3_", "M4_"]:
                        evaluator = pickle_loader(
                            os.path.join(modelpath, str(args.identifier) + str(m) + "eval.pickle")
                        )
                        model = os.path.basename(modelpath) + str(args.identifier) + str(m)
                        try:
                            correlations = main(model, modelpath, evaluator, plot=args.plot)
                            keep.extend(correlations)
                        except Exception as e:
                            print(f"{model} has not been able to be run: {e}")
            if keep:
                df = pd.DataFrame(keep)
                df.fillna(0, inplace=True)
                if args.outofdomain:
                    df = deduce_methods(
                        df,
                        modelpaths=df["version"]
                        .apply(lambda x: os.path.join("/mnt/lerna/models", x.split(args.identifier)[0]))
                        .tolist(),
                        modelroot="",
                    )
                else:
                    df = deduce_methods(df, modelpaths=None, modelroot=args.modeldirectory)
                # ranker(df)
                df = make_style(df, absolute=True)

                out = dataset
                if args.filter:
                    out += "_" + args.filter
                if args.identifier:
                    out += "_" + args.identifier
                if args.benchmark:
                    out += "_benchmark"
                if args.ensembler:
                    out += "_M"
                df.to_excel(os.path.join(args.modeldirectory, out + '_ood.xlsx'), index=False)
        del os.environ['CUDA_VISIBLE_DEVICES']
        sys.exit(1)

    version = args.modeldirectory
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    if args.crossdomain:
        evaluator = merge_crossdomain(modelpath, args.identifier)
    else:
        evaluator = pickle_loader(os.path.join(modelpath, str(args.identifier) + "eval.pickle"))

    if args.M_ablation:
        keep = []
        for i in range(1, 6):
            if i == 1:
                identifier = "M0_"
            elif i == 5:
                identifier = ""
            else:
                identifier = str(i)
            evaluator = pickle_loader(os.path.join(modelpath, identifier + "eval.pickle"))
            model = os.path.basename(modelpath) + str(args.identifier)
            try:
                correlations = main(model, modelpath, evaluator, plot=args.plot)
                for k in range(len(correlations)):
                    correlations[k]["M"] = i
                    correlations[k]["sampling"] = list(reversed(evaluator.stats.keys()))[k]
                keep.extend(correlations)
            except Exception as e:
                print(f"{model} has not been able to be run: {e}")

        if keep:
            df = pd.DataFrame(keep)
            df.fillna(0, inplace=True)
            df = deduce_methods(df, modelpaths=[modelpath] * len(df), modelroot="")
            df.to_excel(os.path.join(modelpath, 'ensemble_ablation.xlsx'), index=False)
            sys.exit(1)

    """
    Out-of-domain detection
    """
    correlations = main(version, modelpath, evaluator, plot=args.plot)
    if "M5" in modelpath and args.ensembler:
        for m in ["M0_", "M1_", "M2_", "M3_", "M4_"]:
            evaluator = pickle_loader(os.path.join(modelpath, str(args.identifier) + str(m) + "eval.pickle"))
            model = os.path.basename(modelpath) + str(args.identifier) + str(m)
            try:
                correlations = main(model, modelpath, evaluator, plot=args.plot)
            except Exception as e:
                print(f"{model} has not been able to be run: {e}")
    del os.environ['CUDA_VISIBLE_DEVICES']
