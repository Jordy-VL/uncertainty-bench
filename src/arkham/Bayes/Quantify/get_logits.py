#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import numpy as np
import regex as re
import pandas as pd
from copy import deepcopy

from tqdm import tqdm
from arkham.utils.utils import timer, pickle_loader, pickle_dumper
from arkham.utils.model_utils import MODELROOT, _get_weights, load_model_path
from arkham.Bayes.Quantify.data import generators_from_directory
from arkham.Bayes.Quantify.evaluate import TouristLeMC
from arkham.Bayes.Quantify.compare import multilabel_prediction, multilabel_encode
from netcal.scaling import TemperatureScaling


def get_concrete_p(model):
    # print(model.losses)
    ps = []
    for layer in model.layers:
        if "concrete" in layer.name:
            assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob
            ps.append(1 / (1 + np.exp(-layer.p_logit.numpy()))[0])  # sigmoid
    print(ps, np.mean(ps))
    return ps


def get_temperature(version):
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    test_evaluator = pickle_loader(os.path.join(modelpath, "calib_eval.pickle"))
    print(modelpath, "\t", test_evaluator.temperature[0])


def get_concrete(version):
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    model, _config = load_model_path(modelpath)
    if _config["dropout_concrete"]:
        get_concrete_p(model)


def ensemble_size():
    best = {
        "20news": "20news_aleatorics_M5",  # aleatorics [non concrete] OR heteroscedastic ensemble
        "imdb": "imdb_baseline_M5",  # YET MC
        "CLINC150": "CLINC150_aleatorics_M5_concrete",  # full combination
        "Reuters_multilabel": "Reuters_multilabel_baseline_M5_concrete",
        "AAPD": "AAPD_nodropout_M5",
    }
    sampling = {
        "20news": "nonbayesian",  # aleatorics [non concrete] OR heteroscedastic ensemble
        "imdb": "mc",  # YET MC
        "CLINC150": "mc",  # full combination
        "Reuters_multilabel": "nonbayesian",
        "AAPD": "nonbayesian",
    }

    for dataset, model in best.items():
        print(dataset)
        modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
        for i in range(0, 6):
            if i == 0:
                identifier = "M0_"
            elif i == 5:
                identifier = ""
            else:
                identifier = str(i)
            evaluator = pickle_loader(os.path.join(modelpath + "_ood", identifier + "eval.pickle"))
            stats = evaluator.stats[sampling[model]]
            # AUPR, AUROC, epistemics_R

        # abl = pd.read_csv("/mnt/lerna/models/" + model + "_ood/" + sampling[dataset] + "oov-doclen.csv")


def calibrate(version, evaluation_data=None, downsampling=0, dump=True, **kwargs):
    """
    standard:

    - create dev evaluator [either single or for all models] (for all statistics)

    if not ensemble, get both logits for dev and test -> temperature scale
    if ensemble, need to get logits for dev for all ensemble members;

    To get the exact T
    https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/calibration_lib.py
    https://github.com/dirichletcal/dirichlet_python/blob/6c750d9fca1cc8c425446055352115fd649ca03b/calib/tempscaling.py
    """
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version

    model, _config = load_model_path(modelpath)
    _config["data_folder"] = re.sub(r'/home/[^/]+', os.path.expanduser("~"), _config["data_folder"])
    _config["out_folder"] = modelpath

    generators, _, _, _ = generators_from_directory(_config["data_folder"], sets=["dev"], **_config)

    evaluators = {}
    for m in range(_config["ensemble"]):
        evaluator = TouristLeMC(
            model,
            generators["dev"],
            _config["idx2voc"],
            _config["idx2label"],
            _config.get("use_aleatorics"),
            posterior_sampling=kwargs.get("posterior_sampling", 10),
            identifier="M" + str(m) + "_",
        )
        evaluator.evaluate(mode="mc")
        evaluator.evaluate(mode="nonbayesian")
        evaluators[m] = evaluator

    test_evaluator = pickle_loader(os.path.join(modelpath, "eval.pickle"))

    """
    df["y"] = df["gold"].apply(lambda x: _config["label2idx"][x])
    temperature = TemperatureScaling()
    temperature.fit(X_train, y_train)
    X_test_cal = temperature.transform(X_test)
    preds, argmax_probs = prob_to_predict(X_test_cal, list(_config["label2idx"].keys()))
    boolean = [1 if _config["label2idx"][preds[i]] == y_test[i] else 0 for i in range(len(preds))]
    """

    """
    At which level to calibrate logits? The final mean softmax; standard [should not calibrate individual model samples NOR subsamples]
    """
    calibrated_evaluator = deepcopy(test_evaluator)
    labels = sorted(list(test_evaluator.idx2label.values()))
    for sampling in list(test_evaluator.stats.keys()):
        X_test = np.array(test_evaluator.stats[sampling]["softmax"])
        if test_evaluator.multilabel:
            y_test = np.array(
                [np.vectorize(_config["label2idx"].get)(idx) for idx in test_evaluator.stats[sampling]["gold"]]
            )
            y_test = multilabel_encode(labels, y_test)
        else:
            y_test = np.vectorize(_config["label2idx"].get)(test_evaluator.stats[sampling]["gold"])
        # np.vectorize(_config["label2idx"].get)()
        if len(evaluators) == 1:
            X_dev = np.array(evaluators[0].stats[sampling]["softmax"])
            if test_evaluator.multilabel:
                y_dev = np.array(
                    [np.vectorize(_config["label2idx"].get)(idx) for idx in evaluators[0].stats[sampling]["gold"]]
                )
                y_dev = multilabel_encode(labels, y_dev)
            else:
                y_dev = np.vectorize(_config["label2idx"].get)(evaluators[0].stats[sampling]["gold"])
        else:
            X_dev = np.vstack([evaluators[i].stats[sampling]["softmax"] for i in range(len(evaluators))])
            # if test_evaluator.multilabel:
            #     y_dev = np.array([np.vectorize(_config["label2idx"].get)(idx) for idx in evaluators[0].stats[sampling]["gold"]])
            # else:
            #     y_dev = np.vectorize(_config["label2idx"].get)(evaluators[0].stats[sampling]["gold"])
            y_dev = np.vectorize(_config["label2idx"].get)(
                np.hstack([evaluators[i].stats[sampling]["gold"] for i in range(len(evaluators))])
            )

        temperature = TemperatureScaling()
        temperature.fit(X_dev, y_dev)
        X_test_cal = temperature.transform(X_test)

        calibrated_evaluator.stats[sampling]["softmax"] = X_test_cal
        if calibrated_evaluator.multilabel:
            calibrated_evaluator.stats[sampling]["predicted"], calibrated_evaluator.stats[sampling][
                "confidence"
            ] = multilabel_prediction(
                calibrated_evaluator.stats[sampling]["softmax"], calibrated_evaluator.idx2label
            )  # 54
            calibrated_evaluator.stats[sampling]["confidence"] = (
                calibrated_evaluator.stats[sampling]["confidence"] * 100
            )
        else:
            calibrated_evaluator.stats[sampling]["confidence"] = (
                np.max(calibrated_evaluator.stats[sampling]["softmax"], -1) * 100
            )
            calibrated_evaluator.stats[sampling]["predicted"] = np.vectorize(calibrated_evaluator.idx2label.get)(
                np.argmax(calibrated_evaluator.stats[sampling]["softmax"], -1)
            )

    calibrated_evaluator.temperature = temperature._weights
    calibrated_evaluator.compute_stats(out_folder=None)
    calibrated_evaluator.dump(modelpath, identifier="calib_")


def main(version, identifier="nonbayesian", evaluation_data=None, downsampling=0, dump=True, **kwargs):
    """
    Load model by version; reupdate params
    """
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    model, _config = load_model_path(modelpath)
    if _config["dropout_concrete"]:
        get_concrete_p(model)
        # could also get diversity here

    _config["data_folder"] = re.sub(r'/home/[^/]+', os.path.expanduser("~"), _config["data_folder"])
    _config["out_folder"] = modelpath  # to ensure if the model has been moved, that the correct folder is being used

    generators, _, _, _ = generators_from_directory(_config["data_folder"], sets=["dev", "test"], **_config)

    evaluators = {}
    for dataset, evaluation_data in generators.items():
        evaluator = TouristLeMC(
            model,
            evaluation_data,
            _config["idx2voc"],
            _config["idx2label"],
            _config.get("use_aleatorics"),
            posterior_sampling=kwargs.get("posterior_sampling", 10),
        )
        evaluator.evaluate(mode=identifier)
        evaluators[dataset] = evaluator.get_logits(identifier=identifier)
        evaluators[dataset]["dataset"] = dataset

    logits = pd.concat([d for dataset, d in evaluators.items()], axis=0)
    # print(logits.head())
    # logits.describe(include="all")
    # logits.dtypes()
    logits.to_pickle(os.path.join(_config["out_folder"], "logits.pkl"))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import argparse

    parser = argparse.ArgumentParser("""Get logits for a model""")
    parser.add_argument("version", type=str, default="")
    parser.add_argument("identifier", nargs="?", type=str, default="nonbayesian")
    parser.add_argument("-s", dest="posterior_samples", type=int, default=10, help="number of forward samples")
    parser.add_argument("-c", dest="calibrator", default=False, action="store_true", help="test temperature scaling")
    parser.add_argument("-t", dest="temperature", default=False, action="store_true", help="get temperature")
    parser.add_argument("-p", dest="concrete", default=False, action="store_true", help="get concrete p")

    args = parser.parse_args()
    # ensemble_size()
    # sys.exit(1)

    if args.temperature:
        get_temperature(args.version)
        sys.exit(1)
    if args.concrete:
        get_concrete(args.version)
        sys.exit(1)
    if args.calibrator:
        calibrate(args.version, dump=True, posterior_sampling=args.posterior_samples)
        del os.environ['CUDA_VISIBLE_DEVICES']
        sys.exit(1)
    main(args.version, args.identifier, dump=True, posterior_sampling=args.posterior_samples)
