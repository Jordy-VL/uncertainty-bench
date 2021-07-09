#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import regex as re
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict
from prettytable import PrettyTable
from scipy.special import softmax
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
import time

from arkham.default_config import MODELROOT, SAVEROOT, DATAROOT

from arkham.utils.model_utils import _get_weights, load_model_path, load_model_config, multiclass_argmaxprobs
from arkham.utils.utils import timer, pickle_loader, pickle_dumper
from arkham.utils.custom_metrics import entropy, exp_entropy, pred_entropy, mutual_info
from arkham.Bayes.Quantify.data import (
    generators_from_directory,
    tokenize_composition,
    encode_text,
    encode_test,
    get_tokenizer,
)

test_texts = {
    "imdb": """ 
In 1989, Tim Burton created the very first Batman movie with great stars like Michael Keaton and Jack Nicholson. The Joker is definitely one of Hollywood's best villains on screen. Jack Nicholson was born for the role, with his psychotic and sick look. Michael Keaton is also great as Batman and is pretty good as Bruce Wayne. Kim Basinger is kind of annoying at times, but she's not the worst damsel in distress ever seen on screen.
Tim Burton has a unique way of doing Batman, and I think most people can agree that it fits the characters and the story. To bad Warner Bros. got rid of him after the 2nd film.
""",  # 8
    "yelp/2013": "this restaurant's food is actually shit, really I never want to go here again.",
    "yelp/2014": """
great unk burger ! amazing service ! brilliant interior
! the burger was delicious but it was a little big . it ’s a
great restaurant good for any occasion .
""",
    "yelp/2015": """
i ’ve bought tires from discount tire for years at different locations and have had a good experience , but
this location was different . i went in to get some new
tires with my fianc . john the sales guy pushed a certain
brand , specifically because they were running a rebate
special . tires are tires , especially on a prius (the rest
134 tokens not shown here due to space)
""",
}


def onehot_multilabel(label_array):
    return np.nonzero(label_array)[0]


def dynamic_wordpiece_mask(array, mask, join=False):
    def most_frequent(l):
        return max(set(l), key=l.count)

    array = np.array(array)
    array_shape = len(array.shape)
    new_array = []
    for indices in mask:
        if isinstance(indices, list):
            if not indices:  # masking pad, cls, and sep
                continue
            if array_shape == 1:
                if isinstance(array[indices[0]], str):
                    if join:
                        masked = "".join([t.replace("##", "") for t in array[indices].tolist()])  # VOTE most frequent
                    else:
                        masked = most_frequent(array[indices].tolist())  # VOTE most frequent
                else:
                    if join:
                        masked = np.mean(array[indices])
                    else:
                        masked = array[indices]
            elif array_shape == 2:
                masked = np.mean(array[indices, :], axis=0)
            elif array_shape == 3:
                masked = np.mean(array[:, indices, :], axis=1)
        else:
            if array_shape == 1:
                masked = array[indices]
            elif array_shape == 2:
                masked = array[indices, :]
            elif array_shape == 3:
                masked = array[:, indices, :]
        new_array.append(masked)

    new_array = np.array(new_array)
    if array_shape == 3:
        new_array = np.transpose(new_array, (1, 0, 2))
    return new_array


def generate_wordpiece_mask(wordpieces):
    """Generate an indices mask for subword tokens
    # if first wordpiece tactic =
        # mask = [i for i, t in enumerate(wordpieces) if not (t.startswith("##") or t in ['[CLS]', '[SEP]', '[PAD]'])]
    HERE: build a mask for subword tokens

    Args:
        wordpieces (list of str): subword tokens

    Returns:
        list of masked indices (int)
    """
    group_level_indices = []
    current = []
    for i, wp in enumerate(reversed(wordpieces)):
        i = abs(i - len(wordpieces) + 1)
        if wp.startswith("##"):
            current.append(i)
        else:
            if wp in ['[CLS]', '[SEP]', '[PAD]']:
                group_level_indices.append([])
                continue

            if current:
                current.append(i)
                current = sorted(current)
                group_level_indices.append(current)
                current = []
            else:
                group_level_indices.append(i)

    dyn_mask = np.array(list(reversed(group_level_indices)), dtype="object")
    return dyn_mask


def decode_x_and_y(batch, labels, idx2voc, idx2label, sequence_labels=False, multilabel=False, tokenizer=None):
    def approximate_originalseq_length(encoded, labels):
        """
        cannot assume we encode all voc words AND at least want to capture last positive label
        """
        activated = sorted(set(np.where(encoded)[0].tolist() + np.where(labels)[0].tolist()))
        if activated:
            return activated[-1] + 1
        else:
            return len(labels)

    batch_texts, batch_golds, batch_seq_lengths = [], [], []

    for i in range(batch.shape[0]):  # batchsize
        if not sequence_labels:
            if multilabel:  # np.count_nonzero(labels) > len(labels)
                indices = onehot_multilabel(labels[i])
                batch_golds.append(np.vectorize(idx2label.get)(indices))
            else:
                if len(labels) > 0:
                    batch_golds.append(idx2label[np.argmax(labels[i], axis=-1)])
            batch_text = ""
            encoded = [batch[i]] if len(batch[i].shape) == 1 else batch[i]
            for b in encoded:
                if not np.any(b):
                    continue
                unpadded = np.trim_zeros(b, 'b')
                stringified = np.vectorize(idx2voc.get)(unpadded)
                joined = " ".join(stringified).replace("PAD", "0")
                batch_text += joined + "\n"
            batch_texts.append(batch_text)
        else:

            if tokenizer:
                wordpieces = tokenizer.convert_ids_to_tokens(batch[i])
                dynamic_mask = generate_wordpiece_mask(wordpieces)
                batch_seq_lengths.append(dynamic_mask)
                text = dynamic_wordpiece_mask(wordpieces, dynamic_mask, join=True)
                gold = dynamic_wordpiece_mask([idx2label.get(l) for l in labels[i]], dynamic_mask)

            else:
                gold = []
                text = []
                l = labels[i]  # [~mask]
                b = batch[i]  # [~mask]
                original_seq_length = approximate_originalseq_length(b, l)
                batch_seq_lengths.append(original_seq_length)

                for k in range(0, original_seq_length):
                    text.append(idx2voc.get(b[k], "/"))
                    gold.append(idx2label[l[k]])

            batch_golds.append(" ".join(gold))  # original_seq_length #160
            batch_texts.append(" ".join(text))  # [:original_seq_length]
    return batch_texts, batch_golds, batch_seq_lengths


def predict(model, _config, test_data):
    def simple_entropy(x):
        return float((-x * np.log2(x)).sum(axis=-1)[0])

    def wrap_model(model, use_aleatorics):
        """
        Depends on the tensorflow version on how to generate predictions
        either MC dropout sampling or heteroscedastic
        """
        from tensorflow.python.keras import backend

        model_aleatoric = None
        if not any(
            [
                x
                for x in model.layers
                if isinstance(x, tf.keras.layers.Dropout)
                or "dropout" in x.name
                or x.get_config().get("dropout")
                or x.get_config().get("layer", {}).get("config", {}).get("dropout")
            ]
        ):
            model_mc = None
            return model_mc, model_aleatoric
        if not use_aleatorics:
            out_layer = [model.output]
            model_mc = tf.keras.backend.function([model.input, backend.symbolic_learning_phase()], out_layer)
        else:
            dist_layer = next(layer for layer in model.layers if "distribution" in layer.name)
            out_layer = [tf.nn.softmax(dist_layer.output[1], axis=-1)]
            model_mc = tf.keras.backend.function(
                [model.input, backend.symbolic_learning_phase()],
                out_layer + [dist_layer.output[0].mean(), dist_layer.output[0].variance()],
            )
            model_aleatoric = tf.keras.backend.function(
                [model.input], out_layer + [tf.reduce_mean(dist_layer.output[0].variance(), axis=-1)]
            )
        return model_mc, model_aleatoric

    def predict_all(model, batched, use_aleatorics, T=10, sequence_labels=False, multilabel=False):
        predictions = {}
        model_mc, model_aleatoric = wrap_model(model, use_aleatorics)

        ### SIMPLE ###
        predictions["simple"] = {}
        if model_aleatoric:
            sample = model_aleatoric([next(iter(batched))])
            prediction = sample[0]
            predictions["simple"]["aleatorics"] = float(sample[1])
        else:
            prediction = model.predict(batched)

        predictions["simple"]["index"] = np.argmax(prediction, axis=-1)[0]
        predictions["simple"]["confidence"] = 100 * np.max(prediction, axis=-1)[0]
        predictions["simple"]["entropy"] = entropy(prediction)
        if not sequence_labels:
            predictions["simple"]["index"] = int(predictions["simple"]["index"])
            predictions["simple"]["confidence"] = float(predictions["simple"]["confidence"])
            predictions["simple"]["entropy"] = simple_entropy(prediction)

        # stats["predicted"].append(" ".join([str(self.idx2label[p]) for p in mean_argmax]))
        # sequence_token_pred = [self.idx2label[p] for p in np.argmax(sequence, axis=-1)]
        # sequence_token_prob = multiclass_argmaxprobs(sequence) * 100
        # stats["predicted"].append(" ".join([str(x) for x in sequence_token_pred]))
        if sequence_labels:
            return predictions

        ### MC ###
        if model_mc:
            samples = [model_mc([next(iter(batched)), True]) for _ in range(T)]

            if len(samples[0]) > 1:
                aggregate, means, variances = zip(*samples)
            else:
                aggregate = zip(*samples)
                means, variances = [], []
            aggregate, means, variances = np.array(aggregate), np.array(means), np.array(variances)

            if len(means) > 0:  # T, B, C
                aleatorics = np.mean(
                    np.mean(variances, axis=0), axis=-1
                )  # batchsize ; first over samples, then over classes
                aggregate = softmax(means, axis=-1)
                epistemics = np.mean(np.var(aggregate, axis=0, ddof=1), axis=-1)  # calculated as softmax variance
                prediction = np.mean(aggregate, axis=0)
                # DEV: alternate version 2: formula as in Kwon et al. (2018)
                # aleatorics, epistemics = formula_samples(aggregate)
            else:
                means = np.mean(aggregate, axis=0)  # softmaxed shape: B x C
                variances = np.var(aggregate, axis=0, ddof=1)  # softmaxed shape: B x C
                epistemics = np.sqrt(np.sum(variances, axis=-1))  # total variance :) #np.var
                aleatorics = None
                prediction = means

            predictions["bayesian"] = {
                "index": int(np.argmax(prediction, axis=-1)[0]),
                "confidence": float(100 * np.max(prediction, axis=-1)),
                "entropy": simple_entropy(prediction),
                "epistemics": float(epistemics),
            }
            if aleatorics:
                predictions["bayesian"]["aleatorics"] = float(aleatorics)
        return predictions

    if not test_data:
        return

    tokenized, encoded = encode_test(test_data, _config)
    if "sentence" in _config["composition"]:
        batched = encoded
        """
        batched = tf.data.Dataset.from_tensors(encoded[0])
        batched.padded_batch(1, padded_shapes=[_config["max_sentences"], _config["max_document_len"]])
        """
    elif _config.get("model_class"):
        batched = encoded

    else:
        batched = tf.data.Dataset.from_tensors(encoded).padded_batch(1, padded_shapes=[_config["max_document_len"]])
    if not _config.get("use_aleatorics"):
        try:
            model._make_predict_function()
        except Exception as e:
            print(e)

    predictions = predict_all(
        model,
        batched,
        _config.get("use_aleatorics"),
        T=_config.get("posterior_sampling"),
        sequence_labels=_config.get("sequence_labels"),
        multilabel=_config['multilabel'],
    )
    for version in predictions:
        predictions[version]["text"] = np.vectorize(_config["idx2label"].get)(predictions[version]["index"])
    """
    print(list(zip(pred_array, prob_array)))
    if len(tokenized) == len(pred_array):
        for i in range(len(tokenized)):
            print(tokenized[i], "\t", pred_array[i], "\t", prob_array[i])
    """
    print(predictions)
    return predictions


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


def determine_ood_classes(modelpath=None, _config=None):
    if _config is None:
        _config = load_model_config(modelpath)
    if str(_config.get("ood", None)) != "None":
        unknown = _config["ood"]
        return unknown
    if _config["label2idx"].get("ood"):
        return 'ood'
    else:
        version = modelpath
        if "twitter" in version:
            unknown = "neutral"
        if "imdb" in version:
            unknown = 5
        if "Reuters" in version:
            unknown = 0
        if "CLINC" in version:
            unknown = 150
        if "AAPD" in version:
            unknown = [0, 1]

    return unknown


def multilabel_prediction(predictions, idx2label):  # thinks its a batch!
    if len(predictions.shape) == 1:  # single prediction
        predictions = np.expand_dims(predictions, 0)
    boolean = predictions > 0.5
    indices = [onehot_multilabel(pred) for pred in boolean]
    pred_array = []
    prob_array = []
    for j, index in enumerate(indices):
        if len(index) == 0:
            index = [np.argmax(predictions[j], -1)]
        probs = predictions[j][index]  # changed
        vector_index = np.vectorize(idx2label.get, otypes=[int])(index)
        pred_array.append(vector_index)
        prob_array.append(probs)
    prob_array = np.array(prob_array, dtype="object")
    pred_array = np.array(pred_array, dtype="object")
    return pred_array, prob_array


def reshape_to_sequencelen(flat, original_seq_lengths):
    nested = []
    end = 0
    for s in original_seq_lengths:
        nested.append(flat[end : end + s])  # .tolist()
        end += s
    return nested


def marginalize_stats(stats, remap_document_level=False):
    # 1. GLOBAL: RESHAPE to sentence/document-level [complicates evaluation]
    original_seq_lengths = np.array([conf_array.shape[0] for conf_array in stats["softmax"]])
    # reshape_to_sequencelen(array, original_seq_lengths)

    # 2. FLATTEN towards level of marginals
    skip = ["filename", "field", "evaluation", "evaluated", "var_mc_overall", "entropy_mc_overall"]
    marginal_level = len(stats["gold"])

    for key in skip:
        if key in stats:
            stats.pop(key)
    for key in stats:
        if len(stats[key]) == marginal_level:
            stats[key] = np.array(stats[key])
            continue
        first = stats[key][0]
        if isinstance(first, np.ndarray):
            stats[key] = np.array([np.array(conf_array) for sequence in stats[key] for conf_array in sequence])

    for key in ["epistemics", "aleatorics", "entropy_mc_overall"]:
        if key not in stats:
            if key == "entropy_mc_overall":
                stats[key] = entropy(stats["softmax"])
            else:
                stats[key] = np.zeros(marginal_level)

    if remap_document_level:
        for key in stats:
            stats[key] = np.array(reshape_to_sequencelen(stats[key], original_seq_lengths))

    return stats


def stats_to_entities(stats, label2idx, entity_level=False):
    """
    At this point can assume that all are maximally 2D structures
    With maybe expection of "raw", but nvm for now

    essentially 2 strategies
    0. general flattening
    1. entity-level masking and averaging at y level
    2. "activated" masking at y_i level
    """
    predicted = np.array(stats["predicted"])
    groundtruth = np.array(stats["gold"])
    stats["y"] = np.array(equality(predicted, groundtruth))
    indices_groundtruth = np.vectorize(label2idx.get)(groundtruth)
    indices_predicted = np.vectorize(label2idx.get)(predicted)
    mask = np.nonzero(indices_groundtruth + indices_predicted)

    if not entity_level:
        print("MAP y_i event level for all activated pred/gold with I(x): ", len(mask[0]))
        for key in stats:
            stats[key] = stats[key][mask]
        stats["confidence"] = np.max(stats["softmax"], -1)
        return stats

    from seqeval.metrics.sequence_labeling import get_entities

    pred_spans = get_entities(predicted.tolist())
    gold_spans = get_entities(groundtruth.tolist())

    gold_mask = [list(range(start, end + 1)) for label, start, end in gold_spans]
    fp_mask = [list(range(p[1], p[2] + 1)) for p in pred_spans if p not in gold_spans]
    events_mask = np.array(gold_mask + fp_mask)  # could afterwards sort all keys by start

    avg_tokens_gold = np.mean([len(x) for x in gold_mask])
    avg_tokens_pred = np.mean([len(x) for x in [list(range(p[1], p[2] + 1)) for p in pred_spans]])
    avg_tokens_fp = np.mean([len(x) for x in fp_mask])

    print(f"MAP y entity event levels in I(x): ", len(events_mask))
    print(
        f"MAP y average tokens gold: {avg_tokens_gold} \n MAP y average tokens pred: {avg_tokens_pred} \n MAP y average tokens fp: {avg_tokens_fp}"
    )
    stats["y"] = np.array([True if g in pred_spans else False for g in gold_spans] + [False] * len(fp_mask))
    stats["evaluation"] = np.array(
        ["TP" if g in pred_spans else "FN" for g in gold_spans] + ["FP"] * len(fp_mask)
    )  # TN are all the O's; no interest
    stats["labels"] = np.array(
        [g[0] if g in pred_spans else g[0] for g in gold_spans] + ["O" for p in pred_spans if p not in gold_spans]
    )
    for key in stats:
        if key in ["y", "evaluation", "labels"]:
            continue
        stats[key] = dynamic_wordpiece_mask(stats[key], events_mask, join=True)

    # stats["confidence"] = np.array([s[0] for s in stats["confidence"]]) #have to unpack
    return stats


def montecarlo_evaluate(
    model,
    _config,
    evaluation_data,
    posterior_sampling=10,
    identifier=None,
    raw=False,
    skip_MC=False,
    predict_logits=False,
    timings=False,
):
    evaluator = TouristLeMC(
        model,
        evaluation_data,
        _config["idx2voc"],
        _config["idx2label"],
        _config.get("use_aleatorics"),
        posterior_sampling,
        identifier=True if raw or identifier else False,
        model_class=_config.get("model_class"),
        predict_logits=predict_logits,
    )
    if not skip_MC:
        evaluator.evaluate(mode="mc")
    evaluator.evaluate(mode="nonbayesian")
    if not timings:
        evaluator.compute_stats(out_folder=_config["out_folder"])
    return evaluator


def dropout_check(model):
    def layer_check(layer):
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if layer_check(sublayer):
                    return True
        if isinstance(layer, tf.keras.layers.Dropout) or "dropout" in layer.name:
            return True

        if hasattr(layer, "get_config"):
            try:
                if layer.get_config().get("dropout") or layer.get_config().get("layer", {}).get("config", {}).get(
                    "dropout"
                ):
                    return True
            except NotImplementedError as e:
                pass

        return False

    for layer in model.layers:
        if layer_check(layer):
            return True
    return False


class TouristLeMC(object):

    """Main class responsible for sampling predictions and crude evaluations
        Creates model wrappers for Monte-Carlo Dropout and Aleatoric Uncertainty
        Contains statistics on (sampled) predictions

    Attributes:
        evaluation_data (tf.batch): generated batches for testset
        idx2label (dict): dictionary mapping idx to labels
        idx2voc (dict): dictionary mapping idx to words
        model (tf.keras.model): loaded model ready for predictions
        model_aleatoric (tf.keras.model.predict): model predict_function ready for predictions and data uncertainty
        model_mc (tf.keras.model.predict): model predict_function ready for predictions and model uncertainty
        posterior_sampling (int): number of Monte-Carlo samples
        stats (dict): dictionary containing predictions, gold and evaluations per identifier
        use_aleatorics (boolean): if model has ability to provide data uncertainty
        identifier: just used to save "RAW" predictions
        predict_logits: let the model only output logits; save in stats as well

    DEV: stuck with intermediate name, many model evals saved which depend on module name.
    """

    def __init__(
        self,
        model,
        evaluation_data,
        idx2voc,
        idx2label,
        use_aleatorics=False,
        posterior_sampling=10,
        identifier=None,
        **kwargs,
    ):
        self.model = model
        self.evaluation_data = evaluation_data
        self.idx2voc = idx2voc
        self.idx2label = idx2label
        self.use_aleatorics = use_aleatorics
        self.posterior_sampling = posterior_sampling
        self.predict_logits = kwargs.get("predict_logits", False)
        if self.model:
            self.determine_task()
            self.create_model_mc()
        self.identifier = identifier
        self.stats = OrderedDict()
        self.tokenizer = kwargs.get("model_class", "")  # implies we require a tokenizer!

    def determine_task(self):
        """
        on the basis of the first task, we can determine if a task is multiclas, multilabel or NER
        """
        test_batch, test_labels = next(iter(self.evaluation_data))
        if isinstance(test_batch, tuple):
            test_encoded_batch = test_batch[0]
            # test_encoded_batch, test_mask_batch = test_batch
        elif isinstance(test_batch, dict):
            test_encoded_batch = test_batch["word_ids"]
        else:
            test_encoded_batch, test_mask_batch = test_batch, np.zeros(test_batch.shape)
        self.sequence_labels = True if test_encoded_batch.shape == test_labels.shape else False
        self.multilabel = True if np.count_nonzero(test_labels) > len(test_labels) else False

    def create_model_mc(self):
        if self.predict_logits:
            self.model.layers[-1].activation = None
            self.model.compile()

        if not dropout_check(self.model):
            self.model_mc = None
            return

        if not self.use_aleatorics:
            self.model_mc = True
        else:
            from tensorflow.python.keras import backend

            dist_layer = next(layer for layer in self.model.layers if "distribution" in layer.name)
            # self.model.layers[-1]
            # ps = np.array([layer.p_logit for layer in model.layers if hasattr(layer, 'p')])
            activation = tf.nn.softmax if not self.multilabel else tf.nn.sigmoid
            out_layer = [activation(dist_layer.output[1])]  # , axis=-1 is default!
            self.model_mc = tf.keras.backend.function(
                [self.model.input, backend.symbolic_learning_phase()],
                out_layer + [dist_layer.output[0].mean(), dist_layer.output[0].variance()],
            )
            self.model_aleatoric = tf.keras.backend.function(
                [self.model.input], out_layer + [tf.reduce_mean(dist_layer.output[0].variance(), axis=-1)]
            )

    def create_stats_collector(self, identifier):
        self.stats[identifier] = {"document": [], "gold": [], "predicted": [], "confidence": [], "softmax": []}
        if "mc" in identifier:
            self.stats[identifier]["mean_mc_array"] = []  # array
            self.stats[identifier]["var_mc_array"] = []  # array
            self.stats[identifier]["var_mc_overall"] = []  # number
            self.stats[identifier]["entropy_mc_overall"] = []  # number and prediction?
            self.stats[identifier]["epistemics"] = []  # number
            self.stats[identifier]["mutual_information"] = []  # number

            if self.identifier:
                self.stats[identifier]["raw"] = []

        if self.use_aleatorics:
            self.stats[identifier]["aleatorics"] = []
            if self.identifier:
                self.stats[identifier]["means"] = []
                self.stats[identifier]["variances"] = []

        return self.stats[identifier]

    def compute_stats(self, out_folder=None):
        def equality(predicted, gold):
            equals = []
            for pred, g in zip(predicted, gold):
                if isinstance(g, np.ndarray):
                    equal = set(pred) == set(g)
                else:
                    equal = pred == g
                equals.append("TP" if equal else "FP")
            return equals

        for identifier, stats in self.stats.items():
            print("Computing stats for {}".format(identifier))
            if self.sequence_labels:
                stats["document"] = [word for text in stats["document"] for word in text.split()]
                stats["gold"] = [label for gold in stats["gold"] for label in gold.split()]
                stats["predicted"] = [
                    pred for sequence_token_pred in stats["predicted"] for pred in sequence_token_pred.split()
                ]
                stats["confidence"] = [
                    prob for sequence_token_prob in stats["confidence"] for prob in sequence_token_prob
                ]

            stats["filename"] = list(range(0, len(stats["document"])))
            stats["field"] = [identifier for i in range(len(stats["document"]))]
            stats["evaluation"] = equality(stats["predicted"], stats["gold"])

            simplex_stats = {
                k: v
                for k, v in stats.items()
                if k in ["document", "filename", "field", "predicted", "gold", "evaluation", "confidence"]
            }

            if out_folder:
                try:
                    pd.DataFrame().from_dict(simplex_stats).to_csv(
                        os.path.join(out_folder, "eval" + identifier + ".csv"), index=False, sep="\t"
                    )
                except Exception as e:
                    print(e)
                    print("DEAL with me later")

    def get_logits(self, identifier, as_frame=True, entity_level=False):
        stats = self.stats[identifier]
        if self.sequence_labels:
            """
            What would be easy is just to convert the stats to entity stats with all similar keys
            This would allow out-of-the-box integration with compare.py
            for each sampling in stats:
                add an extra entry for entity-level

            event-level space =
            1) where gold != 0 #what all the papers probably take
            2) anywhere predicted & gold != 0 #required to account for FPs [overconfident clf]

            Two alternating versions:
            * mask all activated [like dynamic_mask for any d-D shape] (multi-class)
            [event marginals]
            * convert to entity with spans (binary)
            [event probabilities]
            """
            if len(set([len(v) for k, v in stats.items()])) != 1:
                stats = marginalize_stats(stats)

            label2idx = {v: k for k, v in self.idx2label.items()}
            stats = stats_to_entities(stats, label2idx, entity_level=entity_level)
            if as_frame:
                (
                    boolean,
                    argmax_confidence,
                    confidences,
                    epistemics,
                    aleatorics,
                    entropy_mc_overall,
                    predicted,
                    groundtruth,
                ) = (
                    stats["y"],
                    stats["confidence"],
                    stats["softmax"],
                    stats["epistemics"],
                    stats["aleatorics"],
                    stats["entropy_mc_overall"],
                    stats["predicted"],
                    stats["gold"],
                )
        else:
            predicted = np.array(stats["predicted"])
            groundtruth = np.array(stats["gold"])
            confidences = np.array(stats["softmax"])
            argmax_confidence = np.array(stats["confidence"]) / 100
            boolean = np.array(equality(predicted, groundtruth))
            # known = np.array(IID_check(groundtruth, unknown))
            stats["aleatorics"] = (
                stats["aleatorics"] if "aleatorics" in stats else [0 for _ in range(len(stats["predicted"]))]
            )
            stats["epistemics"] = (
                stats["epistemics"] if "epistemics" in stats else [0 for _ in range(len(stats["predicted"]))]
            )
            stats["entropy_mc_overall"] = (
                stats["entropy_mc_overall"] if "entropy_mc_overall" in stats else entropy(confidences)
            )
            aleatorics = stats["aleatorics"]
            epistemics = stats["epistemics"]
            entropy_mc_overall = stats["entropy_mc_overall"]

        if as_frame:
            data = pd.DataFrame(
                list(
                    zip(
                        boolean,
                        argmax_confidence,
                        confidences,
                        aleatorics,
                        epistemics,
                        entropy_mc_overall,
                        predicted,
                        groundtruth,
                    )
                ),
                columns=["y", "confidence", "softmax", "aleatorics", "epistemics", "entropy", "predicted", "gold"],
            )
        else:
            data = stats
        return data

    def compute_uncertainty(mc_pred, means, variances):
        """Calculate both Epistemic and Heteroscedastic Aleatoric uncertainties. """
        # size: num_passes x batch_size x num_classes; T x N x K
        if len(means) == 0:  # MC dropout
            means = np.mean(mc_pred, 0)  # N x K
            variances = np.var(mc_pred, 0)  # N x K
            aleatorics = np.mean(variances, -1)  # N
            epistemics = np.sqrt(np.sum(variances, -1))  # ~total variance

        else:  # HETEROSCEDASTIC extensions
            aleatorics = np.mean(np.mean(variances, 0), -1)
            epistemics = np.mean(np.var(means, 0), -1)
        return aleatorics, epistemics

    def predict_batch_mc(self, batch):

        aggregate, means, variances = [], [], []

        for _ in range(self.posterior_sampling):
            if hasattr(self.model, "submodel"):
                if self.multilabel:
                    from arkham.Bayes.GP.SNGP import mean_field_logits

                    logits, covmat, _ = self.model.call_covmat(
                        batch, training=True, mean_field=False, calc_variance=False
                    )
                    var = tf.linalg.diag_part(covmat)[:, None]
                    sample = tf.nn.sigmoid(mean_field_logits(logits, covmat, mean_field_factor=np.pi / 8.0))
                else:
                    logits, var, sample = self.model.call_covmat(
                        batch, training=True, mean_field=True, calc_variance=True
                    )
                sample = sample.numpy()
                variances.append(var.numpy().flatten())
            else:
                sample = self.model(batch, training=True)

            aggregate.append(sample)

            if self.use_aleatorics:
                means.append(sample[1])
                variances.append(sample[2])

        aggregate, means, variances = np.array(aggregate), np.array(means), np.array(variances)

        # T, N, K
        if len(means) > 0:
            activation = tf.nn.softmax if not self.multilabel else tf.nn.sigmoid
            aggregate = activation(means).numpy()  # take the predictive means instead of samples!

        if not hasattr(self.model, "submodel"):
            aleatorics, epistemics = TouristLeMC.compute_uncertainty(aggregate, means, variances)
        else:
            aleatorics = 0
            epistemics = np.mean(variances, 0)  # T x B

        if len(means) == 0:
            means = np.mean(aggregate, axis=0)  # softmaxed shape: N x K
            variances = np.var(aggregate, axis=0)  # softmaxed shape: N x K

        transposition = (1, 0, 2) if len(aggregate.shape) == 3 else (1, 0, 2, 3)
        sample_aggregator = np.transpose(aggregate, transposition)  # batch_size x T
        return (
            sample_aggregator,
            aleatorics,
            epistemics,
            means,
            variances,
        )  # sampled_batch_prediction  T x batch_size x num_classes

    def score_sampled_prediction(self, pred_mc):
        """
        num_passes (x maxseqlen) x num_classes
        all calculations are done on axis 0 as this is the T "samples" axis
        """
        mean = np.mean(pred_mc, axis=0)
        variance = np.var(pred_mc, axis=0)

        # option 1: posterior X = mean[argmax(mean)]
        mean_argmax = np.argmax(mean, axis=-1)
        mean_argmax_p = np.max(mean, axis=-1)  # mean[mean_argmax]

        overall_variance = np.sqrt(np.sum(variance))  # this is aleatorics at softmax
        try:
            std_argmax_p = variance[mean_argmax]  # std_argmax_confidence
        except Exception as e:  # sequencelevel
            std_argmax_p = np.array([variance[i, m] for i, m in enumerate(mean_argmax)])
        overall_entropy = np.mean(entropy(pred_mc))  # -np.sum(pred_mc * np.log2(pred_mc + 1e-14))
        mutual = mutual_info(pred_mc)
        return mean, variance, overall_variance, mean_argmax, mean_argmax_p, std_argmax_p, overall_entropy, mutual

    def evaluate(self, mode="mc"):
        if mode == "mc" and not self.model_mc:
            logging.debug("No dropout; so no MC dropout")
            return
        stats = self.create_stats_collector(mode)
        sequence_lengths = []

        for i, (test_batch, test_labels) in enumerate(tqdm(self.evaluation_data)):
            if isinstance(test_batch, tuple):
                test_encoded_batch = test_batch[0]
                # test_encoded_batch, test_mask_batch = test_batch
            elif isinstance(test_batch, dict):
                test_encoded_batch, test_mask_batch = test_batch["word_ids"], test_batch["char_ids"]
            else:
                test_encoded_batch, test_mask_batch = test_batch, np.zeros(test_batch.shape)

            # multilabel = True if test_encoded_batch
            batch_texts, batch_golds, batch_sequence_lengths = decode_x_and_y(
                test_encoded_batch.numpy(),
                test_labels.numpy(),
                self.idx2voc,
                self.idx2label,
                sequence_labels=self.sequence_labels,
                multilabel=self.multilabel,
                tokenizer=get_tokenizer(self.tokenizer),
            )
            stats["document"].extend(batch_texts)
            stats["gold"].extend(batch_golds)
            sequence_lengths.extend(batch_sequence_lengths)

            if i == 1:
                start = time.time()

            if mode == "mc":
                pred_mc, aleatorics, epistemics, means, variances = self.predict_batch_mc(test_batch)

                for p, pred in enumerate(pred_mc):
                    if self.sequence_labels:  # remove padding for NER
                        if self.tokenizer:
                            pred = dynamic_wordpiece_mask(pred, batch_sequence_lengths[p])
                        else:
                            pred = pred[:, : batch_sequence_lengths[p], :]
                    (
                        mean,
                        variance,
                        overall_variance,
                        mean_argmax,
                        mean_argmax_p,
                        std_argmax_p,
                        overall_entropy,
                        mutual,
                    ) = self.score_sampled_prediction(pred)
                    if "raw" in stats:
                        stats["raw"].append(pred)

                    if "means" in stats or "variances" in stats:
                        stats["means"].append(means[:, p, :])
                        stats["variances"].append(variances[:, p, :])

                    stats["mean_mc_array"].append(mean)
                    stats["var_mc_array"].append(variance)
                    stats["var_mc_overall"].append(overall_variance)
                    if self.sequence_labels:
                        stats["predicted"].append(" ".join([str(self.idx2label[p]) for p in mean_argmax]))
                        stats["confidence"].append(mean_argmax_p * 100)
                    else:
                        if self.multilabel:
                            pred_array, prob_array = multilabel_prediction(mean, self.idx2label)
                            stats["predicted"].extend(pred_array)
                            stats["confidence"].extend(prob_array * 100)
                        else:
                            stats["predicted"].append(self.idx2label[mean_argmax])
                            stats["confidence"].append(mean_argmax_p * 100)

                    stats["softmax"].append(mean)
                    stats["entropy_mc_overall"].append(overall_entropy)
                    stats["mutual_information"].append(mutual)

                    stats["epistemics"].append(epistemics[p])
                    if self.use_aleatorics:
                        stats["aleatorics"].append(aleatorics[p])

            else:
                # normal predict and convert; YET at batch_level
                if self.use_aleatorics:
                    sample = self.model_aleatoric([test_batch])
                    predictions = sample[0]
                    aleatorics = sample[1]

                elif hasattr(self.model, "submodel"):  # PROXY for SNGP
                    if self.multilabel:
                        from arkham.Bayes.GP.SNGP import mean_field_logits

                        logits, covmat, _ = self.model.call_covmat(
                            test_batch, training=False, mean_field=False, calc_variance=False
                        )
                        var = tf.linalg.diag_part(covmat)[:, None]
                        predictions = tf.nn.sigmoid(mean_field_logits(logits, covmat, mean_field_factor=np.pi / 8.0))
                    else:
                        logits, var, predictions = self.model.call_covmat(
                            test_batch, training=False, mean_field=True, calc_variance=True
                        )
                    if not "epistemics" in stats:
                        stats["epistemics"] = []
                    stats["epistemics"].extend(var.numpy().flatten())
                    predictions = predictions.numpy()
                else:
                    predictions = self.model.predict(test_batch)

                if self.sequence_labels:
                    for s, sequence in enumerate(predictions):
                        if self.tokenizer:
                            sequence = dynamic_wordpiece_mask(sequence, batch_sequence_lengths[s])
                        else:
                            sequence = sequence[: batch_sequence_lengths[s]]

                        sequence_token_pred = [self.idx2label[p] for p in np.argmax(sequence, axis=-1)]
                        sequence_token_prob = multiclass_argmaxprobs(sequence) * 100
                        stats["predicted"].append(" ".join([str(x) for x in sequence_token_pred]))
                        stats["confidence"].append(sequence_token_prob)
                        stats["softmax"].append(sequence)
                else:
                    if self.multilabel:
                        pred_array, prob_array = multilabel_prediction(predictions, self.idx2label)
                    else:
                        prob_array = np.max(predictions, -1) * 100
                        pred_array = [self.idx2label[p] for p in np.argmax(predictions, axis=-1)]
                    stats["predicted"].extend(pred_array)
                    stats["confidence"].extend(prob_array)
                    stats["softmax"].extend(predictions)
                    if self.use_aleatorics:
                        stats["aleatorics"].extend(aleatorics)
            if i == 1:
                timing = round(time.time() - start, 6)
                sample_timing = timing / test_encoded_batch.shape[0]
                print("batchsize: ", test_encoded_batch.shape[0])
                print(f"\nMode {mode} ; Batch Inference time: {timing}s")
                print(f"Mode {mode} ; Sample Inference time: {sample_timing}s")
                print(f"")

        return stats

    def dump(self, out_folder, identifier=""):
        deleteattrs = ["model", "model_mc", "evaluation_data", "model_aleatoric"]  # ""
        for attr in deleteattrs:
            try:
                delattr(self, attr)
            except:
                pass  # ifitdoesnothaveitnoneedtodelete
        if self.posterior_sampling > 10:
            identifier += str(self.posterior_sampling)
        identifier = identifier.replace("/", "-")
        pickle_dumper(self, path=out_folder, filename=str(identifier) + "eval.pickle")

    @staticmethod
    def load(out_folder):
        return pickle_loader(path=os.path.join(out_folder, "eval.pickle"))


def main(
    version,
    test_data=None,
    evaluation_data=None,
    downsampling=0,
    dump=True,
    identifier="",
    raw=False,
    data_identifier="",
    **kwargs,
):
    """
    Load model by version; reupdate params
    """
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    model, _config = load_model_path(modelpath, identifier=identifier)

    _config["data_folder"] = re.sub(r'/home/[^/]+', os.path.expanduser("~"), _config["data_folder"])
    _config["out_folder"] = modelpath  # to ensure if the model has been moved, that the correct folder is being used

    if data_identifier and data_identifier != "dev":
        if _config["identifier"] != data_identifier:
            _config["label2idx"]["ood"] = max(list(_config["idx2label"].keys())) + 1
            _config["idx2label"][max(list(_config["idx2label"].keys())) + 1] = "ood"

        _config["data_folder"] = re.sub(_config["identifier"], data_identifier, _config["data_folder"])
        _config["identifier"] = data_identifier  # clincoos
        evaluation_data = None
        modelpath += data_identifier

    """
    Run simple predict
    """
    test_data = test_texts.get(_config["identifier"]) if not test_data else test_data
    if test_data:
        try:
            predict(model, _config, test_data)  # need to prep it in same way
        except Exception as e:
            print(e)

    if (not evaluation_data) or ("ood" in modelpath) or str(_config.get("ood", None)) != str(None):  # SLOPPY
        _config["downsampling"] = downsampling
        if kwargs.get("oov_corruption"):
            _config["oov_corruption"] = kwargs.get("oov_corruption")

        if str(_config.get("ood", None)) != str(None) or "ood" in modelpath and not data_identifier:
            novel = determine_ood_classes(modelpath=modelpath, _config=_config)
            if isinstance(novel, int):
                _config["label2idx"][novel] = max(list(_config["idx2label"].keys())) + 1
                _config["idx2label"][max(list(_config["idx2label"].keys())) + 1] = novel
            else:
                for novel in _config["ood"]:
                    _config["label2idx"][novel] = max(list(_config["idx2label"].keys())) + 1
                    _config["idx2label"][max(list(_config["idx2label"].keys())) + 1] = novel
            _config["ood"] = None

        # if "twitter" in modelpath:
        if "binary" in _config["data_folder"]:
            _config["data_folder"] = _config["data_folder"].replace("_binary", "")
            _config["label2idx"]["neutral"] = max(list(_config["idx2label"].keys())) + 1
            _config["idx2label"][max(list(_config["idx2label"].keys())) + 1] = "neutral"

        if kwargs.get("raw_data"):
            _config["raw"] = True
        sets = ["test"]
        if data_identifier == "dev":
            sets = ["dev"]
            print("Evaluating on validation set!")
        generators, _, _, _ = generators_from_directory(_config["data_folder"], sets=sets, **_config)
        evaluation_data = generators[sets[0]]

    if kwargs.get("get_evaluation_data"):
        return evaluation_data

    # Tensorflow
    # model.evaluate(evaluation_data)

    if kwargs.get("timings"):
        evaluation_data = [next(iter(evaluation_data)), next(iter(evaluation_data))]

    evaluator = montecarlo_evaluate(
        model,
        _config,
        evaluation_data,
        posterior_sampling=kwargs.get("posterior_sampling", 10),
        identifier=identifier,
        raw=raw,
        skip_MC=kwargs.get("skip_MC", False),
        predict_logits=kwargs.get("predict_logits", False),
        timings=kwargs.get("timings", False),
    )

    if kwargs.get("timings"):
        dump = False
        print("Not saving!")
        del os.environ['CUDA_VISIBLE_DEVICES']  # reset if using cpu!

    if dump:
        out = _config["out_folder"]
        if _config.get("oov_corruption"):
            out = os.path.join(out, str(_config.get("oov_corruption")))
        os.makedirs(out, exist_ok=True)
        identifier += data_identifier
        if raw and not "M" in _config["out_folder"]:
            identifier += "raw"
        evaluator.dump(out, identifier=identifier)
    return evaluator


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("""Evaluation with Monte Carlo Dropout""")
    parser.add_argument("version", type=str, default="")
    parser.add_argument("test_data", nargs="?", type=str, default=None)
    parser.add_argument(
        "-i", dest="identifier", type=str, default="", help="identifier to add to eval.pickle [ensemble]"
    )
    parser.add_argument("-r", dest="raw", action="store_true", default=False, help="save RAW predictions with pickle")
    parser.add_argument("-c", dest="oov_corruption", type=float, default=0, help="OOV corruption")
    parser.add_argument("-s", dest="posterior_samples", type=int, default=10, help="number of forward samples")
    parser.add_argument(
        "-d",
        dest="data_identifier",
        type=str,
        default="",
        help="change data identifier to create ood (CLINC) or evaluate cross-domain [with same dataparams though]",
    )
    parser.add_argument(
        "-l", dest="predict_logits", action="store_true", default=False, help="output model logits [as well]"
    )
    parser.add_argument(
        "--skip", dest="skip_MC", action="store_true", default=False, help="skip computation of MC Dropout stats"
    )
    parser.add_argument("--cpu", dest="cpu", action="store_true", default=False, help="prediction on CPU")
    parser.add_argument("--t", dest="timings", action="store_true", default=False, help="get batch and sample timings")

    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    dump = False if args.timings else True

    main(
        args.version,
        args.test_data,
        identifier=args.identifier,
        dump=dump,
        oov_corruption=args.oov_corruption,
        posterior_sampling=args.posterior_samples,
        raw=args.raw,
        data_identifier=args.data_identifier,
        skip_MC=args.skip_MC,
        predict_logits=args.predict_logits,
        timings=args.timings,
    )
