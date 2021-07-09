import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from itertools import product

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.metrics import f1_score, log_loss, brier_score_loss, mean_squared_error, hamming_loss, accuracy_score

import calibration as cal
from calibration.utils import (
    get_discrete_bins,
    get_equal_bins,
    get_labels_one_hot,
    unbiased_l2_ce,
    bin,
    normal_debiased_ce,
    plugin_ce,
)

from arkham.Bayes.Quantify.inference import beam_search, diverse_beam_search, compare_calibration_measures
from test_evaluation import NER_cases, NER_test_cases

from arkham.StructuredPrediction.SCE import sce, measure_CE, sceV2
from arkham.calibration.evaluator import StructuredLogits, StructuredLogitsStore


def translate_sequence_supervision(y):
    if y.shape == 2:
        y = np.argmax(y, -1)
    return "".join([str(x) for x in y])


def hamming_loss_masked(y_true, y_pred, mask=0):
    structured_loss = tfa.metrics.HammingLoss(mode='multiclass', threshold=None)

    sparsity_mask = (y_true[:, 0] == True).astype(int) & (y_pred[:, 0] == True).astype(
        int
    ) == 0  # you should not be scored for correctly predicting non-entities

    y_true_mask = y_true[sparsity_mask]
    y_pred_mask = y_pred[sparsity_mask]

    structured_loss.update_state(y_true_mask, y_pred_mask)
    inverse_loss = 1 - structured_loss.result().numpy()

    return inverse_loss


def test_structured_calibration_error(beam_width):
    lookup = [
        "perfect",
        "boundary_switch",
        "part_miss",
        "say_nothing",
        "label_boundary_switch",
        "boundary_over",
        "labels_switch",
        "wrong",
    ]

    for index in range(0, 8):
        print(index, lookup[index])

        for case in ["sharp", "unsharp", "weak", "random", "worst"]:

            y_true, y_pred, y_labels, pred_labels = NER_test_cases(
                index=index, calibration=case, tensorize=False, categorical=True
            )

            inverse_loss = hamming_loss_masked(y_true, y_pred)

            # print(y_true)
            # print(y_pred)
            # M = beam_search(y_pred, k=beam_width)
            M = diverse_beam_search(
                y_pred, k=beam_width, n_groups=3, diversity_objective=None, div_strength=0.5, alpha=1
            )

            avg_hamming_loss = np.mean(
                [hamming_loss_masked(y_true, np.eye(y_pred.shape[-1])[np.array(beam)]) for beam in M]
            )

            structerror = sce(
                y_pred,
                y_true,
                M=M,
                structured_loss=hamming_loss_masked,
                lamb=1,  # 0
                debias=False,
                num_bins=None,
                binning_scheme=get_discrete_bins,
                p=1,
                mode='structured',
            )

            merror = sce(
                y_pred,
                y_true,
                M=None,
                structured_loss=hamming_loss_masked,
                lamb=1,  # 0
                debias=False,
                num_bins=None,
                binning_scheme=get_discrete_bins,
                p=1,
                mode='marginal',
            )

            ece = sce(
                y_pred,
                y_true,
                M=None,
                lamb=1,  # 0
                debias=False,
                num_bins=None,
                binning_scheme=get_discrete_bins,
                p=1,
                mode='top-label',
            )

            # merror = cal.get_ece(y_pred, y_true, debias=False, mode='marginal')

            print(
                f"{case} -SCE-: {structerror} | MCE: {merror} ; HS: {inverse_loss}; AHS: {avg_hamming_loss}"
            )  # sanity check

        print()


def test_unit_SCE():
    multiclass = True
    nested = False
    tokenized = "I am Jordy VL".split()

    if nested:
        tokenized = "Lo the king of Belgium".split()
        correct = ["O", "PER", "PER", "PER", "PER|GPE"]
        idx2label = {0: "O", 1: "PER", 2: "GPE"}
        y_pred = np.array([[0.5, 0.2, 0.3], [0.8, 0.1, 0.1], [0.7, 0.1, 0.2], [0.2, 0.6, 0.2], [0.15, 0.4, 0.45]])

        # alternate : add K y_pred
        # welcome to the sigmoid! FAK

        label2idx = {v: k for k, v in idx2label.items()}
        y_true = np.zeros((len(correct), len(idx2label)), dtype=int)
        for i, x in enumerate(correct):
            for y in x.split("|"):
                index = label2idx.get(y)
                y_true[i, index] = 1
    else:
        if multiclass:
            correct = ["O", "O", "B-PER", "I-PER"]
            idx2label = {0: "O", 1: "B-PER", 2: "I-PER"}
            y_pred = np.array([[0.8, 0.1, 0.1], [0.7, 0.1, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])

        else:
            correct = ["O", "O", "PER", "PER"]
            idx2label = {0: "O", 1: "PER"}
            y_pred = np.array([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
            # np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        label2idx = {v: k for k, v in idx2label.items()}

        # fully correct
        y_true = np.vectorize(label2idx.get)(correct)
        y_true = tf.one_hot(y_true, depth=len(idx2label), dtype=np.int32).numpy()

    sceeval = sceV2(
        y_pred,
        y_true,
        M=None,
        structured_loss=hamming_loss,
        lamb=1,
        debias=True,
        num_bins=None,
        binning_scheme=get_discrete_bins,
        p=2,
        mode='structured',
    )

    compare_calibration_measures(y_true, y_pred, beam_width=0, diversity=False)
    compare_calibration_measures(y_true, y_pred, beam_width=5, diversity=False)
    compare_calibration_measures(y_true, y_pred, beam_width=5, diversity=True)


def functional_process_onkey(key, json_object, func):
    if isinstance(json_object, list):
        for list_element in json_object:
            functional_process_onkey(key, list_element, func)
    elif isinstance(json_object, dict):
        if key in json_object:
            json_object[key] = func(json_object[key])
        for dict_value in json_object.values():
            functional_process_onkey(key, dict_value, func)


"""
def flat_mask(nested_list):
    mask = []
    for index, element in enumerate(nested_list):
        if isinstance(element, Iterable) and not element == None:
            for el in element:
                mask.append(index)
        else:
            mask.append(index)
    return mask

"""

"""
import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
"""
"""
def possibly_nested_flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in possibly_nested_flatten(i):
                yield j
        else:
            yield i
"""


def flatten_hierarchy(d, sep="/"):
    import collections

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    return obj

    # def flat_hierarchy(hierarchy, joiner="/"):
    #     flats = []
    #     count = 0
    #     for parent, children in hierarchy.items():
    #         if isinstance(children, list):

    #         for list_element in json_object:
    #         functional_process_onkey(key, list_element, func)
    # elif isinstance(json_object, dict):


def test_unit_hierarchical_SCE():
    flat2idx = {}

    hierarchy = {  # parent: [child1, child2]
        "person": ["manchild", {"famous": ["hollywood", "bollywood", "kessel"]}],
        "software": ["deep-learning"],
        "organization": [],
        "": [],
    }

    tokenized = "Jordy studies structured prediction at Contract.fit".split()
    y_true = ["person/famous/kessel", "", "software/deep-learning", "software/deep-learning", "", "organization"]

    flat = flatten_hierarchy(hierarchy)
    prior = None
    hierarchical_loss = None


def test_structuredlogits():
    tokenized = "I am Jordy VL".split()
    y_true = ["O", "O", "B-PER", "I-PER"]

    y_pred = np.array([[0.8, 0.1, 0.1], [0.7, 0.1, 0.2], [0.2, 0.6, 0.2], [0.8, 0.1, 0.1]])
    idx2label = {0: "O", 1: "B-PER", 2: "I-PER"}
    label2idx = {v: k for k, v in idx2label.items()}
    y_true_idx = np.vectorize(label2idx.get)(y_true)

    logits = StructuredLogits(
        f_x=y_pred,
        y_true=y_true,
        y_hat=None,
        probas=y_pred,
        c=None,
        tokenized=tokenized,
        document_masks=[list(range(len(y_true)))],
    )
    logits.logits_to_pred_idx(idx2label)

    print(logits.to_dict())


def test_exp():
    a = 5
    b = 3
    print(np.exp(a + b))
    assert np.exp(a + b) == np.exp(a) * np.exp(b)
