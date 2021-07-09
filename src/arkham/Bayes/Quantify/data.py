#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"
"""
Important reference: https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
"""
import os
import sys
from collections import Counter, OrderedDict
from itertools import zip_longest
import regex as re
import random

random.seed(42)
from tqdm import tqdm

import pandas as pd
import numpy as np

np.random.seed(42)

import csv

csv.field_size_limit(sys.maxsize)
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from arkham.utils.utils import num_plot, sklearn_split


def generate_vocabulary_label2idx(
    labelsets, lowercase, min_token_len, min_token_freq, max_vocabulary, pretrained_embeddings=None, debug=False
):
    def dump_vocab(freqdist):
        with open("vocab.txt", "w") as f:
            for idx, (word, freq) in enumerate(
                sorted(zip(freqdist.keys(), freqdist.values()), key=lambda tupl: tupl[1], reverse=True)
            ):
                f.write("{}\t{}\t{}\n".format(word, freq, idx))

    labels = set()
    vocabulary = []
    for labelset, documents in labelsets.items():
        for document in documents:
            sequence_pack = zip_longest(*document)
            for iterable, label in sequence_pack:
                if not isinstance(iterable, list):
                    iterable = [iterable]
                for word in iterable:
                    if len(word) >= min_token_len and word != "PAD":
                        word = word if not lowercase else word.lower()
                        vocabulary.append(word)
                    if label is not None:
                        if not isinstance(label, (list, tuple, np.ndarray)):
                            labels.add(label)
                        else:
                            labels.update(label)

    reverse = False if not isinstance(list(labels)[0], str) else True
    label2idx = {label: idx for idx, label in enumerate(sorted(labels, reverse=reverse))}

    freqdist = Counter(vocabulary)

    splitpoint = (
        len(freqdist) + 1
        if not max_vocabulary
        else max_vocabulary + 1
        if max_vocabulary > 1
        else int(max_vocabulary * len(freqdist))
    )  # proportion float is also allwoed
    voc2idx = {
        word: idx + 1
        for idx, (word, freq) in enumerate(
            sorted(zip(freqdist.keys(), freqdist.values()), key=lambda tupl: tupl[1], reverse=True)[:splitpoint]
        )
        if freq > min_token_freq
    }
    voc2idx["PAD"] = 0
    # , **{"PAD": 0, "UNK": 1}}
    if debug:
        print(len(freqdist))
        dump_vocab(freqdist)

    return voc2idx, label2idx


class CharSequenceEncoder(object):
    def __init__(
        self,
        lowercase=True,
        special_chars_to_keep=["€", "£"],
        filler="P",
        standardize_digits=False,
        squeeze_spacing=False,
    ):
        self.lowercase = lowercase
        self.special_chars_to_keep = special_chars_to_keep
        self.filler = filler
        self.standardize_digits = standardize_digits
        self.squeeze_spacing = squeeze_spacing
        self.vocabulary = self.create_character_set()
        self.char2idx = dict((char, i) for i, char in enumerate(self.vocabulary))

    def create_character_set(self):
        digits = ["1"] if self.standardize_digits else string.digits
        letters = string.ascii_lowercase if self.lowercase else string.ascii_letters
        uniquecharset = (
            list(letters) + list(digits) + list(string.punctuation) + ['\n', ' '] + self.special_chars_to_keep
        )
        if self.filler in uniquecharset:
            self.filler = "µ"
        if self.filler:
            uniquecharset[0:0] = [self.filler]
            uniquecharset[1:1] = ["UNK"]
        return uniquecharset


def generate_char2idx_idx2char(as_list=False):
    # print(CharSequenceEncoder(lowercase=False, filler="PAD").char2idx)
    char2idx = {
        'PAD': 0,
        'UNK': 1,
        'a': 2,
        'b': 3,
        'c': 4,
        'd': 5,
        'e': 6,
        'f': 7,
        'g': 8,
        'h': 9,
        'i': 10,
        'j': 11,
        'k': 12,
        'l': 13,
        'm': 14,
        'n': 15,
        'o': 16,
        'p': 17,
        'q': 18,
        'r': 19,
        's': 20,
        't': 21,
        'u': 22,
        'v': 23,
        'w': 24,
        'x': 25,
        'y': 26,
        'z': 27,
        'A': 28,
        'B': 29,
        'C': 30,
        'D': 31,
        'E': 32,
        'F': 33,
        'G': 34,
        'H': 35,
        'I': 36,
        'J': 37,
        'K': 38,
        'L': 39,
        'M': 40,
        'N': 41,
        'O': 42,
        'P': 43,
        'Q': 44,
        'R': 45,
        'S': 46,
        'T': 47,
        'U': 48,
        'V': 49,
        'W': 50,
        'X': 51,
        'Y': 52,
        'Z': 53,
        '0': 54,
        '1': 55,
        '2': 56,
        '3': 57,
        '4': 58,
        '5': 59,
        '6': 60,
        '7': 61,
        '8': 62,
        '9': 63,
        '!': 64,
        '"': 65,
        '#': 66,
        '$': 67,
        '%': 68,
        '&': 69,
        "'": 70,
        '(': 71,
        ')': 72,
        '*': 73,
        '+': 74,
        ',': 75,
        '-': 76,
        '.': 77,
        '/': 78,
        ':': 79,
        ';': 80,
        '<': 81,
        '=': 82,
        '>': 83,
        '?': 84,
        '@': 85,
        '[': 86,
        '\\': 87,
        ']': 88,
        '^': 89,
        '_': 90,
        '`': 91,
        '{': 92,
        '|': 93,
        '}': 94,
        '~': 95,
        '\n': 96,
        ' ': 97,
        '€': 98,
        '£': 99,
    }
    idx2char = {v: k for k, v in char2idx.items()}
    # lowercase_char2idx = {'PAD': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36, '!': 37, '"': 38, '#': 39, '$': 72, '%': 41, '&': 42, "'": 43, '(': 44, ')': 45, '*': 46, '+': 47, ',': 48, '-': 49, '.': 50, '/': 51, ':': 52, ';': 53, '<': 54, '=': 55, '>': 56, '?': 57, '@': 58, '[': 59, '\\': 60, ']': 61, '^': 62, '_': 63, '`': 64, '{': 65, '|': 66, '}': 67, '~': 68, '\n': 69, '€': 70, '£': 71, ' ': 72}
    # DEV: char2idx #<EOS>
    if as_list:
        return [idx2char[i] for i in range(len(char2idx))]
    return char2idx, idx2char


def generate_char_encodings(X, max_chars_len):
    char2idx, _ = generate_char2idx_idx2char()
    char_encodings = np.zeros((len(X), max_chars_len), dtype=np.int32)
    for idx, doc in enumerate(X):
        max_length = min(len(doc), max_chars_len)
        for i in range(0, max_length):
            char_encodings[idx, i] = char2idx.get(doc[i], 0)
    return char_encodings


def generate_case2idx(as_list=False):
    case2idx = {
        'PAD': 0,
        'allLower': 1,
        'allUpper': 2,
        'initialUpper': 3,
        'other': 4,
        'mainly_numeric': 5,
        'contains_digit': 6,
        'numeric': 7,
    }
    idx2case = {v: k for k, v in case2idx.items()}
    if as_list:
        return [idx2case[i] for i in range(len(case2idx))]
    return case2idx, idx2case


# define casing s.t. NN can use case information to learn patterns
def generate_case_encodings(sentence, case2idx):
    def wordcase_encoding(word, case2idx):
        casing = 'other'

        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1

        digitFraction = numDigits / float(len(word))

        if word.isdigit():  # Is a digit
            casing = 'numeric'
        elif digitFraction > 0.5:
            casing = 'mainly_numeric'
        elif word.islower():  # All lower case
            casing = 'allLower'
        elif word.isupper():  # All upper case
            casing = 'allUpper'
        elif word[0].isupper():  # is a title, initial char upper, then all lower
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'

        # DEV: could dig up patternlib from before H_regex

        return case2idx[casing]

    case_encodings = np.zeros((len(sentence)), dtype=np.int32)
    for i, word in enumerate(sentence):
        case_encodings[i] = wordcase_encoding(word, case2idx)
    return case_encodings


def tokenize_composition(text, composition, token_pattern=r"\b\w\S+\b"):
    """
    tokenize text according to compositition

    Args:
        text (str):
        composition (list):
        token_pattern (str, optional):

    Returns:
        TYPE: list of lists * len(composition)

    Raises:
        NotImplementedError for character composition (not yet ready)

    DEV: should make it recursive
    """
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#fit_on_texts
    sentence_split_token = ["<sssss>", "<eos>", "\n"]
    tokenized = [text]  # DEFAULT
    if "sentence" in composition:
        splitted = False
        for splitter in sentence_split_token:
            if splitter in text:
                tokenized = text.split(splitter)
                splitted = True
        if not splitted:
            from nltk import sent_tokenize

            tokenized = sent_tokenize(text)
    else:
        tokenized = [re.sub("(" + "|".join(sentence_split_token) + ")", " ", text)]  # get the tokens out
    if "word" in composition:
        if len(tokenized) == 1 and not "sentence" in composition:
            if token_pattern:
                tokenized = [match.group() for match in re.finditer(token_pattern, tokenized[0])]
            if not token_pattern and len(tokenized) == 1:
                tokenized = [match.group() for match in re.finditer("\S+", tokenized[0])]
        else:
            tokenized = [[match.group() for match in re.finditer(token_pattern, sentence)] for sentence in tokenized]
    return tokenized


def calc_oov(encoded):
    if not isinstance(encoded[0], np.int32):
        for sent in encoded:
            try:
                assert len(sent) > 0
            except AssertionError as e:
                print(e)

        encoded = np.hstack(encoded)
        assert len(encoded) > 0
    return round(len(np.where(encoded == 0)[0]) / encoded.shape[0], 2)


def corrupt_text(encoded, oov_corruption=0.1):
    # replace a random percentage of words with OOV [not the ones already OOV!]
    # random, but NOT those indices, so condition mask replacement? :)
    # base = calc_oov(encoded)  # 0.15
    mask = np.random.choice([False, True], size=encoded.shape[0], p=[oov_corruption, 1 - oov_corruption])
    encoded_mask = np.where(mask, encoded, 0)
    masked = np.where(encoded > 0, encoded_mask, 0)
    return masked


def corrupt_text_stairwise(encoded, oov_corruption=0.1):
    permuted_indices = np.random.permutation(list(range(encoded.shape[0])))
    mask = permuted_indices[: int(encoded.shape[0] * oov_corruption)]
    encoded[mask] = 0
    return encoded


def encode_text(tokenized, voc2idx, oov=0, lowercase=False, oov_corruption=0, composition=["word"]):
    def encode(word):
        return voc2idx.get(word.lower() if lowercase else word, oov)

    # special token sentence boundary?
    # check depth? and len of smallest -> == composition
    if not "sentence" in composition:
        encoded_text = np.vectorize(encode, otypes=[np.int32])(tokenized)

    else:
        encoded_text = np.array(
            [np.vectorize(encode, otypes=[np.int32])(sent) for sent in tokenized if len(sent) > 0], dtype="object"
        )
    version = "v3"
    if oov_corruption:
        # could make into tuple with version
        base = len(np.where(encoded_text == None)[0]) / encoded_text.shape[0]

        # version 1: max(base, corruption)
        if version == "v1":
            if oov_corruption > base:
                corrupted = corrupt_text(encoded_text, oov_corruption=oov_corruption)
            else:
                return encoded_text

        # version 2: (1-base)*corruption #base is already in there
        elif version == "v2":
            corrupted = corrupt_text(encoded_text, oov_corruption=oov_corruption)

        elif version == "v3":
            if oov_corruption > base:
                corrupted = corrupt_text_stairwise(encoded_text, oov_corruption=oov_corruption)
            else:
                return encoded_text
        # print("OOV rate: ", calc_oov(corrupted))
        encoded_text = corrupted
    return encoded_text


def pad_seq(sequence, max_document_len, pad_value=0):
    return np.concatenate(
        (sequence[:max_document_len], [pad_value for _ in range(max_document_len - len(sequence))])
    ).astype(np.int32)


def prepare_composition(labelsets, composition=["word"], token_pattern=r"\b\w\S+\b", ood=None):
    ood = [ood] if isinstance(ood, int) else ood
    texts = []
    for labelset, documents in labelsets.items():
        new_documents = []
        for text, label_array in documents:
            if not text.strip():
                continue
            tokenized = tokenize_composition(text, composition, token_pattern)
            if tokenized:  # it can be that no tokens can be generated -> remove all together
                if ood is not None:
                    if any(x in ood for x in label_array):
                        continue
                    # if len(label_array) > 1:
                    #     if set(ood).issubset(label_array):
                    #         continue
                new_documents.append([tokenized, label_array])
        labelsets[labelset] = new_documents
    return labelsets


def imbalance_ratio(data):
    """
    The C2 measure is a well known index computed for measuring class balance. Here we adopt a version of the measure
    that is also suited for multiclass classification problems [Tanwani and Farooq, 2010]
    LARGER value means more imbalance.
    """
    data = data.tolist()
    c = Counter(data)
    total = len(data)
    imb_ratio = ((len(c) - 1) / len(c)) * sum([freq / (total - freq) for label, freq in c.items()])
    final_ratio = 1 - (1 / imb_ratio)
    return final_ratio


def text_stats(texts, token_pattern):  # apply on text?
    collection = {}
    collection["c/w"] = [len(word) for text in texts for word in tokenize_composition(text, ["word"], token_pattern)]
    collection["chars"] = [len(text) for text in texts]
    collection["words"] = [len(tokenize_composition(text, ["word"], token_pattern)) for text in texts]
    collection["sentences"] = [len(tokenize_composition(text, ["sentence"], token_pattern)) for text in texts]
    collection["w/s"] = [len(x) for text in texts for x in tokenize_composition(text, ["sentence"], token_pattern)]
    for key in collection:
        num_plot(collection[key], key, plot=True)


# def label_stats(labels):
#     collection = {}
#     collection["label"] = [len(word) for text in texts for word in tokenize_composition(text, ["word"], token_pattern)]
#     for key in collection:
#         num_plot(collection[key], key, plot=True)


def encode_labels(labels, label2idx, sequence_labels=False):
    def label_transform(label):
        label_idx = label2idx.get(label, label2idx.get("ood"))
        if label_idx is None:
            if label == -100:  # ignored for BERT
                label_idx = label
        return label_idx

    backoff = label2idx.get("ood")
    multilabel = True if len(labels) > 1 else False
    if sequence_labels:
        # if len(document[1]) == len(encoded_text) and (len(document[1]) >= 1):
        # DEV: sparse categorical encoding for NER for now...
        encoded_labels = np.array([label_transform(label) for label in labels])
    elif multilabel:
        # DEV : binary cross entropy encoding
        encoded_labels = np.zeros(len(label2idx), dtype=int)
        encoded_labels[[label2idx[label] for label in labels]] = 1
    else:
        # DEV: categorical cross entropy encoding
        encoded_labels = tf.keras.utils.to_categorical(
            [label2idx.get(label, backoff) for label in labels], num_classes=len(label2idx), dtype=int
        )[0]
    return encoded_labels


def encode_idx(
    labelset,
    voc2idx,
    label2idx,
    oov=0,
    lowercase=False,
    max_document_len=None,
    max_sentences=None,
    max_chars_len=None,
    oov_corruption=0,
    composition=["word"],
    input_dtype=tf.int32,
    sequence_labels=False,
    skip_short=False,
):
    remove = []
    additional_inputs = {k: [] for k in composition if k not in ["word", "sentence"]}

    for i, document in enumerate(labelset):  # composition for sentences to be avoided for now
        if input_dtype == tf.string:
            encoded_text = np.array(document[0])
            if "sentence" in composition:
                encoded_text = np.array(
                    [
                        " ".join(sent) if not lowercase else " ".join(sent).lower()
                        for sent in encoded_text
                        if len(sent) > 0
                    ],
                    dtype="object",
                )
        else:
            encoded_text = encode_text(
                document[0],
                voc2idx,
                oov=oov,
                lowercase=lowercase,
                oov_corruption=oov_corruption,
                composition=composition,
            )
        encoded_labels = encode_labels(document[1], label2idx, sequence_labels=sequence_labels)
        if sequence_labels:
            assert encoded_labels.shape[0] == encoded_text.shape[0]

        if "sentence" in composition:
            if len(encoded_text) <= 1 or any([len(x) == 0 for x in encoded_text]) and not input_dtype == tf.string:
                logging.debug("Instance with length 0!")
                remove.append(i)
                continue

            if max_sentences:
                encoded_text = encoded_text[:max_sentences]
        else:
            if max_document_len:
                document = document[:max_document_len]
                if sequence_labels:
                    encoded_text = pad_seq(encoded_text, max_document_len, pad_value=0)
                    encoded_labels = pad_seq(encoded_labels, max_document_len, pad_value=0)
                    assert len(encoded_text) == len(encoded_labels)
                else:
                    encoded_text = encoded_text[:max_document_len]

            if not any(encoded_text) or (len(encoded_text) < 5):  # NEED at least 5 tokens for CNN kernel!
                if skip_short:
                    # print(f"SHORT text found: {document}")
                    remove.append(i)
                    continue

        labelset[i] = [encoded_text, encoded_labels]

        if "character" in composition:
            additional_inputs["character"].append(generate_char_encodings(document[0], max_chars_len))
            # could add casing inputs as well; change to dictionary, then pick out both

            if "casing" in composition:
                case2idx, _ = generate_case2idx()
                additional_inputs["casing"].append(generate_case_encodings(document[0], case2idx))

    labelset = [labelset[i] for i in range(len(labelset)) if i not in remove]
    if "character" in additional_inputs:
        additional_inputs["character"] = [
            additional_inputs["character"][i] for i in range(len(additional_inputs["character"])) if i not in remove
        ]
    if "casing" in additional_inputs:
        additional_inputs["casing"] = [
            additional_inputs["casing"][i] for i in range(len(additional_inputs["casing"])) if i not in remove
        ]
    return labelset, additional_inputs


def batch_shuffle_generator(
    series,
    shuffle=True,
    buffer_size=1000,
    batch_size=32,
    max_document_len=None,
    max_sentences=None,
    sequence_labels=None,
    composition=["word"],
    input_dtype=tf.int32,
):
    """
    # SHUFFLE BY EXAMPLES
    if shuffle:
        # presort here by length
        tf.random.set_seed(42)
        series = series.shuffle(buffer_size, reshuffle_each_iteration=True)
    """

    padded_shapes = [[None], [None]]
    # if len(series._flat_shapes) > 2:
    #     padded_shapes[0].insert(0, None)
    # DEV: cannot figure out how to do the batching for NER
    if "sentence" in composition:
        padded_shapes[0].insert(0, max_sentences if max_sentences else None)
        if input_dtype == tf.string:
            padded_shapes[0] = [padded_shapes[0][0]]
    else:
        if max_document_len:
            # train_data.map(lambda x: x[:8])
            padded_shapes[0] = [max_document_len]
    padded_shapes = tuple(padded_shapes)
    batched = series.padded_batch(
        batch_size, padded_shapes=padded_shapes, drop_remainder=True
    )  # .prefetch(tf.data.experimental.AUTOTUNE)

    # SHUFFLE BY BATCHES
    if shuffle:
        tf.random.set_seed(42)
        batched = batched.shuffle(buffer_size, reshuffle_each_iteration=True).prefetch(tf.data.experimental.AUTOTUNE)
    return batched


def hierarchical_tensorize(variable_length_sequences, max_sentences=None, max_document_len=None):
    """
    since support for ragged tensors is still incipient; we need to convert ragged to full tensors
    Args:
        variable_length_sequences (list): 2D variable length "sentences x words" elements
        max_sentences (int, optional):
        max_document_len (int, optional): IS abused for max words per sentence

        -> .to_tensor()[:,:max_sentences,:max_document_len]
    if using ragged tensor -> can only use batch without padding

    Returns:
        2D tensor; no batch_size present yet
    """

    docs = tf.ragged.constant(variable_length_sequences)
    if isinstance(variable_length_sequences[0], str):
        if max_sentences:
            docs = docs[:max_sentences]
        return docs
    docs = docs.to_tensor()
    if max_sentences:
        docs = docs[:max_sentences, :]

    if max_document_len:
        docs = docs[:, :max_document_len]
    return docs


def encode_test(test_data, _config):
    tokenized = tokenize_composition(test_data, _config["composition"], _config["token_pattern"])
    tokenizer = get_tokenizer(_config.get("model_class"))
    if tokenizer:
        from arkham.Bayes.Quantify.BERT import tokenize

        encoded, _ = tokenize(
            [tokenized],
            tokenizer,
            max_document_len=_config["max_document_len"],
            sequence_labels=_config["sequence_labels"],
            labels=None,
        )
    else:
        encoded = encode_text(
            tokenized,
            _config["voc2idx"],
            oov=_config["voc2idx"]["PAD"],
            lowercase=_config["lowercase"],
            composition=_config["composition"],
        )

        if "character" in _config["composition"]:
            encoded = [encoded]
            encoded.append(generate_char_encodings(tokenized, _config["max_chars_len"]))
            if "casing" in _config["composition"]:
                case2idx, _ = generate_case2idx()
                encoded.append(generate_case_encodings(tokenized, case2idx))
            encoded = tuple(encoded)

    ### FIX ME
    #                    new_tuple = (encoded[labelset][i][0], *tuple(additional_inputs[k][i] for k in additional_composition), encoded[labelset][i][1])

    # additional_inputs["character"].append(generate_char_encodings(document[0], max_chars_len))
    # # could add casing inputs as well; change to dictionary, then pick out both

    # if "casing" in composition:
    #     case2idx, _ = generate_case2idx()
    #     additional_inputs["casing"].append(generate_case_encodings(document[0], case2idx))

    if "sentence" in _config["composition"]:
        encoded = hierarchical_tensorize(
            [encoded], max_sentences=_config["max_sentences"], max_document_len=_config["max_document_len"]
        )
    return tokenized, encoded


def get_tokenizer(model_class):
    tokenizer = None
    if model_class:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_class)
    return tokenizer


def infer_split(filename):
    filename = os.path.basename(filename)
    if "train" in filename:
        return "train"
    elif "dev" in filename or "val" in filename:
        return "validation"
    elif "test" in filename:
        return "test"


def generators_from_directory(
    directory,
    sets=["train", "dev", "test"],
    voc2idx=None,
    label2idx=None,
    downsampling=0,
    buffer_size=1000,
    batch_size=32,
    composition=["word"],
    token_pattern=r"\b\w\S+\b",
    lowercase=False,
    min_token_len=2,
    min_token_freq=2,
    max_vocabulary=0,
    overblend_vocabulary=True,
    pretrained_embeddings=None,
    max_document_len=None,
    max_sentences=None,
    max_chars_len=None,
    sequence_labels=False,
    sort_by_seqlen=True,
    column_map=None,
    oov_corruption=0,
    raw=False,
    debug=False,
    ood=None,
    model_class=None,
    **kwargs,
):
    '''
    Read data files and return input/target TF2 generators
    '''

    def process_CONLL(filename):
        # loads a column dataset into list of (tokens, labels)
        # assumes BIO(IOB2) labeling
        def load_dataset_from_column(path, schema='bio'):
            with open(path, 'r', encoding='utf-8') as f:
                sentences = []
                tokens = []
                labels = []
                for line in f.readlines() + ['']:
                    if len(line) == 0 or line.startswith('-DOCSTART-') or line.isspace():
                        if len(tokens) > 0:
                            if schema is not None and schema != 'none':
                                if schema == 'iob':
                                    labels = iob2bio(labels)
                                elif schema == 'iobes':
                                    labels = iobes2bio(labels)
                                validate_bio(labels)
                            sentences.append((tokens, labels))
                        tokens = []
                        labels = []
                    else:
                        splits = line.strip().split()
                        token, label = splits[0], splits[-1]
                        tokens.append(token)
                        labels.append(label)
            return sentences

        documents = []

        with open(filename, "r") as f:
            next(f)
            next(f)

            document = [[], []]
            for line in f:
                input_output_split = line.split()

                if len(input_output_split) == 4:
                    document[0].append(input_output_split[0])
                    document[1].append(input_output_split[-1])

                elif len(input_output_split) == 0:  # document_boundary
                    documents.append(document)
                    document = [[], []]
                else:
                    document = [[], []]
            f.close()
        return documents

    def process_NER_flair(directory):
        import flair.datasets as flair_datasets

        def validate_bio(labels):
            for cur_label, next_label in zip(labels, labels[1:] + ['O']):
                if cur_label[0] == 'O':
                    assert next_label[0] == 'O' or next_label[0] == 'B'
                    continue
                elif cur_label[0] == 'B':
                    assert (
                        next_label[0] == 'O'
                        or next_label[0] == 'B'
                        or (next_label[0] == 'I' and cur_label[1:] == next_label[1:])
                    )
                elif cur_label[0] == 'I':
                    assert (
                        next_label[0] == 'O'
                        or next_label[0] == 'B'
                        or (next_label[0] == 'I' and cur_label[1:] == next_label[1:])
                    )
                else:
                    assert False

        def iob2bio(iob_labels):
            bio_labels = []
            for prev_label, cur_label in zip(['O'] + iob_labels[:-1], iob_labels):
                if (prev_label[0] == 'O' and cur_label[0] == 'I') or (
                    prev_label[0] != 'O' and cur_label[0] == 'I' and prev_label[2:] != cur_label[2:]
                ):
                    bio_labels.append('B' + cur_label[1:])
                else:
                    bio_labels.append(cur_label)
            return bio_labels

        def iobes2bio(iobes_labels):
            bio_labels = []
            for label in iobes_labels:
                if label[0] == 'S':
                    bio_labels.append('B' + label[1:])
                elif label[0] == 'E':
                    bio_labels.append('I' + label[1:])
                else:
                    bio_labels.append(label)
            return bio_labels

        def get_flair_corpus(identifier):
            try:
                corpus = getattr(flair_datasets, identifier)()
            except AttributeError as e:
                available = "\t".join(sorted([x for x in dir(flair_datasets) if x.isupper()]))
                print(f"identifier {identifier} not present as flair corpus {available}")
                raise e
            return corpus

        def construct_ner(data_folder, columns={0: 'text', 1: 'pos', 2: 'ner'}):
            # init a corpus using column format, data folder and the names of the train, dev and test files
            return flair_datasets.ColumnCorpus(data_folder, columns)  # tag_to_biloes="ner"

        def get_flair_corpus_v2(identifier):
            from flair.data_fetcher import NLPTaskDataFetcher, NLPTask

            try:
                corpus = NLPTaskDataFetcher.load_corpus(getattr(NLPTask, identifier))  # (tag_to_bioes=None)
            except AttributeError as e:
                available = "\t".join(sorted([x for x in dir(flair_datasets) if x.isupper()]))
                print(f"identifier {identifier} not present as flair corpus {available}")
                raise e
            return corpus

        def simplify_corpus(corpus, tag_scheme="iobes"):
            collect = []

            if hasattr(corpus, "sentences"):
                iterator = corpus.sentences
            else:
                iterator = corpus.dataset.sentences

            for sentence in iterator:
                # sentence.convert_tag_scheme(tag_type="ner", target_scheme="iob")
                labels = [token.get_tag("ner").value for token in sentence]
                if tag_scheme == "iob":
                    labels = iobes2bio(labels)
                words = [token.text for token in sentence]
                assert len(labels) == len(words)
                collect.append([words, labels])
            return collect

        identifier = os.path.basename(directory).upper()
        if identifier == "ONTONOTES":
            corpus = construct_ner(directory, columns={0: 'text', 1: 'ner'})
        elif identifier == "ACE2005":
            corpus = construct_ner(directory, columns={0: 'text', 2: 'ner'})
        elif identifier == "SROIE2019":
            corpus = construct_ner(directory, columns={0: 'text', 1: 'ner'})
        elif "ACL" in identifier:
            corpus = construct_ner(directory, columns={0: 'text', 1: 'ner'})
        else:
            corpus = get_flair_corpus(identifier)  # construct_ner(directory, columns={0: 'text', 1: 'pos', 2: 'ner'})
        # tags = corpus.make_tag_dictionary(tag_type="ner").get_items()
        # label_dictionary = corpus.make_tag_dictionary(tag_type="ner") #BIOSE
        # stats = corpus.obtain_statistics()
        # print(f"label/tag_dictionary {label_dictionary} \n stats: {stats}")
        # IO format...
        tag_scheme = "iobes"  # "iobes" if not "CONLL" in identifier else "iob"

        labelsets = {}
        labelsets["train"] = simplify_corpus(corpus.train, tag_scheme=tag_scheme)
        labelsets["dev"] = simplify_corpus(corpus.dev, tag_scheme=tag_scheme)
        labelsets["test"] = simplify_corpus(corpus.test, tag_scheme=tag_scheme)

        return labelsets["train"], labelsets["dev"], labelsets["test"]

    def process_huggingface_tokenclf(filename):
        def relabel(sequences, idx2label):
            seqs = []
            for sequence in sequences:
                seqs.append([idx2label[x] for x in sequence])
            return seqs

        from datasets import load_dataset

        dataset = "conll2003" if "conll_03" in filename else ""
        # split = infer_split(filename)
        d = load_dataset(dataset)
        task = "ner"
        label_list = d["train"].features[f"{task}_tags"].feature.names
        idx2label = OrderedDict(enumerate(label_list))
        labelsets = {}
        labelsets["train"] = list(zip(d["train"]["tokens"], relabel(d["train"][f"{task}_tags"], idx2label)))
        labelsets["dev"] = list(zip(d["validation"]["tokens"], relabel(d["validation"][f"{task}_tags"], idx2label)))
        labelsets["test"] = list(zip(d["test"]["tokens"], relabel(d["test"][f"{task}_tags"], idx2label)))
        return labelsets["train"], labelsets["dev"], labelsets["test"]

    def process_IMDB_YELP(filename, **kwargs):
        df = pd.read_csv(
            filename, encoding="utf-8-sig", engine="python", sep="\t\t", names=["user", "product", "label", "text"]
        )
        df = df.sample(frac=1, random_state=42)
        df = df.dropna(subset=["label", "text"]).reset_index(drop=True)  # shuffle the bastard
        df = df[df["text"].str.len() > 20].reset_index(drop=True)  # maybe this is dangerous?
        # df = df[df["label"] != 5].reset_index(drop=True)
        return [[df["text"].iloc[i], [df["label"].iloc[i]]] for i in range(len(df))]  # tokenize text: document -> words
        if not column_map:
            column_map = {"label_process": 0, "text": 1}
        df = pd.read_csv(filename, encoding="utf-8-sig", engine="python", sep="\t")
        df = df.sample(frac=1, random_state=42)
        df = df.dropna(subset=list(column_map.keys())).reset_index(drop=True)  # shuffle the bastard
        df = df[df["text"].str.len() > 20].reset_index(drop=True)  # maybe this is dangerous?
        label_column = [col for col in column_map if col.startswith("label")][0]
        return [
            [df["text"].iloc[i], [df[label_column].iloc[i]]] for i in range(len(df))
        ]  # tokenize text: document -> words

    def process_reuters(filename, **kwargs):
        def lab2idx():
            return {
                'copper': 6,
                'livestock': 28,
                'gold': 25,
                'money-fx': 19,
                'ipi': 30,
                'trade': 11,
                'cocoa': 0,
                'iron-steel': 31,
                'reserves': 12,
                'tin': 26,
                'zinc': 37,
                'jobs': 34,
                'ship': 13,
                'cotton': 14,
                'alum': 23,
                'strategic-metal': 27,
                'lead': 45,
                'housing': 7,
                'meal-feed': 22,
                'gnp': 21,
                'sugar': 10,
                'rubber': 32,
                'dlr': 40,
                'veg-oil': 2,
                'interest': 20,
                'crude': 16,
                'coffee': 9,
                'wheat': 5,
                'carcass': 15,
                'lei': 35,
                'gas': 41,
                'nat-gas': 17,
                'oilseed': 24,
                'orange': 38,
                'heat': 33,
                'wpi': 43,
                'silver': 42,
                'cpi': 18,
                'earn': 3,
                'bop': 36,
                'money-supply': 8,
                'hog': 44,
                'acq': 4,
                'pet-chem': 39,
                'grain': 1,
                'retail': 29,
            }

        def mc_labels(x):
            return np.where(list(int(y) for y in x))[0]

        def ml_labels(x):
            return np.where(list(int(y) for y in x))[0]

        # 46 remaining after filtering freq(y) >= 5
        singletons = [
            0,
            1,
            3,
            4,
            6,
            9,
            10,
            13,
            15,
            17,
            19,
            21,
            22,
            23,
            24,
            25,
            26,
            29,
            31,
            32,
            33,
            34,
            35,
            36,
            38,
            40,
            41,
            43,
            44,
            45,
            46,
            47,
            49,
            54,
            55,
            59,
            66,
            67,
            69,
            71,
            77,
            78,
            83,
            84,
            85,
            87,
            88,
            89,
        ]

        if "multilabel" in filename:
            filepath = os.path.join(os.path.dirname(filename).split("_multilabel")[0], os.path.basename(filename))
        else:
            filepath = filename

        df = pd.read_csv(filepath, encoding="utf-8-sig", sep="\t")
        df = df.sample(frac=1, random_state=42)
        df.columns = ["label", "text"]

        df["label"] = df["label"].apply(lambda x: mc_labels(x))
        if not "multilabel" in filename:
            df = df.loc[df["label"].str.len() == 1].reset_index(drop=True)
            df["label"] = df["label"].apply(lambda x: x[0])
            df = df.loc[df["label"].isin(singletons)].reset_index(drop=True)
            data = [[df["text"].iloc[i], [df["label"].iloc[i]]] for i in range(len(df))]
        else:
            data = [[df["text"].iloc[i], df["label"].iloc[i]] for i in range(len(df))]

        return data

    def process_twitter(filename, identifier):
        if "prep" in identifier:
            filename = filename.split(".")[0] + "_prep.txt"

        df = pd.read_csv(filename, encoding="utf-8-sig", sep="\t")
        df.columns = ["id", "label", "text"]
        df = df.sample(frac=1, random_state=42)
        df = df[df["text"].str.len() > 5].reset_index(drop=True)

        if "binary" in identifier:
            df = df[df["label"] != "neutral"].reset_index(drop=True)
        return [[df["text"].iloc[i], [df["label"].iloc[i]]] for i in range(len(df))]

    def process_amazon_reviews(filename):
        if "_full" in filename and "test" in filename:
            directory = filename.split("_full")[0]
            directory = directory.replace("amazon_reviews-", "amazon_reviews/")
            df = pd.concat(
                tuple(
                    [
                        pd.read_csv(
                            os.path.join(directory, f + ".txt"), encoding="utf-8-sig", sep="\t", names=["label", "text"]
                        )
                        for f in ["train", "dev", "test"]
                    ]
                )
            )
            return [[df["text"].iloc[i], [df["label"].iloc[i]]] for i in range(len(df))]

        if "-" in filename:
            filename = filename.replace("-", "/")
        df = pd.read_csv(filename, encoding="utf-8-sig", sep="\t", names=["label", "text"])
        df = df.sample(frac=1, random_state=42)
        # use = []
        # domains = ["books","dvd","electronics","kitchen"]
        # for domain in domains:
        #     if domain in filename:
        #         use.append(domain)
        return [[df["text"].iloc[i], [df["label"].iloc[i]]] for i in range(len(df))]

    def process_webofscience(filename):
        df = pd.read_csv(filename, encoding="utf-8-sig", sep="\t")
        df = df.sample(frac=1, random_state=42)
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        return [[df["text"].iloc[i], [df["label"].iloc[i]]] for i in range(len(df))]

    def process_standard(filename, label="process_label"):
        def blender(subject, body, attachments):
            max_length = 500
            min_page_tokens = 50
            if not attachments:
                text = subject + "\n" + body[: max_length - len(subject) - 1]
                attachment_text = ""
            else:
                text = subject + "\n" + body[:250]
                sb = len(text)
                pages = attachments.split("\f")
                n_pages = len(pages)
                pagelens = [len(p) for p in pages]

                # how many pages we can at least take 50 tokens
                max_pages = sb % 50

                # then we only take for the max pages
                if n_pages > max_pages:
                    attachment_text = [p[:min_page_tokens] for p in pages[:max_pages]]
                else:
                    attachment_text = [p[: int(sb / n_pages) - n_pages] for p in pages]

                attachment_text = "\f".join(attachment_text)

            text += "\f" + attachment_text

            return text

        df = pd.read_csv(filename, encoding="utf-8-sig", sep="\t")
        df = df.sample(frac=1, random_state=42)
        df.fillna("", inplace=True)
        if "text" not in df.columns:
            df["text"] = df.apply(lambda row: blender(row["subject"], row["body"], row["attachment"]), axis=1)
            # df["text"] = df["subject"] + df["body"] #+df["attachment"]
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        return [[df["text"].iloc[i], [df[label].iloc[i]]] for i in range(len(df))]

    def process_text_csv(filename, column_map):  # column map follows the logic of "text" (X) and "label_XXX" (Y)
        if not column_map:
            column_map = {"label_process": 0, "text": 1}
        df = pd.read_csv(filename, encoding="utf-8-sig", engine="python", sep="\t")
        df = df.sample(frac=1, random_state=42)
        df = df.dropna(subset=list(column_map.keys())).reset_index(drop=True)  # shuffle the bastard
        df = df[df["text"].str.len() > 20].reset_index(drop=True)  # maybe this is dangerous?
        df = df[df.groupby("label_process")["label_process"].transform('size') > 2].reset_index(drop=True)
        label_column = [col for col in column_map if col.startswith("label")][0]
        return [
            [df["text"].iloc[i], [df[label_column].iloc[i]]] for i in range(len(df))
        ]  # tokenize text: document -> words

    def process_SST(filename):
        df = pd.read_csv(filename, encoding="utf-8-sig", sep="\t", names=["label", "text"])
        if "SST-2" in filename:
            df["text"] = df["label"].apply(lambda x: x[2:])
            df["label"] = df["label"].apply(lambda x: x[0])
        df = df.sample(frac=1, random_state=42)
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        df["label"] = df["label"].apply(lambda x: x.replace("__label__", "").strip()).astype(int)
        return [[df["text"].iloc[i], [df["label"].iloc[i]]] for i in range(len(df))]

    def process_huggingface(filename):
        from datasets import load_dataset

        dataset = "rotten_tomatoes" if "rotten_tomatoes" in filename else "yelp_polarity"
        split = infer_split(filename)
        if dataset == "yelp_polarity" and split == "validation":
            return []
        d = load_dataset(dataset, split=split)
        return list(zip(d["text"], [[l] for l in d["label"]]))

    """
    def process_amazon_movies(filename):
        d = load_dataset("amazon_us_reviews", 'Video_DVD_v1_00')
        len(d["train"]["review_body"])
        sum([len(x) for x in d["train"]["review_body"]])/len(d["train"]["review_body"])
    """

    def process_20news(directory):
        generators = {}

        # "ALTERNATE" The topics of the 20 news groups
        """
        categories = [
            "alt.atheism",
            "talk.politics.guns",
            "talk.politics.mideast",
            "talk.politics.misc",
            "talk.religion.misc",
            "soc.religion.christian",
            "comp.sys.ibm.pc.hardware",
            "comp.graphics",
            "comp.os.ms-windows.misc",
            "comp.sys.mac.hardware",
            "comp.windows.x",
            "rec.autos",
            "rec.motorcycles",
            "rec.sport.baseball",
            "rec.sport.hockey",
            "sci.crypt",
            "sci.electronics",
            "sci.space",
            "sci.med",
            "misc.forsale"]
        """

        # ood = ['talk.politics.guns', 'talk.politics.misc', 'soc.religion.christian', 'comp.graphics', 'comp.sys.mac.hardware', 'rec.autos', 'rec.sport.baseball', 'sci.crypt', 'sci.space', 'misc.forsale']

        def apply_cats(indices):
            return [[categories[i]] for i in indices]

        def load_data(filename, stop_words=[]):
            """Load the raw dataset."""

            # Differently from Hendrycks's code, we don't throw away stop words,

            x, y = [], []
            with open(filename, 'r') as f:
                for line in f:
                    line = re.sub(r'\W+', ' ', line).strip()
                    if line[1] == ' ':
                        x.append(line[1:])
                        y.append(line[0])
                    else:
                        x.append(line[2:])
                        y.append(line[:2])
                    x[-1] = ' '.join(word for word in x[-1].split())
            return x, np.array(y, dtype=int)

        categories = [
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'soc.religion.christian',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.politics.misc',
            'talk.religion.misc',
        ]

        # SKLEARN
        """
        #from sklearn.datasets import fetch_20newsgroups


        data_train = fetch_20newsgroups(
            subset='train', categories=categories, remove=('headers'), shuffle=True, random_state=42
        )
        data_test = fetch_20newsgroups(
            subset='test', categories=categories, remove=('headers'), shuffle=True, random_state=42
        )
        # X_train, X_dev, y_train, y_dev, train_indices, dev_indices = sklearn_split(
        #     data_train.data, data_train.target, 0.15, stratified=True
        # )
        # generators["test"] = list(zip(data_test.data, apply_cats(data_test.target)))
        """

        # open from Hendrycks; heavily preprocessed!
        train_X, train_y = load_data(os.path.join(directory, "20ng-train.txt"))
        X_test, y_test = load_data(os.path.join(directory, "20ng-test.txt"))

        X_train, X_dev, y_train, y_dev, train_indices, dev_indices = sklearn_split(
            train_X, train_y, 0.15, stratified=True
        )

        generators["test"] = list(zip(X_test, apply_cats(y_test)))
        generators["train"] = list(zip(X_train, apply_cats(y_train)))
        generators["dev"] = list(zip(X_dev, apply_cats(y_dev)))
        return generators["train"], generators["dev"], generators["test"]

    def process_clinc150(filename):  # separate set
        def ood_label(x):
            return x + 150

        import tensorflow_datasets as tfds

        labelsets = {}
        labelsets["train"], labelsets["dev"], labelsets["test"] = tfds.load(
            name="clinc_oos", as_supervised=True, split=["train", "validation", "test"]
        )  # , batch_size=-1
        labelsets["train"] = [(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(labelsets["train"])]
        labelsets["dev"] = [(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(labelsets["dev"])]
        labelsets["test"] = [(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(labelsets["test"])]
        # next(iter(data["test"].take(1)))
        # (<tf.Tensor: shape=(), dtype=string, numpy=b'do you know how i can change my insurance policy'>, <tf.Tensor: shape=(), dtype=int32, numpy=114>)
        if "ood" in filename:  # SHOULD ALSO LOAD oos sets via split
            train, dev, test = tfds.load(
                name="clinc_oos", as_supervised=True, split=["train_oos", "validation_oos", "test_oos"]
            )  # , batch_size=-1
            labelsets["train"].extend([(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(train)])  # 100
            labelsets["dev"].extend([(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(dev)])  # 100
            labelsets["test"].extend([(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(test)])  # 1000

        return labelsets["train"], labelsets["dev"], labelsets["test"]

    def process_agnews(filename):
        import tensorflow_datasets as tfds

        labelsets = {}
        labelsets["train"], labelsets["dev"], labelsets["test"] = tfds.load(
            name="ag_news_subset", as_supervised=True, split=["train", "validation", "test"]
        )
        labelsets["train"] = [(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(labelsets["train"])]
        labelsets["dev"] = [(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(labelsets["dev"])]
        labelsets["test"] = [(x.decode('utf-8'), [y]) for x, y in tfds.as_numpy(labelsets["test"])]
        return labelsets["train"], labelsets["dev"], labelsets["test"]

    def process_AAPD(filename, **kwargs):
        topic_num_map = {
            "cs.it": 0,
            "math.it": 1,
            "cs.lg": 2,
            "cs.ai": 3,
            "stat.ml": 4,
            "cs.ds": 5,
            "cs.si": 6,
            "cs.dm": 7,
            "physics.soc-ph": 8,
            "cs.lo": 9,
            "math.co": 10,
            "cs.cc": 11,
            "math.oc": 12,
            "cs.ni": 13,
            "cs.cv": 14,
            "cs.cl": 15,
            "cs.cr": 16,
            "cs.sy": 17,
            "cs.dc": 18,
            "cs.ne": 19,
            "cs.ir": 20,
            "quant-ph": 21,
            "cs.gt": 22,
            "cs.cy": 23,
            "cs.pl": 24,
            "cs.se": 25,
            "math.pr": 26,
            "cs.db": 27,
            "cs.cg": 28,
            "cs.na": 29,
            "cs.hc": 30,
            "math.na": 31,
            "cs.ce": 32,
            "cs.ma": 33,
            "cs.ro": 34,
            "cs.fl": 35,
            "math.st": 36,
            "stat.th": 37,
            "cs.dl": 38,
            "cmp-lg": 39,
            "cs.mm": 40,
            "cond-mat.stat-mech": 41,
            "cs.pf": 42,
            "math.lo": 43,
            "stat.ap": 44,
            "cs.ms": 45,
            "stat.me": 46,
            "cs.sc": 47,
            "cond-mat.dis-nn": 48,
            "q-bio.nc": 49,
            "physics.data-an": 50,
            "nlin.ao": 51,
            "q-bio.qm": 52,
            "math.nt": 53,
        }

        def mc_labels(x):
            # float(x) if single label
            # np.vectorize(dict(zip(topic_num_map.values(),topic_num_map.keys())).get)(X)
            return np.where(list(int(y) for y in x))[0]

        df = pd.read_csv(filename, encoding="utf-8-sig", engine="python", sep="\t")
        df = df.sample(frac=1, random_state=42)
        df.columns = ["label", "text"]
        df["label"] = df["label"].apply(lambda x: mc_labels(x))
        df = df[df["text"].str.len() > 20].reset_index(drop=True)
        data = [[df["text"].iloc[i], df["label"].iloc[i]] for i in range(len(df))]
        return data

    # DEV: could sort train & validation by length here to make more easily batches with less padding!
    def dynamic_sort(labelset):
        if len(set(len(x[0]) for x in labelset)) == 1:
            return sorted(labelset, key=lambda x: len(x[0][0]))
        return sorted(labelset, key=lambda x: len(x[0]))

    def generate_raw(composed):
        for word_ids, label_ids in composed:
            yield word_ids

    def generate_train(train, tensorize=False, max_sentences=0, max_document_len=0, additional_composition=False):
        if additional_composition:
            for tupled in train:
                X_ids, label_ids = tupled[:-1], tupled[-1]
                if len(additional_composition) == 1:
                    (word_ids, char_ids) = X_ids
                    X = {"word_ids": word_ids, "char_ids": char_ids}
                elif len(additional_composition) == 2:
                    (word_ids, char_ids, casing_ids) = X_ids
                    X = {"word_ids": word_ids, "char_ids": char_ids, "casing_ids": casing_ids}
                else:
                    raise NotImplementedError("Does not yet have support for more arbitrary features")
                yield X, label_ids
        else:
            for word_ids, label_ids in train:
                if not tensorize:
                    yield word_ids, label_ids
                else:
                    yield hierarchical_tensorize(
                        word_ids, max_sentences=max_sentences, max_document_len=max_document_len
                    ), label_ids

    def generate_dev(dev, tensorize=False, max_sentences=0, max_document_len=0, additional_composition=False):
        if additional_composition:
            for tupled in dev:
                X_ids, label_ids = tupled[:-1], tupled[-1]
                if len(additional_composition) == 1:
                    (word_ids, char_ids) = X_ids
                    X = {"word_ids": word_ids, "char_ids": char_ids}
                elif len(additional_composition) == 2:
                    (word_ids, char_ids, casing_ids) = X_ids
                    X = {"word_ids": word_ids, "char_ids": char_ids, "casing_ids": casing_ids}
                else:
                    raise NotImplementedError("Does not yet have support for more arbitrary features")
                yield X, label_ids
        else:
            for word_ids, label_ids in dev:
                if not tensorize:
                    yield word_ids, label_ids
                else:
                    yield hierarchical_tensorize(
                        word_ids, max_sentences=max_sentences, max_document_len=max_document_len
                    ), label_ids

    def generate_test(test, tensorize=False, max_sentences=0, max_document_len=0, additional_composition=False):
        if additional_composition:
            for tupled in test:
                X_ids, label_ids = tupled[:-1], tupled[-1]
                if len(additional_composition) == 1:
                    (word_ids, char_ids) = X_ids
                    X = {"word_ids": word_ids, "char_ids": char_ids}
                elif len(additional_composition) == 2:
                    (word_ids, char_ids, casing_ids) = X_ids
                    X = {"word_ids": word_ids, "char_ids": char_ids, "casing_ids": casing_ids}
                else:
                    raise NotImplementedError("Does not yet have support for more arbitrary features")
                yield X, label_ids
        else:
            for word_ids, label_ids in test:
                if not tensorize:
                    yield word_ids, label_ids
                else:
                    yield hierarchical_tensorize(
                        word_ids, max_sentences=max_sentences, max_document_len=max_document_len
                    ), label_ids

    labelsets = {k: [] for k in sets}

    """
    LOAD THE DATA
    """
    for labelset in labelsets:
        if "conll" in directory:

            """
            IO format
            """
            # labelsets[labelset] = process_CONLL(os.path.join(directory, labelset + ".txt"))

            """
            BIOSE format
            """
            labelsets["train"], labelsets["dev"], labelsets["test"] = process_NER_flair(directory)
            break

            """
            IOB format: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
            """
            labelsets["train"], labelsets["test"], labelsets["dev"] = process_huggingface_tokenclf(directory)

        elif (
            "WNUT_17" in directory
            or "ontonotes" in directory
            or "ACE2005" in directory
            or "SROIE" in directory
            or "ACL" in directory
        ):
            labelsets["train"], labelsets["dev"], labelsets["test"] = process_NER_flair(directory)
            break

        elif "imdb50k" in directory:
            import tensorflow_datasets as tfds

            labelsets["train"], labelsets["dev"], labelsets["test"] = tfds.load(
                name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True
            )
            break

        elif "rotten_tomatoes" in directory or "yelp_polarity" in directory:
            labelsets[labelset] = process_huggingface(os.path.join(directory, labelset + ".txt"))

        elif "imdb" in directory or "yelp" in directory:
            labelsets[labelset] = process_IMDB_YELP(os.path.join(directory, labelset + ".txt"))

        elif "Reuters" in directory:
            labelsets[labelset] = process_reuters(os.path.join(directory, labelset + ".tsv"))

        elif "twitter" in directory:
            identifier = directory.split("/")[-1]
            if "_" in directory:
                cwd = directory.split("_")[0]
            labelsets[labelset] = process_twitter(os.path.join(cwd, labelset + ".txt"), identifier)

        elif "WOS-46985" in directory:
            labelsets[labelset] = process_webofscience(os.path.join(directory, labelset + ".csv"))

        elif "20news" in directory:
            labelsets["train"], labelsets["dev"], labelsets["test"] = process_20news(directory)
            break

        elif "CLINC" in directory:
            labelsets["train"], labelsets["dev"], labelsets["test"] = process_clinc150(directory)
            labelsets = {k: v for k, v in labelsets.items() if k in sets}
            break

        elif "AGNews" in directory:
            labelsets["train"], labelsets["dev"], labelsets["test"] = process_agnews(directory)
            break

        elif "AAPD" in directory:
            labelsets[labelset] = process_AAPD(os.path.join(directory, labelset + ".tsv"))

        elif "SST" in directory:
            labelsets[labelset] = process_SST(os.path.join(directory, labelset + ".txt"))

        elif "amazon_reviews" in directory:
            labelsets[labelset] = process_amazon_reviews(os.path.join(directory, labelset + ".txt"))
        else:
            raise NotImplementedError

    if downsampling:
        for labelset in labelsets:
            if labelset == "test":
                continue
            labelsets[labelset] = random.sample(
                labelsets[labelset], min(len(labelsets[labelset]), int(len(labelsets[labelset]) * downsampling))
            )
        # kick out before even creating vocabulary! [operate on same label arrays]
        # labels = list(zip(*labelsets[labelset]))[1]
    """
    BUILD COMPOSITIONALITY
    """
    if not isinstance(labelsets[sets[0]][0][0], str):
        logging.warning("Data has been tokenized already to predefined composition: assume word/token-level")
        if debug:
            text_stats(
                [" ".join(text) for labelset, documents in labelsets.items() for text, label in documents],
                token_pattern,
            )

    else:
        if debug:
            text_stats([text for labelset, documents in labelsets.items() for text, label in documents], token_pattern)
            # DEV: label_stats
        labelsets = prepare_composition(labelsets, composition=composition, token_pattern=token_pattern, ood=ood)

    if raw:
        for labelset in labelsets:
            if sequence_labels:
                continue
            labelsets[labelset] = list(
                generate_train(labelsets[labelset])
            )  # only works for test where shuffle is not applied!
            labels = np.array(list(zip(*labelsets[labelset]))[1]).flatten()
            if any(isinstance(x, np.ndarray) for x in labels):
                labels = np.hstack(labels)
            print("imbalance_ratio : ", imbalance_ratio(labels))
        return labelsets, None, None, None

    """
    CREATE VOCABULARY AND LABEL SCHEME
    """
    input_dtype = tf.int32
    if not voc2idx and not label2idx:
        # construct word/character vocabulary ; unique tokens, labels; should be able to handle composition
        voc2idx, label2idx = generate_vocabulary_label2idx(
            labelsets, lowercase, min_token_len, min_token_freq, max_vocabulary, pretrained_embeddings, debug=debug
        )  # make voc on within sentence :/
        if pretrained_embeddings:
            from arkham.Bayes.Quantify.load_pretrained import build_embeddings

            if "tfhub" in pretrained_embeddings:
                input_dtype = tf.string
            pretrained_embeddings, voc2idx = build_embeddings(
                pretrained_embeddings, max_vocabulary, max_document_len, voc2idx, overblend_vocabulary
            )  # returns compiled tf.keras layer and new voc

    """
    ENCODE TEXT AND LABELS with vocabulary (given pre-trained)
    """
    additional_composition = [k for k in composition if k not in ["word", "sentence"]]
    tokenizer = get_tokenizer(model_class)
    encoded = {}
    for labelset in labelsets:
        if tokenizer:
            from arkham.Bayes.Quantify.BERT import tokenize

            if sequence_labels and labelset != "test":
                labelsets[labelset] = sorted(labelsets[labelset], key=lambda x: len(x[0]))

            documents, labels = list(zip(*labelsets[labelset]))
            if not sequence_labels:
                documents = np.array([" ".join(document) for document in documents])

            X, new_y = tokenize(
                documents, tokenizer, max_document_len=max_document_len, sequence_labels=sequence_labels, labels=labels
            )
            if sequence_labels:
                y = np.array([encode_labels(label, label2idx, sequence_labels=sequence_labels) for label in new_y])
            else:
                y = np.array([encode_labels(label, label2idx) for label in labels])
            encoded[labelset] = tf.data.Dataset.from_tensor_slices((X, y)).padded_batch(batch_size)  # list(zip(X, y))
            """
            def example_to_features(input_ids,attention_masks,token_type_ids,y):
                return {"input_ids": input_ids,
                      "attention_mask": attention_masks,
                      "token_type_ids": token_type_ids},y
            """

        else:
            encoded[labelset], additional_inputs = encode_idx(
                labelsets[labelset],
                voc2idx,
                label2idx,
                oov=voc2idx["PAD"],
                lowercase=lowercase,
                max_document_len=max_document_len,
                max_sentences=max_sentences,
                max_chars_len=max_chars_len,
                oov_corruption=oov_corruption,
                composition=composition,
                input_dtype=input_dtype,
                sequence_labels=sequence_labels,
                skip_short=True if "CLINC" in directory else False,
            )
            print("LABELSET: ", labelset, ": ", len(encoded[labelset]))
            print("AVG corruption: ", np.mean([calc_oov(word_ids) for word_ids, label_ids in encoded[labelset]]))

            if additional_inputs:
                """
                these should be concatted as an array to X before any sorting 
                tuple of 2 2D arrays with same shape[0]
                """
                added = []
                for i in range(len(encoded[labelset])):
                    new_tuple = (
                        encoded[labelset][i][0],
                        *tuple(additional_inputs[k][i] for k in additional_composition),
                        encoded[labelset][i][1],
                    )
                    added.append(new_tuple)
                encoded[labelset] = added

            if not sort_by_seqlen or labelset == "test":  # or sequence_labels:
                continue
            if "train" in encoded:
                if len(encoded["train"]) > 60000:
                    continue
            encoded[labelset] = dynamic_sort(encoded[labelset])  # SHOULD NOT DO WHEN TAKING STEPSPEREPOCH

    """
    GENERATOR OBJECTS, BATCHING & PADDING
    """
    # https://github.com/tensorflow/tensorflow/issues/34793 -> pad ragged tensor
    # https://github.com/tensorflow/tensorflow/issues/39163 -> how about converting ragged to tensors
    # https://hanxiao.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/
    generators = {}
    generator_fxs = {"train": generate_train, "dev": generate_dev, "test": generate_test}
    if tokenizer:
        for labelset in labelsets:
            if labelset != "test":
                tf.random.set_seed(42)
                generators[labelset] = (
                    encoded[labelset]
                    .shuffle(buffer_size, reshuffle_each_iteration=True)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                )
            else:
                generators[labelset] = encoded[labelset].prefetch(tf.data.experimental.AUTOTUNE)
        return generators, voc2idx, label2idx, pretrained_embeddings

    sequence_labels = len(label2idx) if sequence_labels else None

    if additional_composition:
        if len(additional_composition) == 1:
            output_types = ({"word_ids": tf.int32, "char_ids": tf.int32}, tf.int32)
            output_shapes = ({"word_ids": None, "char_ids": (None, max_chars_len)}, None)
            padded_shapes = ({"word_ids": [None], "char_ids": [None, max_chars_len]}, [None])
        elif len(additional_composition) == 2:
            output_types = ({"word_ids": tf.int32, "char_ids": tf.int32, "casing_ids": tf.int32}, tf.int32)
            output_shapes = ({"word_ids": None, "char_ids": (None, max_chars_len), "casing_ids": None}, None)
            padded_shapes = ({"word_ids": [None], "char_ids": [None, max_chars_len], "casing_ids": [None]}, [None])
        else:
            raise NotImplementedError("Does not yet have support for more arbitrary features")

    if "train" in sets:
        if not additional_composition:
            generators["train"] = batch_shuffle_generator(
                tf.data.Dataset.from_generator(
                    lambda: generator_fxs["train"](
                        encoded["train"],
                        "sentence" in composition,
                        max_sentences=max_sentences,
                        max_document_len=max_document_len,
                    ),
                    output_types=(input_dtype, tf.int32),
                ),
                shuffle=False if sequence_labels else True,
                buffer_size=buffer_size,
                batch_size=batch_size,
                max_document_len=max_document_len,
                max_sentences=max_sentences,
                sequence_labels=sequence_labels,
                composition=composition,
                input_dtype=input_dtype,
            )
        else:
            series = tf.data.Dataset.from_generator(
                lambda: generator_fxs["train"](encoded["train"], additional_composition=additional_composition),
                output_types=output_types,
                output_shapes=output_shapes,
            ).padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
            tf.random.set_seed(42)
            generators["train"] = series.shuffle(buffer_size, reshuffle_each_iteration=True).prefetch(
                tf.data.experimental.AUTOTUNE
            )
    if "dev" in sets:
        if not additional_composition:
            generators["dev"] = batch_shuffle_generator(
                tf.data.Dataset.from_generator(
                    lambda: generator_fxs["dev"](
                        encoded["dev"],
                        "sentence" in composition,
                        max_sentences=max_sentences,
                        max_document_len=max_document_len,
                    ),
                    output_types=(input_dtype, tf.int32),
                ),
                shuffle=False if sequence_labels else True,
                buffer_size=buffer_size,
                batch_size=batch_size,
                max_document_len=max_document_len,
                max_sentences=max_sentences,
                sequence_labels=sequence_labels,
                composition=composition,
                input_dtype=input_dtype,
            )
        else:
            series = tf.data.Dataset.from_generator(
                lambda: generator_fxs["dev"](encoded["dev"], additional_composition=additional_composition),
                output_types=output_types,
                output_shapes=output_shapes,
            ).padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
            tf.random.set_seed(42)
            generators["dev"] = series.shuffle(buffer_size, reshuffle_each_iteration=True).prefetch(
                tf.data.experimental.AUTOTUNE
            )
    if "test" in sets:
        if not additional_composition:
            generators["test"] = batch_shuffle_generator(
                tf.data.Dataset.from_generator(
                    lambda: generator_fxs["test"](
                        encoded["test"],
                        "sentence" in composition,
                        max_sentences=max_sentences,
                        max_document_len=max_document_len,
                    ),
                    output_types=(input_dtype, tf.int32),
                ),
                shuffle=False,
                buffer_size=buffer_size,
                batch_size=batch_size,
                max_document_len=max_document_len,
                max_sentences=max_sentences,
                sequence_labels=sequence_labels,
                composition=composition,
                input_dtype=input_dtype,
            )
        else:
            series = tf.data.Dataset.from_generator(
                lambda: generator_fxs["test"](encoded["test"], additional_composition=additional_composition),
                output_types=output_types,
                output_shapes=output_shapes,
            ).padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
            tf.random.set_seed(42)
            generators["test"] = series.prefetch(tf.data.experimental.AUTOTUNE)

    if debug:
        debug_generators(generators)

    return generators, voc2idx, label2idx, pretrained_embeddings

def debug_generators(labelsets):
    for labelset, dataset in labelsets.items():
        print("LABELSET: ", labelset)
        try:
            print(next(iter(dataset)))
        except Exception as e:
            raise e
        dataset_length = [i for i, _ in enumerate(dataset)][-1] + 1
        print(dataset_length)
