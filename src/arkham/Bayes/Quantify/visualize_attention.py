#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import regex as re
import pandas as pd

from tqdm import tqdm

# from arkham.utils.utils import timer, pickle_dumper
from arkham.utils.model_utils import MODELROOT, _get_weights, load_model_path
from arkham.Bayes.Quantify.data import generators_from_directory, encode_test
from arkham.Bayes.Quantify.evaluate import decode_x_and_y
from arkham.Bayes.Quantify.HAN import visualize_word_attentions
from tensorflow.python.keras import backend


def main(version, test_data=None, batch_size=2, dump=True, **kwargs):
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    model, _config = load_model_path(modelpath)
    _config["data_folder"] = re.sub(r'/home/[^/]+', os.path.expanduser("~"), _config["data_folder"])
    _config["out_folder"] = modelpath

    if test_data:
        tokenized, encoded = encode_test(test_data, _config)
        batch_texts, batch_golds, _ = decode_x_and_y(
            encoded.numpy(), [], _config["idx2voc"], _config["idx2label"], sequence_labels=False
        )
        visualize_word_attentions(model, _config, encoded, batch_texts, golds=None, predictions=None)

    _config["batch_size"] = 1
    generators, _, _, _ = generators_from_directory(_config["data_folder"], sets=["test"], **_config)
    for batch in iter(generators["test"]):  # tuple of X,y
        batch_texts, batch_golds, _ = decode_x_and_y(
            batch[0].numpy(), batch[1].numpy(), _config["idx2voc"], _config["idx2label"], sequence_labels=False
        )
        visualize_word_attentions(model, _config, batch[0], batch_texts, golds=batch_golds, predictions=None)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("""Visualize attention on test data or a guided sample of test data""")
    parser.add_argument("version", type=str, default="")
    parser.add_argument("test_data", nargs="?", type=str, default=None)
    parser.add_argument("-s", dest="batch_size", type=int, default=2, help="batchsize")
    args = parser.parse_args()
    main(args.version, args.test_data, args.batch_size, dump=True)
