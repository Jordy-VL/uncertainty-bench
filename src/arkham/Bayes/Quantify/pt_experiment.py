#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://github.com/pytorch/tutorials/blob/master/beginner_source/text_sentiment_ngrams_tutorial.py
"""

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2021 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"

import os
import sys
import pandas as pd
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver

from arkham.Bayes.Quantify.data import (
    generators_from_directory,
    tokenize_composition,
    encode_text,
    generate_vocabulary_label2idx,
)

from arkham.utils.utils import generate_out_folder, remove_progress, generate_experiment_folder
from arkham.utils.callbacks import MemoryCallback, ParamsCallback

"""
from arkham.utils.model_utils import (
    multiclass_argmaxprobs,
    get_optimizers,
    _get_weights,
    config_to_json,
    dump_embeddings,
    RelaxedLayer,
)
from arkham.utils.losses import (
    deduce_loss,
    sparse_categorical_accuracy_masked
)
"""
from arkham.Bayes.Quantify.evaluate import main as model_evaluate
from arkham.Bayes.Quantify.experiment import ex, DATAROOT, DEFAULT_EXPERIMENT

import wandb
from wandb.keras import WandbCallback


@ex.named_config
def clf_pytorch():
    identifier = "CLINC-150"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "document_classification"  # or regression

    downsampling = 0.1
    buffer_size = 1000
    batch_size = 32

    token_pattern = r"\b\w\S+\b"
    lowercase = False
    min_token_len = 2
    min_token_freq = 2
    max_vocabulary = 20000
    max_document_len = None  # from code derived: 200 tokens
    max_sentences = None
    composition = ["word"]

    pretrained_embeddings = None
    raw = True
    model = "TextClassificationCNN_simple"


@ex.automain
def run(_config, _run, _seed):
    '''
    1. Create datasets as batchgenerators following specification in _config
    '''

    experiment_name = _run.experiment_info["name"]
    # wandb.init(project="muppets", config=_config, name=experiment_name if experiment_name != "quantify_textCNN" else None)

    generators, _, _, _ = generators_from_directory(_config["data_folder"], **_config)

    voc2idx, label2idx = generate_vocabulary_label2idx(
        generators,
        _config["lowercase"],
        _config["min_token_len"],
        _config["min_token_freq"],
        _config["max_vocabulary"],
        pretrained_embeddings=None,
        debug=False,
    )

    vocab_size = len(voc2idx)
    nb_classes = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    idx2voc = {v: k for k, v in voc2idx.items()}

    '''
    2. Build model structure and hyperparametrization following specification in _config
    '''

    ## Magic
    # wandb.watch(model)

    '''
    3. Compile model with optimizer, loss, callbacks and metrics
    '''

    os.mkdir(os.path.join(_config["out_folder"]))
    params = ParamsCallback(
        _config,
        idx2label,
        idx2voc,
        {
            "command": "python3 "
            + os.path.dirname(os.path.abspath(__file__))
            + "/experiment.py "
            + " ".join(sys.argv)[" ".join(sys.argv).find("with") :]
        },
    )
    params.on_epoch_end(epoch=0)
    # MemoryCallback()

    '''
    4. Train model with all above prepared args
    '''

    '''
    5. Save binaries if observers activated
    '''

    '''
    6. Evaluate model on testset
    '''

    """
    7. Cleaning
    """
    if experiment_name != DEFAULT_EXPERIMENT:
        generate_experiment_folder(_config["out_folder"], experiment_name)
