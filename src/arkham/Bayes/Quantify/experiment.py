#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import numpy as np
import shutil
from collections import Iterable

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
import gc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, LearningRateScheduler

# All internal imports
from arkham.utils.utils import generate_experiment_folder
from arkham.Bayes.Quantify.data import (
    generators_from_directory,
    tokenize_composition,
    encode_text,
    generate_char2idx_idx2char,
)

from arkham.Bayes.Quantify.models import model_descriptors
from arkham.utils.losses import deduce_loss, sparse_categorical_accuracy_masked
from arkham.utils.model_utils import (
    multiclass_argmaxprobs,
    get_optimizers,
    _get_weights,
    config_to_json,
    dump_embeddings,
    RelaxedLayer,
)
from arkham.utils.callbacks import (
    ChunkF1,
    Heteroscedastic_MSE,
    Heteroscedastic_Acc,
    UoF,
    macro_f1,
    LogEpochMetrics,
    UnopenedFieldCallback,
    MemoryCallback,
    ParamsCallback,
    CustomCheckpoint,
)
from arkham.utils.calibration_metrics import ExpectedCalibrationError

from arkham.Bayes.Quantify.evaluate import main as model_evaluate
from arkham.Bayes.Quantify.compare import combine_ensemble

# New SGMCMC methods
from arkham.Bayes.MCMC.sgmcmc import cSGLD, burnout_epochs, Cyclic_Checkpoint, get_lr_metric
from arkham.Bayes.MCMC.sghmc import cSGHMC

# Experimentation imports
from sacred.observers import MongoObserver, SlackObserver
import wandb
from wandb.keras import WandbCallback
from arkham.default_config import MODELROOT, SAVEROOT, DATAROOT
from arkham.Bayes.Quantify.configs import DEFAULT_EXPERIMENT, ex


def set_observers():
    if not "-n" in sys.argv:
        return
    if "mini_imdb" in sys.argv or "epochs=2 " in sys.argv:
        return
    """Set observers for Sacred writing out experiment data"""
    GORDON = os.path.expanduser("~/code/gordon/gordon")
    if os.path.exists(GORDON):
        sys.path.append(GORDON)
        from default_config import get_connection_string, MONGODB_SETTINGS, DEFAULT_WEBHOOK

        # ex.observers.append(MongoObserver(get_connection_string(MONGODB_SETTINGS), db_name="main"))
        ex.observers.append(SlackObserver(DEFAULT_WEBHOOK))


set_observers()


def init_wandb(_config, experiment_name):
    PROJECT = "muppets" if _config["sequence_labels"] else "bayes-bench"
    NAME = experiment_name if experiment_name != "quantify_textCNN" else None
    wandb.init(project=PROJECT, config=_config, name=NAME, entity="jordy-vlan")


def log_metrics(_run, evaluation_data, extra_metrics):  # all between 0 and 1?
    sys.path.append(os.path.expanduser("~/code/gordon/gordon/utils"))
    from metrics import evaluation

    evaluated = evaluation(evaluation_data)

    for field, values in evaluated["fields"].items():
        for metric, value in values.items():
            metricname = ".".join([field, metric])
            if not isinstance(value, Iterable):
                _run.log_scalar(metricname, value)
            elif isinstance(value, list):  # stepwise metric
                if isinstance(value[0], dict):
                    for key in value[0]:
                        vs = [v[key] for v in value]
                        _run.log_scalar(".".join([field, metric, str(key)]), vs)
                else:
                    _run.log_scalar(metricname, value)
            else:
                print("Unsupported: ", metricname)
    for metric, value in extra_metrics.items():
        _run.log_scalar(metric, value)
    return evaluated


def metrics_monitor_mode(metric_config, idx2label, T=10):
    metrics = []
    monitor = "val_loss"
    mode = "min"
    # metrics.append(UoF(max_fp=0.05))

    if "accuracy" in metric_config:
        metrics.append("accuracy")

    if "sparse_categorical_accuracy_masked" in metric_config:
        metrics.append(sparse_categorical_accuracy_masked)
    if "chunk_f1" in metric_config:
        metrics.append(ChunkF1(idx2label, average="weighted"))
        # monitor = "val_chunk_f1"
        # mode = "max"

    if "ECE" in metric_config:
        metrics.append(ExpectedCalibrationError(num_bins=30))

    elif "heteroscedastic_acc" in metric_config:
        metrics.append(Heteroscedastic_Acc(T=T))
        monitor = "val_heteroscedastic_acc"
        mode = "max"

    elif "heteroscedastic_mse" in metric_config:
        metrics.append(Heteroscedastic_MSE(T=T))
        monitor = "val_heteroscedastic_mse"
        mode = "min"

    elif "macro_f1" in metric_config:
        metrics.append(macro_f1)
        monitor = "val_macro_f1"
        mode = "max"
    else:
        if not metrics:
            metrics.extend(metric_config)

    return metrics, monitor, mode


def eval_stats(evaluator, model):  # could fix this with a return statement
    from prettytable import PrettyTable
    from arkham.Bayes.Quantify.compare import evaluate_model, metric_names

    t = PrettyTable(["version"] + metric_names)
    for identifier, stats in evaluator.stats.items():
        print(f"** {identifier} **")
        metrics, k = evaluate_model(model, identifier, stats, evaluator)
        if identifier == "nonbayesian":
            for metric, value in k.items():
                if metric in ["Acc", "MSE(↓)", "F1(m)", "F1(M)", "NLL(↓)", "ECE(↓)", "Brier(↓)", "AUC", "MU(μ)"]:
                    if value == 0:
                        continue
                    try:
                        wandb.log({metric: value})
                    except Exception as e:
                        pass # You probably did not init wandb
        t.add_row([k["version"]] + metrics)
    print(t)
    return t


@ex.automain
def run(_config, _run, _seed):
    print(_config)
    experiment_name = _run.experiment_info["name"]
    if experiment_name != DEFAULT_EXPERIMENT:
        init_wandb(_config, experiment_name)
    out = ""
    '''
    1. Create datasets as batchgenerators following specification in _config
    '''
    if _config.get("ood"):
        assert isinstance(_config.get("ood"), int) or isinstance(_config.get("ood"), list)
    generators, voc2idx, label2idx, embedding_layer = generators_from_directory(_config["data_folder"], **_config)

    vocab_size = len(voc2idx)
    nb_classes = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    idx2voc = {v: k for k, v in voc2idx.items()}

    '''
    2. Build model structure and hyperparametrization following specification in _config
    '''
    for m in range(_config["ensemble"]):
        evaluator = None
        model_identifier = "M" + str(m) + "_" if _config["ensemble"] > 1 else ""
        model = model_descriptors(
            _config, vocab_size, nb_classes, embedding_layer=embedding_layer, calibration=_config["calibration"]
        )
        model.summary()

        metrics, monitor, mode = metrics_monitor_mode(_config["metrics"], idx2label, T=_config["posterior_sampling"])

        callbacks = []
        if m == 0:
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
            tensorboard = TensorBoard(
                log_dir=os.path.join(_config["out_folder"], "TensorBoard"),
                histogram_freq=1 if not _config["sequence_labels"] else 0,
                write_graph=False,
                write_images=False,
                profile_batch=0,
            )
            callbacks.append(tensorboard)  # responsible for creating model directory
            callbacks.append(params)  # responsible for saving model params in first epoch

        savepath = os.path.join(_config["out_folder"], model_identifier + "weights_{epoch:02d}-{val_loss:.5f}.hdf5")
        save_format = "hdf5"
        checkpointer = ModelCheckpoint
        if _config["use_gp_layer"]:
            save_format = "tf"
            savepath = os.path.join(_config["out_folder"], model_identifier + "weights_{epoch:02d}-{val_loss:.5f}")
            checkpointer = CustomCheckpoint

        if _config["cycles"] == 1:
            callbacks.extend(
                [
                    EarlyStopping(
                        monitor=monitor,
                        patience=int(_config["epochs"] / 3) if _config["epochs"] > 10 else _config["epochs"] - 1,
                        mode=mode,
                        restore_best_weights=False,
                    ),
                    checkpointer(
                        filepath=savepath,
                        save_format=save_format,
                        monitor=monitor,
                        save_best_only=True,
                        mode=mode,
                        verbose=1,
                        save_freq="epoch",
                    ),
                    MemoryCallback()
                    # , tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=2, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-6),
                    #  UnopenedFieldCallback(validation_data=generators["dev"], max_fp=0.05)
                ]
            )
        if "BERT" in _config["model"]:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    factor=0.5,
                    patience=2,
                    verbose=0,
                    mode='auto',
                    min_delta=0.001,
                    cooldown=0,
                    min_lr=1e-6,
                )
            )
        if experiment_name != DEFAULT_EXPERIMENT:
            callbacks.append(WandbCallback(save_model=False))

        '''
        3. Compile model with optimizer, loss, callbacks and metrics
        '''
        if _config["optimizer"] == "cSGLD":
            burnout = burnout_epochs(_config["epochs"], _config["cycles"], _config["posterior_sampling"])
            dynamic_callback = Cyclic_Checkpoint(filepath=savepath, burnout=burnout)
            callbacks.append(dynamic_callback)
            # callbacks.append(LR_Callback())

            steps_per_epoch = (
                _config["steps_per_epoch"]
                if _config["steps_per_epoch"]
                else [i for i, _ in enumerate(generators["train"])][-1] + 1
            )
            total_iterations = _config["epochs"] * steps_per_epoch
            noise_burnin = int(
                0.9 * (total_iterations // _config["cycles"])
            )  # 10% of steps per cycle spent on exploration

            lr_decayed_fn = tf.keras.experimental.CosineDecayRestarts(
                _config["learning_rate"],
                first_decay_steps=noise_burnin,
                t_mul=1.0,  # _config["cycles"],
                m_mul=1.0,  # 0.5?
                alpha=0.0,
            )
            if _config["weight_decay"]:
                from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay

                cSGLDW = extend_with_decoupled_weight_decay(cSGLD)
                optimizer = cSGLDW(
                    learning_rate=lr_decayed_fn,
                    momentum=1 - _config["alpha"],
                    data_size=1,
                    burnin=noise_burnin,
                    weight_decay=_config["weight_decay"],
                )
            else:

                optimizer = cSGLD(
                    learning_rate=lr_decayed_fn, momentum=1 - _config["alpha"], data_size=1, burnin=noise_burnin
                )

            metrics.append(get_lr_metric(optimizer))
        else:
            optimizer = get_optimizers(
                _config["optimizer"],
                _config["learning_rate"],
                weight_decay=_config["weight_decay"],
                clipnorm=_config["clipnorm"],
                kwargs={}
                if _config["optimizer"] != "RectifiedAdam"
                else {
                    "steps": [i for i, _ in enumerate(generators["train"].unbatch())][-1] + 1,
                    "epochs": _config["epochs"],
                    "batch_size": _config["batch_size"],
                },
            )

        model.compile(
            optimizer=optimizer,
            loss=deduce_loss(
                _config["loss_fn"],
                nb_classes,
                _config["multilabel"],
                _config["use_aleatorics"],
                gamma=_config["gamma"],
                label_smoothing=_config["label_smoothing"],
            ),
            metrics=metrics,
        )

        """
        after compile checks
        """
        if _config["dropout_concrete"]:
            for layer in model.layers:
                if "concrete" in layer.name:
                    assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob

        traingens = (
            generators["train"].repeat(_config["epochs"])
            if _config["steps_per_epoch"] and _config["batch_size"] >= 8
            else generators["train"]
        )
        devgens = generators["dev"].repeat(_config["epochs"])
        steps_per_epoch = _config["steps_per_epoch"] if _config["batch_size"] >= 8 else None
        validation_steps = int(_config["steps_per_epoch"] / 5) if _config["steps_per_epoch"] else None

        '''
        4. Train model with all above prepared args
        '''
        history = None
        try:
            history = model.fit(
                traingens,
                epochs=_config["epochs"],
                validation_data=devgens,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
            )
        except Exception as e:
            print(e)
            print("stopped training")
        try:
            loss = history.history['loss']
            print(loss)
            print(history.history)
        except Exception as e:
            print(e)

        '''
        5-6. Evaluate model on testset
        '''
        if _config["optimizer"] == "cSGLD":
            from arkham.Bayes.Quantify.evaluate_ensemble import main as ensemble_evaluate

            evaluate_method = ensemble_evaluate
        else:
            evaluate_method = model_evaluate

        try:
            evaluator = evaluate_method(
                _config["out_folder"],
                test_data=None,
                evaluation_data=generators["test"],
                downsampling=0,
                dump=True,
                identifier=model_identifier,
            )
        except Exception as e:
            print(e)

        evaluation_data, extra_metrics = None, {}

        try:
            add_out = eval_stats(evaluator, experiment_name)
            out = str(out) + "\n" + str(add_out)
        except Exception as e:
            print(e)

        tf.keras.backend.clear_session()  # important to clear memory from previous run!; https://github.com/keras-team/keras/issues/2102
        del model  # https://stackoverflow.com/questions/58453793/the-clear-session-method-of-keras-backend-does-not-clean-up-the-fitting-data
        del evaluator
        gc.collect()

    '''
    7. Save artefacts if observers activated
    '''
    if len(ex.observers) > 1:
        _run.add_artifact(os.path.join(_config["out_folder"], "params.json"))
        _run.add_artifact(_get_weights(_config["out_folder"]))
        if evaluation_data:
            out = log_metrics(_run, evaluation_data, extra_metrics)

    # try:
    #     dump_embeddings(model, idx2voc, _config["out_folder"])
    # except Exception as e:
    #     print(e)

    if _config["ensemble"] > 1:
        combine_ensemble(_config["out_folder"])

    experiment_name = experiment_name if experiment_name != DEFAULT_EXPERIMENT else ""
    new_experiment_path = generate_experiment_folder(_config["out_folder"], experiment_name, return_path=True)
    print("Moving to ", new_experiment_path)
    try:
        shutil.move(_config["out_folder"], new_experiment_path)
    except Exception as e:
        pass

    return out
