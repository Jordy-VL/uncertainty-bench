#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import json
from tensorflow.python.lib.io.file_io import FileIO
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import AdamW, RectifiedAdam

from arkham.default_config import MODELROOT, SAVEROOT, DATAROOT

from arkham.utils.callbacks import ChunkF1, Heteroscedastic_Acc, Heteroscedastic_MSE
from arkham.utils.regularization import (
    EmbeddingDropout,
    ConcreteDropout,
    SpatialConcreteDropout,
    SpatialConcreteDropout2D,
)
from arkham.Bayes.Quantify.HAN import HierarchicalAttention
from arkham.Bayes.Quantify.transformer import TokenAndPositionEmbedding, TransformerBlock
from arkham.utils.losses import deduce_loss


def multiclass_argmaxprobs(output_probabilities):
    if len(output_probabilities.shape) < 2:
        output_probabilities = output_probabilities.reshape(output_probabilities.shape[0], 1)
    a_argmax = np.expand_dims(np.argmax(output_probabilities, axis=1), axis=1)
    return np.ravel(np.take_along_axis(output_probabilities, a_argmax, 1))


def get_optimizers(optimizer, learning_rate, weight_decay=None, clipnorm=None, kwargs=None):
    if optimizer == "RectifiedAdam":
        decay_steps = int(kwargs["epochs"] * kwargs["steps"] / kwargs["batch_size"])  # bsz
        warmup_steps = int(0.1 * decay_steps)
        return RectifiedAdam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            weight_decay=weight_decay,
            amsgrad=False,
            sma_threshold=5.0,
            total_steps=decay_steps,
            warmup_proportion=0.1,
            min_lr=1e-9,
            name='RectifiedAdam',
            clipnorm=clipnorm,
        )

    elif weight_decay:
        return AdamW(lr=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
    else:

        optimizer_map = {e.lower(): e for e in ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax"]}
        optimize_fn = getattr(optimizers, optimizer_map[optimizer])(lr=learning_rate, clipnorm=clipnorm)
    return optimize_fn


def _get_weights(out_folder, identifier="", best_only=True):
    weights = [
        os.path.join(out_folder, basename)
        for basename in os.listdir(out_folder)
        if basename.endswith(".hdf5") and identifier in basename
    ]

    if not weights:  # must be TF format!
        weights = [
            os.path.join(out_folder, basename)
            for basename in os.listdir(out_folder)
            if "weights" in basename and identifier in basename
        ]

    if best_only:
        weights = max(weights, key=os.path.getctime)
    return weights


def check_h5(name):
    import h5py

    def print_structure(weight_file_path):
        """
        Prints out the structure of HDF5 file.

        Args:
          weight_file_path (str) : Path to the file to analyze
        """
        f = h5py.File(weight_file_path)
        try:
            if len(f.attrs.items()):
                print("{} contains: ".format(weight_file_path))
                print("Root attributes:")

            print("  f.attrs.items(): ")
            for key, value in f.attrs.items():
                print("  {}: {}".format(key, value))

            if len(f.items()) == 0:
                print("  Terminate # len(f.items())==0: ")
                return

            print("  layer, g in f.items():")
            for layer, g in f.items():
                print("  {}".format(layer))
                print("    g.attrs.items(): Attributes:")
                for key, value in g.attrs.items():
                    print("      {}: {}".format(key, value))

                try:
                    print("    Dataset:")
                    for p_name in g.keys():
                        param = g[p_name]
                        subkeys = param.keys()
                        print("    Dataset: param.keys():")
                        for k_name in param.keys():
                            print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
                except:
                    pass
        finally:
            f.close()

    print_structure(name)

    f = h5py.File(name, 'r')
    print(f['model_weights'].keys())


def _load_weights(out_folder, identifier="", custom_objects=None, optimizer=None, loss=None, metrics=None):
    """
    DEV: great idea but does not work with TouristLeMC

    if "cSGLD" in custom_objects:
        from arkham.Bayes.MCMC.sgmcmc import EnsembleWrapper
        checkpoints = _get_weights(out_folder, identifier=identifier, best_only=False)
        #compile= False
        loaded_checkpoints = [load_model(checkpoint, custom_objects=custom_objects) for checkpoint in checkpoints]
        return EnsembleWrapper(loaded_checkpoints)
    """

    model_weights_path = _get_weights(out_folder, identifier=identifier)

    print("using weights: ", model_weights_path)
    needs_compile = False if not metrics else True

    # check_h5(temp_weights_file.name)
    if not needs_compile:  # did not save model_config
        try:
            model = load_model(model_weights_path, custom_objects=custom_objects)  # STILL COMPILES!
        except Exception as e:
            print("Moving along without compiling")
            model = load_model(model_weights_path, custom_objects=custom_objects, compile=False)
    else:
        model = load_model(model_weights_path, compile=False, custom_objects=custom_objects)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def config_to_json(_config, idx2label, idx2voc, kwargs={}):
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def convert(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        raise TypeError

    params = {**_config, **{"idx2label": idx2label, "idx2voc": idx2voc}, **kwargs}
    with open(os.path.join(_config["out_folder"], "params.json"), "w") as f:
        json.dump(params, f, indent=4, default=convert)  # cls=NumpyEncoder


def dump_embeddings(model, idx2voc, out_folder):
    e = next(x for x in model.layers if isinstance(x, tf.keras.layers.Embedding))
    weights = e.get_weights()[0]

    import io

    out_v = io.open(os.path.join(out_folder, 'vecs.tsv'), 'w', encoding='utf-8')
    out_m = io.open(os.path.join(out_folder, 'meta.tsv'), 'w', encoding='utf-8')

    for num in range(len(idx2voc)):
        word = idx2voc[num]
        vec = weights[num]  # could skip 0, it's padding.
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()


def load_model_config(modelpath):
    with open(os.path.join(modelpath, "params.json"), "r") as f:
        _config = json.load(f)

    _config["idx2voc"] = {int(k): v for k, v in _config["idx2voc"].items()}
    _config["idx2label"] = {int(k): v for k, v in _config["idx2label"].items()}
    _config["voc2idx"] = {v: k for k, v in _config["idx2voc"].items()}
    _config["label2idx"] = {v: k for k, v in _config["idx2label"].items()}
    return _config


def load_model_path(modelpath, identifier=""):
    _config = load_model_config(modelpath)
    optimizer = None
    loss = None
    metrics = None
    custom_objects = {}
    if _config.get("embedding_dropout"):
        custom_objects["EmbeddingDropout"] = EmbeddingDropout(_config["embedding_dropout"])
    if _config.get("dropout_concrete"):
        custom_objects["ConcreteDropout"] = ConcreteDropout
        custom_objects["SpatialConcreteDropout"] = SpatialConcreteDropout
        custom_objects["SpatialConcreteDropout2D"] = SpatialConcreteDropout2D

    if _config.get("weight_decay"):
        custom_objects["AdamW"] = AdamW(lr=_config["learning_rate"], weight_decay=_config["weight_decay"])
    if _config.get("use_aleatorics"):
        from tensorflow_probability.python.layers import DistributionLambda

        custom_objects[_config["loss_fn"]] = deduce_loss(
            _config["loss_fn"], len(_config["idx2label"]), _config["multilabel"], _config["use_aleatorics"]
        )
    if not "entropy" in _config["loss_fn"]:
        custom_objects["internal_loss"] = deduce_loss(
            _config["loss_fn"], len(_config["idx2label"]), _config["multilabel"], _config["use_aleatorics"]
        )
    if _config["optimizer"] == "cSGLD":
        from arkham.Bayes.MCMC.sgmcmc import cSGLD

        custom_objects["cSGLD"] = cSGLD
        optimizer = custom_objects["cSGLD"]

    if "Hierarchical" in _config["model"]:
        custom_objects["HierarchicalAttention"] = HierarchicalAttention

    if "Transformer" in _config["model"]:
        custom_objects["TokenAndPositionEmbedding"] = TokenAndPositionEmbedding
        custom_objects["TransformerBlock"] = TransformerBlock

    if "BERT" in _config["model"]:
        from transformers import TFBertMainLayer

        custom_objects["TFBertMainLayer"] = TFBertMainLayer  # Custom>TFBertMainLayer
        if _config.get("sequence_labels"):
            custom_objects[_config["loss_fn"]] = deduce_loss(
                _config["loss_fn"], len(_config["idx2label"]), _config["multilabel"], _config["use_aleatorics"]
            )
            from arkham.utils.losses import sparse_categorical_accuracy_masked

            metrics = [sparse_categorical_accuracy_masked]
            optimizer = get_optimizers(
                _config["optimizer"],
                _config["learning_rate"],
                weight_decay=_config["weight_decay"],
                clipnorm=_config["clipnorm"],
            )
    if "SNGP" in _config["model"]:
        custom_objects['tf'] = tf  # required for flatten layer

    if _config["metrics"] == ["chunk_f1"]:
        # https://github.com/tensorflow/tensorflow/issues/34068#event-2983714135
        metrics = [ChunkF1(_config["idx2label"])]
        optimizer = custom_objects.get("AdamW", "adam")
        loss = deduce_loss(
            _config["loss_fn"], len(_config["idx2label"]), _config["multilabel"], _config["use_aleatorics"]
        )
    if "heteroscedastic_acc" in _config["metrics"]:
        metrics = [Heteroscedastic_Acc(T=_config["posterior_sampling"])]
        optimizer = custom_objects.get("AdamW", "adam")
        loss = deduce_loss(
            _config["loss_fn"], len(_config["idx2label"]), _config["multilabel"], _config["use_aleatorics"]
        )

    elif "heteroscedastic_mse" in _config["metrics"]:
        metrics = [Heteroscedastic_MSE(T=_config["posterior_sampling"])]
        optimizer = custom_objects.get("AdamW", "adam")
        loss = deduce_loss(
            _config["loss_fn"], len(_config["idx2label"]), _config["multilabel"], _config["use_aleatorics"]
        )

    elif "ECE" in _config["metrics"]:
        from arkham.utils.calibration_metrics import ExpectedCalibrationError

        custom_objects["ECE"] = ExpectedCalibrationError(num_bins=30)

        if "decay_lr" in _config["metrics"]:
            from arkham.Bayes.MCMC.sgmcmc import get_lr_metric

            custom_objects["decay_lr"] = get_lr_metric

    model = _load_weights(
        modelpath, identifier=identifier, custom_objects=custom_objects, optimizer=optimizer, loss=loss, metrics=metrics
    )
    return model, _config


def reset_seeds(_seed):
    np.random.seed(_seed)
    random.seed(_seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(_seed)


class MCEnsembleWrapper:
    """
    This class wraps a list of models all of which are mc models
    """

    def __init__(self, modellist, n_mc):
        self.ms = modellist
        self.n_mc = n_mc

    def predict(self, X):
        mc_preds = np.concatenate([np.stack([m.predict(X) for _ in range(self.n_mc)]) for m in self.ms], axis=0)
        return mc_preds.mean(axis=0)

    def get_results(self, X):
        mc_preds = np.concatenate([np.stack([m.predict(X) for _ in range(self.n_mc)]) for m in self.ms], axis=0)
        preds = mc_preds.mean(axis=0)
        ent = -1 * np.sum(preds * np.log(preds + 1e-10), axis=-1)
        bald = ent - np.mean(-1 * np.sum(mc_preds * np.log(mc_preds + 1e-10), axis=-1), axis=0)
        return preds, ent, bald

    def __call__(self, X):
        """
        Returns the mean prediction of the entire ensemble as a keras tensor to allow differentiation
        """
        return K.mean(K.stack([K.mean(mc_dropout_preds(m, X, n_mc=self.n_mc), axis=0) for m in self.ms]), axis=0)


class RelaxedLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RelaxedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.W = self.add_weight(
            shape=(1,),
            name="temperature",  # Create a trainable weight variable for this layer.
            initializer="one",
            trainable=True,
        )
        # Create a trainable weight variable for this layer.
        self.relaxer = tf.constant(1.0, shape=(1,), name="relaxer", dtype="float32")
        super(RelaxedLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return tf.multiply(x, tf.divide(self.relaxer, self.W))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class SampleNormal(tf.keras.layers.Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_avg, z_log_var):
        eps = tf.keras.backend.random_normal(shape=tf.shape(z_avg), mean=0.0, stddev=1.0)
        return z_avg + tf.exp(z_log_var) * eps

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        return self._sample_normal(z_avg, z_log_var)
