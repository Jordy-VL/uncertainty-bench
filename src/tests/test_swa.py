"""
Sources: 

https://github.com/simon-larsson/keras-swa
https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SWA
https://stackoverflow.com/questions/66001157/how-to-update-weights-in-stochastic-weight-averaging-swa-on-tensorflow
=> no uncertainty
https://github.com/wjmaddox/swa_gaussian

References:

Nemeth, C., & Fearnhead, P. (2019). Stochastic gradient Markov chain Monte Carlo, 1–31.
Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubon, D. B. (n.d.). Bayesian Data Analysis Third Edition.
"""
# https://github.com/simon-larsson/keras-swa/issues/1


import os
import sys
import pytest
import numpy as np
import re
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

sys.path.append(os.path.expanduser("~/code/gordon/arkham"))
from arkham.Bayes.Quantify.data import generators_from_directory

# from arkham.Bayes.MCMC.swa import SWA


def get_model(vocab_size, nb_classes, functional=False):

    e = tf.keras.layers.Embedding(vocab_size, 50, trainable=True, mask_zero=True)
    d = tf.keras.layers.Dropout(0.25)
    encoder = tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation="relu")
    p = tf.keras.layers.GlobalMaxPooling1D()
    out = tf.keras.layers.Dense(nb_classes, activation="softmax")

    if functional:
        i = layers.Input(shape=(None,), dtype='int32', name='input')
        x = e(i)
        x = d(x)
        x = encoder(x)
        x = p(x)
        prediction = out(x)
        model = tf.keras.Model(inputs=i, outputs=prediction, name='custom')
    else:
        model = tf.keras.Sequential([e, d, encoder, p, out])
    return model


def get_data(identifier="20news", downsampling=0.1):
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join("/mnt/lerna/data", identifier),
        # os.path.join(pytest.DATAROOT, "20news"),
        downsampling=downsampling,  # .1,
        max_vocabulary=20000,
        composition=["word"],
        debug=False,
    )
    return generators, voc2idx, label2idx


def get_model_data(identifier="20news", downsampling=0.1):
    generators, voc2idx, label2idx = get_data(identifier=identifier, downsampling=downsampling)

    vocab_size = len(voc2idx)
    nb_classes = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    idx2voc = {v: k for k, v in voc2idx.items()}

    model = get_model(vocab_size, nb_classes)
    # print(model.summary())
    return model, generators

    vocab_size = len(voc2idx)
    nb_classes = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    idx2voc = {v: k for k, v in voc2idx.items()}

    model = get_model(vocab_size, nb_classes)
    # print(model.summary())
    return model, generators


def test_swa_tf():
    model, generators = get_model_data()
    epochs = 5
    batch_size = 32
    modelpath = f"/tmp/weights_{epochs:02d}.hdf5"

    """
    SWA: base implementation
    """
    start_epoch = int(epochs * (0.75))
    cycles = 2
    sgd = tf.keras.optimizers.Adam(0.001)
    stochastic_avg_sgd = tfa.optimizers.SWA(sgd, start_averaging=start_epoch, average_period=cycles)

    norm_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"/tmp/weights_{epochs:02d}-swa.hdf5",
        save_best_only=False,  # if _config["epochs"] > 1 else False,
        verbose=1,
        save_freq="epoch",
    )
    avg_callback = tfa.callbacks.AverageModelCheckpoint(
        filepath=modelpath, update_weights=True, verbose=1
    )  # set the average to continue
    model.compile(optimizer=stochastic_avg_sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    """
    alternate, non-functional

    swa = SWA(start_epoch=start_epoch, 
              lr_schedule='cyclic', 
              swa_lr=0.001,
              swa_lr2=0.003,
              swa_freq=cycles,
              verbose=1)
    """

    model.fit(
        generators["train"],
        epochs=epochs,
        validation_data=generators["dev"].repeat(epochs),
        steps_per_epoch=None,
        validation_steps=None,
        callbacks=[avg_callback],
    )
    model.evaluate(generators["test"])

    # save and load
    # model.save(modelpath)
    # del model

    loaded = tf.keras.models.load_model(modelpath)
    loaded.evaluate(generators["test"])
