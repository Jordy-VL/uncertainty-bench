import tensorflow as tf
import numpy as np
import os
import sys
import pytest

from arkham.utils.regularization import *

# Task: predict whether each sentence is a question or not.
sentences = tf.constant(
    ['What makes you think she is a witch?', 'She turned me into a newt.', 'A newt?', 'Well, I got better.']
)
is_question = tf.constant([True, False, True, False])

# Preprocess the input strings.
hash_buckets = 1000
words = tf.strings.split(sentences, ' ')
hashed_words = tf.strings.to_hash_bucket_fast(words, hash_buckets).to_tensor()  # get rid of ragged


def test_embedding_dropout():
    embedded = tf.keras.layers.Embedding(hash_buckets, 5)(hashed_words)
    dropped = EmbeddingDropout(0.5)(embedded, training=True)
    print(hashed_words.shape)
    print(embedded)
    print(embedded.shape)
    print(dropped.shape)
    print(dropped)
    # what would we expect it to drop?
    assert np.sum(embedded) != np.sum(dropped)

    model = True
    # Build the Keras model.
    if model:
        keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=[None], dtype=tf.int64, ragged=False),
                tf.keras.layers.Embedding(hash_buckets, 16),
                tf.keras.layers.Dropout(0.5),  # EmbeddingDropout(0.5),
                # embedding dropout
                tf.keras.layers.LSTM(32, use_bias=False),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        keras_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        keras_model.fit(hashed_words, is_question, epochs=5)
        print(keras_model.predict(hashed_words))


def test_concrete_dropout():
    def build_model(out_size=1):
        input_layer = tf.keras.layers.Input(shape=[None], dtype=tf.int64, ragged=False)
        x = tf.keras.layers.Embedding(hash_buckets, 16)(input_layer)
        x = tf.keras.layers.LSTM(32, use_bias=False)(x)
        x, regularization_loss = ConcreteDropout(tf.keras.layers.Dense(32, activation="relu"))(x)
        output = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=input_layer, outputs=output, name='dummy_model')
        model.add_loss(regularization_loss)
        return model

    # Build the Keras model.
    model = build_model()
    losses = []
    print()
    print("before: ", model.losses)
    for layer in model.layers:
        if "concrete" in layer.name:
            assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob
            print("p: ", layer.p)
            print("plogit: ", layer.p_logit)
    print("inbetween: ", model.losses)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(hashed_words, is_question, epochs=5)
    print("after: ", model.losses)
    for layer in model.layers:
        if "concrete" in layer.name:
            assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob
            print("p: ", layer.p)
            print("plogit: ", layer.p_logit)
    print(model.predict(hashed_words))


def test_dropconnect():
    # Build the Keras model.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[None], dtype=tf.int64, ragged=False),
            tf.keras.layers.Embedding(hash_buckets, 16),
            DropConnect(0.5),
            tf.keras.layers.LSTM(32, use_bias=False),
            DropConnect(0.5),
            tf.keras.layers.Dense(32, activation="relu"),
            DropConnect(0.5),
            tf.keras.layers.Dense(1),
        ]
    )
    """
    print("before: ", model.losses)
    for layer in model.layers:
        if "concrete" in layer.name:
            assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob
        print(layer.p)
        print(layer.p_logit)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(hashed_words, is_question, epochs=5)
    print("after: ", model.losses)
    for layer in model.layers:
    if "concrete" in layer.name:
        assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob
        print(layer.p)
        print(layer.p_logit)
    print(model.predict(hashed_words))
    """


# how can I be sure it does what it says?
# Make small test on array!
