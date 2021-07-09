# -*- coding: utf-8 -*-
"""
Universal Sentence Encoder - pipeline implemented in TensorFlow 2.
This implementation is based on [2]


References
----------
- [1] Universal Sentence Encoder. Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Céspedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil. arXiv:1803.11175, 2018.
- [2] [Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/pdf/1907.04307.pdf)

alternative:
- [3] Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Narveen Ari, Wei Wang. Language-agnostic BERT Sentence Embedding. July 2020
# https://tfhub.dev/google/LaBSE/1

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


def load_from_hub(pretrained_embeddings="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
    # Import the Universal Sentence Encoder's TF Hub module
    # embed = hub.Module(module_url)
    import tensorflow_hub as hub
    import tensorflow_text

    hub_layer = hub.KerasLayer(
        pretrained_embeddings,
        input_shape=(None,),
        trainable=True,
        dtype=tf.string,
        name=pretrained_embeddings.replace("https://", ""),
    )  # , arguments={"mask_zero": True})
    return hub_layer


"""
import tensorflow_hub as hub
import tensorflow_text
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
"""


def UniversalEmbedding(x):
    return tf.expand_dims(
        embed(tf.squeeze(tf.cast(x, tf.string))), axis=0
    )  # , signature="default", as_dict=True)["default"]
    # embedding = layers.Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)


class UniversalSentenceEncoder:
    def __init__(
        self,
        max_document_len=200,
        max_sentences=15,
        dropout=0,
        dropout_nonlinear=0,
        embedding_dropout=0,
        nb_classes=None,
        use_aleatorics=False,
        batchnorm=None,
    ):
        """
        Arguments:
            max_document_len  : Max length of word sequence (default: 200)
            max_sentences   :   Max number of sentences per document (default: 15)
            dropout    : If defined, dropout will be added after embedding layer & concatenation (default: 0)
            nb_classes      : Number of classes which can be predicted
            dropout_nonlinear : Add dropout between all nonlinear layers for montecarlo dropout evaluation (default: 0)
        """
        self.embedding_layer = load_from_hub()
        self.max_document_len = max_document_len
        self.max_sentences = max_sentences

        # General
        self.dropout = dropout
        self.dropout_nonlinear = dropout_nonlinear
        self.embedding_dropout = embedding_dropout
        self.batchnorm = batchnorm
        self.nb_classes = nb_classes
        self.use_aleatorics = use_aleatorics

    def embedder(self, x):
        return self.embedding_layer(
            tf.squeeze(x)
        )  # tf.squeeze(tf.cast(x, tf.string)))  # , signature="default", as_dict=True)["default"]

    def build_model(self):
        """
        Build a non-compiled model
        Returns:
            Model : tensorflow.keras model instance
        """

        # Checks
        # (None if not self.max_sentences else self.max_sentences, None if not self.max_document_len else self.max_document_len)
        document_input = layers.Input(
            shape=(None if not self.max_sentences else self.max_sentences,), ragged=False, dtype='string'
        )  # ragged input with lambda :/

        # x = layers.TimeDistributed(self.embedding_layer)(document_input)  # should apply per sentence, then average
        x = layers.Lambda(self.embedder)(document_input)  # , output_shape=(embed_size,)

        x = layers.GlobalAveragePooling1D()(tf.expand_dims(x, axis=0))
        # x = layers.Dense(256, activation='relu')(x)
        # x = layers.GlobalAveragePooling1D()(x)
        # x = layers.Bidirectional(tf.keras.layers.LSTM(100))(x)
        # ValueError: Input 0 of layer bidirectional is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: [None, 512]

        """
        optional bottleneck layer
        """

        if not self.use_aleatorics:
            prediction = layers.Dense(self.nb_classes, activation='softmax')(x)
        else:
            import tensorflow_probability as tfp

            mu = layers.Dense(self.nb_classes, activation=None, name="mu")(x)
            sigma = layers.Dense(self.nb_classes, activation=None, name="sigma")(x)
            if self.batchnorm:
                mu = layers.BatchNormalization()(mu)
                sigma = layers.BatchNormalization()(sigma)
            prediction = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=tf.exp(0.5 * t[1]))
            )([mu, sigma])

        document_encoder = tf.keras.Model(inputs=document_input, outputs=prediction, name='USE')
        return document_encoder

    def example_fit(model):
        sentences = tf.ragged.constant(
            ['What makes you think she is a witch?', 'She turned me into a newt.', 'A newt?', 'Well, I got better.']
        )
        # problem = not the same input; this works easily; yet we feed sequences of inputs [multiple ragged sentences] -> confused batchsize
        is_question = tf.constant([True, False, True, False])

        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        model.fit(sentences, is_question, epochs=3, batch_size=2, verbose=2)
        model.predict(sentences)


if __name__ == '__main__':
    model = UniversalSentenceEncoder(nb_classes=3).build_model()
    print(model.summary())
    UniversalSentenceEncoder.example_fit(model)
