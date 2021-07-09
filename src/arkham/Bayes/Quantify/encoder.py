# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from arkham.utils.regularization import EmbeddingDropout, ConcreteDropout, SpatialConcreteDropout
from arkham.Bayes.Quantify.text_cnn import construct_output_layer
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class SimpleEncoder:
    def __init__(
        self,
        embedding_layer=None,
        vocab_size=None,
        embed_dim=None,
        max_document_len=100,
        encoder="lstm",
        projection_nodes=32,
        use_char=False,
        max_chars_len=100,
        alphabet_size=None,
        char_embed_dim=30,
        char_kernel_sizes=None,
        char_feature_maps=None,
        composition=None,
        dropout=0,
        dropout_nonlinear=0,
        dropout_concrete=False,
        embedding_dropout=0,
        nb_classes=None,
        multilabel=False,
        use_aleatorics=False,
        batchnorm=None,
    ):
        """
        Arguments:
            embedding_layer : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            vocab_size       : Maximal amount of words in the vocabulary (default: None)
            embed_dim   : Dimension of word representation (default: None)
            max_document_len  : Max length of word sequence (default: 100)

            encoder = 2  # Number of attention heads
            projection_nodes    : Hidden layer size in feed forward network inside SimpleEncoder (default: 32)

            dropout    : If defined, dropout will be added after embedding layer & concatenation (default: None)
            nb_classes      : Number of classes which can be predicted
            dropout_nonlinear : Add dropout between all nonlinear layers for montecarlo dropout evaluation (default: False)
            dropout_concrete: Add ConcreteDropout between all nonlinear layers for montecarlo dropout evaluation (default: False)
        """

        # WORD-level
        self.embedding_layer = embedding_layer
        self.vocab_size = vocab_size
        self.max_document_len = max_document_len
        self.embed_dim = embed_dim

        # General
        self.encoder = encoder
        self.projection_nodes = projection_nodes

        # CHAR-level
        self.use_char = True if "character" in composition else False
        self.max_chars_len = max_chars_len
        self.alphabet_size = alphabet_size
        self.char_embed_dim = char_embed_dim
        self.char_kernel_sizes = char_kernel_sizes
        self.char_feature_maps = char_feature_maps

        self.composition = composition
        self.dropout = dropout  # on input to each LSTM block
        self.embedding_dropout = embedding_dropout
        self.dropout_concrete = dropout_concrete
        self.auxiliary_losses = []
        self.dropout_nonlinear = dropout_nonlinear  # on recurrent input signal -> recurrent
        self.nb_classes = nb_classes
        self.multilabel = multilabel
        self.use_aleatorics = use_aleatorics
        self.batchnorm = batchnorm

    def build_model(self):
        """
        Build a non-compiled model

        Returns:
            Model : tensorflow.keras model instance
        """

        # Checks
        if not self.embedding_layer and (not self.vocab_size or not self.embed_dim):
            raise Exception('Please define `vocab_size` and `embed_dim` if you not using a pre-trained embedding.')

        # WORD-level
        word_input = layers.Input(
            shape=(None if not self.max_document_len else self.max_document_len,), dtype='int32', name='word_input'
        )

        # Building word-embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                mask_zero=True,
                # input_length=self.max_document_len,
                weights=None,
                trainable=True,
                name="word_embedding",
            )

        x = self.embedding_layer(word_input)
        if self.embedding_dropout:
            x = EmbeddingDropout(self.embedding_dropout)(
                x
            )  # layers.Dropout(self.dropout, noise_shape=(None, self.embed_dim))(x)  # missing batch_size

        # Create classifier model using SimpleEncoder layer
        if self.encoder == "pooling_baseline":
            encoder = tf.keras.layers.GlobalAveragePooling1D()
        elif "lstm" in self.encoder:
            encoder = tf.keras.layers.LSTM(
                self.projection_nodes, dropout=self.dropout_nonlinear, return_sequences=False
            )  # , recurrent_dropout=self.dropout
            if self.encoder == "bilstm":
                encoder = tf.keras.layers.Bidirectional(encoder)
        else:
            raise NotImplementedError

        x = encoder(x)

        if self.dropout_nonlinear or self.dropout and not self.dropout_concrete:
            v = self.dropout_nonlinear if self.dropout_nonlinear else self.dropout
            x = layers.Dropout(v)(x)

        prediction, self.auxiliary_losses = construct_output_layer(
            x,
            self.nb_classes,
            use_aleatorics=self.use_aleatorics,
            multilabel=self.multilabel,
            dropout_concrete=self.dropout_concrete,
            auxiliary_losses=self.auxiliary_losses,
            batchnorm=self.batchnorm,
        )
        model = tf.keras.Model(inputs=word_input, outputs=prediction, name=self.encoder + 'Encoder')
        for loss in self.auxiliary_losses:
            model.add_loss(loss)
        return model

    def build_sequence_model(self):
        if not self.embedding_layer and (not self.vocab_size or not self.embed_dim):
            raise Exception('Please define `vocab_size` and `embed_dim` if you not using a pre-trained embedding.')

        if self.use_char:

            char_input = layers.Input(
                shape=(None, None if not self.max_chars_len else self.max_chars_len), dtype='int32', name='char_input'
            )
            x_char = layers.TimeDistributed(
                layers.Embedding(
                    input_dim=self.alphabet_size + 1,
                    output_dim=self.char_embed_dim,
                    mask_zero=False,  # CONV1D does not support masking
                    embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
                    # embeddings_regularizer='l2',
                    name='char_embedding',
                )
            )(char_input)

            x_char = layers.Dropout(self.dropout)(x_char)

            # CNN char encoder
            conv1d_out = layers.TimeDistributed(
                layers.Conv1D(
                    filters=self.char_feature_maps[0],
                    kernel_size=self.char_kernel_sizes[0],
                    padding='same',
                    activation='tanh',
                    strides=1,
                ),
                name="CharConv",
            )(x_char)
            maxpool_out = layers.TimeDistributed(layers.MaxPooling1D(self.max_chars_len), name="Maxpool")(conv1d_out)
            char_output = layers.TimeDistributed(layers.Flatten(), name="Flatten")(maxpool_out)
            char_output = layers.Dropout(self.dropout)(char_output)
            """
            #DEV: alternate form
            x_char = self.building_block(x_char, self.char_kernel_sizes, self.char_feature_maps)
            """
            """
            #LSTM char encoder
            char_enc = layers.TimeDistributed(layers.LSTM(units=self.char_embed_dim, return_sequences=False,
                                            recurrent_dropout=self.dropout))(x_char)
            """
        if "casing" in self.composition:
            # DEV: bad coding but got to move fast
            casing_dim = len(
                {
                    'PAD': 0,
                    'allLower': 1,
                    'allUpper': 2,
                    'initialUpper': 3,
                    'other': 4,
                    'mainly_numeric': 5,
                    'contains_digit': 6,
                    'numeric': 7,
                }
            )
            casing_input = layers.Input(shape=(None,), dtype='int32', name='casing_input')
            casing_output = layers.Embedding(
                input_dim=casing_dim,
                output_dim=casing_dim,
                mask_zero=False,
                trainable=False,
                weights=[np.identity(casing_dim, dtype='float32')],
                name='casing_embedding',
            )(casing_input)
            #
        # WORD-level
        word_input = layers.Input(
            shape=(None if not self.max_document_len else self.max_document_len,), dtype='int32', name='word_input'
        )

        # Building word-embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                mask_zero=True,
                # input_length=self.max_document_len,
                weights=None,
                trainable=True,
                name="word_embedding",
            )

        x = self.embedding_layer(word_input)

        if self.use_char:
            if "casing" in self.composition:
                x = layers.concatenate([x, char_output, casing_output])
            else:
                x = layers.concatenate([x, char_output])
            # if self.embedding_dropout:
            # x = SpatialDropout1D(0.3)(x)

        hidden_steps = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.projection_nodes,
                dropout=self.dropout,
                recurrent_dropout=self.dropout_nonlinear,
                return_sequences=True,
            )
        )(x)

        # tf.keras.layers.Dense(100, activation="relu")(hidden_steps)
        prediction = tf.keras.layers.Dense(self.nb_classes, activation="softmax")(hidden_steps)

        if self.use_char:
            inputs = {"char_inputs": char_input, "word_input": word_input}
            encoder_name = 'CharEncoder'
            if "casing" in self.composition:
                inputs["casing_input"] = casing_input
                encoder_name = "Case" + encoder_name
            model = tf.keras.Model(inputs=inputs, outputs=prediction, name=self.encoder + encoder_name)
        else:
            model = tf.keras.Model(inputs=word_input, outputs=prediction, name=self.encoder + 'Encoder')
        return model

    def building_block(self, input_layer, kernel_sizes, feature_maps):
        """
        Creates several CNN channels in parallel and concatenate them

        Arguments:
            input_layer : Layer which will be the input for all convolutional blocks
            kernel_sizes: Array of kernel sizes (working as n-gram filter)
            feature_maps: Array of feature maps

        Returns:
            x           : Building block with one or several channels
        """
        channels = []
        for ix in range(len(kernel_sizes)):
            x = self.create_channel(input_layer, kernel_sizes[ix], feature_maps[ix])
            channels.append(x)

        # Check how many channels, one channel doesn't need a concatenation
        if len(channels) > 1:
            x = layers.concatenate(channels)
        return x

    def create_channel(self, x, kernel_size, feature_map):
        """
        Creates a layer, working channel wise
        "complex" adds a more efficient Conv1D operator + global and average pooling with a linear projection layer.

        Arguments:
            x           : Input for convolutional channel
            kernel_size : Kernel size for creating Conv1D
            feature_map : Feature map

        Returns:
            x           : Channel including (Conv1D + {GlobalMaxPooling & GlobalAveragePooling} + Dense [+ Dropout])
        """

        if self.version == "complex":
            conv = layers.SeparableConv1D(
                feature_map, kernel_size=kernel_size, activation='relu', strides=1, padding='valid', depth_multiplier=4
            )
            # In the code: The depth_multiplier is used to reduce the number of channels at each layer. So the depth_multiplier corresponds the width multiplier α.
        else:
            conv = layers.Conv1D(feature_map, kernel_size, strides=1, padding='valid', activation=None)

        if self.dropout_concrete:
            x, l = SpatialConcreteDropout(conv)(x)
            self.auxiliary_losses.append(l)
        else:
            x = conv(x)

        if self.dropout_nonlinear and not self.dropout_concrete:
            x = layers.Dropout(self.dropout_nonlinear)(x)

        if self.version == "complex":
            x1 = layers.GlobalMaxPooling1D()(x)
            x2 = layers.GlobalAveragePooling1D()(x)
            x = layers.concatenate([x1, x2])
            x = layers.Dense(self.projection_nodes)(x)
        else:
            x = layers.GlobalMaxPooling1D()(x)
            x = layers.Activation('relu')(x)

        # DEV: was included before, does not make sense to apply dropout here.
        # if self.dropout:
        #     x = layers.Dropout(self.dropout_nonlinear)(x)
        return x


def example_fit():
    """## Download and prepare dataset"""

    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    n_classes = 2
    model = SimpleEncoder(
        vocab_size=vocab_size, embed_dim=100, nb_classes=2, max_document_len=200, dropout_nonlinear=0.5
    ).build_model()

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    """## Train and Evaluate"""
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
    print(history)


if __name__ == '__main__':
    SimpleEncoder(vocab_size=10000, embed_dim=100, nb_classes=2, dropout_nonlinear=0.5).build_model()
    model = SimpleEncoder(
        vocab_size=10000, embed_dim=100, nb_classes=2, dropout_nonlinear=0.5, dropout_concrete=True
    )  # .build_model()
    model.build_model()
    example_fit()
