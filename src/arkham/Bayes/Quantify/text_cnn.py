# -*- coding: utf-8 -*-
"""
CNN model for text classification implemented in TensorFlow 2.
This implementation is based on the original paper of Yoon Kim [1] for classification using words.
Besides I add character level input [2]. In a next iteration, hierarchical extension with sentence-level attention will be added.

References
----------
- [1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [2] [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

"""

import tensorflow as tf
from tensorflow.keras import layers
from arkham.utils.model_utils import SampleNormal
from arkham.utils.regularization import EmbeddingDropout, ConcreteDropout, SpatialConcreteDropout
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


def construct_output_layer(
    x, nb_classes, use_aleatorics=False, multilabel=False, dropout_concrete=None, auxiliary_losses=None, batchnorm=False
):
    if not use_aleatorics:
        if dropout_concrete:
            prediction, l = ConcreteDropout(layers.Dense(nb_classes, activation='softmax'))(x)
            auxiliary_losses.append(l)
        else:
            activation = "sigmoid" if multilabel else "softmax"
            prediction = layers.Dense(nb_classes, activation=activation)(x)
    else:

        mu = layers.Dense(nb_classes, activation=None, name="mu")
        sigma = layers.Dense(nb_classes, activation=None, name="sigma")
        if dropout_concrete:
            mu, l = ConcreteDropout(mu)(x)
            auxiliary_losses.append(l)
            if not "1layer" in str(use_aleatorics):
                sigma, l = ConcreteDropout(sigma)(x)
                auxiliary_losses.append(l)
        else:
            mu = mu(x)
            sigma = sigma(x)
        if batchnorm:
            mu = layers.BatchNormalization()(mu)
            sigma = layers.BatchNormalization()(sigma)

        K_layers = [mu, sigma]
        if not "softplus" in str(use_aleatorics):
            function_to_scale = lambda var: tf.exp(0.5 * var)
        elif "nofunc" in str(use_aleatorics):
            function_to_scale = lambda var: var
        else:
            function_to_scale = lambda var: tf.math.softplus(var)
        use_lambda = True

        if use_aleatorics in [True, "softplus", "nofunc"]:
            distribution_fn = lambda t: tfd.Normal(loc=t[0], scale=function_to_scale(t[1]))

        elif use_aleatorics == "independent-lambda":
            distribution_fn = lambda t: tfd.Independent(tfd.Normal(loc=t[0], scale=function_to_scale(t[1])))

        elif use_aleatorics in ["multivar", "multivar-softplus"]:
            distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=function_to_scale(t[1]))

        elif use_aleatorics == "multivar-nofunc":
            distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1])

        elif use_aleatorics == "multivar-independent":
            distribution_fn = lambda t: tfd.Independent(
                tfd.MultivariateNormalDiag(loc=t[0], scale_diag=function_to_scale(t[1]))
            )
        elif use_aleatorics == "1layer":
            K_layers = mu
            distribution_fn = lambda t: tfd.Normal(loc=t, scale=function_to_scale(t))
        elif use_aleatorics == "1layer_multivar":
            K_layers = mu
            distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=function_to_scale(t))

        elif use_aleatorics == "1layer_multivar-nofunc":
            K_layers = mu
            distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=function_to_scale(t))

        prediction = tfpl.DistributionLambda(distribution_fn)(K_layers)
    return prediction, auxiliary_losses


class TextClassificationCNN:
    def __init__(
        self,
        embedding_layer=None,
        vocab_size=None,
        embed_dim=None,
        max_document_len=100,
        kernel_sizes=[3, 4, 5],
        feature_maps=[100, 100, 100],
        projection_nodes=100,
        dropout=None,
        dropout_nonlinear=None,
        dropout_concrete=None,
        embedding_dropout=None,
        nb_classes=None,
        multilabel=False,
        use_aleatorics=False,
        batchnorm=None,
        use_char=False,
        max_chars_len=200,
        alphabet_size=None,
        char_kernel_sizes=[3, 10, 20],
        char_feature_maps=[100, 100, 100],
        version="simple",
    ):
        """
        Arguments:
            embedding_layer : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            vocab_size       : Maximal amount of words in the vocabulary (default: None)
            embed_dim   : Dimension of word representation (default: None)
            max_document_len  : Max length of word sequence (default: 100)
            filter_sizes    : An array of filter sizes per channel (default: [3,4,5])
            feature_maps    : Defines the feature maps per channel (default: [100,100,100])
            use_char        : If True, char-based model will be added to word-based model
            max_chars_len : Max length of char sequence (default: 200)
            alphabet_size   : Amount of different chars used for creating embeddings (default: None)
            projection_nodes    : Hidden units per convolution channel (default: 100)
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
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        # CHAR-level
        self.use_char = use_char
        self.max_chars_len = max_chars_len
        self.alphabet_size = alphabet_size
        self.char_kernel_sizes = char_kernel_sizes
        self.char_feature_maps = char_feature_maps
        # General
        self.projection_nodes = projection_nodes
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.dropout_concrete = dropout_concrete
        self.auxiliary_losses = []
        self.dropout_nonlinear = dropout_nonlinear  # could use abstract wrapper "doing nothing" if None
        self.nb_classes = nb_classes
        self.multilabel = multilabel
        self.use_aleatorics = use_aleatorics
        self.batchnorm = batchnorm
        self.version = version

    def build_model(self):
        """
        Build a non-compiled model

        Returns:
            Model : tensorflow.keras model instance
        """

        # Checks
        if len(self.kernel_sizes) != len(self.feature_maps):
            raise Exception('Please define `kernel_sizes` and `feature_maps` with the same amount.')
        if not self.embedding_layer and (not self.vocab_size or not self.embed_dim):
            raise Exception('Please define `vocab_size` and `embed_dim` if you not using a pre-trained embedding.')
        if self.use_char and (not self.max_chars_len or not self.alphabet_size):
            raise Exception('Please define `max_chars_len` and `alphabet_size` if you are using char.')

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

        # WORD-level
        word_input = layers.Input(
            shape=(None if not self.max_document_len else self.max_document_len,), dtype='int32', name='word_input'
        )

        x = self.embedding_layer(word_input)

        if self.embedding_dropout:
            x = EmbeddingDropout(self.embedding_dropout)(
                x
            )  # layers.Dropout(self.dropout, noise_shape=(None, self.embed_dim))(x)  # missing batch_size

        x = self.building_block(x, self.kernel_sizes, self.feature_maps)
        x = layers.Activation('relu')(x)

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

        # CHAR-level
        if self.use_char:
            char_input = layers.Input(
                shape=(None if not self.max_chars_len else self.max_chars_len,), dtype='int32', name='char_input'
            )
            x_char = layers.Embedding(
                input_dim=self.alphabet_size + 1,
                output_dim=50,  # DEV: char_embed_size
                mask_zero=True,
                input_length=self.max_chars_len,
                name='char_embedding',
            )(char_input)
            x_char = self.building_block(x_char, self.char_kernel_sizes, self.char_feature_maps)
            x_char = layers.Activation('relu')(x_char)
            x_char = layers.Dense(self.nb_classes, activation='softmax')(x_char)

            prediction = layers.Average()([prediction, x_char])
            return tf.keras.Model(inputs=[word_input, char_input], outputs=prediction, name='CNN_Word_Char')
        model = tf.keras.Model(inputs=word_input, outputs=prediction, name='CNN_Word')
        for loss in self.auxiliary_losses:
            model.add_loss(loss)
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
    model = TextClassificationCNN(
        vocab_size=vocab_size, embed_dim=100, nb_classes=2, dropout_nonlinear=0.5, max_document_len=200
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
    example_fit()
    """
    TextClassificationCNN(
        vocab_size=10000, embed_dim=100, nb_classes=2, dropout_nonlinear=0.5, embedding_dropout=0.5
    ).build_model()

    model = TextClassificationCNN(
        vocab_size=10000, embed_dim=100, nb_classes=2, dropout_nonlinear=0.5, dropout_concrete=True
    ).build_model()
    print(model.losses)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print(model.losses)
    """
