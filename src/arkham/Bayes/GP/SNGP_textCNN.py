from copy import deepcopy

import tensorflow as tf
from tensorflow.keras import layers
from arkham.utils.model_utils import SampleNormal
from arkham.utils.regularization import EmbeddingDropout, ConcreteDropout, SpatialConcreteDropout2D
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

from arkham.Bayes.GP.SNGP import SNGP, spec_norm_kwargs, conv_norm_kwargs, gp_kwargs, SNGP_wrapper
from arkham.Bayes.GP.concrete_dropnorm import SpectralNormalizationConcreteDropoutConv2D
from arkham.utils.callbacks import CustomCheckpoint

import edward2 as ed
from functools import partial


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

        prediction = tfpl.DistributionLambda(distribution_fn)(K_layers)
    return prediction, auxiliary_losses


class TextClassificationCNN_SNGP:
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
        spec_norm_kwargs=None,
        spec_norm_multipliers=[1, 2],
        # gp_kwargs=None,
        use_gp_layer=True,
        implementation="v2",
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
        self.batchnorm = False
        self.version = version
        self.implementation = implementation
        self.set_SNGP_kwargs(spec_norm_kwargs, spec_norm_multipliers, use_gp_layer)

    def set_SNGP_kwargs(self, spec_norm_kwargs, spec_norm_multipliers, use_gp_layer):
        self.spec_norm_kwargs = spec_norm_kwargs
        self.spec_norm_kwargs["norm_multiplier"] = spec_norm_multipliers[0]
        self.conv_norm_kwargs = deepcopy(self.spec_norm_kwargs)
        if self.implementation == "v2":
            self.conv_norm_kwargs.pop("inhere_layer_name")
        self.conv_norm_kwargs["norm_multiplier"] = spec_norm_multipliers[1]
        # could find a "c" larger than 1 if lipschitz bound can be respected
        self.use_gp_layer = use_gp_layer

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
            x = EmbeddingDropout(self.embedding_dropout)(x)

        if self.implementation == "v2":
            # ISSUE: reshape with None!
            x = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)
            """
            x = tf.keras.layers.Reshape(
                (None, self.embed_dim, 1), name='add_channel')(x)
            """

        x = self.building_block(x, self.kernel_sizes, self.feature_maps)

        if self.implementation == "v2":
            x = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')(x)
        else:
            x = layers.Activation('relu')(x)  # DEV: has no real purpose?

        if self.dropout_nonlinear or self.dropout and not self.dropout_concrete:
            v = self.dropout_nonlinear if self.dropout_nonlinear else self.dropout
            x = layers.Dropout(v)(x)

        prediction, self.auxiliary_losses = construct_output_layer(
            x,
            self.nb_classes,
            use_aleatorics=self.use_aleatorics,
            multilabel=self.multilabel,
            dropout_concrete=self.dropout_concrete if not self.use_gp_layer else False,
            auxiliary_losses=self.auxiliary_losses,
            batchnorm=self.batchnorm,
        )

        model = tf.keras.Model(inputs=word_input, outputs=prediction, name='CNN_Word')

        if not self.use_gp_layer:
            for loss in self.auxiliary_losses:
                model.add_loss(loss)

        # print("full: ", model.summary())
        if self.use_gp_layer:
            model = SNGP_wrapper(model, use_gp_layer=True)
            # print("with GP: ", model.summary())
            for loss in self.auxiliary_losses:
                model.submodel.add_loss(loss)
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

        # denser = partial(ed.layers.SpectralNormalization, **self.spec_norm_kwargs)
        # conver = partial(ed.layers.SpectralNormalization, **self.conv_norm_kwargs)
        if self.implementation == "v2":
            # from arkham.Bayes.GP.convnorm import SpectralNormalizationConv2D
            conver = partial(ed.layers.SpectralNormalizationConv2D, **self.conv_norm_kwargs)
        else:
            conver = partial(ed.layers.SpectralNormalization, **self.conv_norm_kwargs)

        if self.version == "complex":
            conv = layers.SeparableConv1D(
                feature_map, kernel_size=kernel_size, activation='relu', strides=1, padding='valid', depth_multiplier=4
            )
            # In the code: The depth_multiplier is used to reduce the number of channels at each layer. So the depth_multiplier corresponds the width multiplier α.
        elif self.implementation == "v2":
            filter_shape = (kernel_size, self.embed_dim)
            max_pool_shape = (self.max_document_len - kernel_size + 1, 1)

            conv = tf.keras.layers.Conv2D(
                feature_map,
                filter_shape,
                strides=(1, 1),
                padding='valid',
                data_format='channels_last',
                activation='relu',
                kernel_initializer='glorot_normal',
                bias_initializer=tf.keras.initializers.constant(0.1),
                name='convolution_{:d}'.format(kernel_size),
            )
        else:
            conv = layers.Conv1D(feature_map, kernel_size, strides=1, padding='valid', activation=None)

        if self.dropout_concrete:
            if self.conv_norm_kwargs["norm_multiplier"] > 0:
                x, l = SpectralNormalizationConcreteDropoutConv2D(conv, **self.conv_norm_kwargs)(x)
            else:
                x, l = SpatialConcreteDropout2D(conv)(x)  # incorporate into each other
            self.auxiliary_losses.append(l)
        else:
            if self.conv_norm_kwargs["norm_multiplier"] > 0:
                conv = conver(conv)
            x = conv(x)

        if self.dropout_nonlinear and not self.dropout_concrete:
            x = layers.Dropout(self.dropout_nonlinear)(x)

        if self.version == "complex":
            x1 = layers.GlobalMaxPooling1D()(x)
            x2 = layers.GlobalAveragePooling1D()(x)
            x = layers.concatenate([x1, x2])
            # denser()
            x = layers.Dense(self.projection_nodes)(x)

        elif self.implementation == "v2":
            x = tf.keras.layers.MaxPool2D(
                pool_size=max_pool_shape,
                strides=(1, 1),
                padding='valid',
                data_format='channels_last',
                name='max_pooling_{:d}'.format(kernel_size),
            )(x)
            # x = layers.Activation('relu')(x)

        else:
            x = layers.GlobalMaxPooling1D()(x)
            x = layers.Activation('relu')(x)

        return x


def example_fit():
    """## Download and prepare dataset"""

    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    n_classes = 2
    epochs = 2

    """
    model = TextClassificationCNN_SNGP(
        vocab_size=vocab_size,
        embed_dim=100,
        nb_classes=2,
        dropout_nonlinear=0.5,
        max_document_len=200,
        spec_norm_kwargs=spec_norm_kwargs,
        use_gp_layer=True,
        dropout_concrete=False,
        implementation="v2"
    ).build_model()

    spec_norm_kwargs["norm_multiplier"] = 0
    model = TextClassificationCNN_SNGP(
        vocab_size=vocab_size,
        embed_dim=100,
        nb_classes=2,
        dropout_nonlinear=0.5,
        max_document_len=200,
        spec_norm_kwargs=spec_norm_kwargs,
        spec_norm_multipliers=[0,0],
        use_gp_layer=False,
        dropout_concrete=True,
        implementation="v2"
    ).build_model()
    """
    model = TextClassificationCNN_SNGP(
        vocab_size=vocab_size,
        embed_dim=100,
        nb_classes=2,
        dropout_nonlinear=0.5,
        max_document_len=200,
        spec_norm_kwargs=spec_norm_kwargs,
        # spec_norm_multipliers=[15,15],
        use_gp_layer=True,
        dropout_concrete=True,
        implementation="v2",
    ).build_model()
    """
    spec_norm_kwargs["norm_multiplier"] = 0
    """
    # dropout_concrete

    # model.submodel.summary()

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    x_train, y_train = x_train[:100], y_train[:100]

    """## Train and Evaluate"""
    loss = "sparse_categorical_crossentropy"
    if hasattr(model, "submodel"):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    norm_callback = CustomCheckpoint(
        filepath="/home/jordy/woef/woef3",  #  {epoch:02d}",
        save_format="tf",
        # filepath=f"//home/jordy/woef/weights_{epochs:02d}-sngp.hdf5",
        save_best_only=False,  # if _config["epochs"] > 1 else False,
        verbose=1,
        save_freq="epoch",
        save_weights_only=False,
    )

    for layer in model.layers:
        if "concrete" in layer.name:
            assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob

    model.compile("adam", loss, metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_val, y_val), callbacks=[norm_callback]
    )
    print(history)

    x_val, y_val = x_val[:1000], y_val[:1000]

    sngp_logits, sngp_covmat = model(x_val, return_covmat=True, mean_field=False)
    sngp_variance = tf.linalg.diag_part(sngp_covmat)[:, None]

    for layer in model.layers:
        if "concrete" in layer.name:
            assert len(layer.trainable_weights) == 3  # kernel, bias, and dropout prob
    # modelpath = "./weights"
    # model.save(modelpath, save_format="tf")

    loaded = tf.keras.models.load_model("/home/jordy/woef/woef3")  # , custom_objects={"SNGP_wrapper":SNGP_wrapper})
    loaded.evaluate(x_val, y_val)

    loaded.call_covmat(x_val, True, False, True)

    sngp_logits, sngp_covmat = loaded(x_val, return_covmat=True, mean_field=False)
    sngp_variance = tf.linalg.diag_part(sngp_covmat)[:, None]


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
