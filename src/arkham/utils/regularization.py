#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

tf.random.set_seed(42)


class ConcreteDropout(layers.Wrapper):
    # adapted from https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb
    # can only use in FUNCTIONAL api! need to manually collect layer loss to add to model
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(
        self,
        layer,
        weight_regularizer=1e-6,
        dropout_regularizer=1e-5,
        init_min=0.1,
        init_max=0.1,
        is_mc_dropout=True,
        **kwargs
    ):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        # self.p = None
        self.init_min = np.log(init_min) - np.log(1.0 - init_min)
        self.init_max = np.log(init_max) - np.log(1.0 - init_max)

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        # initialise p
        self.p_logit = self.layer.add_weight(
            name='p_logit',
            shape=(1,),
            initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
            trainable=True,
        )
        # self.p = tf.math.sigmoid(self.p_logit)
        # self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1  # why this temp constant?

        unif_noise = tf.random.uniform(shape=tf.shape(x))
        drop_prob = (
            tf.math.log(self.p + eps)
            - tf.math.log(1.0 - self.p + eps)
            + tf.math.log(unif_noise + eps)
            - tf.math.log(1.0 - unif_noise + eps)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temp)
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    # mc_dropout arg is new
    def call(self, inputs, training=None):
        self.p = tf.math.sigmoid(self.p_logit)  # why sigmoid?

        # initialise regulariser / prior KL term
        input_dim = inputs.shape[-1]  # last dim #np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1.0 - self.p)
        dropout_regularizer = self.p * tf.math.log(self.p)
        dropout_regularizer += (1.0 - self.p) * tf.math.log(1.0 - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)

        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs)), regularizer
        else:

            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))

            return K.in_train_phase(relaxed_dropped_inputs, self.layer.call(inputs), training=training), regularizer


class SpatialConcreteDropout(layers.Wrapper):
    """This wrapper allows to learn the dropout probability for any given Conv1D input layer.
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(
        self,
        layer,
        weight_regularizer=1e-6,
        dropout_regularizer=1e-5,
        init_min=0.1,
        init_max=0.1,
        is_mc_dropout=True,
        data_format=None,
        **kwargs
    ):
        assert 'kernel_regularizer' not in kwargs
        super(SpatialConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = np.log(init_min) - np.log(1.0 - init_min)
        self.init_max = np.log(init_max) - np.log(1.0 - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'
        self.input_dim = None

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(SpatialConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(
            name='p_logit',
            shape=(1,),
            initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
            trainable=True,
        )
        self.p = tf.math.sigmoid(self.p_logit)
        """
        What changes is the number of spatial dimensions of your input that is convolved:

        With Conv1D, one dimension only is used, so the convolution operates on the first axis (size 68). #batch_shape + (steps, input_dim)
        With Conv2D, two dimensions are used, so the convolution operates on the two axis defining the data (size (68,2))
        """

        # initialise regulariser / prior KL term
        assert len(input_shape) >= 3, 'this wrapper is adapted to ONLY Conv1D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[2]
        # self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 2.0 / 3.0  # WHY???

        input_shape = K.shape(x)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1)
        else:
            noise_shape = (input_shape[0], 1, input_shape[2])
        unif_noise = K.random_uniform(shape=noise_shape)

        drop_prob = (
            K.log(self.p + eps) - K.log(1.0 - self.p + eps) + K.log(unif_noise + eps) - K.log(1.0 - unif_noise + eps)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temp)
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        self.p = tf.math.sigmoid(self.p_logit)

        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1.0 - self.p)
        dropout_regularizer = self.p * tf.math.log(self.p)
        dropout_regularizer += (1.0 - self.p) * tf.math.log(1.0 - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)

        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs)), regularizer
        else:

            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs))

            return K.in_train_phase(relaxed_dropped_inputs, self.layer.call(inputs), training=training), regularizer


class SpatialConcreteDropout2D(layers.Wrapper):
    """This wrapper allows to learn the dropout probability for any given Conv1D input layer.
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(
        self,
        layer,
        weight_regularizer=1e-6,
        dropout_regularizer=1e-5,
        init_min=0.1,
        init_max=0.1,
        is_mc_dropout=True,
        data_format=None,
        **kwargs
    ):
        assert 'kernel_regularizer' not in kwargs
        super(SpatialConcreteDropout2D, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = np.log(init_min) - np.log(1.0 - init_min)
        self.init_max = np.log(init_max) - np.log(1.0 - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'
        self.input_dim = None

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(
            SpatialConcreteDropout2D, self
        ).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(
            name='p_logit',
            shape=(1,),
            initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
            trainable=True,
        )
        self.p = tf.math.sigmoid(self.p_logit)
        """
        What changes is the number of spatial dimensions of your input that is convolved:

        With Conv1D, one dimension only is used, so the convolution operates on the first axis (size 68). #batch_shape + (steps, input_dim)
        With Conv2D, two dimensions are used, so the convolution operates on the two axis defining the data (size (68,2))
        """

        # initialise regulariser / prior KL term
        assert len(input_shape) >= 3, 'this wrapper is adapted to ONLY Conv2D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[3]
        # self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 2.0 / 3.0  # WHY???

        input_shape = K.shape(x)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], self.input_dim, 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, self.input_dim)
        unif_noise = K.random_uniform(shape=noise_shape)

        drop_prob = (
            K.log(self.p + eps) - K.log(1.0 - self.p + eps) + K.log(unif_noise + eps) - K.log(1.0 - unif_noise + eps)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temp)
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        self.p = tf.math.sigmoid(self.p_logit)

        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1.0 - self.p)
        dropout_regularizer = self.p * tf.math.log(self.p)
        dropout_regularizer += (1.0 - self.p) * tf.math.log(1.0 - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)

        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs)), regularizer
        else:

            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs))

            return K.in_train_phase(relaxed_dropped_inputs, self.layer.call(inputs), training=training), regularizer


class DropConnect(layers.Layer):
    def __init__(self, p, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.p = p

    def build(self, input_shape):
        super(DropConnect, self).build(input_shape)

    def dropconnect(self, W):
        return tf.nn.dropout(W, rate=self.p) * self.p

    def dropconnect_hard(W, p):
        M_vector = tf.multinomial(tf.log([[1 - p, p]]), np.prod(W_shape))
        M = tf.reshape(M_vector, W_shape)
        M = tf.cast(M, tf.float32)
        return M * W

    def call(self, x, training=None):
        if training:
            return self.dropconnect(x)
        return x


class EmbeddingDropout(layers.Layer):
    # https://github.com/keras-team/keras/issues/7290
    # word_embeddings = SpatialDropout1D(0.50)(word_embeddings) # and possibly drop some dimensions on every single embedding (timestep)

    def __init__(self, p, **kwargs):
        super(EmbeddingDropout, self).__init__(**kwargs)
        self.p = p

    def build(self, input_shape):
        super(EmbeddingDropout, self).build(input_shape)

    def call(self, x, training=None):
        if training:
            return tf.nn.dropout(x, rate=self.p)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({'p': self.p})
        return config

    """
    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape
    """
