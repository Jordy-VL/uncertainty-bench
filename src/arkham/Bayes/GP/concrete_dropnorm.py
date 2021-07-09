# coding=utf-8
# Copyright 2021 Jordy Van Landeghem

"""Spectral Normalized Concrete Dropout

## References:

[1] Yuichi Yoshida, Takeru Miyato. Spectral Norm Regularization for Improving
    the Generalizability of Deep Learning.
    _arXiv preprint arXiv:1705.10941_, 2017. https://arxiv.org/abs/1705.10941

[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.
    Spectral normalization for generative adversarial networks.
    In _International Conference on Learning Representations_, 2018.

[3] Henry Gouk, Eibe Frank, Bernhard Pfahringer, Michael Cree.
    Regularisation of neural networks by enforcing lipschitz continuity.
    _arXiv preprint arXiv:1804.04368_, 2018. https://arxiv.org/abs/1804.04368

[4] Gal, Yarin, Jiri Hron, and Alex Kendall. "Concrete dropout." arXiv:1705.07832 2017.
#CD adapted from https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

tf.random.set_seed(42)


class SpectralNormalizationConcreteDropoutConv2D(tf.keras.layers.Wrapper):
    """Implements spectral normalization for Conv2D layer based on [3] and concrete dropout based on ."""

    def __init__(
        self,
        layer,
        iteration=1,
        norm_multiplier=0.95,
        training=True,
        aggregation=tf.VariableAggregation.MEAN,
        weight_regularizer=1e-6,
        dropout_regularizer=1e-5,
        init_min=0.1,
        init_max=0.1,
        is_mc_dropout=True,
        **kwargs,
    ):
        """Initializer.

        Args:
          layer: (tf.keras.layers.Layer) A TF Keras layer to apply normalization to.
          iteration: (int) The number of power iteration to perform to estimate
            weight matrix's singular value.
          norm_multiplier: (float) Multiplicative constant to threshold the
            normalization. Usually under normalization, the singular value will
            converge to this value.
          training: (bool) Whether to perform power iteration to update the singular
            value estimate.
          aggregation: (tf.VariableAggregation) Indicates how a distributed variable
            will be aggregated. Accepted values are constants defined in the class
            tf.VariableAggregation.

          **kwargs: (dict) Other keyword arguments for the layers.Wrapper class.
        """
        # SN params
        self.norm_multiplier = norm_multiplier
        self.iteration = iteration
        self.do_power_iteration = training
        self.aggregation = aggregation

        # CD params
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = np.log(init_min) - np.log(1.0 - init_min)
        self.init_max = np.log(init_max) - np.log(1.0 - init_max)
        self.data_format = layer.data_format
        self.input_dim = None

        # Set layer attributes.
        layer._name += '_spec_norm_CD'

        # Checks
        if not isinstance(layer, tf.keras.layers.Conv2D):
            raise ValueError(
                'layer must be a `tf.keras.layer.Conv2D` instance. You passed: {input}'.format(input=layer)
            )
        assert 'kernel_regularizer' not in kwargs
        super(SpectralNormalizationConcreteDropoutConv2D, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        # self.init_shape = tf.keras.layers.InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(SpectralNormalizationConcreteDropoutConv2D, self).build()

        self.layer.kernel._aggregation = self.aggregation  # pylint: disable=protected-access
        self._dtype = self.layer.kernel.dtype

        # Shape (kernel_size_1, kernel_size_2, in_channel, out_channel).
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.strides = self.layer.strides

        # Set the dimensions of u and v vectors.
        batch_size = input_shape[0]
        uv_dim = 1

        # Resolve shapes.
        in_height = input_shape[1]
        in_width = input_shape[2]
        in_channel = self.w_shape[2]

        out_height = in_height // self.strides[0]
        out_width = in_width // self.strides[1]
        out_channel = self.w_shape[3]

        self.in_shape = (uv_dim, in_height, in_width, in_channel)
        self.out_shape = (uv_dim, out_height, out_width, out_channel)

        self.uv_initializer = tf.initializers.random_normal()

        self.v = self.add_weight(
            shape=self.in_shape,
            initializer=self.uv_initializer,
            trainable=False,
            name='v',
            dtype=self.dtype,
            aggregation=self.aggregation,
        )

        self.u = self.add_weight(
            shape=self.out_shape,
            initializer=self.uv_initializer,
            trainable=False,
            name='u',
            dtype=self.dtype,
            aggregation=self.aggregation,
        )

        self.p_logit = self.add_weight(
            name='p_logit',
            shape=(1,),
            initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
            trainable=True,
        )
        self.p = tf.math.sigmoid(self.p_logit)

        # initialise regulariser / prior KL term
        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[3]
        # self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def update_weights(self):
        """Computes power iteration for convolutional filters based on [3]."""
        # Initialize u, v vectors.
        u_hat = self.u
        v_hat = self.v

        # shape update?
        # in_shape update

        if self.do_power_iteration:
            for _ in range(self.iteration):
                # Updates v.
                v_ = tf.nn.conv2d_transpose(
                    u_hat, self.w, output_shape=self.in_shape, strides=self.strides, padding='SAME'
                )
                v_hat = tf.nn.l2_normalize(tf.reshape(v_, [1, -1]))
                v_hat = tf.reshape(v_hat, v_.shape)

                # Updates u.
                u_ = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding='SAME')
                u_hat = tf.nn.l2_normalize(tf.reshape(u_, [1, -1]))
                u_hat = tf.reshape(u_hat, u_.shape)

        v_w_hat = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding='SAME')

        sigma = tf.matmul(tf.reshape(v_w_hat, [1, -1]), tf.reshape(u_hat, [-1, 1]))
        # Convert sigma from a 1x1 matrix to a scalar.
        sigma = tf.reshape(sigma, [])

        u_update_op = self.u.assign(u_hat)
        v_update_op = self.v.assign(v_hat)

        w_norm = tf.cond(
            (self.norm_multiplier / sigma) < 1,
            lambda: (self.norm_multiplier / sigma) * self.w,  # pylint:disable=g-long-lambda
            lambda: self.w,
        )

        w_update_op = self.layer.kernel.assign(w_norm)

        return u_update_op, v_update_op, w_update_op

    def restore_weights(self):
        """Restores layer weights to maintain gradient update (See Alg 1 of [1])."""
        return self.layer.kernel.assign(self.w)

    def spatial_concrete_dropout(self, x):
        """
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """
        eps = K.cast_to_floatx(K.epsilon())
        temp = 2.0 / 3.0  # HARDCODED value from original implementation

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

    def concrete_call(self, inputs, training=None):
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
            if training:
                return self.layer.call(self.spatial_concrete_dropout(inputs)), regularizer
            else:
                return self.layer.call(inputs), regularizer

    def call(self, inputs, training=None):
        u_update_op, v_update_op, w_update_op = self.update_weights()
        output, regularizer = self.concrete_call(inputs, training=training)  # what is the correct order here?
        w_restore_op = self.restore_weights()

        # Register update ops.
        self.add_update(u_update_op)
        self.add_update(v_update_op)
        self.add_update(w_update_op)
        self.add_update(w_restore_op)
        return output, regularizer


if __name__ == '__main__':
    l = SpectralNormalizationConcreteDropoutConv2D(tf.keras.layers.Conv2D(5, (10, 1), data_format="channels_last"))

    x = K.random_uniform((32, 150, 1, 100))

    so = l(x)
    print(so)
    print(l.get_config())
