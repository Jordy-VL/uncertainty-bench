"""SNGP
Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.
## References:
[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Ashish Vaswani et al. Attention Is All You Need.
     _Neural Information Processing System_, 2017.
     https://papers.nips.cc/paper/7181-attention-is-all-you-need
"""


from copy import deepcopy
import numpy as np
import tensorflow as tf

import edward2 as ed
from functools import partial

spec_norm_kwargs = {"iteration": 1, "norm_multiplier": 0.95, "inhere_layer_name": True}

conv_norm_kwargs = {"iteration": 1, "norm_multiplier": 4, "inhere_layer_name": True}

gp_kwargs = dict(
    # units=num_labels #will be set from model
    num_inducing=1024,
    scale_random_features=False,
    use_custom_random_features=True,
    kernel_initializer='glorot_uniform',
    custom_random_features_initializer=(tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
    gp_cov_momentum=-1,
    # scale_random_features=True,
    # normalize_input=False,
    # gp_cov_ridge_penalty=1.,
    # l2_regularization=1e-6,
)


class ResetCovarianceCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        """Resets covariance matrix at the begining of the epoch."""
        if epoch > 0:
            self.model.classifier.reset_covariance_matrix()


def mean_field_logits(logits, covariance_matrix=None, mean_field_factor=1.0):
    """Adjust the model logits so its softmax approximates the posterior mean [1].
    [1]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
         Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
         https://arxiv.org/abs/2006.07584
    Arguments:
      logits: A float tensor of shape (batch_size, num_classes).
      covariance_matrix: The covariance matrix of shape (batch_size, batch_size).
        If None then it assumes the covariance_matrix is an identity matrix.
      mean_field_factor: The scale factor for mean-field approximation, used to
        adjust the influence of posterior variance in posterior mean
        approximation. If covariance_matrix=None then it is used as the
        temperature parameter for temperature scaling.
    Returns:
      Tensor of adjusted logits, shape (batch_size, num_classes).
    """
    if mean_field_factor is None or mean_field_factor < 0:
        return logits

    # Compute standard deviation.
    if covariance_matrix is None:
        variances = 1.0
    else:
        variances = tf.linalg.diag_part(covariance_matrix)

    # Compute scaling coefficient for mean-field approximation.
    logits_scale = tf.sqrt(1.0 + variances * mean_field_factor)

    if len(logits.shape) > 1:
        # Cast logits_scale to compatible dimension.
        logits_scale = tf.expand_dims(logits_scale, axis=-1)

    return logits / logits_scale


def compute_posterior_mean_probability(logits, covmat, lambda_param=np.pi / 8.0, softmax=True):
    # Computes uncertainty-adjusted logits using the built-in method.
    # As an alternative to expensive monte carlo sampling
    logits_adjusted = mean_field_logits(logits, covmat, mean_field_factor=lambda_param)
    if softmax:
        probs = tf.nn.softmax(logits_adjusted, axis=-1)
    """
    else:
        probs = tf.nn.sigmoid(logits_adjusted)
    if probs.shape[-1] == 2:
        probs = probs[:, 0]
    """
    return probs


class SNGP(tf.keras.Model):
    def __init__(self, model, spec_norm_kwargs=None, gp_kwargs=None, use_gp_layer=True, **kwargs):
        super().__init__()
        self.spec_norm_kwargs = spec_norm_kwargs
        self.gp_kwargs = gp_kwargs
        self.use_gp_layer = use_gp_layer
        self.parse_model(model)  # featurizer
        self.build(model.layers[0].input_shape)

    def parse_model(self, model):
        num_labels = model.output_shape[-1]
        self.gp_kwargs["units"] = num_labels
        denser = partial(ed.layers.SpectralNormalization, **self.spec_norm_kwargs)
        conv_norm_kwargs = deepcopy(self.spec_norm_kwargs)
        conv_norm_kwargs["norm_multiplier"] = 1  # could find a "c" larger than 1 if lipschitz bound can be respected
        conver = partial(ed.layers.SpectralNormalization, **conv_norm_kwargs)
        gp_layer = ed.layers.RandomFeatureGaussianProcess(**self.gp_kwargs)

        def wrap_layer(model):
            is_final_layer = lambda l_i, L: True if l_i == L else False
            batch_shape = model.layers[0].input_shape
            if isinstance(batch_shape, list):
                batch_shape = batch_shape[0]
            new_input = tf.keras.layers.Input(batch_shape=batch_shape)
            prev_layer = new_input
            for i, layer in enumerate(model.layers):
                current_layer = layer
                if is_final_layer(i, len(model.layers) - 1):
                    continue
                if "dense" in layer.name:
                    current_layer = denser(layer)
                if "conv" in layer.name:
                    current_layer = conver(layer)
                prev_layer = current_layer(prev_layer)
            return tf.keras.Model(inputs=new_input, outputs=prev_layer, name='SPNN')

        self.submodel = wrap_layer(model)
        if self.use_gp_layer:
            self.classifier = gp_layer
        else:
            self.classifier = model.layers[-1]

    def fit(self, *args, **kwargs):
        """Adds ResetCovarianceCallback to model callbacks."""
        kwargs["callbacks"] = list(kwargs.get("callbacks", []))
        if self.use_gp_layer:
            kwargs["callbacks"].append(ResetCovarianceCallback())

        return super().fit(*args, **kwargs)

    def call(self, inputs, training=False, return_covmat=False, mean_field=False):
        penultimate = self.submodel.call(inputs)

        if self.use_gp_layer:
            # Gets logits and covariance matrix from GP layer.

            logits, covmat = self.classifier(penultimate)

            if mean_field:
                probs = compute_posterior_mean_probability(logits, covmat)
                return probs

            if not training and return_covmat:
                return logits, covmat

        else:
            logits = self.classifier(penultimate)

        # Returns only logits during training.
        return logits


# base_model = simple_CNN(input_shape=(28, 28, 1))

## Deterministic
# base_model.summary()

## Spectral normalized
# new_model = SNGP(base_model, spec_norm_kwargs=spec_norm_kwargs, gp_kwargs=gp_kwargs, use_gp_layer=False)
# new_model.submodel.summary()
# new_model.summary()


def penultimate_layer(model):
    return [i for i, l in enumerate(model.layers) if "dense" in l.name][-1] - 1


class SNGP_wrapper(tf.keras.Model):
    # Added spectral norm; use this to instantiate final layer
    """
    Assume final layer is Dense(nb_classes, activation=[sigmoid/softmax])
    """

    def __init__(self, model, use_gp_layer=True):  # gp_kwargs=None,
        super().__init__(name="SNGP_wrapper")
        # POP last layer!
        self.submodel = tf.keras.Model(inputs=model.input, outputs=model.layers[penultimate_layer(model)].output)
        self.gp_kwargs = gp_kwargs
        nb_classes = model.output_shape[-1]
        self.gp_kwargs["units"] = nb_classes
        self.use_gp_layer = use_gp_layer
        if self.use_gp_layer:
            self.classifier = ed.layers.RandomFeatureGaussianProcess(**self.gp_kwargs)
        else:
            self.classifier = model.layers[-1]
        self.build(model.layers[0].input_shape)

    def fit(self, *args, **kwargs):
        """Adds ResetCovarianceCallback to model callbacks."""
        kwargs["callbacks"] = list(kwargs.get("callbacks", []))
        if self.use_gp_layer:
            kwargs["callbacks"].append(ResetCovarianceCallback())

        return super().fit(*args, **kwargs)

    def call(self, inputs, training=False, return_covmat=False, mean_field=False):
        penultimate = self.submodel(inputs)  # .call

        if self.use_gp_layer:
            # Gets logits and covariance matrix from GP layer.

            logits, covmat = self.classifier(penultimate)

            if mean_field:
                probs = compute_posterior_mean_probability(logits, covmat)
                return probs

            if not training and return_covmat:
                return logits, covmat

        else:
            logits = self.classifier(penultimate)

        # Returns only logits during training.
        return logits

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        )
    )
    def call_covmat(self, inputs, training=False, mean_field=False, calc_variance=True):
        penultimate = self.submodel(inputs, training=training)
        logits, covmat = self.classifier(penultimate)
        if mean_field:
            probs = compute_posterior_mean_probability(logits, covmat, softmax=True)
        else:
            probs = tf.constant(0, dtype=tf.float32)
        if calc_variance:
            covmat = tf.linalg.diag_part(covmat)[:, None]  # SNGP variance
        return logits, covmat, probs

    def get_config(self):
        return {"model": self.submodel, "use_gp_layer": self.use_gp_layer}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def test_SNGP():
    def simple_CNN(input_shape=(28, 28, 1)):
        num_labels = 10
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_labels, activation="softmax"),
            ]
        )
        return model

    base_model = simple_CNN()
    new_model = SNGP(base_model, spec_norm_kwargs=spec_norm_kwargs, gp_kwargs=gp_kwargs, use_gp_layer=True)
    new_model.submodel.summary()
    new_model.summary()
    new_model.classifier


if __name__ == '__main__':
    test_SNGP()
"""
if hasattr(model, "submodel"):
    sngp_logits, sngp_covmat = model(test_images, return_covmat=True, mean_field=False)
    sngp_variance = tf.linalg.diag_part(sngp_covmat)[:, None]
    return model, sngp_logits, sngp_covmat, test_labels
else:
    probs = model(test_images)
    return model, probs, None, test_labels
"""
