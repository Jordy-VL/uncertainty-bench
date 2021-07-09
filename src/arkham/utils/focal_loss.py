"""Multiclass focal loss implementation."""
#    __                          _     _
#   / _|                        | |   | |
#  | |_    ___     ___    __ _  | |   | |   ___    ___   ___
#  |  _|  / _ \   / __|  / _` | | |   | |  / _ \  / __| / __|
#  | |   | (_) | | (__  | (_| | | |   | | | (_) | \__ \ \__ \
#  |_|    \___/   \___|  \__,_| |_|   |_|  \___/  |___/ |___/

"""
Alternate source: 
https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss_adaptive_gamma.py
"""

import itertools
from typing import Any, Optional

from scipy.special import lambertw
import numpy as np

import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()


def shape_list(tensor: tf.Tensor):
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def mask_padding(y_true, y_pred, mask_value=-100):
    """
    Deal with padding in a clean way: mask both logits & labels
    """
    active_loss = tf.reshape(y_true, (-1,)) != mask_value
    reduced_logits = tf.boolean_mask(tf.reshape(y_pred, (-1, shape_list(y_pred)[-1])), active_loss)  # -2
    y_true = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
    return y_true, reduced_logits


def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1 - p) ** (1 - (1 - p) / (p * np.log(p))) / (p * np.log(p))) * np.log(1 - p)
    gamma_complex = (1 - p) / (p * np.log(p)) + lambertw(-y + 1e-12, k=-1) / np.log(1 - p)
    gamma = np.real(gamma_complex)  # gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma


# adaptive component
ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1


def get_gamma_list(pt, default_gamma=2):
    gamma_tensor = tf.ones_like(tf.shape(pt)) * default_gamma
    for key in sorted(gamma_dic.keys(), reverse=True):
        gamma_tensor = tf.where(
            tf.less_equal(pt, tf.cast(key, tf.float32)), tf.cast(gamma_dic[key], tf.int32), gamma_tensor
        )
    return tf.cast(gamma_tensor, dtype=tf.float32)

    """

    # make a boolean definition; here it assumes each pt_sample is 1 example
    # what if I flatten it all, then convert it back to batch_size?
    original_size = pt.shape 
    gamma_list = []

    flat_pt = tf.reshape(pt, [-1])

    for i in range(flat_pt.shape[0]):
        pt_sample = flat_pt[i]
        if pt_sample >= max(gamma_dic.keys()):
            gamma_list.append(default_gamma)
            continue
        # Choosing the [highest] gamma for the sample
        for key in sorted(gamma_dic.keys()):
            if pt_sample < key:
                gamma_list.append(gamma_dic[key])
                break
    gamma_tensor = tf.convert_to_tensor(gamma_list, dtype=tf.float32)
    return tf.reshape(gamma_tensor, original_size)
    """


def sparse_categorical_focal_loss(
    y_true,
    y_pred,
    gamma,
    *,
    class_weight: Optional[Any] = None,
    from_logits: bool = False,
    label_smoothing=0,
    mask=None,
    adaptive=False,
    axis: int = -1,
) -> tf.Tensor:
    r"""Focal loss function for multiclass classification with integer labels.

    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter called the *focusing parameter* that allows
    hard-to-classify examples to be penalized more heavily relative to
    easy-to-classify examples.

    See :meth:`~focal_loss.binary_focal_loss` for a description of the focal
    loss in the binary setting, as presented in the original work [1]_.

    In the multiclass setting, with integer labels :math:`y`, focal loss is
    defined as

    .. math::

        L(y, \hat{\mathbf{p}})
        = -\left(1 - \hat{p}_y\right)^\gamma \log(\hat{p}_y)

    where

    *   :math:`y \in \{0, \ldots, K - 1\}` is an integer class label (:math:`K`
        denotes the number of classes),
    *   :math:`\hat{\mathbf{p}} = (\hat{p}_0, \ldots, \hat{p}_{K-1})
        \in [0, 1]^K` is a vector representing an estimated probability
        distribution over the :math:`K` classes,
    *   :math:`\gamma` (gamma, not :math:`y`) is the *focusing parameter* that
        specifies how much higher-confidence correct predictions contribute to
        the overall loss (the higher the :math:`\gamma`, the higher the rate at
        which easy-to-classify examples are down-weighted).

    The usual multiclass softmax cross-entropy loss is recovered by setting
    :math:`\gamma = 0`.

    Parameters
    ----------
    y_true : tensor-like
        Integer class labels.

    y_pred : tensor-like
        Either probabilities or logits, depending on the `from_logits`
        parameter.

    gamma : float or tensor-like of shape (K,)
        The focusing parameter :math:`\gamma`. Higher values of `gamma` make
        easy-to-classify examples contribute less to the loss relative to
        hard-to-classify examples. Must be non-negative. This can be a
        one-dimensional tensor, in which case it specifies a focusing parameter
        for each class.

    class_weight: tensor-like of shape (K,)
        Weighting factor for each of the :math:`k` classes. If not specified,
        then all classes are weighted equally.

    from_logits : bool, optional
        Whether `y_pred` contains logits or probabilities.

    axis : int, optional
        Channel axis in the `y_pred` tensor.

    Returns
    -------
    :class:`tf.Tensor`
        The focal loss for each example.

    Examples
    --------

    This function computes the per-example focal loss between a one-dimensional
    integer label vector and a two-dimensional prediction matrix:

    >>> import numpy as np
    >>> from focal_loss import sparse_categorical_focal_loss
    >>> y_true = [0, 1, 2]
    >>> y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
    >>> loss = sparse_categorical_focal_loss(y_true, y_pred, gamma=2)
    >>> np.set_printoptions(precision=3)
    >>> print(loss.numpy())
    [0.009 0.032 0.082]

    Warnings
    --------
    This function does not reduce its output to a scalar, so it cannot be passed
    to :meth:`tf.keras.Model.compile` as a `loss` argument. Instead, use the
    wrapper class :class:`~focal_loss.SparseCategoricalFocalLoss`.

    References
    ----------
    .. [1] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár. Focal loss for
        dense object detection. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 2018.
        (`DOI <https://doi.org/10.1109/TPAMI.2018.2858826>`__)
        (`arXiv preprint <https://arxiv.org/abs/1708.02002>`__)

    See Also
    --------
    :meth:`~focal_loss.SparseCategoricalFocalLoss`
        A wrapper around this function that makes it a
        :class:`tf.keras.losses.Loss`.
    """
    # Process focusing parameter
    if gamma == "adaptive":
        scalar_gamma = True  # not true but tensor to be created
    else:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
        gamma_rank = gamma.shape.rank
        scalar_gamma = gamma_rank == 0

    # Process class weight
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight, dtype=tf.dtypes.float32)

    # Process prediction tensor
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred_rank = y_pred.shape.rank
    if y_pred_rank is not None:
        axis %= y_pred_rank
        if axis != y_pred_rank - 1:
            # Put channel axis last for sparse_softmax_cross_entropy_with_logits
            perm = list(itertools.chain(range(axis), range(axis + 1, y_pred_rank), [axis]))
            y_pred = tf.transpose(y_pred, perm=perm)
    elif axis != -1:
        raise ValueError(
            f'Cannot compute sparse categorical focal loss with axis={axis} on '
            'a prediction tensor with statically unknown rank.'
        )
    y_pred_shape = tf.shape(y_pred)

    # Process ground truth tensor
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank

    if y_true_rank is None:
        raise NotImplementedError(
            'Sparse categorical focal loss not supported ' 'for target/label tensors of unknown rank'
        )

    reshape_needed = y_true_rank is not None and y_pred_rank is not None and y_pred_rank != y_true_rank + 1
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if mask is not None:
        y_true, y_pred = mask_padding(y_true, y_pred, mask_value=mask)

    if from_logits:
        logits = y_pred
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    if label_smoothing:
        num_classes = shape_list(y_pred)[-1]  # make onehot for label smoothing; default!
        y_true_onehot = tf.one_hot(y_true, num_classes)
        y_true_smooth = tf.math.add(
            tf.math.multiply(y_true_onehot, 1 - label_smoothing), (label_smoothing / num_classes)
        )
        xent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_smooth, logits=logits)
    else:
        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)
    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)

    if not tf.is_tensor(gamma):
        gamma = get_gamma_list(probs, default_gamma=2)

    focal_modulation = (1 - probs) ** gamma
    loss = focal_modulation * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0, batch_dims=y_true_rank)
        loss *= class_weight

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    r"""Focal loss function for multiclass classification with integer labels.

    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter :math:`\gamma` (gamma), called the
    *focusing parameter*, that allows hard-to-classify examples to be penalized
    more heavily relative to easy-to-classify examples.

    This class is a wrapper around
    :class:`~focal_loss.sparse_categorical_focal_loss`. See the documentation
    there for details about this loss function.

    Parameters
    ----------
    gamma : float or tensor-like of shape (K,)
        The focusing parameter :math:`\gamma`. Higher values of `gamma` make
        easy-to-classify examples contribute less to the loss relative to
        hard-to-classify examples. Must be non-negative. This can be a
        one-dimensional tensor, in which case it specifies a focusing parameter
        for each class.

    class_weight: tensor-like of shape (K,)
        Weighting factor for each of the :math:`k` classes. If not specified,
        then all classes are weighted equally.

    from_logits : bool, optional
        Whether model prediction will be logits or probabilities.

    **kwargs : keyword arguments
        Other keyword arguments for :class:`tf.keras.losses.Loss` (e.g., `name`
        or `reduction`).

    Examples
    --------

    An instance of this class is a callable that takes a rank-one tensor of
    integer class labels `y_true` and a tensor of model predictions `y_pred` and
    returns a scalar tensor obtained by reducing the per-example focal loss (the
    default reduction is a batch-wise average).

    >>> from focal_loss import SparseCategoricalFocalLoss
    >>> loss_func = SparseCategoricalFocalLoss(gamma=2)
    >>> y_true = [0, 1, 2]
    >>> y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
    >>> loss_func(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.040919524>

    Use this class in the :mod:`tf.keras` API like any other multiclass
    classification loss function class that accepts integer labels found in
    :mod:`tf.keras.losses` (e.g.,
    :class:`tf.keras.losses.SparseCategoricalCrossentropy`:

    .. code-block:: python

        # Typical usage
        model = tf.keras.Model(...)
        model.compile(
            optimizer=...,
            loss=SparseCategoricalFocalLoss(gamma=2),  # Used here like a tf.keras loss
            metrics=...,
        )
        history = model.fit(...)

    See Also
    --------
    :meth:`~focal_loss.sparse_categorical_focal_loss`
        The function that performs the focal loss computation, taking a label
        tensor and a prediction tensor and outputting a loss.
    """

    def __init__(
        self,
        gamma,
        class_weight: Optional[Any] = None,
        from_logits: bool = False,
        label_smoothing=0,
        mask=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.mask = mask

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary containing the configuration of a
        layer. The same layer can be re-instantiated later (without its trained
        weights) from this configuration.

        Returns
        -------
        dict
            This layer's config.
        """
        config = super().get_config()
        config.update(
            gamma=self.gamma,
            class_weight=self.class_weight,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            mask=self.mask,
        )
        return config

    def call(self, y_true, y_pred):
        """Compute the per-example focal loss.

        Parameters
        ----------
        y_true : tensor-like, shape (N,)
            Integer class labels.

        y_pred : tensor-like, shape (N, K)
            Either probabilities or logits, depending on the `from_logits`
            parameter.

        Returns
        -------
        :class:`tf.Tensor`
            The per-example focal loss. Reduction to a scalar is handled by
            this layer's
            :meth:`~focal_loss.SparseCateogiricalFocalLoss.__call__` method.
        """
        return sparse_categorical_focal_loss(
            y_true=y_true,
            y_pred=y_pred,
            class_weight=self.class_weight,
            gamma=self.gamma,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            mask=self.mask,
        )
