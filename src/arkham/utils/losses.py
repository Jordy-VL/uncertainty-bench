import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import Any, Optional
import numpy as np
import tensorflow as tf

from arkham.utils.focal_loss import sparse_categorical_focal_loss


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


def logsumexp(a):  # sort of normalize by max
    a_max = tf.reduce_max(a, axis=0)
    return tf.math.log(tf.reduce_sum(tf.exp(a - a_max), axis=0)) + a_max


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def attenuated_learned_loss(y_true, y_pred, T=10):
    """
    Attenuated loss for classification described in https://arxiv.org/pdf/1703.04977.pdf

    Args:
        targets: target labels of size (batch_size [ x num_classes)]
        y_pred: sampled distribution of logits resulted from passing
            batch of data through the model. size (sample_size x batch_size [ x num_classes)]
        (sample_size: number of samples used)

    """
    sampled = y_pred.sample(T)
    sampled_loss = tf.stack(
        [tf.nn.softmax_cross_entropy_with_logits(y_true, sampled[t]) for t in range(T)]
    )  # T x batch_size x maxlen (x labels)
    batch_losses = logsumexp(-sampled_loss)
    likelihood_loss = -tf.reduce_mean(batch_losses) + tf.math.log(tf.cast(T, "float32"))
    return likelihood_loss


def attenuated_learned_loss_multilabel(y_true, y_pred, T=10):
    """
    Attenuated loss for classification described in https://arxiv.org/pdf/1703.04977.pdf

    Args:
        targets: target labels of size (batch_size [ x num_classes)]
        y_pred: sampled distribution of logits resulted from passing
            batch of data through the model. size (sample_size x batch_size [ x num_classes)]
        (sample_size: number of samples used)

    """
    sampled = y_pred.sample(T)
    sampled_loss = tf.stack(
        [tf.nn.sigmoid_cross_entropy_with_logits(y_true, sampled[t]) for t in range(T)]
    )  # T x batch_size x maxlen (x labels)
    batch_losses = logsumexp(-sampled_loss)
    likelihood_loss = -tf.reduce_mean(batch_losses) + tf.math.log(tf.cast(T, "float32"))
    return likelihood_loss


def original_attenuated_learned_loss(y_true, y_pred, T=10):
    """
    Attenuated loss for classification described in https://arxiv.org/pdf/1703.04977.pdf

    Args:
        targets: target labels of size (batch_size [ x num_classes)]
        y_pred: sampled distribution of logits resulted from passing
            batch of data through the model. size (sample_size x batch_size [ x num_classes)]
        (sample_size: number of samples used)

    """
    sampled = y_pred.sample(T)
    sampled_loss = tf.stack(
        [tf.nn.softmax_cross_entropy_with_logits(y_true, sampled[t]) for t in range(T)]
    )  # T x batch_size x maxlen (x labels)
    batch_losses = logsumexp(-sampled_loss)
    likelihood_loss = -tf.reduce_mean(batch_losses) + tf.math.log(tf.cast(T, "float32"))
    return likelihood_loss


def sequential_attenuated_learned_loss(y_true, y_pred, T=10):
    """
    Same as above, yet applied to sequential level with sparse entropy
    """

    # print(f"before: pred = {y_pred.shape} true = {y_true.shape}")
    sampled = y_pred.sample(T)
    print(f"before: pred = {sampled.shape} true = {y_true.shape}")
    print(y_true)
    print(sampled[0].shape)
    try:
        sampled_loss = tf.stack(
            [tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, sampled[t]) for t in range(T)]
        )  # T x batch_size x maxlen (x labels)
    except Exception as e:
        print(e, "just for compilation purposes, let it fail")
        sampled_loss = tf.stack(
            [tf.nn.softmax_cross_entropy_with_logits(y_true, sampled[t]) for t in range(T)]
        )  # T x batch_size x maxlen (x labels)
    batch_likelihood_loss = tf.reduce_mean(logsumexp(-sampled_loss), axis=-1)  # reduces to batch_size
    loss = -tf.reduce_mean(batch_likelihood_loss) + tf.math.log(tf.cast(T, "float32"))
    return loss


@tf.keras.utils.register_keras_serializable()
class SparseMaskCatCELoss(tf.keras.losses.Loss):
    def __init__(
        self, label_smoothing=0, mask=-100, class_weight: Optional[Any] = None, from_logits: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.class_weight = class_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.mask = mask

    def get_config(self):
        config = super().get_config()
        config.update(
            class_weight=self.class_weight,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            mask=self.mask,
        )
        return config

    def call(self, y_true, y_pred):
        return sparse_categorical_focal_loss(
            y_true=y_true,
            y_pred=y_pred,
            class_weight=self.class_weight,
            gamma=0,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            mask=self.mask,
        )


def sparse_crossentropy_masked_v2(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False  # , reduction=tf.keras.losses.Reduction.NONE
    )
    y_true, reduced_logits = mask_padding(y_true, y_pred, mask_value=-100)
    return loss_fn(y_true, reduced_logits)


def sparse_categorical_accuracy_masked(y_true, y_pred):
    y_true, reduced_logits = mask_padding(y_true, y_pred)
    reduced_logits = tf.cast(tf.argmax(reduced_logits, axis=-1), tf.keras.backend.floatx())
    equality = tf.equal(y_true, reduced_logits)
    return tf.reduce_mean(tf.cast(equality, tf.keras.backend.floatx()))


def sparse_crossentropy_masked(y_true, y_pred):
    mask_value = -100
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, mask_value))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, mask_value))
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))


@tf.keras.utils.register_keras_serializable()
class AdaptiveDiceLoss(tf.keras.losses.Loss):
    r"""Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Math Function:
        https://arxiv.org/abs/1911.02855.pdf
        adaptive_dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} (1 - p_i) ** alpha * p_i * y_i + smooth
            denominator = \sum_{1}^{t} (1 - p_i) ** alpha * p_i + \sum_{1} ^{t} y_i + smooth

    Source: 
        https://github.com/ShannonAI/mrc-for-flat-nested-ner/issues/59
        https://github.com/fursovia/self-adj-dice 

    """

    def __init__(
        self,
        gamma=1e-8,
        alpha=0.1,
        label_smoothing=0,
        mask=-100,
        linear_interpolation=0,
        class_weight: Optional[Any] = None,
        from_logits: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.class_weight = class_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.mask = mask
        self.linear_interpolation = linear_interpolation

    def get_config(self):
        config = super().get_config()
        config.update(
            gamma=self.gamma,
            alpha=self.alpha,
            linear_interpolation=self.linear_interpolation,
            class_weight=self.class_weight,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            mask=self.mask,
        )
        return config

    def dice_coef(y_true, y_pred):
        smooth = 1
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        t = y_true_f * y_pred_f
        intersection = tf.reduce_sum(t)
        return (2 * intersection + smooth) / (
            tf.reduce_sum(y_true_f * y_true_f) + tf.reduce_sum(y_pred_f * y_pred_f) + smooth
        )

    def call(self, y_true, y_pred):
        if self.mask:
            y_true, y_pred = mask_padding(y_true, y_pred, mask_value=self.mask)  # flattens batch

        if y_pred.shape.rank == 3:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])

        probs = tf.gather_nd(
            y_pred, tf.cast(tf.expand_dims(y_true, 1), tf.int32), batch_dims=1
        )  # probs of correct classes; size N
        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (
            probs_with_factor + 1 + self.gamma
        )  # instead of 1; shape of flattened y?

        # intersection = torch.sum((1-flat_input)**self.alpha * flat_input * flat_target, -1) + self.smooth
        # denominator = torch.sum((1-flat_input)**self.alpha * flat_input) + flat_target.sum() + self.smooth
        # return 1 - 2 * intersection / denominator

        if self.linear_interpolation:
            if self.label_smoothing:
                num_classes = shape_list(y_pred)[-1]  # make onehot for label smoothing; default!
                y_true_onehot = tf.one_hot(y_true, num_classes)
                y_true_smooth = tf.math.add(
                    tf.math.multiply(y_true_onehot, 1 - label_smoothing), (label_smoothing / num_classes)
                )
                xent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_smooth, logits=logits)
            else:
                xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
            lamb_dice = self.linear_interpolation[0]
            lamb_CE = self.linear_interpolation[1]
            loss = (lamb_dice * loss) + (lamb_CE * xent_loss)

        return tf.reduce_mean(loss)

def deduce_loss(loss_fn, n_classes, multilabel, use_aleatorics, **kwargs):
    # https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
    loss = "categorical_crossentropy" if not multilabel else "binary_crossentropy"

    # Sequence losses
    if loss_fn == "sparse_categorical_crossentropy_0":
        loss = SparseMaskCatCELoss(label_smoothing=kwargs.get("label_smoothing", 0), mask=0)

    elif loss_fn == "sparse_crossentropy_masked_v2":
        return sparse_crossentropy_masked_v2

    elif "crossentropy" in loss_fn:
        loss_temp = (
            tf.keras.losses.SparseCategoricalCrossentropy
            if "sparse" in loss_fn
            else tf.keras.losses.CategoricalCrossentropy
        )
        if multilabel:
            loss_temp = tf.keras.losses.BinaryCrossentropy
        from_logits = True if "logits" in loss_fn else False
        return loss_temp(from_logits=from_logits)

    # Heteroscedastic losses
    elif loss_fn == "attenuated_learned_loss":
        loss = attenuated_learned_loss  # posterior = 100?
    elif loss_fn == "alternate_attenuated_learned_loss":
        loss = alternate_attenuated_learned_loss
    elif loss_fn == "sequential_attenuated_learned_loss":
        loss = sequential_attenuated_learned_loss
    elif loss_fn == "attenuated_learned_loss_multilabel":
        loss = attenuated_learned_loss_multilabel
    # Focal losses
    elif loss_fn == "sparse_focal_loss":
        from arkham.utils.focal_loss import SparseCategoricalFocalLoss

        loss = SparseCategoricalFocalLoss(
            gamma=kwargs.get("gamma", 2),
            class_weight=None,
            from_logits=False,
            label_smoothing=kwargs.get("label_smoothing", 0),
            mask=kwargs.get("pad_value", -100),
        )  # default try to mask

    elif loss_fn == "dice_loss":
        loss = AdaptiveDiceLoss(
            gamma=1e-8,
            alpha=0.1,
            class_weight=None,  # alpha
            from_logits=False,
            label_smoothing=kwargs.get("label_smoothing", 0),
            mask=kwargs.get("pad_value", -100),
        )  # default try to mask

    elif loss_fn == "SparseMaskCatCELoss":
        loss = SparseMaskCatCELoss(
            label_smoothing=kwargs.get("label_smoothing", 0), mask=kwargs.get("pad_value", -100)
        )  # default try to mask

    # Assymmetric losses
    elif loss_fn == "asymmetric_loss":
        loss = AsymmetricLossTF_full(gamma_pos=1, gamma_neg=2, clip=0, eps=1e-8)
    elif loss_fn == "asymmetric_loss_single":
        loss = ASLSingleLabelTF_full(gamma_pos=1, gamma_neg=2, eps=0, reduction="mean")

    if loss_fn != loss:
        print(f"***WARNING: loss_fn {loss_fn} changed to {loss}***")
    return loss


def ASLSingleLabelTF_full(gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean"):  # ASLSingleLabelTF
    # https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0

    def internal_loss(y_true, y_pred):
        log_preds = tf.nn.log_softmax(y_pred, axis=-1)  # like softmax

        """
        => Categorical encoding for labels; can assume this is already done
        """
        # scatter_(dim, index, src)
        # tensor_scatter_add(tf.zeros(shape, values.dtype), indices, values)
        # tf.scatter_nd(indices, updates, shape)
        # ensor tells us that parameters include the dim, index tensor, and the source tensor.
        # Scatter updates into a new tensor according to indices.
        # tf.zeros_like(y_pred)
        # tf.expand_dims(tf.cast(y_true, tf.int64), 1)
        # targets_classes = tf.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        # targets = targets_classes

        # ASL weights
        anti_targets = 1 - y_true
        xs_pos = tf.math.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * y_true
        xs_neg = xs_neg * anti_targets

        asymmetric_w = tf.math.pow(1 - xs_pos - xs_neg, gamma_pos * y_true + gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if eps > 0:  # label smoothing
            num_classes = y_true.shape[-1]
            if num_classes:
                y_true = tf.math.add(tf.math.multiply(y_true, 1 - eps), (eps / num_classes))

        # loss calculation
        loss = -tf.math.multiply(y_true, log_preds)
        loss = tf.math.reduce_sum(loss, axis=-1)

        if reduction == "mean":
            loss = tf.math.reduce_mean(loss)
        return loss

        internal_loss.__name__ = "asymmetric_loss_single"

    return internal_loss


def ASLSingleLabelTF(y_true, y_pred, gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean"):
    log_preds = tf.nn.log_softmax(y_pred, axis=-1)  # like softmax

    # ASL weights
    anti_targets = 1 - y_true
    xs_pos = tf.math.exp(log_preds)
    xs_neg = 1 - xs_pos
    xs_pos = xs_pos * y_true
    xs_neg = xs_neg * anti_targets

    asymmetric_w = tf.math.pow(1 - xs_pos - xs_neg, gamma_pos * y_true + gamma_neg * anti_targets)
    log_preds = log_preds * asymmetric_w

    if eps > 0:  # label smoothing
        num_classes = y_true.shape[-1]
        y_true = tf.math.add(tf.math.multiply(y_true, 1 - eps), (eps / num_classes))

    # loss calculation
    loss = tf.math.multiply(y_true, log_preds)
    loss = tf.math.reduce_sum(loss, axis=-1)

    if reduction == "mean":
        loss = tf.math.reduce_mean(loss)
    return loss


def AsymmetricLossTF(y_true, y_pred, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
    """"
    Parameters
    ----------
    y_pred: input logits
    y: targets (multi-label binarized vector)
    """
    # Calculating Probabilities
    xs_pos = y_pred
    xs_neg = 1 - xs_pos

    # Asymmetric Clipping
    if clip is not None and clip > 0:
        xs_neg = tf.clip_by_value((xs_neg + clip), clip_value_min=eps, clip_value_max=1)

    # Basic CE calculation
    los_pos = y_true * tf.math.log(tf.clip_by_value(xs_pos, clip_value_min=eps, clip_value_max=float('inf')))
    los_neg = (1 - y_true) * tf.math.log(tf.clip_by_value(xs_neg, clip_value_min=eps, clip_value_max=float('inf')))
    loss = los_pos + los_neg

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = xs_pos * y_true
        pt1 = xs_neg * (1 - y_true)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
        one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w

    return tf.math.reduce_sum(loss)


def AsymmetricLossTF_full(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
    def internal_loss(y_true, y_pred):
        """"
        Parameters
        ----------
        y_pred: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        xs_pos = y_pred
        xs_neg = 1 - xs_pos

        # Asymmetric Clipping
        if clip is not None and clip > 0:
            xs_neg = tf.clip_by_value((xs_neg + clip), clip_value_min=eps, clip_value_max=1)

        # Basic CE calculation
        los_pos = y_true * tf.math.log(tf.clip_by_value(xs_pos, clip_value_min=eps, clip_value_max=float('inf')))
        los_neg = (1 - y_true) * tf.math.log(tf.clip_by_value(xs_neg, clip_value_min=eps, clip_value_max=float('inf')))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if gamma_neg > 0 or gamma_pos > 0:
            pt0 = xs_pos * y_true
            pt1 = xs_neg * (1 - y_true)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
            one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return tf.math.reduce_sum(loss)

    internal_loss.__name__ = "asymmetric_loss"
    return internal_loss


# ----------------------------------------------------------------------------------


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    If you need a loss function that takes in parameters beside y_true and y_pred, 
    you can subclass the tf.keras.losses.Loss class and implement the following two methods:

    __init__(self) —Accept parameters to pass during the call of your loss function
    call(self, y_true, y_pred) —Use the targets (y_true) and the model predictions (y_pred) to compute the model's loss
    Parameters passed into __init__() can be used during call() when calculating loss.
    """

    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss form logits or the probability.
      reduction: Type of tf.tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(
        self,
        pos_weight,
        weight,
        from_logits=False,
        reduction=tf.keras.losses.Reduction.AUTO,
        name='weighted_binary_crossentropy',
    ):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if not self.from_logits:
            # Manually calculate the weighted cross entropy.
            # Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            # where z are labels, x is logits, and q is the weight.
            # Since the values passed are from sigmoid (assuming in this case)
            # sigmoid(x) will be replaced by y_pred

            # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
            x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6)
            # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
            x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + 1e-6)
            return tf.add(x_1, x_2) * self.weight
        # Use built in function
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight