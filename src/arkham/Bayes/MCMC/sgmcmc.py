# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer

# from tensorflow.python.keras.optimizer_v2 import optimizer_v2
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export

# from tensorflow_probability.python.internal import distribution_util
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer?version=nightly
# https://stackoverflow.com/questions/63878497/how-to-update-trainable-variables-on-my-custom-optimizer-using-tensorflow-2
# https://www.kdnuggets.com/2018/01/custom-optimizer-tensorflow.html
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/notebooks/classification/SGLD_MNIST.ipynb

## General updates
# _resource_apply_dense (update variable given gradient tensor is a dense tf.Tensor)
# _resource_apply_sparse (update variable given gradient tensor is a sparse tf.IndexedSlices. The most common way for this to happen is if you are taking the gradient through a tf.gather.)
# _create_slots (if your optimizer algorithm requires additional variables)
# get_config (serialization of the optimizer, include all hyper parameters)

# _create_slots() and  _prepare() create and initialise additional variables, such as momentum.
# _apply_dense(), and _apply_sparse() implement the actual Ops, which update the variables.

## Weight decay
# https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/DecoupledWeightDecayExtension
# extend_with_decoupled_weight_decay(tf.keras.optimizers.cSGLD,
#                                   weight_decay=weight_decay)


def burnout_epochs(epochs, cycles, T):
    """Calculate for which epochs to save weights

    Args:
        epochs (int): Number of total epochs
        cycles (int): Number of cycles to save T weight sets
        T (int): Number of weight sets to save per cycle
    """
    assert (cycles * T) < epochs
    burnout_epochs = []
    for epoch in range(epochs):  # start from 0 or 1?
        if int(epoch) % (epochs // cycles) >= ((epochs // cycles) - T):
            burnout_epochs.append(epoch)  # +1
    return burnout_epochs


def get_lr_metric(optimizer, decay=True):
    def lr(y_true, y_pred):
        return K.get_value(optimizer.lr)  # I use ._decayed_lr method instead of .lr

    def decay_lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)  # I use ._decayed_lr method instead of .lr

    if decay:
        return decay_lr
    return lr


class Cyclic_Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, burnout):
        super(Cyclic_Checkpoint, self).__init__()
        self.filepath = filepath
        self.current_epoch = K.variable(0)
        self.burnout = burnout
        self.is_burnout_epoch = K.variable(0)

    def on_epoch_begin(self, epoch, log={}):
        if epoch > self.current_epoch:
            K.set_value(self.current_epoch, epoch)
        if epoch in self.burnout:
            K.set_value(self.is_burnout_epoch, 1)
        print(
            "\nCurrent Epoch is "
            + str(int(K.get_value(self.current_epoch)))
            + (" Burnout" if K.get_value(self.is_burnout_epoch) else " Burnin")
        )

    def _save_model(self, epoch, logs):
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}
        out = self.filepath.format(epoch=epoch + 1, **logs)
        self.model.save(out)

    def on_epoch_end(self, epoch, log={}):
        if self.is_burnout_epoch:  # save weights
            print(f"\nSaving Burnout epoch {epoch} weights")
            # DEV: do pathlib like replacement
            self._save_model(epoch, log)
            """
            self.model.save(
                self.filepath.replace("{epoch:02d}", f"{epoch+1:02d}")
            )  # proper way with pathlib
            """
            K.set_value(self.is_burnout_epoch, 0)
        else:
            print(f"\nBurnin epoch {epoch}")


@keras_export("keras.optimizers.cSGLD")
class cSGLD(tf.keras.optimizers.Optimizer):
    """
    Args:
      learning_rate: Scalar `float`-like `Tensor`. The base learning rate for the
        optimizer. Must be tuned to the specific function being minimized.

      #momentum 1-alpha
      alpha: Scalar `float`-like `Tensor`. The exponential
        decay rate of the rescaling of the preconditioner (RMSprop). (This is
        "alpha" in Li et al. (2016)). Should be smaller than but nearly `1` to
        approximate sampling from the posterior. (Default: `0.95`)

      data_size: Scalar `int`-like `Tensor`. The effective number of
        points in the data set. Assumes that the loss is taken as the mean over a
        minibatch. Otherwise if the sum was taken, divide this number by the
        batch size. If a prior is included in the loss function, it should be
        normalized by `data_size`. Default value: `1`.

      burnin: Scalar `int`-like `Tensor`. The number of iterations to collect
        gradient statistics to update the preconditioner before starting to draw
        noisy samples. (Default: `25`)

    #### References
    [1]: Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin.
         Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural
         Networks. In _Association for the Advancement of Artificial
         Intelligence_, 2016. https://arxiv.org/abs/1512.07666
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate,
        momentum=1,
        data_size=1,  # to calculate temperature 1/data_size
        burnin=25,  # might need to calculate differently
        name='cSGLD',
        **kwargs,
    ):
        super(cSGLD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)
        self.alpha = 1 - momentum

        # DEV: set the other variables
        self._data_size = tf.convert_to_tensor(data_size, name='data_size')
        self._burnin = tf.convert_to_tensor(burnin, name='burnin', dtype=tf.int64)

    def _create_slots(self, var_list):
        """
        Many optimizer subclasses, such as Adam and Adagrad allocate and manage additional variables
        associated with the variables to train. These are called Slots. Slots have names
        and you can ask the optimizer for the names of the slots that it uses.
        Once you have a slot name you can ask the optimizer for the variable it created to hold the slot value.
         for v in var_list:
         self._zeros_slot(v, "m", self._name)
         self._zeros_slot(v, "v", self._name)
        """
        # Create slots for allocation and later management of additional
        # variables associated with the variables to train.
        # for example: the first and second moments.
        # if self._momentum: # with SGLD, always create this variable
        for var in var_list:
            self.add_slot(var, "momentum", initializer=tf.random_normal_initializer(stddev=1.0))  # rms

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # LR handling copied from optimizer_v2.OptimizerV2
        if "learning_rate" in config:
            if isinstance(config["learning_rate"], dict):
                config["learning_rate"] = tf.keras.optimizers.schedules.deserialize(
                    config["learning_rate"], custom_objects=custom_objects
                )

        if "weight_decay" in config:
            if isinstance(config["weight_decay"], dict):
                config["weight_decay"] = tf.keras.optimizers.schedules.deserialize(
                    config["weight_decay"], custom_objects=custom_objects
                )

        return cls(**config)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        #    TypeError: _prepare() missing 2 required positional arguments: 'var_dtype' and 'apply_state'
        super(cSGLD, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(self._get_hyper("momentum", var_dtype))

        """
        if "learning_rate" in self._hyper:
            apply_state[(var_device, var_dtype)]["learning_rate"] = array_ops.identity(
                self._get_hyper("learning_rate", var_dtype)
            )
        """
        # wd_t = tf.identity(self._decayed_wd(var_dtype))
        # apply_state[(var_device, var_dtype)]["wd_t"] = wd_t
        """
            if "weight_decay" in self._hyper:
                wd_t = tf.identity(self._decayed_wd(var_dtype))
                apply_state[(var_device, var_dtype)]["wd_t"] = wd_t

        def _decayed_lr(self, var_dtype):
            wd_t = self._get_hyper("weight_decay", var_dtype)

            if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
                wd_t = tf.cast(wd_t(self.iterations), var_dtype)

            return wd_t
        """

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # if no momentum, this one applies!
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(
            var_device, var_dtype
        )

        momentum_var = self.get_slot(var, "momentum")
        new_grad = self._apply_noisy_update(grad, var, lr=coefficients["lr_t"])

        if self._momentum:
            return gen_training_ops.ResourceApplyKerasMomentum(
                var=var.handle,
                accum=momentum_var.handle,
                lr=coefficients["lr_t"],
                grad=new_grad,
                momentum=coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=False,  # DEV: already fixed
            )
        else:
            return gen_training_ops.ResourceApplyGradientDescent(
                var=var.handle, alpha=coefficients["lr_t"], delta=new_grad, use_locking=self._use_locking
            )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(
            var_device, var_dtype
        )

        momentum_var = self.get_slot(var, "momentum")
        new_grad = self._apply_noisy_update(grad, var, lr=coefficients["lr_t"], indices=indices)

        return gen_training_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=coefficients["lr_t"],
            grad=new_grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=False,
        )

    @tf.function
    def _apply_noisy_update(self, grad, var, lr=None, indices=None):
        ## DEV: check iplementation and see if all tensors prepared.
        # Compute and apply the gradient update following
        # (preconditioned) Langevin dynamics
        """
        def noise_loss(lr, alpha):
            noise_loss = 0.0
            noise_std = (2 / lr * alpha) ** 0.5
            for var in net.parameters():
                means = torch.zeros(var.size()).cuda(device_id)
                noise_loss += torch.sum(var * torch.normal(means, std=noise_std).cuda(device_id))
            return noise_loss

        loss_noise = noise_loss(lr, args.alpha) * (args.temperature / datasize) ** 0.5
        loss = criterion(outputs, targets) + loss_noise

        """

        # 1. calculate cyclic decay steps

        # # check equality of array in t1 for each array in t2 element by element. This is possible because the equal function supports broadcasting.
        # equal =  tf.math.equal(t1, t2)
        # # checking if all elements from  t1 match for all elements of some array in t2
        # equal_all = tf.reduce_all(equal, axis=2)
        # contains = tf.reduce_any(equal_all)

        # {'initial_learning_rate': 0.5, 'first_decay_steps': 108, 't_mul': 2, 'm_mul': 1.0, 'alpha': 0.0, 'name': None}
        if lr is None:
            print("backtracking to decayed lr")
            lr = self._decayed_lr(grad.dtype)

        condition = tf.squeeze(tf.math.floormod(self.iterations, tf.cast(self._burnin, tf.int64)) > 0)
        passing = tf.cast(
            # tf.math.rsqrt(2 / (self._decayed_lr(grad.dtype) * tf.cast(self.alpha, grad.dtype))),
            0.5 ** (2 / (lr * tf.cast(self.alpha, grad.dtype))),
            grad.dtype,
        )
        reject = tf.zeros([], grad.dtype)
        stddev = tf.where(condition, passing, reject)

        new_grad = grad + tf.random.normal(mean=0, stddev=stddev, shape=tf.shape(grad), dtype=grad.dtype)
        # where to apply temperature???
        # (args.temperature / datasize) ** 0.5
        # scaled_grad = new_grad
        # for preconditioned update.
        # decay_tensor = tf.cast(self.alpha, grad.dtype)
        # new_mom = decay_tensor * mom + (1.0 - decay_tensor) * tf.square(grad)
        # lr_t = self._decayed_lr(grad.dtype)

        # Scale the gradient according to the data size
        # scaled_grad = grad * tf.cast(self._data_size, grad.dtype)
        # langevin_noise = tf.random.normal(shape=scaled_grad.shape, dtype=scaled_grad.dtype)

        # new grad = True, learning rate updates happen in dense!
        # STANDARD gaussian
        return new_grad

        # var.assign_sub(lr_t* (0.5 * scaled_grad) - 2 * lr_t * langevin_noise)

        # state_ops.assign_sub(var, lr_t*grad/2 - lr_t * lr_t * pnoise)

        # SGLD: lr_t*grad/2 + lr_t*lr_t*tf.random_normal(shape=tf.shape(grad))
        # pSGLD:
        # m_t = m.assign(tf.sqrt(epsilon_t + decay_t * m + (1 - decay_t) * tf.square(grad)))
        # lr_t*grad/2 + lr_t*lr_t*tf.divide(tf.random_normal(shape=tf.shape(grad)), m))
        # current_epoch = self.iterations // self._steps_per_epoch
        # eq = tf.math.equal(self._burnout_epochs, tf.cast(current_epoch, tf.int32))
        # tf.math.equal(self._burnout_epochs, tf.cast(current_epoch, tf.int32))

    def get_config(self):
        config = super(cSGLD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self._serialize_hyperparameter("momentum"),
            }
        )
        return config


class EnsembleWrapper:
    """
    This class wraps a list of models all of which are mc models
    """

    def __init__(self, modellist):
        self.ms = modellist
        self.n_mc = len(modellist)

    def predict(self, X):
        mc_preds = np.concatenate([np.stack([m.predict(X) for _ in range(self.n_mc)]) for m in self.ms], axis=0)
        return mc_preds.mean(axis=0)

    def get_results(self, X):
        mc_preds = np.concatenate([np.stack([m.predict(X) for _ in range(self.n_mc)]) for m in self.ms], axis=0)
        preds = mc_preds.mean(axis=0)
        ent = -1 * np.sum(preds * np.log(preds + 1e-10), axis=-1)
        bald = ent - np.mean(-1 * np.sum(mc_preds * np.log(mc_preds + 1e-10), axis=-1), axis=0)
        return preds, ent, bald

    def __call__(self, X):
        """
        Returns the mean prediction of the entire ensemble as a keras tensor to allow differentiation
        """
        return K.mean(K.stack([K.mean(mc_dropout_preds(m, X, n_mc=self.n_mc), axis=0) for m in self.ms]), axis=0)


# cSGLDW = tfa.optimizers.extend_with_decoupled_weight_decay(cSGLD)
