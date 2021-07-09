import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import gen_training_ops

# https://github.com/gergely-flamich/BVAE/blob/master/code/adaptive_sghmc_v2.py


class cSGHMC(tf.optimizers.Optimizer):
    """
    In this implementation we assume that the scaled gradient noise variance
    (beta_hat in the original paper) is set to 0.

    DEV: have to convert to TF2!
    
    - fix __init__ to TF2 (set_hyper)
    - add burnin since only applies some epochs
    """

    def __init__(
        self,
        learning_rate,
        alpha=0.01,  # momentum decay
        burnin=0,  # steps to apply momentum
        data_size=1,
        name="cSGHMC",
        **kwargs
    ):

        super(cSGHMC, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        # self._set_hyper("momentum", 1-alpha)

        """
        self._momentum = False
        momentum = 1-alpha
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        """

        # DEV: set the other variables
        self._data_size = tf.convert_to_tensor(data_size, name='data_size', dtype=tf.int64)
        self._burnin = tf.convert_to_tensor(burnin, name='burnin', dtype=tf.int64)
        self.alpha = tf.convert_to_tensor(alpha, name="alpha")

        """
        with tf.name_scope(name):
            #self._learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
            self._data_size = tf.convert_to_tensor(data_size, name="data_size", dtype=tf.int64)
            self.alpha = tf.convert_to_tensor(momentum_decay, name="alpha", )

            super().__init__(name=name, **kwargs)
        """

    def get_config(self):
        pass

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum", initializer=tf.random_normal_initializer(stddev=1.0))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        #    TypeError: _prepare() missing 2 required positional arguments: 'var_dtype' and 'apply_state'
        super(cSGHMC, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(self._get_hyper("momentum", var_dtype))

    def _resource_apply_dense(self, grad, var, apply_state=None):

        momentum = self.get_slot(var, "momentum")

        return self._sghmc_step(grad=grad, variable=var, momentum=momentum)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):

        momentum = self.get_slot(var, "momentum")

        return self._sghmc_step(grad=grad, variable=var, momentum=momentum, indices=indices)

    """

    def _resource_scatter_update(self, resource, indices, update):
        return self._resource_scatter_operate(
            resource, indices, update, tf.raw_ops.ResourceScatterUpdate
        )

    def _resource_scatter_sub(self, resource, indices, update):
        return self._resource_scatter_operate(
            resource, indices, update, tf.raw_ops.ResourceScatterSub
        )

    def _resource_scatter_operate(self, resource, indices, update, resource_scatter_op):
        resource_update_kwargs = {
            "resource": resource.handle,
            "indices": indices,
            "updates": update,
        }
        return resource_scatter_op(**resource_update_kwargs)
    """

    # @tf.function
    def _sghmc_step(self, grad, variable, momentum, indices=None):

        # Scale the gradient according to the data size
        scaled_grad = grad * tf.cast(self._data_size, grad.dtype)

        noise_stddev = tf.sqrt(self._decayed_lr(grad.dtype) * self.alpha)
        momentum_noise = tf.random.normal(shape=tf.shape(scaled_grad), dtype=scaled_grad.dtype)

        """
        if indices is not None:
            mom = tf.gather(momentum, indices)
        else:
            mom = momentum
        """
        # tf.gather(new_momentum, indices)

        momentum_delta = (
            -0.5 * self.alpha * momentum + self._decayed_lr(grad.dtype) * scaled_grad + noise_stddev * momentum_noise
        )

        new_variable = variable + momentum
        new_momentum = momentum + momentum_delta

        # Dense moment update
        if indices is None:
            # Note the minus sign on the delta argument. This is because we wish to perform gradient ascent.
            update_ops = [variable.assign(new_variable).op, momentum.assign(new_momentum).op]

        else:
            update_ops = [
                self._resource_scatter_update(variable, indices, new_variable),  # tf.gather(new_variable, indices)),
                self._resource_scatter_update(momentum, indices, new_momentum),  # tf.gather(new_momentum, indices))
            ]

            """
            update_ops = [gen_training_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum.handle,
            lr=coefficients["lr_t"],
            grad=new_grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=False,
        )
            training_ops.resource_sparse_apply_momentum(
            variable.handle,
            momentum.handle,
            new_variable,
            grad,
            indices,
            new_momentum,
            use_locking=self._use_locking,
            use_nesterov=False,
            )
            """
        return tf.group(update_ops)


"""
ValueError: Shape must be rank 0 but is rank 2 for '{{node cSGHMC/cSGHMC/update/ResourceSparseApplyMomentum}} = ResourceSparseApplyMomentum[T=DT_FLOAT, Tindices=DT_INT32, use_locking=true, use_nesterov=false](sequential/embedding/embedding_lookup/975, cSGHMC/cSGHMC/update/mul_3/ReadVariableOp/resource, cSGHMC/cSGHMC/update/add_2, cSGHMC/cSGHMC/update/UnsortedSegmentSum, cSGHMC/cSGHMC/update/Unique, cSGHMC/cSGHMC/update/add_3)' with input shapes: [], [], [20002,50], [?,50], [?], [20002,50].

"""
