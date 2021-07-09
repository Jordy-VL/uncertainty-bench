"""
Sources: 

https://github.com/ruqizhang/csgmcmc #original
https://www.tensorflow.org/api_docs/python/tf/keras/experimental/CosineDecayRestarts
https://eide.ai/bayesian/neural%20nets/mcmc/2020/11/13/bayesian-deep-learning.html #pytorch simple

Cyclic Cosine learning rate
https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/CyclicalLearningRate
https://github.com/keras-team/keras/issues/2595
https://www.tensorflow.org/api_docs/python/tf/keras/experimental/CosineDecay
https://www.tensorflow.org/api_docs/python/tf/keras/experimental/NoisyLinearCosineDecay
https://stackoverflow.com/questions/63665686/how-to-use-cosinedecayrestarts-in-tf2-2
https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SGDW
https://mancap314.github.io/cyclical-learning-rates-with-tensorflow-implementation.html
https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
https://stackoverflow.com/questions/51643726/which-of-these-is-the-correct-implementation-of-cosine-decay-learning-rate-rewe
https://www.jeremyjordan.me/nn-learning-rate/ 
https://github.com/titu1994/Snapshot-Ensembles/blob/master/optimize_cifar100.ipynb
https://github.com/henripal/sgld_tf/blob/7afa607e486d8c1290897a49d3851748df10557d/sgld_tf/optimizers.py

https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
%https://stackoverflow.com/questions/9455111/define-a-method-outside-of-class-definition/9455442

Custom optimizer/callback: 
https://cloudxlab.com/blog/writing-custom-optimizer-in-tensorflow-and-keras/
https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
https://towardsdatascience.com/tensorflow-2-2-and-a-custom-training-logic-16fa72934ac3
https://stackoverflow.com/questions/60801746/tensorflow-2-0-learning-rate-scheduler-with-tf-gradienttape
https://stackoverflow.com/questions/60996892/how-to-replace-loss-function-during-training-tensorflow-keras
https://stackoverflow.com/questions/61859671/is-it-possible-to-update-the-learning-rate-each-batch-based-on-batch-label-y

SGLD:
https://stackoverflow.com/questions/38727656/how-do-i-implement-weight-noise-in-tensorflow?rq=1 #noise
https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/optimizer/sgld.py#L35-L297
sgd = tfp.optimizer.StochasticGradientLangevinDynamics(
    start_lr, preconditioner_decay_rate=0.95, data_size=1, burnin=burnin_epoch,
    diagonal_bias=1e-08, name=None, parallel_iterations=10
)
#NotImplementedError: Eager execution currently not supported for  SGLD optimizer.

Noise loss:
https://stackoverflow.com/questions/55406146/resume-training-with-different-loss-function


More advanced:
https://github.com/WayneDW/Contour-Stochastic-Gradient-Langevin-Dynamics
https://arxiv.org/pdf/2010.06772.pdf



References:

Nemeth, C., & Fearnhead, P. (2019). Stochastic gradient Markov chain Monte Carlo, 1–31.
Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubon, D. B. (n.d.). Bayesian Data Analysis Third Edition.
Zhang, R., Li, C., Zhang, J., Chen, C., & Wilson, A. G. (2019). Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning, (2017)
"""
# https://github.com/simon-larsson/keras-swa/issues/1

"""
Notes: 

Overarching class: SG-MCMC algorithms

SGLD:  gaussian noise at step k (dependent on stepsize) | momentum 0 (alpha=1) | langevin | noise_loss 
SGHMC: no noisy loss | momentum > 0 | gamma for noise estimate   --> performs best, no need for noise update | hamiltonian | update_params

"""


import os
import sys
import pytest
import re
import numpy as np

np.random.seed(42)

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from test_swa import get_model_data
import math

from arkham.Bayes.MCMC.sgmcmc import cSGLD, burnout_epochs, Cyclic_Checkpoint, get_lr_metric
from arkham.Bayes.MCMC.sghmc import cSGHMC

"""
# SGLD
def noise_loss(lr,alpha):
    noise_loss = 0.0
    noise_std = (2/lr*alpha)**0.5
    for var in net.parameters():
        means = torch.zeros(var.size()).cuda(device_id)
        noise_loss += torch.sum(var * torch.normal(means, std = noise_std).cuda(device_id))
    return noise_loss

def adjust_learning_rate(optimizer, epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# SGHMC
def update_params(lr,epoch):
    for p in net.parameters():
        if not hasattr(p,'buf'):
            p.buf = torch.zeros(p.size()).cuda(device_id)
        d_p = p.grad.data
        d_p.add_(weight_decay, p.data)
        buf_new = (1-args.alpha)*p.buf - lr*d_p
        if (epoch%50)+1>45:
            eps = torch.randn(p.size()).cuda(device_id)
            buf_new += (2.0*lr*args.alpha*args.temperature/data_size)**.5*eps
        p.data.add_(buf_new)
        p.buf = buf_new

def adjust_learning_rate_sghmc(epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0
    return lr

def decayed_learning_rate(step):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed
"""


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        # includes the forward pass, loss calculation, backpropagation, and metric updates.
        # We can change the default convention for parameters (tuple x, y and weights)
        # and use any data we want.
        x, y = data

        # In the following code, we use compiled loss, metrics and optimizer.
        # This is mandatory! You are free to experiment.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # additional noise loss dependent on epoch

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        #    # K.set_value(model.optimizer.learning_rate, 0.001)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # can make an ensemble wrapper, save uncertainty too!
        x, y = data
        y_pred = self(x, training=False)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def test_burnout_epochs(epochs=50, cycles=2, T=3):
    burnout = burnout_epochs(epochs, cycles, T)
    print(f"Burnout: {burnout}")
    assert 49 in burnout
    assert len(burnout) == T * cycles


def test_burnout_steps(epochs=50, cycles=2, T=3, burnin=25, steps_per_epoch=20):
    """Calculate for which epochs to save weights

    Args:
        epochs (int): Number of total epochs
        cycles (int): Number of cycles to save T weight sets
        T (int): Number of weight sets to save per cycle
    """
    assert (cycles * T) < epochs
    total_iterations = epochs * steps_per_epoch
    cycle_length = total_iterations // cycles

    noise_burnin = int(0.9 * (total_iterations // cycles))  # 10% of steps per cycle spent on exploration
    epoch_burnin = int(0.95 * (total_iterations // cycles))  # last 5% of steps per cycle used for

    burnout_steps = []

    # rcounter = epoch * num_batch + batch_idx
    # cos_inner = np.pi * (rcounter % (T // M))
    for epoch in range(epochs):  # start from 0 or 1?
        for step in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + step
            if (current_step % noise_burnin) > 0:
                pass  # if cycles, use modulo!
        # if int(epoch) % math.floor(epochs/cycles) >= (math.floor(epochs/cycles) - T):
        # burnout_epochs.append(epoch) #+1
    return burnout_steps


def test_sgmcmc_tf():
    model, generators = get_model_data(downsampling=0)

    """
    cSGMCMC: base implementation
    """
    epochs = 20
    # burnin_epoch = int(epochs * (0.75))
    batch_size = 32
    steps_per_epoch = [i for i, _ in enumerate(generators["train"])][-1] + 1
    data_size = steps_per_epoch * batch_size
    total_iterations = epochs * steps_per_epoch  # total number of iterations
    start_lr = 0.5  # higher starter learning rate
    cycles = 2 + 1  # divide epochs into cycles
    wd = 1e-4
    temperature = 1  # for calculating noise loss
    T = 3  # models to save per cycle
    burnout = burnout_epochs(epochs, cycles, T)  # which epochs to save; not yet perfect if not divisable
    print(burnout)

    alpha = 0.95
    method = "SGLD" if alpha == 1 else "HMC"

    modelpath = "/tmp/weights_{epoch:02d}_" + f"{method}" + ".hdf5"

    noise_burnin = int(0.9 * (total_iterations // cycles))  # 10% of steps per cycle spent on exploration
    epoch_burnin = int(0.95 * (epochs // cycles))  # last 5% of steps per cycle used for
    print('epoch burnin: ', epoch_burnin)

    """
    The learning rate multiplier first decays from 1 to alpha for first_decay_steps steps.
    Then, a warm restart is performed. 
    Each new warm restart runs for t_mul times more steps
     and with m_mul times smaller initial learning rate.

    initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
    first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
    t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
    m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
    alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.
    name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.

    #https://stackoverflow.com/questions/63665686/how-to-use-cosinedecayrestarts-in-tf2-2
    """
    # first_decay_steps = int((total_iterations/cycles)*0.75)
    # steps_per_epoch*0.75 #iterations/cycles -> decay each epoch
    # iterations -> never stop decaying
    # sgd = tfa.optimizers.SGDW(learning_rate=lr_decayed_fn, weight_decay=1e-4, momentum=1-alpha)

    lr_decayed_fn = tf.keras.experimental.CosineDecayRestarts(
        start_lr, first_decay_steps=noise_burnin, t_mul=cycles, m_mul=1.0, alpha=0.0
    )

    # lr_callback = LR_Callback()
    # dlr_metric = get_lr_metric(sgd, decay=True)

    # require callback to save last T models per cycle!
    dynamic_callback = Cyclic_Checkpoint(filepath=modelpath, burnout=burnout)
    TB = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join("/tmp", "TensorBoard"),
        histogram_freq=1,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )

    """
    norm_callback = tf.keras.callbacks.ModelCheckpoint(
        save_best_only=False,  # if _config["epochs"] > 1 else False,
        verbose=1,
        save_freq="epoch",
    )get_lr_metric
    """
    # (tf.keras.optimizers.SGD,
    #

    # wd = lambda: 1e-4 * schedule(step)
    cSGLDW = tfa.optimizers.extend_with_decoupled_weight_decay(cSGLD)

    sgd = cSGLDW(learning_rate=lr_decayed_fn, momentum=1 - alpha, data_size=1, burnin=noise_burnin, weight_decay=wd)

    """
    cSGHMCW = tfa.optimizers.extend_with_decoupled_weight_decay(SGHMC)
    sgd = cSGHMCW(
            learning_rate=lr_decayed_fn,
            alpha=0.05,
            data_size=1,
            #burnin=noise_burnin,
            weight_decay=wd
        )
    """

    dlr_metric = get_lr_metric(sgd, decay=True)

    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy", dlr_metric])

    model.fit(
        generators["train"],
        epochs=epochs,
        validation_data=generators["dev"].repeat(epochs),
        steps_per_epoch=None,
        validation_steps=None,
        callbacks=[dynamic_callback, TB],  # lr_callback
        verbose=1,
    )

    model.evaluate(generators["test"])

    # LOAD all weights and get predictions

    # avg_callback = tfa.callbacks.AverageModelCheckpoint(filepath=modelpath, update_weights=True, verbose=1) #set the average

    # save and load
    # model.save(modelpath)
    # del model
    # loaded = tf.keras.models.load_model(modelpath)
    # loaded.evaluate(generators["test"])

    # del model
    # model, _ = get_model_data()
    """
    model = CustomModel(inputs=model.input, outputs=model.output) #custom steps
    # print(model.summary())
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        generators["train"],
        epochs=epochs,
        validation_data=generators["dev"].repeat(epochs),
        steps_per_epoch=None,
        validation_steps=None,
        callbacks=[norm_callback],
    )

    loaded = tf.keras.models.load_model(modelpath, custom_objects={"CustomModel":CustomModel})
    loaded.evaluate(generators["test"])
    """

    # sgd = tf.keras.optimizers.Adam(0.001)
    # sgd = tfa.optimizers.AdamW(lr=lr, weight_decay=wd, clipnorm=None)

    # lr_schedule = tfa.optimizers.CyclicalLearningRate(
    #     initial_learning_rate=0,#: Union[FloatTensorLike, Callable],
    #     maximal_learning_rate=0,#: Union[FloatTensorLike, Callable],
    #     step_size=0, #tfa.types.FloatTensorLike,
    #     #scale_fn: Callable,
    #     #scale_mode: str = 'cycle',
    #     #name: str = 'CyclicalLearningRate'
    # )

    """
    approach, we propose the cyclical cosine stepsize schedule for SG-MCMC. The stepsize at iteration $k$ is defined as:
    $$
    \alpha_{k}=\frac{\alpha_{0}}{2}\left[\cos \left(\frac{\pi \bmod (k-1,\lceil K / M\rceil)}{\lceil K / M\rceil}\right)+1\right]
    $$
    where $\alpha_{0}$ is the initial stepsize, $M$ is the number of cycles and $K$ is the number of total iterations (Loshchilov \& Hutter, 2016; Huang et al., 2017).
    """

    # tf.keras.experimental.NoisyLinearCosineDecay(
    #    initial_learning_rate, decay_steps, initial_variance=1.0, variance_decay=0.55,
    #    num_periods=0.5, alpha=0.0, beta=0.001, name=None
    # )

    # clr = CyclicLR(base_lr=0.001, max_lr=0.006,
    #                             step_size=2000., mode='triangular')
    # K.set_value(model.optimizer.learning_rate, 0.001)

    """
    def scheduler(epoch, lr):
      if epoch < 10:
        return lr
      else:
        return lr * tf.math.exp(-0.1)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

    def loss_wrapper(t_change, current_epoch): #in
        def custom_loss(y_true, y_pred):
            # compute loss_1 and loss_2
            bool_case_1=K.less(current_epoch,t_change)
            num_case_1=K.cast(bool_case_1,"float32")
            loss = (num_case_1)*loss_1 + (1-num_case_1)*loss_2
            return loss
        return custom_loss

    """


def test_SGHMC_tf():
    model, generators = get_model_data(downsampling=0)
    epochs = 20
    batch_size = 32
    steps_per_epoch = [i for i, _ in enumerate(generators["train"])][-1] + 1
    data_size = steps_per_epoch * batch_size
    total_iterations = epochs * steps_per_epoch  # total number of iterations
    start_lr = 0.5  # higher starter learning rate
    cycles = 2 + 1  # divide epochs into cycles
    wd = 1e-4
    temperature = 1  # for calculating noise loss
    T = 3  # models to save per cycle
    burnout = burnout_epochs(epochs, cycles, T)  # which epochs to save; not yet perfect if not divisable
    print(burnout)

    alpha = 0.95
    method = "SGLD" if alpha == 1 else "HMC"

    modelpath = "/tmp/weights_{epoch:02d}_" + f"{method}" + ".hdf5"

    noise_burnin = int(0.9 * (total_iterations // cycles))  # 10% of steps per cycle spent on exploration
    epoch_burnin = int(0.95 * (epochs // cycles))  # last 5% of steps per cycle used for
    print('epoch burnin: ', epoch_burnin)
    lr_decayed_fn = tf.keras.experimental.CosineDecayRestarts(
        start_lr, first_decay_steps=noise_burnin, t_mul=cycles, m_mul=1.0, alpha=0.0
    )
    dynamic_callback = Cyclic_Checkpoint(filepath=modelpath, burnout=burnout)

    # cSGHMCW = tfa.optimizers.extend_with_decoupled_weight_decay(cSGHMC)
    sgd = cSGHMC(
        learning_rate=lr_decayed_fn,
        alpha=0.05,
        data_size=1,
        burnin=noise_burnin
        # weight_decay=wd
    )
    dlr_metric = get_lr_metric(sgd, decay=True)

    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy", dlr_metric])

    model.fit(
        generators["train"],
        epochs=epochs,
        validation_data=generators["dev"].repeat(epochs),
        steps_per_epoch=None,
        validation_steps=None,
        callbacks=[dynamic_callback],  # lr_callback
        verbose=1,
    )

    model.evaluate(generators["test"])


def test_fit_linear():
    model = tf.keras.experimental.LinearModel(
        units=1,
        activation=None,
        use_bias=True,
        kernel_initializer='zeros',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    model.compile(optimizer='sgd', loss='mse')
    model.fit(x, y, epochs=epochs)
    x = np.random.random((2, 3))
    y = np.random.randint(0, 2, (2, 2))
    print(model.metrics_names)
