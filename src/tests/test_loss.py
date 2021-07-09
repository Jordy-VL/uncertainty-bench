import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp

tfd = tfp.distributions

from arkham.utils.losses import (
    attenuated_learned_loss,
    sequential_attenuated_learned_loss,
    logsumexp,
    ASLSingleLabelTF,
    AsymmetricLossTF,
)
from arkham.utils.callbacks import ChunkF1, Heteroscedastic_MSE
from sklearn.metrics import accuracy_score, f1_score
import torch


def test_mse():
    sample_batch_x = tf.convert_to_tensor(np.random.random((32, 5)), dtype=tf.float32)
    sample_batch_y = tf.convert_to_tensor(
        np.random.choice([0.0, 1.0], size=(32, 5), p=[1 / 2, 1 / 2]), dtype=tf.float32
    )

    print(sample_batch_y.numpy())

    metric = Heteroscedastic_MSE(T=10)
    distribution = tfd.Normal(loc=sample_batch_x, scale=sample_batch_x)

    for i in range(5):
        metric.update_state(sample_batch_y, distribution)
    tf_result = metric.result().numpy()
    print(tf_result)

    sampled = distribution.sample(10)
    tf_sampled = metric.fast_compute(sample_batch_y, sampled)
    print(tf_sampled)

    from sklearn.metrics import mean_squared_error

    regular_mse = np.mean([mean_squared_error(sample_batch_y, sampled[i]) for i in range(10)])
    print(regular_mse)

    assert np.equal(tf_sampled, regular_mse)


def test_chunk_f1():
    n_classes = 3
    sample_batch_x = tf.convert_to_tensor(np.random.random((1, 10, n_classes)), dtype=tf.float32)
    sample_batch_y = tf.convert_to_tensor(
        np.random.choice(list(range(n_classes)), size=(1, 10), p=[1 / n_classes for x in range(n_classes)]),
        dtype=tf.int32,
    )

    print(np.argmax(sample_batch_x.numpy(), axis=-1))
    print(sample_batch_y.numpy())

    idx2label = {v: k for k, v in dict(enumerate(range(n_classes))).items()}
    metric = ChunkF1(idx2label)
    metric.update_state(sample_batch_y, sample_batch_x)
    result = metric.result().numpy()

    np_pred = np.argmax(sample_batch_x.numpy(), axis=-1)[0]
    np_true = sample_batch_y.numpy()[0]

    mask = np.nonzero(np_true)
    np_mask_pred = np_pred[mask]
    np_mask_true = np_true[mask]

    print("custom macro: ", result)

    metric = ChunkF1(idx2label, average="weighted")
    metric.update_state(sample_batch_y, sample_batch_x)
    result = metric.result().numpy()

    print("custom weighted: ", result)

    for average in ["micro", "macro", "weighted"]:
        regular_f1 = f1_score(np_pred, np_true, average=average)
        mask_f1 = f1_score(np_mask_pred, np_mask_true, average=average)
        print(average + " reg: ", regular_f1)
        print(average + " mask: ", mask_f1)

    print()
    regular_accuracy = accuracy_score(np_pred, np_true)
    mask_accuracy = accuracy_score(np_mask_pred, np_mask_true)
    print("reg acc: ", regular_accuracy)
    print("mask acc: ", mask_accuracy)


def test_sequential_attenuated_loss():
    sample_batch_x = tf.convert_to_tensor(np.random.random((32, 100, 5)), dtype=tf.float32)
    n_classes = 5
    sample_batch_y = tf.convert_to_tensor(
        np.random.choice(list(range(n_classes)), size=(32, 100), p=[1 / n_classes for x in range(n_classes)]),
        dtype=tf.int32,
    )
    # categorical:
    # sample_batch_y = tf.convert_to_tensor(np.random.choice([0., 1.], size=(32, 100, 5), p=[1 / 2, 1 / 2]), dtype=tf.float32)
    print(sample_batch_x.numpy())
    print(sample_batch_y.numpy())
    distribution = tfd.Normal(loc=sample_batch_x, scale=sample_batch_x)
    y_true = sample_batch_y
    loss = sequential_attenuated_learned_loss(y_true, distribution)
    print(loss)


configs = [
    [sequential_attenuated_learned_loss, ["chunkf1"]],
    # ["sparse_categorical_crossentropy", ["chunkf1"]]
    # ["sparse_categorical_crossentropy", ["accuracy"]]
]


@pytest.mark.parametrize(["loss", "metrics"], configs)
def test_ner_model(loss, metrics):
    model = [
        tf.keras.layers.Embedding(1000, 100, trainable=True, mask_zero=True),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(200, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)
        ),
    ]
    if loss == "sparse_categorical_crossentropy":
        model.append(tf.keras.layers.Dense(8, activation="softmax"))
    else:
        # (100, 100, 8) NICE! distributionlambda is timedistributed as well!
        model.extend(
            [
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8, activation=None)),
                tfp.layers.DistributionLambda(
                    make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t, scale=tf.exp(0.5 * t))
                ),
            ]
        )
    model = tf.keras.Sequential(model)

    sequences, labels = generate_sequence_data_labels(
        samples=100, vocab_size=1000, max_len=100, unique=8, sequence_data=True
    )
    labels = np.argmax(labels, axis=-1)  # if sparse:
    sequences = tf.convert_to_tensor(sequences, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    if "chunkf1" in metrics:
        idx2label = {v: k for k, v in dict(enumerate(np.unique(labels))).items()}
        metrics = [ChunkF1(idx2label)]

    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    print(model.summary())
    model.fit(sequences, labels, batch_size=32, epochs=2, validation_split=0.1)

    predictions = model.predict(sequences)
    print(predictions.shape)
    print(predictions)


def generate_sequence_data_labels(samples=1028, vocab_size=74, max_len=80, unique=1, sequence_data=False):
    sequences = np.random.randint(0, vocab_size, (samples, max_len))
    if sequence_data:
        labels = np.random.choice([0.0, 1.0], size=(samples, max_len, unique), p=[3 / 4, 1 / 4])
    else:
        labels = np.random.choice([0.0, 1.0], size=(samples, unique), p=[1 / 2, 1 / 2])
    return sequences, labels


def build_attenuated_model(out_size, softmax=False):
    input_layer = layers.Input(shape=(5,), name='input')
    x = layers.Dense(20, activation='relu')(input_layer)
    x = layers.Dropout(0.5)(x)
    if softmax:
        output = layers.Dense(out_size, activation="softmax")(x)
    else:
        # x = layers.Dense(out_size, activation=None, name="logits")(x)
        mu = layers.Dense(out_size, activation=None, name="mu")(x)
        sigma = layers.Dense(out_size, activation=None, name="sigma")(x)
        # following https://github.com/tensorflow/probability/issues/511
        # output = tfp.layers.DistributionLambda(lambda t: tfd.Categorical(logits=t))(x)
        # NOT 0 and 1!!!

        output = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=tf.exp(0.5 * t[1]))
        )([mu, sigma])
        # , convert_to_tensor_fn=lambda s: s.sample(10)
        # output = tfp.distributions.Normal(loc=mu, scale=tf.exp(0.5 * std))
    model = tf.keras.Model(inputs=input_layer, outputs=output, name='attenuated_model')
    return model


def simple_loss(y_true, y_pred):
    print(f"before: pred = {y_pred.shape} true = {y_true.shape}")
    try:
        returner = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    except Exception as e:
        print(f"Softmax calculation failed {e}")
    else:
        return returner

    logits = tf.reduce_mean(y_pred.sample(10), axis=0)  # sample(10)
    targets = y_true  # tf.tile(y_true, (10, 1))

    print(f"after: pred = {logits.shape} true = {targets.shape}")
    print(logits.shape)

    try:
        returner = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    except Exception as e:
        print(f"Softmax calculation failed {e}")
    else:
        return returner

    try:
        returner = tf.nn.sigmoid_cross_entropy_with_logits(targets, logits)
    except Exception as e:
        print(f"Sigmoid calculation failed {e}")
    return returner


def neg_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def test_clf_attenuated_loss():
    sample_batch_x = tf.convert_to_tensor(np.random.random((32, 5)), dtype=tf.float32)
    sample_batch_y = tf.convert_to_tensor(
        np.random.choice([0.0, 1.0], size=(32, 5), p=[1 / 2, 1 / 2]), dtype=tf.float32
    )

    print(sample_batch_x.numpy())
    print(sample_batch_y.numpy())

    distribution = tfd.Normal(loc=sample_batch_x, scale=sample_batch_x)

    y_true = sample_batch_y

    loss = attenuated_learned_loss(y_true, distribution)
    print(loss)


def test_normalized_predict(batch_size=32, out_size=5, sample_size=10, max_len=20, samples=400):
    x_train = tf.convert_to_tensor(np.random.randint(0, 20, (samples, out_size)), dtype=tf.float32)
    y_train = tf.convert_to_tensor(
        np.random.choice([0.0, 1.0], size=(samples, out_size), p=[1 / 2, 1 / 2]), dtype=tf.float32
    )

    print(x_train.shape)
    print(y_train.shape)

    use_aleatorics = True

    model = build_attenuated_model(out_size, softmax=False if use_aleatorics else True)
    loss = attenuated_learned_loss if use_aleatorics else "categorical_crossentropy"
    # attenuated_learned_loss if predictive_variance else "categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss)  # , metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=20, validation_split=0.1)

    predictions = model.predict(x_train)
    print(predictions.shape)
    print(predictions)


def test_sampling():
    def random_input(seed, batch_size, n_classes):
        np.random.seed(seed)
        sample_batch_x = np.random.random((batch_size, n_classes))
        return sample_batch_x

    def sample_normal(z_avg, z_log_var):
        eps = tf.cast(tf.keras.backend.random_normal(shape=tf.shape(z_avg), mean=0.0, stddev=1.0), tf.float64)
        return z_avg + tf.exp(z_log_var)  # * eps

    import matplotlib.pyplot as plt

    seed = 42
    np.random.seed(seed)
    tf_pred = random_input(seed, 2, 3)
    tf_dist = tfd.Normal(loc=tf_pred, scale=tf.exp(0.5 * tf_pred))

    tf_dist2 = tfd.MultivariateNormalDiag(loc=tf_pred, scale_diag=tf.exp(0.5 * tf_pred))
    print(tf_dist2.sample())
    import pdb

    pdb.set_trace()  # breakpoint d247f1a8 //

    plt_sample = tf_dist2.sample(10000)
    plt.scatter(plt_sample[:, 0], plt_sample[:, 1], marker='.', alpha=0.05)
    plt.axis('equal')
    plt.show()

    tf_sample = tf_dist.sample()
    layer_sample = sample_normal(tf_pred, 0.5 * tf_pred)

    print(tf_sample)
    print(np.mean(tf_sample, axis=-1))
    print(np.var(tf_sample, axis=-1))

    print(layer_sample)
    print(np.mean(layer_sample, axis=-1))
    print(np.var(layer_sample, axis=-1))


def test_singlelabel_losses():
    np.random.seed(42)
    torch.manual_seed(42)
    n_classes = 5
    sample_batch_x = np.random.random((32, n_classes))
    sample_batch_y = np.random.choice(list(range(n_classes)), size=(32,), p=[1 / n_classes for x in range(n_classes)])
    tf_true = tf.keras.utils.to_categorical(sample_batch_y, num_classes=n_classes, dtype="float32")

    # torch_pred = torch.from_numpy(sample_batch_x)
    # torch_true = torch.from_numpy(sample_batch_y)
    # pt_loss = ASLSingleLabel(gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean")
    # torch_loss = pt_loss(torch_pred, torch_true)
    # print("Torch: ", torch_loss)

    tf_pred = tf.convert_to_tensor(sample_batch_x, dtype=tf.float32)
    tf_loss = ASLSingleLabelTF(tf_true, tf_pred, gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean")
    print("TF: ", tf_loss)


def test_multilabel_losses():
    # TENSORFLOW
    np.random.seed(42)
    torch.manual_seed(42)
    n_classes = 5
    sample_batch_x = np.random.random((32, n_classes))
    sample_batch_y = np.random.choice(list(range(n_classes)), size=(32,), p=[1 / n_classes for x in range(n_classes)])
    tf_true = tf.keras.utils.to_categorical(sample_batch_y, num_classes=n_classes, dtype="float32")

    """
    torch_pred = torch.from_numpy(sample_batch_x)
    torch_true = torch.from_numpy(tf_true)
    pt_loss = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)
    torch_loss = pt_loss(torch_pred, torch_true)
    print("Torch: ", torch_loss)
    """

    tf_pred = tf.convert_to_tensor(sample_batch_x, dtype=tf.float32)
    tf_loss = AsymmetricLossTF(tf_true, tf_pred, gamma_neg=2, gamma_pos=2, clip=0)
    print("TF: ", tf_loss)


def generate_sequence_data_labels(samples=1028, vocab_size=74, max_len=80, unique=1, sequence_data=False):
    # from pdb import set_trace
    # set_trace()
    sequences = np.eye(vocab_size)[np.random.randint(0, 30, (samples, max_len))]
    if sequence_data:
        labels = np.random.choice([0.0, 1.0], size=(samples, max_len, unique), p=[3 / 4, 1 / 4])
    else:
        labels = np.random.choice([0.0, 1.0], size=(samples, unique), p=[1 / 2, 1 / 2])
    return sequences, labels


def test_ner_losses():
    np.random.seed(42)
    torch.manual_seed(42)
    n_classes = 3
    sample_batch_x = tf.convert_to_tensor(np.random.random((1, 10, n_classes)), dtype=tf.float32)
    sample_batch_y = tf.convert_to_tensor(
        np.random.choice(list(range(n_classes)), size=(1, 10), p=[1 / n_classes for x in range(n_classes)]),
        dtype=tf.int32,
    )
    raise NotImplementedError


def plot_covariance():  # how would we expect to see it change? [more concentrated for easy samples; less concentrated for hard -> check softmax as well]
    # model.compile(experimental_run_tf_function=False, loss="mse") => True
    import matplotlib.pyplot as plt

    tf_dist2 = tfd.MultivariateNormalDiag(loc=tf_pred, scale_diag=tf.exp(0.5 * tf_pred))
    plt_sample = tf_dist2.sample(10000)
    plt.scatter(plt_sample[:, 0], plt_sample[:, 1], marker='.', alpha=0.05)


"""
ALTERNATIVE model without tensorflow probability:

    # Output layers has predictive mean and variance sigma^2
    output_layer = tf.layers.dense(fc2, units=2)

    predictions = tf.expand_dims(output_layer[:, 0], -1)
    log_variance = tf.expand_dims(output_layer[:, 1], -1)

    return predictions, log_variance

    loss = tf.reduce_sum(0.5 * tf.exp(-1 * log_variance) * tf.square(tf.abs(y_placeholder - prediction))
                             + 0.5 * log_variance)

ALTERNATIVE 2: (yet is for regression)

mean = ConcreteDropout(Dense(D), weight_regularizer=wd,
                       dropout_regularizer=dd, trainable=True)(x, training=True)
log_var = ConcreteDropout(Dense(D), weight_regularizer=wd,
                          dropout_regularizer=dd, trainable=True)(x, training=True)
pred = tf.concat([mean, log_var], -1, name='main_output')

def heteroscedastic_loss(true, pred):
    mean = pred[:, :D]
    log_var = pred[:, D:]
    precision = tf.exp(-log_var)
    reg_losses = tf.reduce_sum(tf.losses.get_regularization_losses())
    return tf.reduce_sum(precision * (true - mean)**2. + log_var + reg_losses, -1)

----------------------------------------------------------------
"""
