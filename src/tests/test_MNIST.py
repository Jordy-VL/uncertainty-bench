import tensorflow as tf


def alexnet():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                96,
                (11, 11),
                strides=(4, 4),
                activation='relu',
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                input_shape=(224, 224, 3),
            ),
            tf.keras.layers.MaxPooling2D(3, strides=2),
            tf.keras.layers.Conv2D(
                256,
                (5, 5),
                activation='relu',
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer='ones',
            ),
            tf.keras.layers.MaxPooling2D(3, strides=2),
            tf.keras.layers.Conv2D(
                384, (3, 3), activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)
            ),
            tf.keras.layers.Conv2D(
                384,
                (3, 3),
                activation='relu',
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer='ones',
            ),
            tf.keras.layers.Conv2D(
                384,
                (3, 3),
                activation='relu',
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer='ones',
            ),
            tf.keras.layers.MaxPooling2D(3, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                4096, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), bias_initializer='ones'
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                4096, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), bias_initializer='ones'
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                10, activation='softmax', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)
            ),
        ]
    )
    return model


def MNIST():
    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images[:1000]
    training_labels = training_labels[:1000]
    test_images = test_images[:100]
    test_labels = test_labels[:100]

    training_images = tf.map_fn(lambda i: tf.stack([i] * 3, axis=-1), training_images).numpy()
    test_images = tf.map_fn(lambda i: tf.stack([i] * 3, axis=-1), test_images).numpy()

    training_images = tf.image.resize(training_images, [224, 224]).numpy()
    test_images = tf.image.resize(test_images, [224, 224]).numpy()

    training_images = training_images.reshape(1000, 224, 224, 3)
    training_images = training_images / 255.0
    test_images = test_images.reshape(100, 224, 224, 3)
    test_images = test_images / 255.0

    training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    num_len_train = int(0.8 * len(training_images))

    ttraining_images = training_images[:num_len_train]
    ttraining_labels = training_labels[:num_len_train]

    valid_images = training_images[num_len_train:]
    valid_labels = training_labels[num_len_train:]

    training_images = ttraining_images
    training_labels = ttraining_labels

    return training_images, training_labels, valid_images, valid_labels, test_images, test_labels


def test_MNIST():
    training_images, training_labels, valid_images, valid_labels, test_images, test_labels = MNIST()

    model = alexnet()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(5)],
    )

    print(model.summary())

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001)

    model.fit(
        training_images,
        training_labels,
        batch_size=128,
        validation_data=(valid_images, valid_labels),
        epochs=90,
        callbacks=[reduce_lr],
    )

    model.evaluate(test_images, test_labels)


@tf.keras.utils.register_keras_serializable()
class NoiseLoss(tf.keras.losses.Loss):  # wrong
    """For SGLD
    
    Attributes:
        from_logits (TYPE): Description

    #alpha, temperature, datasize, lr
    #lr = self.model.optimizer._decayed_lr
    #loss_noise = noise_loss(lr, args.alpha) * (args.temperature / datasize) ** 0.5

    Source: 
    https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/optimizer/sgld.py#L35-L297
    """

    def __init__(self, datasize, alpha=0, temperature=1, from_logits: bool = False, current_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits

    def get_config(self):
        config = super().get_config()
        config.update(from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):
        # need learning rate!
        # model.trainable_variables
        # current epoch
        return lambda: 0


"""

@tf.function
def train(train_dataset, model, loss_fn, epochs):
    optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(
        0.5, preconditioner_decay_rate=0.95, data_size=1, burnin=3,
        diagonal_bias=1e-08, name=None, parallel_iterations=10
    )
    for epoch in range(epochs):
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                y_hat = model(x)
                loss = loss_fn(y, y_hat)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
"""
