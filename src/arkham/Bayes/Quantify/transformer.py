# -*- coding: utf-8 -*-
"""
Transformer block as a tf.keras layer for text classification implemented in TensorFlow 2.

References
----------
- [1] [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
- [2] [BERT](https://arxiv.org/pdf/1810.04805.pdf)


"""

import tensorflow as tf
from tensorflow.keras import layers
from arkham.utils.regularization import EmbeddingDropout, ConcreteDropout, SpatialConcreteDropout
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


"""## Implement multi head self attention as a tf.keras layer"""


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads})
        return config


"""## Implement a Transformer block as a layer"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.construct(None)

    def construct(self, input_shape):
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.ffn = tf.keras.Sequential([layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        # https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
        config = super().get_config().copy()
        config.update(
            {'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'rate': self.rate}
        )
        return config


"""## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.construct(None)

    def construct(self, input_shape):
        self.token_emb = layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero=True, trainable=True, name="word_embedding"
        )
        self.pos_emb = layers.Embedding(
            input_dim=self.maxlen, output_dim=self.embed_dim, name="position_embedding"
        )  # mask_zero=True, trainable=True,

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):

        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim, 'maxlen': self.maxlen, 'vocab_size': self.vocab_size})
        return config


class Transformer:
    def __init__(
        self,
        embedding_layer=None,
        vocab_size=None,
        embed_dim=None,
        max_document_len=100,
        num_heads=2,
        projection_nodes=32,
        dropout=0,
        dropout_nonlinear=0,
        dropout_concrete=False,
        embedding_dropout=0,
        nb_classes=None,
        use_aleatorics=False,
        batchnorm=None,
    ):
        """
        Arguments:
            embedding_layer : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            vocab_size       : Maximal amount of words in the vocabulary (default: None)
            embed_dim   : Dimension of word representation (default: None)
            max_document_len  : Max length of word sequence (default: 100)

            num_heads = 2  # Number of attention heads
            projection_nodes    : Hidden layer size in feed forward network inside transformer (default: 32)

            dropout    : If defined, dropout will be added after embedding layer & concatenation (default: None)
            nb_classes      : Number of classes which can be predicted
            dropout_nonlinear : Add dropout between all nonlinear layers for montecarlo dropout evaluation (default: False)
            dropout_concrete: Add ConcreteDropout between all nonlinear layers for montecarlo dropout evaluation (default: False)
        """

        # WORD-level
        self.embedding_layer = embedding_layer
        self.vocab_size = vocab_size
        self.max_document_len = max_document_len
        self.embed_dim = embed_dim

        # General
        self.num_heads = num_heads
        self.projection_nodes = projection_nodes

        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.dropout_concrete = dropout_concrete
        self.auxiliary_losses = []
        self.dropout_nonlinear = dropout_nonlinear  # could use abstract wrapper "doing nothing" if None
        self.nb_classes = nb_classes
        self.use_aleatorics = use_aleatorics
        self.batchnorm = batchnorm

    def build_model(self):
        """
        Build a non-compiled model

        Returns:
            Model : tensorflow.keras model instance
        """

        # Checks
        if not self.embedding_layer and (not self.vocab_size or not self.embed_dim):
            raise Exception('Please define `vocab_size` and `embed_dim` if you not using a pre-trained embedding.')

        # WORD-level
        word_input = layers.Input(
            shape=(None if not self.max_document_len else self.max_document_len,), dtype='int32', name='word_input'
        )

        # Building word-embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = TokenAndPositionEmbedding(self.max_document_len, self.vocab_size, self.embed_dim)

        """## Create classifier model using transformer layer

        Transformer layer outputs one vector for each time step of our input sequence.
        Here, we take the mean across all time steps and
        use a feed forward network on top of it to classify text.
        """
        x = self.embedding_layer(word_input)

        if self.embedding_dropout:
            x = EmbeddingDropout(self.embedding_dropout)(
                x
            )  # layers.Dropout(self.dropout, noise_shape=(None, self.embed_dim))(x)  # missing batch_size

        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.projection_nodes)
        x = transformer_block(x)

        x = layers.GlobalAveragePooling1D()(x)
        if self.dropout_nonlinear and not self.dropout_concrete:
            x = layers.Dropout(self.dropout_nonlinear)(x)
        # x = layers.Dropout(0.1)(x)
        x = layers.Dense(self.projection_nodes, activation="relu")(x)
        if self.dropout_nonlinear and not self.dropout_concrete:
            x = layers.Dropout(self.dropout_nonlinear)(x)

        x = layers.Dropout(self.dropout)(x)

        if not self.use_aleatorics:
            if self.dropout_concrete:
                prediction, l = ConcreteDropout(layers.Dense(self.nb_classes, activation='softmax'))(x)
                self.auxiliary_losses.append(l)
            else:
                prediction = layers.Dense(self.nb_classes, activation='softmax')(x)
        else:

            mu = layers.Dense(self.nb_classes, activation=None, name="mu")
            sigma = layers.Dense(self.nb_classes, activation=None, name="sigma")
            if self.dropout_concrete:
                mu, l = ConcreteDropout(mu)(x)
                self.auxiliary_losses.append(l)
                if not "1layer" in str(self.use_aleatorics):
                    sigma, l = ConcreteDropout(sigma)(x)
                    self.auxiliary_losses.append(l)
            else:
                mu = mu(x)
                sigma = sigma(x)
            if self.batchnorm:
                mu = layers.BatchNormalization()(mu)
                sigma = layers.BatchNormalization()(sigma)

            K_layers = [mu, sigma]
            if not "softplus" in str(self.use_aleatorics):
                function_to_scale = lambda var: tf.exp(0.5 * var)
            elif "nofunc" in str(self.use_aleatorics):
                function_to_scale = lambda var: var
            else:
                function_to_scale = lambda var: tf.math.softplus(var)
            use_lambda = True

            if self.use_aleatorics in [True, "softplus", "nofunc"]:
                distribution_fn = lambda t: tfd.Normal(loc=t[0], scale=function_to_scale(t[1]))

            elif self.use_aleatorics == "independent-lambda":
                distribution_fn = lambda t: tfd.Independent(tfd.Normal(loc=t[0], scale=function_to_scale(t[1])))

            elif self.use_aleatorics in ["multivar", "multivar-softplus"]:
                distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=function_to_scale(t[1]))

            elif self.use_aleatorics == "multivar-nofunc":
                distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1])

            elif self.use_aleatorics == "multivar-independent":
                distribution_fn = lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc=t[0], scale_diag=function_to_scale(t[1]))
                )
            elif self.use_aleatorics == "1layer":
                K_layers = mu
                distribution_fn = lambda t: tfd.Normal(loc=t, scale=function_to_scale(t))
            elif self.use_aleatorics == "1layer_multivar":
                K_layers = mu
                distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=function_to_scale(t))

            elif self.use_aleatorics == "1layer_multivar-nofunc":
                K_layers = mu
                distribution_fn = lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=function_to_scale(t))

            prediction = tfpl.DistributionLambda(distribution_fn)(K_layers)

        model = tf.keras.Model(inputs=word_input, outputs=prediction, name='TransformerNN')
        for loss in self.auxiliary_losses:
            model.add_loss(loss)
        return model

    def build_sequence_model(self):
        if not self.embedding_layer and (not self.vocab_size or not self.embed_dim):
            raise Exception('Please define `vocab_size` and `embed_dim` if you not using a pre-trained embedding.')

        word_input = layers.Input(
            shape=(None if not self.max_document_len else self.max_document_len,), dtype='int32', name='word_input'
        )
        # Building word-embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = TokenAndPositionEmbedding(self.max_document_len, self.vocab_size, self.embed_dim)

        x = self.embedding_layer(word_input)
        if self.embedding_dropout:
            x = EmbeddingDropout(self.embedding_dropout)(x)

        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.projection_nodes)
        x = transformer_block(x)
        x = layers.Dropout(self.dropout)(x)
        prediction = layers.Dense(self.nb_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=word_input, outputs=prediction, name='SequenceTransformerNN')
        return model


def example_fit():
    """## Download and prepare dataset"""

    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    n_classes = 2
    model = Transformer(
        vocab_size=vocab_size, embed_dim=100, nb_classes=2, max_document_len=200, dropout_nonlinear=0.5
    ).build_model()

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    """## Train and Evaluate"""
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
    print(history)


if __name__ == '__main__':
    Transformer(vocab_size=10000, embed_dim=100, nb_classes=2, dropout_nonlinear=0.5).build_model()
    model = Transformer(
        vocab_size=10000, embed_dim=100, nb_classes=2, dropout_nonlinear=0.5, dropout_concrete=True
    )  # .build_model()
    model.build_model()
    example_fit()
