# -*- coding: utf-8 -*-
"""
Inspiration:

-    https://github.com/arunarn2/HierarchicalAttentionNetworks
-    https://www.kaggle.com/sermakarevich/hierarchical-attention-network
-    https://github.com/LukeZhuang/Hierarchical-Attention-Network #visualization
-    https://github.com/FlorisHoogenboom/keras-han-for-docla/blob/master/ #contextvector focus
-    https://github.com/stevewyl/comparative-reviews-classification/blob/master #also self-attention

Hierarchical Attention Network implemented in TensorFlow 2.
This implementation is based on the original paper [1] for classification using word and sentence-level attention.

1. Embedding layer
2. Word Encoder: word level bi-directional GRU to get rich representation of words
3. Word Attention:word level attention to get important information in a sentence
4. Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences
5. Sentence Attention: sentence level attention to get important sentence among sentences
6. Fully Connected layer + Softmax

"That is, we first feed the word annotation hit through a one-layer MLP to get uit as a hidden representation of hit, then we measure the importance of
the word as the similarity of uit with a word level context vector uw and get a normalized importance
weight αit through a softmax function. After that, we compute the sentence vector si (we abuse the notation here)
 as a weighted sum of the word annotations based on the weights."

References
----------
- [1] [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)


"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


def dot_product(x, kernel):
    """
    Wrapper for dot product operation
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)


class HierarchicalAttention(layers.Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.

    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    """

    def __init__(self, **kwargs):
        # self.attention_dim = attention_dim
        super(HierarchicalAttention, self).__init__(**kwargs)
        self.supports_masking = True
        # self._supports_ragged_inputs = True

    def build(self, input_shape):
        assert len(input_shape) == 3  # to make sure it is recurrent
        # input_shape[-1] : size of bidrectional GRU hidden nodes/timesteps

        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', name="W", trainable=True
        )
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zero', name="b", trainable=True)
        self.u = self.add_weight(
            shape=(input_shape[-1],), initializer='random_normal', name="u", trainable=True  # glorot_uniform
        )
        super(HierarchicalAttention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def _get_attention_weights(self, x, mask=None):
        """
        Can use it for visualization!
        """
        uit = K.tanh(tf.nn.bias_add(dot_product(x, self.W), self.b))  # feedforward logits
        ait = K.exp(dot_product(uit, self.u))  # can also make alternate version with softmax!
        if mask is not None:
            ait *= tf.cast(mask, K.floatx())

        ait /= tf.cast(tf.reduce_sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return ait

    def call(self, x, mask=None):
        """
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        # ait = tf.exp(tf.squeeze(K.dot(uit, self.u), -1)) #exp
        uit = K.tanh(tf.nn.bias_add(dot_product(x, self.W), self.b))  # feedforward logits
        ait = K.exp(dot_product(uit, self.u))

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= tf.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        ait /= tf.cast(tf.reduce_sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        """
        ait = self._get_attention_weights(x, mask=mask)
        ait = K.expand_dims(ait)  # tf.expand_dims(ait, axis=-1)
        weighted_input = x * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        return output


class HierarchicalAttentionNetwork:
    def __init__(
        self,
        embedding_layer=None,
        vocab_size=None,
        embed_dim=None,
        max_document_len=200,
        max_sentences=15,
        dropout=0,
        dropout_nonlinear=0,
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
            max_document_len  : Max length of word sequence (default: 200)
            max_sentences   :   Max number of sentences per document (default: 15)
            dropout    : If defined, dropout will be added after embedding layer & concatenation (default: 0)
            nb_classes      : Number of classes which can be predicted
            dropout_nonlinear : Add dropout between all nonlinear layers for montecarlo dropout evaluation (default: 0)
        """

        # WORD-level
        self.embedding_layer = embedding_layer
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_document_len = max_document_len
        self.max_sentences = max_sentences

        # General
        self.recurrent_nodes = 100  # paper also reports 50
        self.nb_classes = nb_classes
        self.dropout = dropout
        self.dropout_nonlinear = dropout_nonlinear
        self.embedding_dropout = embedding_dropout
        self.batchnorm = batchnorm
        self.use_aleatorics = use_aleatorics

    def build_model(self):
        """
            maxlen = 100
            max_sentences = 15
            max_words = 20000
            embedding_dim = 100
            glove_dir = "./glove.6B"
        """
        """
        Build a non-compiled model
        Returns:
            Model : tensorflow.keras model instance
        """

        # Checks
        if not self.embedding_layer and (not self.vocab_size or not self.embed_dim):
            raise Exception('Please define `vocab_size` and `embed_dim` if you not using a pre-trained embedding.')

        # Building word-embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                mask_zero=True,
                weights=None,
                trainable=True,
                name="word_embedding",
            )

        # WORD-level [sentence input]; maxdoclen => maxsentencelen
        sentence_input = layers.Input(
            shape=(None if not self.max_document_len else self.max_document_len,),
            ragged=False,
            dtype='int32',
            name='word/sentence_input',
        )
        x = self.embedding_layer(sentence_input)  # embedded sequences
        if self.embedding_dropout:
            x = EmbeddingDropout(self.embedding_dropout)(x)

        # or dropout mixed?
        lstm_word = layers.Bidirectional(
            layers.GRU(
                self.recurrent_nodes,
                dropout=self.dropout,
                recurrent_dropout=self.dropout_nonlinear,
                return_sequences=True,
            )
        )(x)
        attn_word = HierarchicalAttention(name="word_attention")(lstm_word)
        sentence_encoder = tf.keras.Model(
            inputs=sentence_input, outputs=attn_word, name='HAN_SentWord'
        )  # from arbitrary length sequence to encoded + attended sequence
        print(sentence_encoder.summary())

        # ENTRY point: list of sentences with list of encoded words
        document_input = layers.Input(
            shape=(
                None if not self.max_sentences else self.max_sentences,
                None if not self.max_document_len else self.max_document_len,
            ),
            ragged=False,
            dtype='int32',
        )
        document_encoder = layers.TimeDistributed(sentence_encoder)(document_input)
        lstm_sentence = layers.Bidirectional(
            layers.GRU(
                self.recurrent_nodes,
                dropout=self.dropout,
                recurrent_dropout=self.dropout_nonlinear,
                return_sequences=True,
            )
        )(document_encoder)
        attn_sentence = HierarchicalAttention(name="sentence_attention")(lstm_sentence)
        x = attn_sentence

        """
        if self.dropout_nonlinear:
            x = layers.Dropout(self.dropout_nonlinear)(x)
        """
        # could make a generic output layer switch :)
        if not self.use_aleatorics:
            prediction = layers.Dense(self.nb_classes, activation='softmax')(x)
        else:
            import tensorflow_probability as tfp

            mu = layers.Dense(self.nb_classes, activation=None, name="mu")(x)
            sigma = layers.Dense(self.nb_classes, activation=None, name="sigma")(x)
            if self.batchnorm:
                mu = layers.BatchNormalization()(mu)
                sigma = layers.BatchNormalization()(sigma)
            prediction = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=tf.exp(0.5 * t[1]))
            )([mu, sigma])

        document_encoder = tf.keras.Model(inputs=document_input, outputs=prediction, name='HAN_DocSent')
        return sentence_encoder, document_encoder

        def example_fit(self, input=None, output=None, batch_size=2, epochs=5):
            self.model = self.build_model()
            self.model.fit(input, output, epochs=epochs, batch_size=batch_size, verbose=2)
            return


def plot_attention_doc(doc, attentions, converter, title=""):
    from matplotlib import pyplot as plt

    def heatmap(data, text, title="", xlabel="", ylabel=""):
        plt.figure()
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        c = plt.pcolor(data, edgecolors='k', linewidths=4, cmap='jet')  # 'YlGnBu')  # 'RdBu')  # , vmin=0.0, vmax=1.0)
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    '%s' % text[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    color="black" if data[y, x] < 0.6 or data[y, x] > 0.2 else "white",
                )
        plt.colorbar(c)
        plt.show()
        return c

    # remove padding
    padding = np.all(doc == 0, axis=1)
    doc = doc[~padding]
    attentions = attentions[~padding]
    converter[0] = ""
    textuals = np.vectorize(converter.get)(doc)
    c = heatmap(attentions, textuals, title=title)


def visualize_word_attentions(model, _config, inputs, texts, golds=None, predictions=None):
    sent_model = tf.keras.Model.from_config(
        model.layers[1].get_config()["layer"]["config"], custom_objects={"HierarchicalAttention": HierarchicalAttention}
    )
    sent_model.set_weights(model.layers[1].get_weights())
    embedder = K.function([model.layers[0].input], model.layers[0].output)
    pre_word_attentions = K.function(
        [sent_model.layers[0].input], sent_model.layers[-2].output  # could apply on any B x words?
    )
    # sentmodel weights?

    # either batch test else input -> just convert
    # if not isinstance(inputs, tuple):
    #     raise NotImplementedError
    """
    from arkham.Bayes.Quantify.evaluate import decode_x_and_y
    batch_texts, batch_golds, _ = decode_x_and_y(inputs[0].numpy(), inputs[1].numpy(), _config["idx2voc"], _config["idx2label"], sequence_labels=False)
    batch_x = inputs[0]
    """
    docs = embedder(inputs)
    for i, doc in enumerate(docs):
        preatts = pre_word_attentions(doc)  # (100, 16, 200); sentences x words x timesteps RNN -> *2 for bidirectional
        atts = sent_model.layers[-1]._get_attention_weights(tf.expand_dims(preatts, axis=0))[
            0
        ]  # requires batchsize again... 4dim
        assert doc.shape == atts.shape
        most_attended_words_per_sentence = np.argmax(atts, axis=1)
        indices = np.take_along_axis(doc, np.expand_dims(most_attended_words_per_sentence, 1), 1)
        print(texts[i])
        print(np.vectorize(_config["idx2voc"].get)([x for x in indices if x]))
        plot_attention_doc(doc, atts, converter=_config["idx2voc"], title=golds[i] if golds else "")


"""
# visualize attention
def get_attention(sent_model, doc_model, sequences, model_name, topN=5):
    sent_before_att = K.function([sent_model.layers[0].input, K.learning_phase()],
                                 [sent_model.layers[2].output])
    cnt_reviews = sequences.shape[0]

    # 导出这个句子每个词的权重
    sent_att_w = sent_model.layers[3].get_weights()
    sent_all_att = []
    for i in range(cnt_reviews):
        sent_each_att = sent_before_att([sequences[i], 0])
        sent_each_att = cal_att_weights(sent_each_att, sent_att_w, model_name)
        sent_each_att = sent_each_att.ravel()
        sent_all_att.append(sent_each_att)
    sent_all_att = np.array(sent_all_att)
    if model_name in ['HAN', 'MHAN']:
        doc_before_att = K.function([doc_model.layers[0].input, K.learning_phase()],
                                    [doc_model.layers[2].output])
        # 找到重要的分句
        doc_att_w = doc_model.layers[3].get_weights()
        doc_sub_att = doc_before_att([sequences, 0])
        doc_att = cal_att_weights(doc_sub_att, doc_att_w, model_name)
        return sent_all_att, doc_att
"""

"""
class Self_Attention(Layer):

    def __init__(self, ws1, ws2, punish, init='glorot_normal', **kwargs):
        self.kernel_initializer = initializers.get(init)
        self.weight_ws1 = ws1
        self.weight_ws2 = ws2
        self.punish = punish
        super(Self_Attention, self).__init__(** kwargs)

    def build(self, input_shape):
        self.Ws1 = self.add_weight(shape=(input_shape[-1], self.weight_ws1),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='{}_Ws1'.format(self.name))
        self.Ws2 = self.add_weight(shape=(self.weight_ws1, self.weight_ws2),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='{}_Ws2'.format(self.name))
        self.batch_size = input_shape[0]
        super(Self_Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.tanh(K.dot(x, self.Ws1))
        ait = K.dot(uit, self.Ws2)
        ait = K.permute_dimensions(ait, (0, 2, 1))
        A = softmax(ait, axis=1)
        M = K.batch_dot(A, x)
        if self.punish:
            A_T = K.permute_dimensions(A, (0, 2, 1))
            tile_eye = K.tile(K.eye(self.weight_ws2), [self.batch_size, 1])
            tile_eye = K.reshape(
                tile_eye, shape=[-1, self.weight_ws2, self.weight_ws2])
            AA_T = K.batch_dot(A, A_T) - tile_eye
            P = K.l2_normalize(AA_T, axis=(1, 2))
            return M, P
        else:
            return M

    def compute_output_shape(self, input_shape):
        if self.punish:
            out1 = (input_shape[0], self.weight_ws2, input_shape[-1])
            out2 = (input_shape[0], self.weight_ws2, self.weight_ws2)
            return [out1, out2]
        else:
            return (input_shape[0], self.weight_ws2, input_shape[-1])
"""

if __name__ == '__main__':
    layer = HierarchicalAttention().build((32, 15, 100))  # 15 or 100?
    model = HierarchicalAttentionNetwork(vocab_size=10000, embed_dim=100, nb_classes=3).build_model()
    print(model.summary())


# class defining the custom attention layer
"""
def attention(inputs, att_size, time_major=False, return_alphas=False):

    # Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hiddensize = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hiddensize, att_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
"""
