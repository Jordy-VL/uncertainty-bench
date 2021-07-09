import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

"""
# For the keras Lambda
def UniversalEmbedding(x):
    embed = hub.load(module_url)
    results = embed(tf.squeeze(tf.cast(x, tf.string)))["outputs"]
    print(results)
    return keras.backend.concatenate([results])
"""


def load_from_hub(pretrained_embeddings="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
    # https://github.com/tensorflow/hub/issues/355
    # https://groups.google.com/a/tensorflow.org/forum/#!topic/hub/6ZH6UzKtwgA
    # https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/
    import tensorflow_hub as hub
    import tensorflow_text

    hub_layer = hub.KerasLayer(
        pretrained_embeddings,
        input_shape=(None,),
        trainable=True,
        dtype=tf.string,
        name=pretrained_embeddings.replace("https://", ""),
    )  # , arguments={"mask_zero": True})
    return hub_layer  # input_shape=(None,),


def build_embeddings(pretrained_embeddings, max_vocabulary, max_document_len, voc2idx, overblend_vocabulary=True):
    # /media/hdd/experiments/vectors/glove.6B.50d.txt
    """
    scenarios:

    1) only use pretrained vocabulary and weights
    2) merge pretrained vocabulary and weights with custom vocabulary FULL [e.g. CONLL]
    3) hybridize to cutoff max_vocabulary point (only applicable to custom vocabulary)
    4) take blend vocabulary of pretrained and voc2idx based on frequencies [need original input then]

    How to initialize random unique vectors? https://arxiv.org/pdf/1711.09160.pdf
    [-0.5, 0.5) range which is then normalized by the size of your embedding layer (300 being the default value)

    can extend with blend factor
    """

    def determine_vocabulary_blend(
        pretrained_voc2idx, custom_voc2idx, max_vocabulary, overblend_vocabulary=True
    ):  # blendfactor possible
        """
        if maxvoc
            take as many from pretrained as maxvoc
        if blend
            take as many from custom as maxvoc allows
            if maxvoc 0

        """
        if max_vocabulary:
            keep_pretrained = {
                w: i + 1
                for i, w in enumerate([v for v in pretrained_voc2idx if v in custom_voc2idx])
                if i < max_vocabulary
            }  # or v.lower() in custom_voc2idx ?
            pretrained_size = len(keep_pretrained) + 1  # for padding
            canstilltake = 0 if not overblend_vocabulary else max(0, (max_vocabulary - pretrained_size))
        else:
            keep_pretrained = {w: i + 1 for i, w in enumerate([v for v in pretrained_voc2idx if v in custom_voc2idx])}
            pretrained_size = len(keep_pretrained) + 1  # for padding
            canstilltake = 10e32 if overblend_vocabulary else max(0, (max_vocabulary - pretrained_size))

        keep_custom = {}
        filtered_custom_voc2idx = {w: i for w, i in custom_voc2idx.items() if w not in keep_pretrained and w != "PAD"}
        for new_i, (w, og) in enumerate(
            sorted(
                zip(filtered_custom_voc2idx.keys(), filtered_custom_voc2idx.values()),
                key=lambda pair: pair[1],
                reverse=False,
            )
        ):
            if new_i == canstilltake:
                break
            keep_custom[w] = new_i + pretrained_size

        new_voc2idx = {
            **{"PAD": 0},
            **keep_pretrained,
            **keep_custom,
        }  # this one's size becomes embedding vocabulary shape

        if max_vocabulary:
            try:
                assert len(new_voc2idx) == max_vocabulary + 1
            except AssertionError as e:
                print(e)
        return new_voc2idx

    print('Pretrained embeddings are loading...')
    if "tfhub" in pretrained_embeddings:
        embedding_layer = load_from_hub(pretrained_embeddings)
        return embedding_layer, voc2idx
        # >>> np.inner(embed(["mijn kat is ziek"]), embed(["mijn                   is ziek"])) -> same; so PAD -> " "

    pretrained_voc2idx = {}
    embeddings_index = {}
    with open(pretrained_embeddings) as pretrained_embedding_file:  # ('glove.6B.%id.txt' % embed_dim)
        for i, line in enumerate(pretrained_embedding_file.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            pretrained_voc2idx[word] = i
    print('Found %s word vectors in embedding\n' % len(embeddings_index))

    # at this point determine blend!
    # remapping contains new voc index to original coefficient index to ensure we load the correct one
    new_voc2idx = determine_vocabulary_blend(
        pretrained_voc2idx, voc2idx, max_vocabulary, overblend_vocabulary=overblend_vocabulary
    )

    print("Embedding vocabulary size ", len(new_voc2idx))

    # if pretrained_weights, make sure dimensions are the same
    embed_dim = coefs.shape[0]
    embedding_matrix = np.zeros((len(new_voc2idx), embed_dim))

    for word, i in new_voc2idx.items():  # tokenizer.word_index
        # could use as an absolute cutoff point on final size YET should then have a better voc selection method
        # if max_vocabulary and i >= max_vocabulary:
        #     continue
        if word in embeddings_index:
            embedding_vector = embeddings_index[word]
        elif word.lower() in embeddings_index:
            embedding_vector = embeddings_index[word.lower()]  # could be good if its lowercase exists
        else:
            embedding_vector = np.random.normal(0, 0.1, embed_dim)  # fallback = unique random vector N 0 sigma 0.1
        embedding_matrix[i] = embedding_vector

    # https://stackoverflow.com/questions/51876460/getting-tensorflow-s-is-not-valid-scope-name-error-while-i-am-trying-to-creat
    embedding_layer = Embedding(
        input_dim=len(new_voc2idx),
        output_dim=embed_dim,
        weights=[embedding_matrix],
        trainable=True,
        mask_zero=True,
        name="glove",
    )
    return embedding_layer, new_voc2idx
