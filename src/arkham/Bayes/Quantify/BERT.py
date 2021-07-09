# -*- coding: utf-8 -*
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from arkham.utils.regularization import EmbeddingDropout, ConcreteDropout, SpatialConcreteDropout
from arkham.Bayes.Quantify.text_cnn import construct_output_layer
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TFAutoModel,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
)
from arkham.utils.losses import sparse_crossentropy_masked_v2

import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class BERT:
    def __init__(
        self,
        model_class=None,
        finetune=False,
        encoder="dense",
        max_document_len=100,
        projection_nodes=32,
        dropout=0,
        dropout_nonlinear=0,
        dropout_concrete=False,
        embedding_dropout=0,
        nb_classes=None,
        multilabel=False,
        use_aleatorics=False,
        batchnorm=None,
        sequence_labels=False,
    ):
        """
        Arguments:
            model_class    : Reference in Huggingface transformers
            embedding_only : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            finetune       : Maximal amount of words in the vocabulary (default: None)
            model_class   : Dimension of word representation (default: None)
            max_document_len  : Max length of word sequence (default: 100)

            encoder = "dense"
            projection_nodes    : Hidden layer size in feed forward network inside BERT (default: 32)

            dropout    : If defined, dropout will be added after embedding layer & concatenation (default: None)
            nb_classes      : Number of classes which can be predicted
            dropout_nonlinear : Add dropout between all nonlinear layers for montecarlo dropout evaluation (default: False)
            dropout_concrete: Add ConcreteDropout between all nonlinear layers for montecarlo dropout evaluation (default: False)
        """
        self.model_class = model_class
        self.embedding_only = True if finetune == "embedding_only" else False
        self.finetune = finetune
        self.max_document_len = max_document_len

        # General
        self.encoder = encoder
        self.projection_nodes = projection_nodes

        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.dropout_concrete = dropout_concrete
        self.auxiliary_losses = []
        self.dropout_nonlinear = dropout_nonlinear
        self.nb_classes = nb_classes
        self.multilabel = multilabel
        self.use_aleatorics = use_aleatorics
        self.batchnorm = batchnorm
        self.sequence_labels = sequence_labels  # token_classification

    def build_transformer(self):
        self.config = AutoConfig.from_pretrained(self.model_class, num_labels=self.nb_classes)

        if self.dropout > 0:
            self.config.attention_probs_dropout_prob = self.dropout  # 0.1 default
            self.config.hidden_dropout_prob = self.dropout  # 0.1 default

        # DEV: might reset layer normalization for heteroscedasticity
        """            
        if self.use_aleatorics:
           self.config.layer_norm_eps = 0
        """

        if self.embedding_only:
            self.config.output_hidden_states = True
        else:
            self.config.output_hidden_states = False

        if self.embedding_only or self.finetune:
            model = TFAutoModel
        else:
            if self.sequence_labels:
                model = TFAutoModelForTokenClassification
            else:
                model = TFAutoModelForSequenceClassification

        # DEV: ensure correct loading of huggingface model in TF2
        # source: https://stackoverflow.com/questions/62482511/tfbertmainlayer-gets-less-accuracy-compared-to-tfbertmodel
        self.transformer = model.from_pretrained(self.model_class, config=self.config)

        if self.embedding_only:
            self.transformer.bert.embeddings.trainable = False
            for i, encoder in enumerate(self.transformer.bert.encoder.layer):
                encoder.trainable = False
            self.transformer.bert.pooler.trainable = False

        if self.finetune == "freeze":
            self.transformer.bert.embeddings.trainable = False
            # https://github.com/huggingface/transformers/issues/5421 -> can take hidden states per layer and do whatever we want
            for i, encoder in enumerate(self.transformer.bert.encoder.layer):
                encoder.trainable = False
            # self.transformer.bert.encoder.trainable = True #disable full encoder
            self.transformer.bert.pooler.trainable = False

    def nameit(self):
        identifier = []
        identifier.append(self.model_class)
        if self.finetune:
            if self.finetune == "freeze":
                identifier.append(self.finetune)
            elif self.embedding_only:
                identifier.append("feature-based")
            else:
                identifier.append("finetune")
        else:
            element = "Sequence" if not self.sequence_labels else "Token"
            identifier.append(element + "Classification")
        if self.finetune or self.embedding_only:
            identifier.append(str(self.encoder))
        return "_".join(identifier)

    def build_model(self):
        self.build_transformer()

        input_ids_in = tf.keras.layers.Input(shape=(self.max_document_len,), name='input_token', dtype='int32')
        input_masks_in = tf.keras.layers.Input(shape=(self.max_document_len,), name='masked_token', dtype='int32')
        token_type_ids_in = (
            tf.keras.layers.Input(shape=(self.max_document_len,), name='type_token', dtype='int32')
            if self.sequence_labels
            else None
        )
        inputs = [input_ids_in, input_masks_in]
        if self.sequence_labels:
            inputs.append(token_type_ids_in)

        if self.embedding_only:
            embedding_layer = self.transformer.bert(
                input_ids_in, attention_mask=input_masks_in, token_type_ids=token_type_ids_in
            )
            x = layers.concatenate(embedding_layer.hidden_states[-4:])  # all until last 4 encoders
        else:

            x = self.transformer.bert(
                input_ids_in, attention_mask=input_masks_in, token_type_ids=token_type_ids_in
            )  # only works if bert model, else need to check which model "call"
            if not self.finetune:
                # Type of output depends on which transformer model
                # TFTokenClassifierOutput
                # TFBaseModelOutputWithPooling(last_hidden_state=<tf.Tensor 'bert/Identity:0' shape=(None, None, 768) dtype=float32>, pooler_output=None, hidden_states=None, attentions=None)
                if self.sequence_labels:
                    x = x.last_hidden_state
                else:
                    x = x.pooler_output
            else:
                x = x[0]  # frozen embedding

        if self.encoder == "average-pooling":
            encoder = tf.keras.layers.GlobalAveragePooling1D()
        elif self.encoder == "max-pooling":
            encoder = tf.keras.layers.GlobalMaxPooling1D()
        elif "lstm" in self.encoder:
            encoder = tf.keras.layers.LSTM(
                self.projection_nodes,
                dropout=self.dropout_nonlinear,
                return_sequences=True if self.sequence_labels else False,  # , recurrent_dropout=self.dropout
            )
            if self.encoder == "bilstm":
                encoder = tf.keras.layers.Bidirectional(encoder)
        elif self.encoder == "dense":
            encoder = tf.keras.layers.Dense(self.projection_nodes, activation='relu')
        else:
            raise NotImplementedError
        if self.finetune:  # embedding or frozen
            x = encoder(x)

        if self.dropout_nonlinear or self.dropout and not self.dropout_concrete:
            v = self.dropout_nonlinear if self.dropout_nonlinear else self.dropout
            x = layers.Dropout(v)(x)

        prediction, self.auxiliary_losses = construct_output_layer(
            x,
            self.nb_classes,
            use_aleatorics=self.use_aleatorics,
            multilabel=self.multilabel,
            dropout_concrete=self.dropout_concrete,
            auxiliary_losses=self.auxiliary_losses,
            batchnorm=self.batchnorm,
        )
        model = tf.keras.Model(inputs=inputs, outputs=prediction, name=self.nameit())
        for loss in self.auxiliary_losses:
            model.add_loss(loss)
        return model


def deprecate_tokenize(sentences, tokenizer, max_document_len=512):
    if not max_document_len:
        max_document_len = 512
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_document_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
        )
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        # input_segments.append(inputs['token_type_ids'])

    return (
        np.asarray(input_ids, dtype='int32'),
        np.asarray(input_masks, dtype='int32'),
    )  # , np.asarray(input_segments, dtype='int32')


def tokenize(
    sentences,
    tokenizer,
    max_document_len=512,
    sequence_labels=False,
    labels=None,
    label_all_tokens=True,
    is_split_into_words=True,
):
    def align_labels(tokenized_inputs, labels, label_all_tokens=True, ignore=-100):
        label_ids = []
        previous_word_idx = None
        for word_idx in tokenized_inputs.word_ids():
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(ignore)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(labels[word_idx] if label_all_tokens else ignore)
            previous_word_idx = word_idx

        return label_ids

    if sequence_labels:
        input_labels = []
    input_ids, input_masks, input_segments = [], [], []

    for i, sentence in tqdm(enumerate(sentences)):
        if sequence_labels:
            inputs = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                is_split_into_words=True if (sequence_labels and is_split_into_words) else False,
                max_length=max_document_len,
                padding="max_length",  # if max_document_len else "longest",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True if sequence_labels else False,
            )
        else:
            inputs = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=max_document_len,
                padding="max_length",  # if max_document_len else "longest",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
            )

        # DEV: debug tokenization
        # tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
        # tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
        # original_subword_length = len(tokens)

        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        if sequence_labels:
            input_segments.append(inputs['token_type_ids'])
            if labels:
                input_labels.append(align_labels(inputs, labels[i], label_all_tokens=label_all_tokens, ignore=-100))

    if sequence_labels:
        return (
            (
                np.asarray(input_ids, dtype='int32'),
                np.asarray(input_masks, dtype='int32'),
                np.asarray(input_segments, dtype='int32'),
            ),
            input_labels,
        )

    return (np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')), None


def example_fit():
    """## Download and prepare dataset"""
    model_class = "bert-base-uncased"
    max_document_len = 50
    batch_size = 2

    import os
    from arkham.Bayes.Quantify.data import generators_from_directory

    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join("/mnt/lerna/models", "CLINC150"),
        downsampling=0.1,
        composition=["word"],
        model_class="bert-base-uncased",
        batch_size=batch_size,
        max_document_len=max_document_len,  # _ood
    )

    """
    Finetune NOT frozen weights in BERT; cannot do freezing here, since we have a classification HEAD present; most common unfrozen, best performance
    """
    model = BERT(
        model_class=model_class,
        projection_nodes=100,
        nb_classes=len(label2idx),
        max_document_len=max_document_len,
        dropout_nonlinear=0.1,
    ).build_model()
    model.summary()
    model.save("./saved_model")  # , save_format='tf')
    del model
    print('**************************')
    model = tf.keras.models.load_model("./saved_model")
    del model

    """
    """

    del model

    """## Train and Evaluate"""
    # model.compile(tf.keras.optimizers.Adam(learning_rate=2e-5), "categorical_crossentropy", metrics=["accuracy"])  # sparse_
    # history = model.fit(generators["train"], validation_data=generators["dev"], epochs=1, callbacks=None)
    # print(history)
    # model.evaluate(generators["test"])

    """
    Finetune with custom classification HEAD; can do freezing here, since we can disable training in the AutoModel
    """
    model = BERT(
        model_class=model_class,
        projection_nodes=len(label2idx),
        nb_classes=len(label2idx),
        max_document_len=max_document_len,
        dropout_nonlinear=0.1,
        finetune=True,
    ).build_model()
    model.summary()
    del model

    """
    Finetune FROZEN weights with custom classification HEAD
    """
    model = BERT(
        model_class=model_class,
        encoder="average_pooling",
        projection_nodes=len(label2idx),
        nb_classes=len(label2idx),
        max_document_len=max_document_len,
        dropout_nonlinear=0.1,
        finetune="freeze",
    ).build_model()
    model.summary()
    del model

    """
    Feature-based approach: embeddings + bilstm; by default can set freeze to True
    """
    model = BERT(
        model_class=model_class,
        encoder="bilstm",
        projection_nodes=len(label2idx),
        nb_classes=len(label2idx),
        max_document_len=max_document_len,
        dropout_nonlinear=0.1,
        finetune="embedding_only",
    ).build_model()
    model.summary()
    del model


def frozen_fit():
    model_class = "bert-base-uncased"
    max_document_len = 50
    batch_size = 2
    import os
    from arkham.Bayes.Quantify.data import generators_from_directory

    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join("/mnt/lerna/models", "CLINC150"),
        downsampling=0.1,
        composition=["word"],
        model_class="bert-base-uncased",
        batch_size=batch_size,
        max_document_len=max_document_len,  # _ood
    )
    """
    Finetune FROZEN weights with custom classification HEAD
    """
    model = BERT(
        model_class=model_class,
        encoder="bilstm",
        projection_nodes=len(label2idx),
        nb_classes=len(label2idx),
        max_document_len=max_document_len,
        dropout_nonlinear=0.1,
        finetune="freeze",
    ).build_model()
    model.summary()
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=2e-2), "categorical_crossentropy", metrics=["accuracy"]
    )  # sparse_
    history = model.fit(generators["train"], validation_data=generators["dev"], epochs=2, callbacks=None)
    print(history)
    model.evaluate(generators["test"])


def sample_fit():
    model_class = "bert-base-uncased"
    max_document_len = 50
    batch_size = 2
    import os
    from arkham.Bayes.Quantify.data import generators_from_directory

    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join("/mnt/lerna/models", "CLINC150"),
        downsampling=0.1,
        composition=["word"],
        model_class="bert-base-uncased",
        batch_size=batch_size,
        max_document_len=max_document_len,  # _ood
    )
    """
    Finetune FROZEN weights with custom classification HEAD
    """
    model = BERT(
        model_class=model_class,
        projection_nodes=len(label2idx),
        nb_classes=len(label2idx),
        max_document_len=max_document_len,
        dropout=0.3,
        dropout_nonlinear=0.5,
    ).build_model()
    model.summary()
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=2e-5), "categorical_crossentropy", metrics=["accuracy"]
    )  # sparse_
    history = model.fit(generators["train"], validation_data=generators["dev"], epochs=2, callbacks=None)
    print(history)
    model.evaluate(generators["test"])


def tokenclf_fit():
    """
    inspiration: https://github.com/kamalkraj/BERT-NER-TF
    """
    model_class = "bert-base-uncased"
    max_document_len = 100
    batch_size = 2
    import os
    from arkham.Bayes.Quantify.data import generators_from_directory

    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join("/mnt/lerna/models", "conll_03"),
        downsampling=0.1,
        composition=["word"],
        model_class="bert-base-uncased",
        batch_size=batch_size,
        max_document_len=max_document_len,
        sequence_labels=True,  # _ood
    )
    """
    Finetune FROZEN weights with custom classification HEAD
    """
    model = BERT(
        model_class=model_class,
        projection_nodes=len(label2idx),
        nb_classes=len(label2idx),
        max_document_len=max_document_len,
        dropout=0.1,
        dropout_nonlinear=0.1,
        sequence_labels=True,
    ).build_model()
    model.summary()

    # model.save("./saved_model")  # , save_format='tf')
    # del model
    # print('**************************')
    # model = tf.keras.models.load_model("./saved_model")
    # del model
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=2e-4), loss=sparse_crossentropy_masked_v2, metrics=["accuracy"]
    )  # sparse_
    history = model.fit(generators["train"], validation_data=generators["dev"], epochs=2, callbacks=None)
    print(history)
    model.evaluate(generators["test"])


if __name__ == '__main__':
    tokenclf_fit()
    frozen_fit()
    sample_fit()
    example_fit()
