import tensorflow as tf
from tensorflow.keras.models import Sequential, Model


def model_descriptors(_config, vocab_size, nb_classes, embedding_layer=None, calibration=False):
    """
    Allowed values:
    [
        "TextClassificationCNN_simple", "TextClassificationCNN_complex",
        "HierarchicalAttentionNetwork", "Transformer", "UniversalSentenceEncoder",
        "BERT.*", SNGP, ...
    ]
    """
    if _config["model"] in ["cnn_baseline"]:  # , "sequential_bilstm"
        baseline = [
            tf.keras.layers.Embedding(
                input_dim=vocab_size, output_dim=_config["embed_dim"], trainable=True, weights=None, mask_zero=True
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(nb_classes, activation="softmax"),
        ]
        if embedding_layer:
            baseline[0] = embedding_layer
        if calibration:
            baseline[-1] = tf.keras.layers.Dense(nb_classes)
            scaled = [tf.keras.layers.Activation("softmax", name="softmax"), RelaxedLayer(name="temperature_scale")]
            baseline.extend(scaled)

        if _config["model"] == "cnn_baseline":
            encoder = [
                tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation="relu"),
                tf.keras.layers.GlobalMaxPooling1D(),
            ]
            baseline[1:1] = encoder
            model = tf.keras.Sequential(baseline)

        elif _config["model"] == "sequential_bilstm":
            model = [
                baseline[0],
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        100, dropout=_config["dropout"], recurrent_dropout=_config["dropout"], return_sequences=True
                    )
                ),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Dense(nb_classes, activation="softmax"),
            ]
            if _config.get("use_aleatorics"):
                model[-1] = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nb_classes, activation=None))
                model.append(
                    tfp.layers.DistributionLambda(
                        make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t, scale=tf.exp(0.5 * t))
                    )
                )
            model = tf.keras.Sequential(model)
        return model

    if "TextClassificationCNN" in _config["model"]:
        from arkham.Bayes.Quantify.text_cnn import TextClassificationCNN

        model = TextClassificationCNN(
            # WORD-level
            embedding_layer=embedding_layer,
            vocab_size=vocab_size,
            max_document_len=_config["max_document_len"],
            embed_dim=_config["embed_dim"],
            kernel_sizes=_config["kernel_sizes"],
            feature_maps=_config["feature_maps"],
            # General
            projection_nodes=_config["projection_nodes"],
            dropout=_config["dropout"],
            embedding_dropout=_config["embedding_dropout"],
            dropout_nonlinear=_config["dropout_nonlinear"],
            dropout_concrete=_config["dropout_concrete"],
            nb_classes=nb_classes,
            multilabel=_config["multilabel"],
            use_aleatorics=_config["use_aleatorics"],
            batchnorm=_config["batchnorm"],
            # CHAR-level
            use_char=_config["use_char"],
            max_chars_len=_config["max_chars_len"],
            alphabet_size=len(_config["char_voc"]),
            char_kernel_sizes=_config["char_kernel_sizes"],
            char_feature_maps=_config["char_feature_maps"],
            version=_config["model"].split("_")[-1],
        ).build_model()

    elif "HierarchicalAttentionNetwork" in _config["model"]:
        from arkham.Bayes.Quantify.HAN import HierarchicalAttentionNetwork

        assert "sentence" in _config["composition"]
        _, model = HierarchicalAttentionNetwork(
            embedding_layer=embedding_layer,
            vocab_size=vocab_size,
            embed_dim=_config["embed_dim"],
            max_document_len=_config["max_document_len"],
            max_sentences=_config["max_sentences"],
            dropout=_config["dropout"],
            dropout_nonlinear=_config["dropout_nonlinear"],
            embedding_dropout=_config["embedding_dropout"],
            nb_classes=nb_classes,
            use_aleatorics=_config["use_aleatorics"],
            batchnorm=_config["batchnorm"],
        ).build_model()

    elif "UniversalSentenceEncoder" in _config["model"]:
        from arkham.Bayes.Quantify.USE import UniversalSentenceEncoder

        assert "sentence" in _config["composition"]
        model = UniversalSentenceEncoder(
            max_document_len=_config["max_document_len"],
            max_sentences=_config["max_sentences"],
            dropout=_config["dropout"],
            dropout_nonlinear=_config["dropout_nonlinear"],
            embedding_dropout=_config["embedding_dropout"],
            nb_classes=nb_classes,
            use_aleatorics=_config["use_aleatorics"],
            batchnorm=_config["batchnorm"],
        ).build_model()

    elif "Transformer" in _config["model"]:
        from arkham.Bayes.Quantify.transformer import Transformer

        model = Transformer(
            embedding_layer=embedding_layer,
            vocab_size=vocab_size,
            max_document_len=_config["max_document_len"],
            embed_dim=_config["embed_dim"],
            projection_nodes=_config["projection_nodes"],
            num_heads=_config["num_heads"],
            dropout=_config["dropout"],
            embedding_dropout=_config["embedding_dropout"],
            dropout_nonlinear=_config["dropout_nonlinear"],
            dropout_concrete=_config["dropout_concrete"],
            nb_classes=nb_classes,
            use_aleatorics=_config["use_aleatorics"],
            batchnorm=_config["batchnorm"],
        )
        if "Sequence" in _config["model"]:
            model = model.build_sequence_model()
        else:
            model = model.build_model()

    elif "BERT" in _config["model"]:
        from arkham.Bayes.Quantify.BERT import BERT

        model = BERT(
            model_class=_config["model_class"],
            finetune=_config["finetune"],
            encoder=_config["model"].split("_")[-1],
            projection_nodes=_config["projection_nodes"],
            max_document_len=_config["max_document_len"],
            dropout=_config["dropout"],
            embedding_dropout=_config["embedding_dropout"],
            dropout_nonlinear=_config["dropout_nonlinear"],
            dropout_concrete=_config["dropout_concrete"],
            nb_classes=nb_classes,
            multilabel=_config["multilabel"],
            use_aleatorics=_config["use_aleatorics"],
            batchnorm=_config["batchnorm"],
            sequence_labels=_config["sequence_labels"],
        ).build_model()
    else:
        from arkham.Bayes.Quantify.encoder import SimpleEncoder

        model = SimpleEncoder(
            embedding_layer=embedding_layer,
            vocab_size=vocab_size,
            max_document_len=_config["max_document_len"],
            embed_dim=_config["embed_dim"],
            projection_nodes=_config["projection_nodes"],
            encoder=_config["model"],
            composition=_config["composition"],
            max_chars_len=_config["max_chars_len"],
            alphabet_size=len(_config["char_voc"]),
            char_embed_dim=_config["char_embed_dim"],
            char_kernel_sizes=_config["char_kernel_sizes"],
            char_feature_maps=_config["char_feature_maps"],
            dropout=_config["dropout"],
            embedding_dropout=_config["embedding_dropout"],
            dropout_nonlinear=_config["dropout_nonlinear"],
            dropout_concrete=_config["dropout_concrete"],
            nb_classes=nb_classes,
            multilabel=_config["multilabel"],
            use_aleatorics=_config["use_aleatorics"],
            batchnorm=_config["batchnorm"],
        )
        if "sequential" in _config["model"]:
            model = model.build_sequence_model()
        else:
            model = model.build_model()
    return model
