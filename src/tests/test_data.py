import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.expanduser("~/code/gordon/arkham"))
from arkham.Bayes.Quantify.data import generators_from_directory


def test_generators():
    for dataset in ["conll_03", "imdb", "yelp/2013", "yelp/2014", "yelp/2015"]:
        print("DATASET: ", dataset)
        generators, voc2idx, label2idx, _ = generators_from_directory(
            os.path.join(pytest.DATAROOT, dataset), downsampling=0.1, debug=False
        )  # describe
        for batch, (x, y) in enumerate(generators["test"]):
            if batch == 0:
                print("batch: ", batch, "xshape", x.shape, "yshape", y.shape)


def test_yelp():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "yelp/2013"), downsampling=0.1, composition=["word", "sentence"], debug=True
    )


def test_ACE2005_data():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "ACE2005"),
        token_pattern="",
        min_token_len=0,
        sequence_labels=True,
        max_document_len=0,
        downsampling=0,
        raw=False,
    )


def test_conll_data():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "conll_03"),
        token_pattern="",
        min_token_len=0,
        sequence_labels=True,
        max_document_len=0,
        downsampling=0,
        raw=True,
    )

    for batch, (x, y) in enumerate(generators["test"]):
        if batch == 0:
            print("batch: ", batch, "xshape", x.shape, "yshape", y.shape)


def test_conll_chars_casing():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "conll_03"),
        token_pattern="",
        min_token_len=0,
        sequence_labels=True,
        max_document_len=0,
        downsampling=0,
        raw=False,
        max_chars_len=52,
        composition=["word", "character", "casing"],
        debug=True,
    )

    for batch, (x, y) in enumerate(generators["test"]):
        if batch == 0:
            print("batch: ", batch, "x", x, "yshape", y.shape)


def test_ontonotes():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "ontonotes"),
        token_pattern="",
        min_token_len=0,
        sequence_labels=True,
        max_document_len=0,
        downsampling=0,
        raw=False,
        max_chars_len=52,
        composition=["word", "character"],
        debug=True,
        pretrained_embeddings="/mnt/lerna/embeddings/glove.6B.50d.txt",
    )

    print(len(voc2idx))


def test_conll_corrected_data():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "conll_03_corrected"), downsampling=0, debug=True
    )


def test_imdb():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "imdb"), downsampling=0, debug=True
    )


def test_wos():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "WOS-46985"), downsampling=0, debug=True
    )


def test_twitter():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "twitter_prep"), downsampling=0, max_document_len=100, debug=True
    )


def test_reuters_debug():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "Reuters"), downsampling=0, debug=True
    )


def test_reuters_multilabel():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "Reuters_multilabel"), downsampling=0, debug=True, raw=False
    )


def test_agnews():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "AGNews"), downsampling=0, composition=["word"], debug=True
    )


def test_clinc_oos():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "CLINC150"), downsampling=0, composition=["word"], debug=True, raw=True
    )  # _ood


def test_AAPD():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "AAPD"), downsampling=0, composition=["word"], debug=True, raw=True
    )


def test_20news():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "20news"), downsampling=0, composition=["word"], debug=True
    )


def test_SST():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "SST-2"), downsampling=0, composition=["word"], debug=True
    )
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "SST-5"), downsampling=0, composition=["word"], debug=True
    )


def test_amazon_reviews():
    vocd = {}
    vocz = {}
    for domain in ["books", "dvd", "electronics", "kitchen"]:
        print(domain)
        generators, voc2idx, label2idx, _ = generators_from_directory(
            os.path.join(pytest.DATAROOT, "amazon_reviews/" + domain), downsampling=0, composition=["word"], debug=True
        )
        vocz.update(voc2idx)
        vocd[domain] = [k for k, v in voc2idx.items() if v < 5000]
    for domain in ["books", "dvd", "electronics", "kitchen"]:
        for domain2 in ["books", "dvd", "electronics", "kitchen"]:
            if domain == domain2:
                continue
            overlap = len(set(vocd[domain]).intersection(set(vocd[domain2])))
            rel = overlap / len(vocd[domain])
            print(f"{domain}-{domain2}: {rel}")
        # overlap?


def test_huggingface():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "rotten_tomatoes"), downsampling=0, composition=["word"], debug=True
    )

    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "yelp_polarity"), downsampling=0, composition=["word"], sets=["test"], debug=True
    )


def test_bert():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "CLINC150"), downsampling=0, composition=["word"], model_class="bert-base-uncased"
    )  # _ood

    import pdb

    pdb.set_trace()  # breakpoint 5eaa00c7 //


def test_sentence_comp():
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "20news"), downsampling=0.1, sets=["dev"], composition=["word"], debug=True
    )
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "20news"),
        downsampling=0.1,
        sets=["dev"],
        composition=["word", "sentence"],
        debug=True,
    )
    generators, voc2idx, label2idx, _ = generators_from_directory(
        os.path.join(pytest.DATAROOT, "20news"),
        downsampling=0,
        sets=["dev"],
        composition=["word", "sentence"],
        max_sentences=100,
        debug=True,
    )


def test_correct_encoded():
    good = [[0, 2, 2], [3, 4, 5]]
    bad = [[], [3, 4, 5]]
    bad2 = []
    print(not (not all(good) or not any(good)))
    print(not (not all(bad) or not any(bad)))
    print(not all(bad2) or not any(bad2))


def test_databatcher():
    def generate_encoded_doc(n_sentences):
        return np.array(
            [np.array([np.random.randint(100) for i in range(x + 1)], dtype=np.int32) for x in range(n_sentences)]
        )

    import tensorflow as tf

    batch_size = 2
    max_sentences = None
    max_document_len = None

    sentences = [generate_encoded_doc(10), generate_encoded_doc(12), generate_encoded_doc(15), generate_encoded_doc(13)]

    labels = [[0], [1], [2], [2]]
    dataset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(sentences), labels))
    dataset.padded_batch(batch_size, ((None, None), (None)))
