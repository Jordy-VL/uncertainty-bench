import os
import pickle
import numpy as np
from scipy.stats import kurtosis, skew, describe
from scipy.special import softmax

from arkham.Bayes.Quantify.evaluate import dynamic_wordpiece_mask
from arkham.Bayes.Quantify.predict import get_logits
from arkham.Bayes.Quantify.compare import plot_entity_confidence

from arkham.Bayes.Quantify.inference import beam_search, diverse_beam_search

"""
The flow of this script is as follows:

For 1 example, typically a sentence, we require the following:

* text/tokenized
* idx2label/other decoding function (if nested)
* ... 

---------------------------------------------------

Next step: diverse beam search [group splitting]
diversity can be expressed in terms of the vocabulary or neigbour similarity
"""


def divide_chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Args:
        l (Iterable): sequence to be chunked
        n (TYPE): number of chunks

    Yields:
        Iterable: chunk

    # group_idx = list(divide_chunks(list(range(len(logits))), n_groups))
    # groups = list(divide_chunks(logits, n_groups))

    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def parallel_version_of_function(function_for_single_uuid, inputs, return_value=list):
    import multiprocessing
    from joblib import Parallel, delayed

    n_cores = int(multiprocessing.cpu_count() - 2)
    results = Parallel(n_jobs=n_cores)(
        delayed(function_for_single_uuid, check_pickle=False)(inputs[i]) for i in tqdm(range(len(inputs)))
    )
    return return_value(results)


def test_beam_search(beam_width):
    # Option 1: get logits for a specific example
    modelpath = "/mnt/lerna/models/conll_03-BERT_NER_biose_all_focal"
    inputted = "Hal Jordan was the best Green Lantern ever"

    gold = ["B-PER", "I-PER", "O", "O", "O", "B-MISC", "I-MISC", "O"]
    gold_es = [('PER', 0, 1), ('MISC', 5, 6)]

    tokenized, encoded, logits, unpadded = get_logits(modelpath, inputted)

    idx2label = {
        int(k): v
        for k, v in {
            "0": "S-PER",
            "1": "S-ORG",
            "2": "S-MISC",
            "3": "S-LOC",
            "4": "O",
            "5": "I-PER",
            "6": "I-ORG",
            "7": "I-MISC",
            "8": "I-LOC",
            "9": "E-PER",
            "10": "E-ORG",
            "11": "E-MISC",
            "12": "E-LOC",
            "13": "B-PER",
            "14": "B-ORG",
            "15": "B-MISC",
            "16": "B-LOC",
        }.items()
    }

    entity_level = False

    for approach in ["beam search", "DivMBest"]:
        if approach == "beam search":
            beams = beam_search(unpadded, k=beam_width)
        else:
            beams = diverse_beam_search(
                unpadded, k=beam_width, n_groups=beam_width, diversity_objective=None, div_strength=1, alpha=1
            )
        softmaxed = softmax(unpadded, axis=-1)

        decoded = [
            (np.vectorize(idx2label.get)(beam), np.ravel(np.take_along_axis(softmaxed, np.expand_dims(beam, 1), 1)))
            for beam in beams
        ]
        print(decoded)

        from seqeval.metrics.sequence_labeling import get_entities, f1_score

        confidences = []
        correctness = []

        for i in range(len(beams)):
            print(beams[i])
            print(decoded[i])

            es = get_entities(decoded[i][0].tolist())
            print(es)

            f1 = f1_score([gold], [decoded[i][0].tolist()])
            print(f1)

            if entity_level:
                # ENTITY-LEVEL
                pred_mask = [list(range(start, end + 1)) for label, start, end in es]
                confs = dynamic_wordpiece_mask(decoded[i][1], pred_mask, join=True)
                corrects = [True if e in gold_es else False for e in es]  # how about marginal correctness?

            else:
                # MARGINAL-LEVEL
                confs = decoded[i][1]
                corrects = [True if decoded[i][0][j] == gold[j] else False for j in range(len(decoded[i][0]))]

            print(confs)
            print()

            correctness.extend(corrects)
            confidences.extend(confs.tolist())

        print(describe(confidences))

        from matplotlib import pyplot as plt
        import seaborn as sns

        plot_entity_confidence(np.array(confidences), np.array(correctness), title=f"{approach} k={beam_width} G=3")

    # print(res)
    # Option 2:
    """
    with open(path + "/raw.pickle", 'rb') as input_file:
        p = pickle.load(input_file)
    """


"""
    # Decoders
    if args.decoding == "CRF": # conditional random fields
        output_layer = tf.layers.dense(hidden_layer_dropout, num_tags)
        weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            output_layer, self.tags, self.sentence_lens)
        loss = tf.reduce_mean(-log_likelihood)
        self.predictions, viterbi_score = tf.contrib.crf.crf_decode(
            output_layer, transition_params, self.sentence_lens)
        self.predictions_training = self.predictions
"""
