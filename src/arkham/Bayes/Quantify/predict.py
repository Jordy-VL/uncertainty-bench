import sys
import numpy as np

from arkham.Bayes.Quantify.data import encode_test, get_tokenizer
from arkham.Bayes.Quantify.evaluate import generate_wordpiece_mask, dynamic_wordpiece_mask
from arkham.Bayes.Quantify.evaluate import predict as model_predict
from arkham.utils.model_utils import load_model_path
from arkham.utils.custom_metrics import entropy


def predict(inputted):
    content_out = {}
    out = {}
    model = load_model_path(sys.argv[1])

    out = model_predict(*model, inputted)
    # try:
    #     content_out["success"] = True
    # except Exception as e:
    #     logging.error(e)
    content_out = {**content_out, **out}
    return content_out


def get_logits(modelpath, inputted):
    model, _config = load_model_path(modelpath)
    tokenized, encoded = encode_test(inputted, _config)
    """
    prediction = model.predict(encoded)
    predictions = {}
    predictions["index"] = np.argmax(prediction, axis=-1)[0]
    predictions["class"] = np.vectorize(_config["idx2label"].get)(predictions["index"])
    predictions["confidence"] = 100 * np.max(prediction, axis=-1)[0]
    predictions["entropy"] = entropy(prediction)
    """
    model.layers[-1].activation = None
    model.compile()
    logits = model.predict(encoded)

    tokenizer = get_tokenizer(_config.get("model_class"))
    if tokenizer:
        wordpieces = tokenizer.convert_ids_to_tokens(encoded[0][0])
        dynamic_mask = generate_wordpiece_mask(wordpieces)
        unpadded_logits = dynamic_wordpiece_mask(logits, dynamic_mask, join=False)
    else:
        unpadded_logits = None

    return tokenized, encoded, logits, unpadded_logits[0]


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    modelpath = sys.argv[1]
    inputted = sys.argv[2]
    if len(sys.argv) > 3:
        get_logits(modelpath, inputted)
    else:
        predict(inputted)

"""
#~/code/gordon/arkham/arkham/Bayes/Quantify/scripts/timediff.sh weights_02* weights_03*
py predict.py /mnt/lerna/models/conll_03_charCNN-biLSTM-glove200d-20K-casing_smooth01 "Jeremy Watson is a great guy" #cannot
py predict.py /mnt/lerna/models/conll_03-BERT_NER_biose_all_focal "Jeremy Watson is a great guy" logits
"""
