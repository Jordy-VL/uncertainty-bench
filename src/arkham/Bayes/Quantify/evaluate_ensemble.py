import os
import regex as re
from arkham.Bayes.Quantify.evaluate import main as model_evaluate
from arkham.Bayes.Quantify.experiment import eval_stats
from arkham.Bayes.Quantify.compare import merge_stats
from arkham.utils.model_utils import MODELROOT, _get_weights
from arkham.utils.utils import pickle_loader


def unique_weights(modelpath):
    uniques = set()
    for x in os.listdir(modelpath):
        if "nan" in x:  # skip those with nan validation loss
            continue
        m = re.match("(?:weights)(_\d+)", x)
        if m:
            uniques.add(m.group(1))
    return sorted(uniques)


"""
def eval_stats(evaluator, model):  # could fix this with a return statement
    from prettytable import PrettyTable
    from arkham.Bayes.Quantify.compare import evaluate_model, metric_names
    t = PrettyTable(["version"] + metric_names)
    for identifier, stats in evaluator.stats.items():
        metrics, k = evaluate_model(model, identifier, stats, evaluator)
        t.add_row([k["version"]] + metrics)
    print(t)
    return t
"""


def main(
    version,
    test_data=None,
    evaluation_data=None,
    downsampling=0,
    dump=True,
    identifier="",
    raw=False,
    data_identifier="",
    **kwargs,
):
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    checkpoint_ids = unique_weights(modelpath)

    # Load or save checkpoint evaluator
    evaluators = []
    for checkpoint_id in checkpoint_ids:
        checkpoint_path = os.path.join(modelpath, checkpoint_id + "eval.pickle")
        if os.path.exists(checkpoint_path):
            evaluator = pickle_loader(checkpoint_path)
        else:
            evaluator = model_evaluate(
                version,
                test_data,
                identifier=checkpoint_id,
                dump=dump,
                raw=raw,
                data_identifier=data_identifier,
                **kwargs,
            )
        evaluators.append(evaluator)

    # Load or save ensemble [just eval.pickle]
    ensemble_identifier = ""  # "M" + str(len(checkpoint_ids)) #evaluators
    ensemble_path = os.path.join(modelpath, ensemble_identifier + "eval.pickle")
    if os.path.exists(ensemble_path):
        ensemble = pickle_loader(ensemble_path)
    else:
        ensemble = merge_stats(evaluators, modelpath, M=ensemble_identifier, identifier="")

    eval_stats(ensemble, ensemble_identifier)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("""Evaluation for Ensemble""")
    parser.add_argument("version", type=str, default="")
    parser.add_argument("test_data", nargs="?", type=str, default=None)
    parser.add_argument("-r", dest="raw", action="store_true", default=False, help="save RAW predictions with pickle")
    parser.add_argument("-c", dest="oov_corruption", type=float, default=0, help="OOV corruption")
    parser.add_argument("-s", dest="posterior_samples", type=int, default=10, help="number of forward samples")
    parser.add_argument(
        "-d",
        dest="data_identifier",
        type=str,
        default="",
        help="change data identifier to create ood (CLINC) or evaluate cross-domain [with same dataparams though]",
    )
    parser.add_argument(
        "-l", dest="predict_logits", action="store_true", default=False, help="output model logits [as well]"
    )
    parser.add_argument(
        "--skip", dest="skip_MC", action="store_true", default=False, help="skip computation of MC Dropout stats"
    )
    parser.add_argument("--cpu", dest="cpu", action="store_true", default=False, help="prediction on CPU")
    parser.add_argument("--t", dest="timings", action="store_true", default=False, help="get batch and sample timings")

    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    dump = False if args.timings else True

    main(
        args.version,
        args.test_data,
        dump=dump,
        oov_corruption=args.oov_corruption,
        posterior_sampling=args.posterior_samples,
        raw=args.raw,
        data_identifier=args.data_identifier,
        skip_MC=args.skip_MC,
        predict_logits=args.predict_logits,
        timings=args.timings,
    )
