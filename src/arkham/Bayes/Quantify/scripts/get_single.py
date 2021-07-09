#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"

import os
import shutil
import json
import regex as re
from arkham.utils.model_utils import MODELROOT, _get_weights, load_model_path, load_model_config
from arkham.Bayes.Quantify.evaluate import main as model_evaluate


def make_new_dir(new_name):
    try:
        os.mkdir(new_name)
    except Exception as e:
        raise e


def main(version, identifier):
    """
    Load model by version; reupdate params
    """
    modelpath = os.path.join(MODELROOT, version) if not os.path.exists(version) else version
    model_weights_path = _get_weights(modelpath, identifier=identifier)

    if "M" in identifier:
        new_name = modelpath.replace("_M5", "")
        make_new_dir(new_name)
        _config = load_model_config(modelpath)
        _config["ensemble"] = 1
        with open(os.path.join(new_name, "params.json"), "w") as f:
            json.dump(_config, f, indent=4)
    else:
        new_name = modelpath + identifier
        make_new_dir(new_name)
        shutil.copyfile(
            os.path.join(modelpath, "params.json"), os.path.join(new_name, "params.json"), follow_symlinks=False
        )

    shutil.copyfile(
        os.path.join(modelpath, identifier + "eval.pickle"),
        os.path.join(new_name, "eval.pickle"),
        follow_symlinks=False,
    )

    try:
        shutil.copyfile(
            os.path.join(model_weights_path),
            os.path.join(new_name, os.path.basename(model_weights_path)),
            follow_symlinks=False,
        )
    except Exception as e:
        print(e)
        shutil.copytree(
            os.path.join(model_weights_path),
            os.path.join(new_name, os.path.basename(model_weights_path)),
            copy_function=shutil.copy,
        )

    # model_evaluate(new_name)
    print(f"Saved identifier with params & weights & eval to {new_name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("""Evaluation with Monte Carlo Dropout""")
    parser.add_argument("version", type=str, default="")
    parser.add_argument("test_data", nargs="?", type=str, default=None)
    parser.add_argument(
        "-i", dest="identifier", type=str, default="M0_", help="identifier to add to eval.pickle [ensemble]"
    )
    parser.add_argument(
        "-d",
        dest="data_identifier",
        type=str,
        default="",
        help="change data identifier to create ood (CLINC) or evaluate cross-domain [with same dataparams though]",
    )
    args = parser.parse_args()
    main(
        args.version,
        # args.data_identifier,
        identifier=args.identifier,
    )
