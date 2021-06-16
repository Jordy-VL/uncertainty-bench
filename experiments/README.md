```python
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"
```

# Predictive Uncertainty for Probabilistic Novelty Detection in Text Classification

This repository is the official implementation of [Predictive Uncertainty for Probabilistic Novelty Detection in Text Classification](). 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
mkvirtualenv -p /usr/bin/python3.6 -a $HOME/code/arkham arkham
pip3 install poetry
workon arkham; poetry install
```

Main requirements are: 
* TensorFlow 2
* Sacred #for experiment tracking

Additionally, setting up MODELROOT and DATAROOT:
* MODELROOT needs to be altered in utils/model_utils.py
* DATAROOT is assumed to be a directory "data" at the same level as experiment.py; we advice to use a symbolic link to any folder desired 


To test setup:

```sh
python3 experiment.py with clf_default "identifier=mini_imdb" "model=cnn_baseline" "steps_per_epoch=None"
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python3 experiment.py with clf_default identifier=<path_to_data> 
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate a trained model on its testset, run:

```eval
python3 evaluate.py <path_to_model>
```

This will write out an eval.pickle to <path_to_model> containing all the raw data to generate statistics and visualizations


To evaluate and compare multiple models for a dataset:

```compare
python3 compare.py <path_to_model> <identifier>
#e.g. Reuters
```

Output looks like the following example: 
```
+-------------------------------------------+----------+----------+------------+------------+--------+--------+--------+--------+-----------+---------+--------+
|                  version                  | accuracy |   mse    | F1 (micro) | F1 (macro) |  NLL   |  ECE   | Brier  | µ_conf | µ_entropy | µ_model | µ_data |
+-------------------------------------------+----------+----------+------------+------------+--------+--------+--------+--------+-----------+---------+--------+
|      <path-to-model>_mc                   |  0.724   | 407.1914 |   0.7239   |   0.706    | 1.2844 | 0.0683 | 0.3982 | 0.6666 |   2.1333  |  0.0006 | 0.0094 |
|      <path-to-model>_nonbayesian          |  0.7233  | 423.0255 |   0.7229   |   0.7044   | 1.3522 | 0.1195 | 0.4106 | 0.609  |   2.7714  |   0.0   | 0.0181 |
+-------------------------------------------+----------+----------+------------+------------+--------+--------+--------+--------+-----------+---------+--------+
```

To run evaluations on a model with Out-of-Distribution data: 
```ood
python3 ood.py <path_to_model>
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Details

PAPER link again. 


## Citing

If you use any part of this code in your research, please cite it using the following BibTex entry
```cite
@misc{ContractfitUquant,
  author = {Jordy Van Landeghem},
  title = {Predictive Uncertainty for Probabilistic Novelty Detection in Text Classification},
  year = {2020},
  publisher = {ICML},
  journal = {ICML workshop UDL},
  howpublished = {\url{https://github.com/Jordy-VL}},
}
```

## License
Apache License 2.0