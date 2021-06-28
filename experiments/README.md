```python
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "1.0"
```

# Benchmarking Scalable Predictive Uncertainty in Text Classification

📋 This repository is the official implementation of [Benchmarking Scalable Predictive Uncertainty in Text Classification](). 
The original library is called `arkham`, for continued development I have kept this framework structure for the opensource as well.


## Requirements

To install requirements:

```setup
git clone git@github.com:Jordy-VL/uncertainty-bench.git 
mv uncertainty-bench/src $HOME/code/arkham
mkvirtualenv -p /usr/bin/python3.6 -a $HOME/code/arkham arkham
pip3 install poetry
workon arkham; poetry install
```

Main requirements are: 
* TensorFlow 2
* Sacred #for experiment tracking

### Setup

Add a `configfile.py` in `$HOME/code/arkham` with the following *GLOBALS*:
* MODELROOT as destination for saving model artefacts and re-loading for evaluation
* DATAROOT can be a directory "data" at the same level as `experiment.py`; use a symbolic link if needed

### Configs

In Sacred, configs are used to detail hyperparameter configs to the framework. Based on certain switches, different models and uncertainty methods will be activated.
Additionally, from the commandline, you can override a certain parameter. For example runs, see `benchmark_runs.md`.

#### To test the default setup:

```sh
python3 experiment.py with clf_default "identifier=mini_imdb" "model=cnn_baseline" "steps_per_epoch=None"
```

## Repository structure

The most important implementations and helper functions are in these subdirectories:

*  arkham/arkham/Bayes/Quantify
*  arkham/arkham/utils
*  arkham/arkham/Bayes/GP
*  arkham/arkham/Bayes/MCMC


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


### Advanced Instructions

#### Adding a new dataset

Any new dataset should be a subdirectory of DATAROOT, with an identifier, which triggers the appropriate dataloader in `data.py`. 
There are plently examples there. The subdirectory itself should contain any one of three labelsets `train`, `dev` and/or `test`.
We include support for `flair`, `tensorflow-datasets` and `huggingface-nlp` datasets as examples.

#### Adding a new uncertainty method

Highly dependent on what architectural, optimization or additional hyperparameter changes are required.
Generally, add a standalone .py file with the `tf.keras.model` with `tf.keras.optimizers.optimizerV2`, callbacks, loss and compilation code. 
All other functionality should already run out-of-the-box.
Ideally, you include a new model switch in `experiment.py`, based on the `model_identifier` parameter in the default or a new config.
In order to load the model, this new object creation code should be imported in `model_utils.py`.


### More details [TBD]

Link to paper


#### Reproducing main results

Run the commands in `benchmark_runs.md`


## Citing

If you use any part of this code in your research, please cite it using the following BibTex entry
```cite
@inproceedings{VanLandeghem2020a,
  TITLE = {Benchmarking Scalable Predictive Uncertainty in Text Classification},
  AUTHOR = {Van Landeghem, Jordy and Blaschko, Matthew B. and Anckaert, Bertrand and Moens, Marie-Francine},
  BOOKTITLE = {Submitted to Journal of Machine Learning Research},
  YEAR = {2020/1}
}

@misc{ContractfitUquant,
  author = {Jordy Van Landeghem},
  title = {Benchmarking Scalable Predictive Uncertainty in Text Classification},
  year = {2020},
  publisher = {ICML},
  journal = {ICML workshop UDL},
  howpublished = {\url{https://github.com/Jordy-VL}},
}
```
