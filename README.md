# uncertainty-bench
Repository for **Benchmarking Scalable Predictive Uncertainty in Text Classification**, by Jordy Van Landeghem, Matthew Blaschko, Bertrand Anckaert and Marie-Francine Moens, JMLR 2020 (submitted).

It contains the source code of the paper, the experiments therein, and datasets used 
in the folders [`paper`](./paper), [`src`](./src), [`experiments`](./experiments), and [`datasets`](./datasets) respectively.


### Changelog

- [x] boilerplate repo
- [x] raw evaluation data
- [x] link or host datasets
- [x] update experiment instructions
- [ ] ONGOING re-implementation, see Disclaimer
- [ ] assign proper LICENSE


## Motivation

Perfect predictive accuracy is unattainable for most text classification problems, explaining the need for reliable ML solutions that can communicate predictive uncertainty when dealing with noisy or unknown inputs. In the quest for a simple, principled and scalable uncertainty method, which one to choose, when and why? 

Our survey on Bayesian Deep Learning methods and benchmarking on 6 different text classification datasets aims to help practicioners make this decision and have future researchers spurred to continue investigations into hybrid uncertainty methods. 

## Methods

<p align="middle">
<img src="/paper/images/legend_diversity.png" width="45%" alt="Methods and identifiers">
<img src="/paper/images/sngp_legend_only.png" width="45%" alt="SNGP Methods">
</p>


## Installation

[`Requirements and setup`](./experiments/README.md)


## Usage

[`Detail usage`](./experiments/README.md)


### Training a model
_main file: `experiment.py`_

Example command:
```
python3 experiment.py CONFIG_NAME
```

### Extending with your uncertainty method [TBD]


## Citation
```
@inproceedings{VanLandeghem2020a,
  TITLE = {Benchmarking Scalable Predictive Uncertainty in Text Classification},
  AUTHOR = {Van Landeghem, Jordy and Blaschko, Matthew B. and Anckaert, Bertrand and Moens, Marie-Francine},
  BOOKTITLE = {Submitted to Journal of Machine Learning Research},
  YEAR = {2020/1}
}
```

## Results

<img src="/paper/images/single.png" alt="KDE plots of uncertainty in OOD detection task">


## Disclaimer
The code was originally run in a corporate environment*, now reimplemented and open-sourced for aiding the research community. 
There will be small changes between the current output & results presented in the paper.


<img src="https://contract.fit/wp-content/uploads/2019/11/logo-2.png" width="350" alt="CF logo">
