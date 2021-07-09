# uncertainty-bench
Repository for **Benchmarking Scalable Predictive Uncertainty in Text Classification**, by Jordy Van Landeghem, Matthew Blaschko, Bertrand Anckaert and Marie-Francine Moens, JMLR 2020 (submitted).

It contains the source code, the experiments, and datasets used 
in  [`src`](./src), [`experiments`](./experiments), and [`datasets`](./datasets) respectively.

## Motivation

Perfect predictive accuracy is unattainable for most text classification problems, explaining the need for reliable ML solutions that can communicate predictive uncertainty when dealing with noisy or unknown inputs. In the quest for a simple, principled and scalable uncertainty method, which one to choose, when and why? 

Our survey on Bayesian Deep Learning methods and benchmarking on 6 different text classification datasets aims to help practicioners make this decision and have future researchers spurred to continue investigations into hybrid uncertainty methods. 

## Methods

<p align="middle">
<img src="/images/legend_diversity.png" width="45%" alt="Methods and identifiers">
<img src="/images/sngp_legend_only.png" width="45%" alt="SNGP Methods">
</p>


## Installation

[`Requirements and setup`](./src/README.md)


## Usage

[`Detail usage`](./src/README.md)


### Training a model
_main file: `experiment.py`_

Example command:
```
python3 experiment.py with CONFIG_NAME identifier=DATASET 
```

## Citation
```
@inproceedings{VanLandeghem2021,
  TITLE = {Benchmarking Scalable Predictive Uncertainty in Text Classification},
  AUTHOR = {Van Landeghem, Jordy and Blaschko, Matthew B. and Anckaert, Bertrand and Moens, Marie-Francine},
  BOOKTITLE = {Submitted to Journal of Machine Learning Research},
  YEAR = {2021}
}
```

## Results

<img src="/images/single.png" alt="KDE plots of uncertainty in OOD detection task">


## Disclaimer
The code was originally run in a corporate environment*, now reimplemented and open-sourced for aiding the research community. 
There will be small changes between the current output & results presented in the paper.


<img src="https://contract.fit/wp-content/uploads/2019/11/logo-2.png" width="350" alt="CF logo">

<!---
### Changelog

- [x] boilerplate repo
- [x] raw evaluation data
- [x] link or host datasets
- [x] update experiment instructions
- [x] assign proper LICENSE
- [x] re-implementation, see Disclaimer
---!>
