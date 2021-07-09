import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pytest
import numpy as np
from arkham.Bayes.Quantify.evaluate import TouristLeMC
from arkham.Bayes.Quantify.compare import deduce_methods, make_style

import pickle
import pandas as pd
from scipy.stats import pointbiserialr, pearsonr


from arkham.utils.custom_metrics import entropy, exp_entropy, pred_entropy, mutual_info, AUROC_PR


import matplotlib
from matplotlib import pyplot as plt

# plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
import seaborn as sns

sns.set(color_codes=True)


def bilaj(mc_pred):
    # T/M x N x K
    sample_variance = np.var(mc_pred, axis=0)  # N x K
    sample_mean = np.mean(mc_pred, axis=0)  # N x K

    ensemble_mean = np.mean(sample_mean, axis=-1)  # N

    total_variance = (
        np.mean([(np.var(s, axis=-1) + np.mean(s, axis=-1) ** 2) for s in mc_pred], axis=0) - ensemble_mean ** 2
    )
    return np.zeros(len(total_variance)), total_variance


def bilaj_v2(mc_pred):
    # T/M x N x K
    sample_variance = np.var(mc_pred, axis=0)  # N x K
    sample_mean = np.mean(mc_pred, axis=0)  # N x K

    ensemble_mean = np.mean(sample_mean, axis=-1)  # N

    total_variance = (
        np.mean([(np.var(s, axis=-1) + np.mean(s, axis=-1) ** 2) for s in mc_pred], axis=0) - ensemble_mean ** 2
    )

    # % following definition in Bachstein and derived from total variance formulation above [reorder mean]
    aleatorics = np.mean([np.var(s, axis=-1) for s in mc_pred], axis=0)  # variance at K averaged over samples -> #N
    epistemics = np.mean([np.mean(s, axis=-1) ** 2 for s in mc_pred], axis=0) - ensemble_mean ** 2

    """
    try:
        assert np.testing.assert_almost_equal(aleatorics + epistemics, total_variance)
    except AssertionError as e:
        print(e)
        print(aleatorics)
        print(epistemics)
        print(aleatorics + epistemics)
        print(total_variance)
    NOT equivalent, almost close though! 
    """

    return aleatorics, epistemics


def xiaoetal(mc_pred, means=None, variances=None):
    """Calculate both Epistemic and Heteroscedastic Aleatoric uncertainties. """
    # size: num_passes x batch_size x num_classes

    """
    Pytorch source:
    ---------------

    means = torch.cat([dist.loc.unsqueeze(0) for dist in dists], dim=0)  # .loc
    aleatoric_uncertainties = torch.cat(
        [dist.variance.unsqueeze(0) for dist in dists], dim=0)  # .variance

    aleatorics = aleatoric_uncertainties.mean(0).mean(-1)  # almost entropy
    epistemics = means.var(0).mean(-1)  # mean variance
    return aleatorics, epistemics # batch_size
    """
    if means is None:  # then recalculate them...   MC DROPOUT
        means = np.mean(mc_pred, axis=0)  # N x K
        variances = np.var(mc_pred, axis=0)  # N x K

        aleatorics = np.mean(variances, -1)  # N
        epistemics = np.sqrt(np.sum(variances, axis=-1))  # total variance :) #np.var

    else:  # HETEROSCEDASTIC extensions
        aleatorics = np.mean(np.mean(variances, 0), -1)
        epistemics = np.mean(np.var(means, 0), -1)
    return aleatorics, epistemics


def kwonetal(batch_samples):
    def kwon(samples):
        """
         p_hat with dimension (number of estimates, dimension of features)
        binary case:

        epistemic = np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2
        aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

        formula:
        \[\frac{1}{T} \sum_{t=1}^{T}[\operatorname{diag}\{p(y^{*} | x^{*}, \hat{\omega}_{t})\}-p(y^{*} | x^{*}, \hat{\omega}_{t})^{\otimes 2}]\]
        \[\frac{1}{T} \sum_{t=1}^{T}\{p(y^{*} | x^{*}, \hat{\omega}_{t})-\hat{p}_{\hat{\theta}}(y^{*} | x^{*})\}^{\otimes 2}\]

        """
        # something about covariance! X*X.T => instead of variance
        p_mean = np.mean(samples, axis=0)  # average over samples #still does not seem correct?
        aleatorics = np.mean([(np.diag(s) - s) ** 2 for s in samples])
        epistemics = np.mean([(s - p_mean) ** 2 for s in samples])
        return aleatorics, epistemics

    batch_size = batch_samples.shape[1]
    epistemics, aleatorics = np.zeros((batch_size)), np.zeros((batch_size))
    for i in range(batch_size):  # batchsize
        samples = batch_samples[:, i, :]
        aleatorics[i], epistemics[i] = kwon(samples)
        """
        p_mean = np.mean(samples, axis=0)  # average over samples #still does not seem correct?
        aleatorics[i] = np.mean([np.diag(s) - (s * s.T) for s in samples])
        epistemics[i] = np.mean([((s - p_mean) * (s - p_mean).T) for s in samples])
        """
    return aleatorics, epistemics


def current_epistemic(mc_pred, means=None, variances=None):  # LOOKS INCORRECT!!!; seems to be aleatoric?
    epistemics = np.mean(np.var(mc_pred, axis=0, ddof=1), axis=-1)  # calculated as softmax variance
    return epistemics


def current_aleatoric(mc_pred, means=None, variances=None):  # looks very much correct and in line with formulas
    variances = np.var(
        mc_pred, axis=0
    )  # N x K; whereas K layer variances = T x N x K; so have to mean first there as well
    aleatorics = np.mean(variances, axis=-1)
    # batchsize ; first over samples, then over classes
    return aleatorics


def current(mc_pred, means=None, variances=None):
    return (
        current_aleatoric(mc_pred, means=means, variances=variances),
        current_epistemic(mc_pred, means=means, variances=variances),
    )


def samples_first(tensors):
    return np.transpose(np.array(tensors), (1, 0, 2))


def test_uquantities():
    path = "/mnt/lerna/models/20news_aleatoric_multivar_0_ood"
    # path = "/mnt/lerna/models/Reuters_aleatoric_multivar_ood"

    with open(path + "/eval.pickle", 'rb') as input_file:
        p = pickle.load(input_file)

    stats = p.stats["mc"]
    raw = np.array(stats["raw"])
    # how about also taking means and variances? :) or just the distributions
    print(raw.shape)  # N x T x K

    """
    sample = raw[:5]
    transposed = samples_first(sample)  # T x N x K

    for method in [bilaj, bilaj_v2, xiaoetal, kwonetal, current]:
        name = method.__name__
        aleatorics, epistemics = method(transposed)
        print(f"{name}: \nepistemics: {epistemics} \naleatorics: {aleatorics}")
        print()
    # transpose

    print("with means and variances")
    means = samples_first(stats["means"][:5])
    variances = samples_first(stats["variances"][:5])

    for method in [xiaoetal, current]:
        name = method.__name__ + "_aleatorics"
        aleatorics, epistemics = method(transposed, means=means, variances=variances)
        print(f"{name}: \nepistemics: {epistemics} \naleatorics: {aleatorics}")
        print()

    """

    print("***" * 50)

    collect = []
    groundtruth = stats["gold"]
    unknown = [
        'comp.graphics',
        'comp.sys.ibm.pc.hardware',
        'comp.windows.x',
        'rec.autos',
        'rec.sport.baseball',
        'sci.crypt',
        'sci.med',
        'soc.religion.christian',
        'talk.politics.mideast',
        'talk.religion.misc',
    ]
    # [0]
    known = np.array([1 if groundtruth[i] in unknown else 0 for i in range(0, len(groundtruth))])
    unk = np.where(np.isin(stats["gold"], unknown))[0]
    raw = samples_first(raw)
    means = samples_first(stats["means"])
    variances = samples_first(stats["variances"])

    for i, method in enumerate([bilaj, bilaj_v2, xiaoetal, kwonetal, current, xiaoetal]):
        name = method.__name__
        if i in [5, 6]:
            aleatorics, epistemics = method(raw, means=means, variances=variances)
            name += "_heteroscedastic"
        else:
            aleatorics, epistemics = method(raw)
        # print(f"{name}: \nepistemics: {epistemics} \naleatorics: {aleatorics}")

        for n, uncertainties in [("aleatorics", aleatorics), ("epistemics", epistemics)]:
            correlations = {}
            namen = name + "_" + n
            correlation, p = pointbiserialr(known, uncertainties)
            auroc, aupr = AUROC_PR(uncertainties[~unk], uncertainties[unk], pos_label=0)
            print(f"{namen}: \nCorrelation: {correlation} \nAUPR: {aupr}")
            # print(f"{name}: \nepistemics: {epistemics} \naleatorics: {aleatorics}")
            print()

            correlations["version"] = namen
            correlations[n + "_R"] = correlation
            correlations[n + "_p"] = (
                lambda p: "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
            )(p)
            correlations["AUPR"] = aupr
            # MAYBE ADD confidences / entropy?
            collect.append(correlations)

    df = pd.DataFrame(collect)
    print(df)
    df["quantity"] = df["version"].apply(lambda x: "aleatorics" if "aleatorics" in x else "epistemics")

    for y in ["aleatorics_R", "epistemics_R"]:  # , "AUPR"]:
        x = "version"
        plt.title(y)
        plt.xticks(rotation=90)
        sns.stripplot(x=x, y=y, hue="quantity", data=df.sort_values(by=[y, x], ascending=True))
        plt.show()

    df.fillna(0, inplace=True)
    df = make_style(df, absolute=True)
    df.to_excel(path + "/uncertainties.xlsx", index=False)


# test_uquantities()
# means and variances are not saved :o

"""
Aleatorics seems to be correct :) 
Epistemics in 
MC dropout == closely aleatorics
Heteroscedastic == too close to aleatorics 

try with saving means and variances
"""
