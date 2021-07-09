# Imports and variables
import numpy as np

np.random.seed(42)
import itertools
import random
from copy import deepcopy
from scipy import stats
from scipy.special import softmax
import torch

from arkham.StructuredPrediction.synthetic import generate_sequence_simplex
from arkham.StructuredPrediction.MMD import median_heuristic, kernel_two_sample_test, plot_MMD
from arkham.StructuredPrediction.stochastic_sampling import gumbel_top_k_sampling, strategy_entropy


def test_MMD_exact():
    """
    EXACT yet rand
    """
    L = 4  # sequence of coin flips, translation of sequence; assume FIXED length
    K = 2  # binary: H/T OR some categorical
    domain_size = K ** L  # D
    exact_options = itertools.product(np.arange(0, K), repeat=L)
    M = 2  # observations from probability simplex per sample
    n = 100  # samples
    trials = 1
    sample_with_replacement = False  # relevant if more observations than domain size

    # Here: what we will change between P and Q [next to different random seeds]
    connectivities = [1, 0.5]

    """
    1. Construct P and Q
    """
    # create prior on K and on conditional dependencies for gold
    _, _, prior = generate_sequence_simplex(num_nodes=L, connectivity=1, K=K, seed=42)
    d = torch.distributions.bernoulli.Bernoulli(prior)
    gold = torch.stack([d.sample() for i in range(n)])

    # over multiple trials, use different base seeds
    for seed in range(trials):
        np.random.seed(seed)
        P = []
        Q = []
        ## only relevant to collect if we want to compare against another metric
        # P_samples = []
        # Q_samples = []

        # each element in P or Q is a 1D vector with probability of the class in the exponential simplex?
        # OR some flattened vector of combinatorial size
        for sample in range(n):
            sample_seed = n + seed + sample  # for a random/norm logits seed

            modelP, graphP, _ = generate_sequence_simplex(
                num_nodes=L, connectivity=connectivities[0], K=K, seed=sample_seed
            )
            """
            # Relevant once sampling M

            result = gumbel_top_k_sampling(modelP, graphP, M, m=1, strategy=strategy_entropy) #for 1 example; scores won't change if exact
            samples, G_phi, iterations = zip(*result) # tuple with preds, tuple with scores,  
            """
            flat_phi_xi = torch.flatten(modelP)
            P.append(flat_phi_xi)

            # double the sample seed to ensure no overlap
            modelQ, graphQ, _ = generate_sequence_simplex(
                num_nodes=L, connectivity=connectivities[1], K=K, seed=sample_seed + sample_seed
            )
            """
            result = gumbel_top_k_sampling(modelP, graphQ, M, m=1, strategy=strategy_entropy) #for 1 example; scores won't change
            samples, G_phi, iterations = zip(*result) # tuple with preds, tuple with scores,
            """
            flat_phi_xi = torch.flatten(modelQ)
            Q.append(flat_phi_xi)

        P = torch.stack(P)
        Q = torch.stack(Q)

        """
        2. Median heuristic, Gaussian kernel MMD²(u)
        """
        sigma2 = median_heuristic(P, Q)

        """
        3. Apply Gaussian kernel MMD²(u)
        """
        mmd2u, mmd2u_null, p_value = kernel_two_sample_test(
            P, Q, kernel_function='rbf', gamma=1.0 / sigma2, verbose=True
        )
        plot_MMD(mmd2u, mmd2u_null, p_value)


def test_sampling_logits():
    # cannot do this with large simplices
    M = 20
    modelP, graphP, _ = generate_sequence_simplex(num_nodes=5, connectivity=1, K=3, seed=42)
    result = gumbel_top_k_sampling(
        modelP, graphP, M, m=1, strategy=strategy_entropy
    )  # for 1 example; scores won't change if exact
    samples, G_phi, iterations = zip(*result)  # tuple with preds, tuple with scores,

    import pdb

    pdb.set_trace()  # breakpoint 44454c5e //


# test_sampling_logits()
# test_MMD_exact()
