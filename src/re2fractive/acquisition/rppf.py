import numpy as np

import re2fractive.acquisition.utils as utils


def rppf_y(y, rho=0.0, wind=0.01, Tstar=0.01):
    objnum = len(y[0])

    # min-max normalization for each objective function
    E = y.T
    E_minmax = (E - np.min(E, axis=1, keepdims=True)) / (
        np.max(E, axis=1, keepdims=True) - np.min(E, axis=1, keepdims=True)
    )

    # Definition of alpha
    weights = utils.generate_weights(M=objnum, wind=wind)

    ###################################
    ##### free energy evaluations #####
    ###################################

    # Eq.(5) - augmented weighted Tchebycheff
    H_all = utils.matmul_max(weights, E_minmax)
    H_all += rho * np.ones(weights.shape) @ E_minmax

    # Eq.(6) - min-max standardization
    H_all_minmax = (H_all - np.min(H_all, axis=1, keepdims=True)) / (
        np.max(H_all, axis=1, keepdims=True) - np.min(H_all, axis=1, keepdims=True)
    )

    # Eq.(1),(5) - one of the Pareto solutions is located at the optimal solution depending on α
    pareto_list = np.argmin(H_all, axis=1)

    # Eq.(7)-(8) - calculation of MIPS score

    FT = -Tstar * np.log(
        np.sum(
            np.exp(
                -H_all_minmax / Tstar
                - np.max(-H_all_minmax / Tstar, axis=1, keepdims=True)
            ),
            axis=1,
        )
    ) + np.max(-H_all_minmax / Tstar, axis=1)

    #####################
    ##### opt value #####
    #####################

    arg_index = np.argsort(FT)[::-1]
    sorted_pareto = pareto_list[arg_index]

    # Get unique values without sorting
    uniques, indices = np.unique(sorted_pareto, return_index=True)

    # Sort the indices to get the original order of the sorted pareto list
    sorted_indices = np.sort(indices)
    ranking_index = sorted_pareto[sorted_indices]
    ranking_MIPS = FT[arg_index][sorted_indices]

    return ranking_index, ranking_MIPS
