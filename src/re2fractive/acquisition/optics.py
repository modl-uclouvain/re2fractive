"""A place to collect optics-specific acquisition functions."""

import numpy as np
from optimade.adapters import Structure


def from_w_eff(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    include_std: bool = False,
    num_to_select: int = 1,
    order="max",
):
    """Rank by Naccarato et al fit for w_eff."""
    y = np.array([d["predictions"]["refractive_index"] for d in candidate_pool])
    x = np.array([d["predictions"]["band_gap"] for d in candidate_pool])

    w_eff = (((y**2) - 1) ** (1 / 3)) * (x + 6.74 + (-1.19 / x))

    reverse = False
    if order == "max":
        reverse = True

    sorted_pool = np.argsort(w_eff).tolist()
    if reverse:
        sorted_pool = reversed(sorted_pool)

    selected = []
    for _ in range(num_to_select):
        idx = sorted_pool.pop()
        selected.append(candidate_pool[idx])
    return selected


def fom_high_k_leakage(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    include_std: bool = False,
    num_to_select: int = 1,
    order="max",
):
    """Returns the top `num_to_select` structures according to the figure of merit eps*Eg (current leakage), accounting
    for uncertainty if available AND if include_std==True.

    Parameters:
        candidate_pool: The set of structures that can be selected for further investigation, that are currently
            missing values of the desired properties from the oracles.
        decorated_structures: The set of structures (potentially with precomputed properties) that can be used
            to inform the selection process, but cannot themselves be selected.
        include_std: True if the uncertainty should be considered when optimizing.
        num_to_select: How many candidates to suggest.
        order: Either `max` or `min` to indicate whether to maximize or minimize the FOM.

    Returns:
        A list of suggested structures.

    """
    if order not in ("max", "min"):
        raise RuntimeError("order must be either 'max' or 'min'")

    reverse = False
    if order == "max":
        reverse = True

    return sorted(
        candidate_pool,
        key=lambda s: (
            s["predictions"]["refractive_index"] ** 2
            + s["predictions"].get("refractive_index_std", 0.0) * include_std
        )
        * (
            s["predictions"]["band_gap"]
            + s["predictions"].get("band_gap_std", 0.0) * include_std
        ),
        reverse=reverse,
    )[:num_to_select]


def fom_high_k_energy(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    include_std: bool = False,
    num_to_select: int = 1,
    order="max",
):
    """Returns the top `num_to_select` structures according to the figure of merit eps*sqrt(Eg) (stored energy), accounting
    for uncertainty if available AND if include_std==True.

    Parameters:
        candidate_pool: The set of structures that can be selected for further investigation, that are currently
            missing values of the desired properties from the oracles.
        decorated_structures: The set of structures (potentially with precomputed properties) that can be used
            to inform the selection process, but cannot themselves be selected.
        include_std: True if the uncertainty should be considered when optimizing.
        num_to_select: How many candidates to suggest.
        order: Either `max` or `min` to indicate whether to maximize or minimize the FOM.

    Returns:
        A list of suggested structures.

    """
    if order not in ("max", "min"):
        raise RuntimeError("order must be either 'max' or 'min'")

    reverse = False
    if order == "max":
        reverse = True

    return sorted(
        candidate_pool,
        key=lambda s: (
            s["predictions"]["refractive_index"] ** 2
            + s["predictions"].get("refractive_index_std", 0.0) * include_std
        )
        * np.sqrt(
            s["predictions"]["band_gap"]
            + s["predictions"].get("band_gap_std", 0.0) * include_std
        ),
        reverse=reverse,
    )[:num_to_select]
