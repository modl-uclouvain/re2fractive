"""A place to collect optics-specific acquisition functions."""

import numpy as np
import random
from re2fractive.acquisition.rppf import rppf_y

from optimade.adapters import Structure



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
        key=lambda s: (s["predictions"]["refractive_index"]**2
        + s["predictions"].get("refractive_index_std", 0.0) * include_std)
        * (s["predictions"]["band_gap"]
        + s["predictions"].get("band_gap_std", 0.0) * include_std),
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
        key=lambda s: (s["predictions"]["refractive_index"]**2
        + s["predictions"].get("refractive_index_std", 0.0) * include_std)
        * np.sqrt(s["predictions"]["band_gap"]
        + s["predictions"].get("band_gap_std", 0.0) * include_std),
        reverse=reverse,
    )[:num_to_select]


