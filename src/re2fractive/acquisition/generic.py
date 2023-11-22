"""A place to collect generic acquisition functions."""

import numpy as np
import random
from re2fractive.acquisition.rppf import rppf_y

from optimade.adapters import Structure



def exploration(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    property: str,
    num_to_select: int = 1,
):
    """Returns the top `num_to_select` structures with the largest uncertainties on "property".

    Parameters:
        candidate_pool: The set of structures that can be selected for further investigation, that are currently
            missing values of the desired properties from the oracles.
        decorated_structures: The set of structures (potentially with precomputed properties) that can be used
            to inform the selection process, but cannot themselves be selected.
        property: The field name of the property whose uncertainty should be considered.
        num_to_select: How many candidates to suggest.

    Returns:
        A list of suggested structures.

    """
    return sorted(
        candidate_pool,
        key=lambda s: s["predictions"][f"{property}_std"],
        reverse=True,
    )[:num_to_select]


def extremise_expected_value(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    property: str,
    include_std: bool = False,
    num_to_select: int = 1,
    order="max",
):
    """Returns the top `num_to_select` structures according to the maximum/minimum feasible value of `property`, accounting
    for uncertainty if available AND if include_std==True.

    Parameters:
        candidate_pool: The set of structures that can be selected for further investigation, that are currently
            missing values of the desired properties from the oracles.
        decorated_structures: The set of structures (potentially with precomputed properties) that can be used
            to inform the selection process, but cannot themselves be selected.
        property: The field name of the property to maximize. If prsent, the associated uncertainty `<property>_std`
            will be used.
        num_to_select: How many candidates to suggest.
        order: Either `max` or `min` to indicate whether to maximize or minimize the property.

    Returns:
        A list of suggested structures.

    """
    if order not in ("max", "min"):
        raise RuntimeError("order must be either 'max' or 'min'")

    reverse = False
    sign = -1
    if order == "max":
        reverse = True
        sign = 1

    return sorted(
        candidate_pool,
        key=lambda s: s["predictions"][property]
        + sign * s["predictions"].get(f"{property}_std", 0.0) * include_std,
        reverse=reverse,
    )[:num_to_select]


def random_selection(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    num_to_select: int = 1,
) -> list[Structure]:
    """Returns `num_to_select` random structures from the candidate pool.

    Parameters:
        candidate_pool: The set of structures that can be selected for further investigation, that are currently
            missing values of the desired properties from the oracles.
        decorated_structures: The set of structures (potentially with precomputed properties) that can be used
            to inform the selection process, but cannot themselves be selected.
        num_to_select: How many candidates to suggest.

    Returns:
        A list of suggested structures.

    """
    return random.sample(candidate_pool, num_to_select)


def rppf(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    properties: dict[str,str],
    include_std: bool = False,
    num_to_select: int = 1,
    rho:   float = 0.0,
    wind:  float = 0.01,
    Tstar: float = 0.01
):
    """Returns the top `num_to_select` structures according to the Most Isolated Pareto structure Score (MIPS) calculated from projection free energy generalized to any dimension (number of properties). Does not take uncertainty into account. TODO maybe?

    Parameters:
        candidate_pool: The set of structures that can be selected for further investigation, that are currently
            missing values of the desired properties from the oracles.
        decorated_structures: The set of structures (potentially with precomputed properties) that can be used
            to inform the selection process, but cannot themselves be selected.
        properties: The field names of the properties (keys) to maximize/minimize associated to their optimization type ('min' or 'max' as values).
        include_std: True if the uncertainty should be considered when optimizing.
        num_to_select: How many candidates to suggest.
        rho: Parameter in augmented weighted Tchebycheff.
        wind: Window of weights.
        Tstar: Temperature for the calculation of the projection free energy.

    Returns:
        A list of suggested structures.

    """
    orders = list(properties.values())
    for order in orders:
        if order not in ("max", "min"):
            raise RuntimeError("The values of 'properties' must be either 'max' or 'min'")

    # Replace max and min by 1 and -1 since rppf minimizes the objectives by default
    properties_sign = {key: -1 if value == 'max' else 1 for key, value in properties.items()}

    y = np.empty((len(candidate_pool), len(properties_sign)))
    for i, s in enumerate(candidate_pool):
        for j, (key, value) in enumerate(properties_sign.items()):
            y[i,j] = s["predictions"].get(key, 0.0) * value
            if include_std:
                y[i,j] -= s["predictions"].get(f"{key}_std", 0.0)
        

    ranking_index, ranking_MIPS = rppf_y(y, rho=rho, wind=wind, Tstar=Tstar)
    
    selected = []
    for count, idx in enumerate(ranking_index):
        selected.append(candidate_pool[idx])
        count += 1
        if count==num_to_select:
            break

    return selected
