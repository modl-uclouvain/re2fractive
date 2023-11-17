"""A place to collect acquisition functions."""

import random

from optimade.adapters import Structure


def random_selection(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    num_to_select: int = 1,
) -> list[Structure]:
    """Returns `num_to_select` random structures from the candidate pool.

    Parameters:
        candidate_pool: The set of structures that can be selected for further invesitigation, that are currently
            missing values of the desired properties from the oracles.
        decorated_structures: The set of structures (potentially with precomputed properties) that can be used
            to inform the selection process, but cannot themselves be selected.
        num_to_select: How many candidates to suggest.

    Returns:
        A list of suggested structures.

    """
    return random.sample(candidate_pool, num_to_select)


def extremise_expected_value(
    candidate_pool: list[Structure],
    decorated_structures: list[Structure],
    property: str,
    num_to_select: int = 1,
    order="max",
):
    """Returns the top `num_to_select` structures according to the maximum/minimum feasible value of `property`, accounting
    for uncertainty if available.

    Parameters:
        candidate_pool: The set of structures that can be selected for further invesitigation, that are currently
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
        + sign * s["predictions"].get(f"{property}_std", 0.0),
        reverse=reverse,
    )[:num_to_select]
