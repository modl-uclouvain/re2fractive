from atomate2.vasp.jobs.core import DielectricMaker
from atomate2.vasp.sets.core import StaticSetGenerator

__all__ = ("Re2DielectricMaker",)


class Re2StaticSetGenerator(StaticSetGenerator):
    """Some minor tweaks to the default static set generator for better
    agreement with Naccarato et al. (DOI: 10.1103/PhysRevMaterials.3.044602).

    """

    user_potcar_functional: str = "PBE_64"

    user_incar_settings: dict = {
        "KPAR": 8,
        "EDIFF": 1e-6,
        "GGA": None,
        "LREAL": False,
        "ALGO": "All",
        "ISMEAR": 0,
        "SIGMA": 0.03,
        "LAECHG": False,
        "LELF": False,
        "LVTOT": False,
        "LWAVE": False,
        "PREC": "Accurate",
        "IBRION": -1,
        "NSW": 0,
    }

    user_kpoints_settings: dict = {"grid_density": 1500}


class Re2DielectricMaker(DielectricMaker):
    """A tweaked version of the default dielectric maker for better agreement
    with Naccarato et al. (DOI: 10.1103/PhysRevMaterials.3.044602).

    """

    input_set_generator = Re2StaticSetGenerator
