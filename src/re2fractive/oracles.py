from atomate2.vasp.jobs.core import DielectricMaker
from atomate2.vasp.sets.core import StaticSetGenerator


class Re2StaticSetGenerator(StaticSetGenerator):
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
    input_set_generator = Re2StaticSetGenerator
