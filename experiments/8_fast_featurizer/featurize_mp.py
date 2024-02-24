"""Download the latest MP structures that obey the filters
from the previous dataset.

"""
import contextlib
from pathlib import Path
import numpy as np
import json
import os
import pandas as pd
from optimade.adapters import Structure
import modnet.featurizers


data_path = Path(__file__).parent.parent / "data" / "mp2024_structures.json"

class MatminerFastFeaturizer(modnet.featurizers.MODFeaturizer):
    """A set of efficient featurizers for features implemented in matminer
    at time of creation (matminer v0.10.0 from early 2024).

    Removes featurizers that are known to be slow (i.e., orders of magnitude
    more intensive to compute than the rest of the featurizers).

    """

    def __init__(
        self,
        fast_oxid: bool = True,
        continuous_only: bool = False,
    ):
        """Creates the featurizer and imports all featurizer functions.

        Parameters:
            fast_oxid: Whether to use the accelerated oxidation state parameters within
                pymatgen when constructing features that constrain oxidation states such
                that all sites with the same species in a structure will have the same
                oxidation state (recommended if featurizing any structure
                with large unit cells).
            continuous_only: Whether to keep only the features that are continuous
                with respect to the composition (only for composition featurizers).
                Discontinuous features may lead to discontinuities in the model predictions.

        """

        super().__init__()
        print(self.drop_allnan)
        self.fast_oxid = fast_oxid
        self.continuous_only = continuous_only
        self.load_featurizers()

    def load_featurizers(self):
        with contextlib.redirect_stdout(None):
            from pymatgen.analysis.local_env import VoronoiNN
            from matminer.featurizers.composition import (
                AtomicOrbitals,
                # AtomicPackingEfficiency,  # Slow
                BandCenter,
                CationProperty,
                ElectronAffinity,
                ElectronegativityDiff,
                ElementFraction,
                ElementProperty,
                IonProperty,
                # Meredig,  # Included in others
                # Miedema,
                OxidationStates,
                Stoichiometry,
                TMetalFraction,
                ValenceOrbital,
                WenAlloys,
                # YangSolidSolution,  # Included in WenAlloys
            )
            from matminer.featurizers.structure import (
                # BagofBonds,  # Leads to >24 000 features
                # BondFractions, # Slow
                # ChemicalOrdering,  # Slow
                # CoulombMatrix,  # Redundant with SineCoulombMatrix, which is better for periodic systems
                DensityFeatures,
                # Dimensionality,
                ElectronicRadialDistributionFunction,
                EwaldEnergy,
                # GlobalInstabilityIndex,  # Still experimental?
                GlobalSymmetryFeatures,
                # JarvisCFID,
                # MaximumPackingEfficiency,
                # MinimumRelativeDistances,
                # OrbitalFieldMatrix,  # Buggy
                # PartialRadialDistributionFunction,  # Leads to >198 000 features
                # RadialDistributionFunction,  # Slow
                # SineCoulombMatrix,
                # SiteStatsFingerprint,  # Done in featurizers.py
                StructuralComplexity,
                # StructuralHeterogeneity,  # Slow
                # XRDPowderPattern,
            )

            from matminer.featurizers.site import (
                AGNIFingerprints,
                # AngularFourierSeries,  # Redundant with GaussianSymmFunc
                AverageBondAngle,
                AverageBondLength,
                BondOrientationalParameter,
                ChemEnvSiteFingerprint,
                # ChemicalSRO,  # Buggy
                CoordinationNumber,
                CrystalNNFingerprint,
                EwaldSiteEnergy,
                GaussianSymmFunc,
                GeneralizedRadialDistributionFunction,
                IntersticeDistribution,
                LocalPropertyDifference,
                OPSiteFingerprint,
                # SiteElementalProperty,  # Already included in composition featurizers
                # SOAP, # Leads to >260 000 features...
                VoronoiFingerprint,
            )

            # Get additional ElementProperty featurizer, but
            # get only the features that are not yet present with another featurizer.
            # For this reason, we cannot rely on the Matminer presets for those.
            # Also in the case of continuous features, use only the mean and avg_dev for the statistics.
            from matminer.utils.data import (
                PymatgenData,
                DemlData,
                # MatscholarElementData,
                # MEGNetElementData,
            )

            pymatgen_features = [
                "block",
                "mendeleev_no",
                "electrical_resistivity",
                "velocity_of_sound",
                "thermal_conductivity",
                "bulk_modulus",
                "coefficient_of_linear_thermal_expansion",
            ]

            deml_features = [
                "atom_radius",
                "molar_vol",
                "heat_fusion",
                "boiling_point",
                "heat_cap",
                "first_ioniz",
                "electric_pol",
                "GGAU_Etot",
                "mus_fere",
                "FERE correction",
            ]

            if self.continuous_only:
                magpie_featurizer = ElementProperty.from_preset("magpie")
                magpie_featurizer.stats = ["mean", "avg_dev"]

                pymatgen_featurizer = ElementProperty(
                    data_source=PymatgenData(),
                    stats=["mean", "avg_dev"],
                    features=pymatgen_features,
                )

                deml_featurizer = ElementProperty(
                    data_source=DemlData(),
                    stats=["mean", "avg_dev"],
                    features=deml_features,
                )

                # matscholar_featurizer = ElementProperty(
                #     data_source=MatscholarElementData(),
                #     stats=["mean", "avg_dev"],
                #     features=MatscholarElementData().prop_names,
                # )
                #
                # megnet_featurizer = ElementProperty(
                #     data_source=MEGNetElementData(),
                #     stats=["mean", "avg_dev"],
                #     features=MEGNetElementData().prop_names,
                # )

                self.composition_featurizers = (
                    BandCenter(),
                    ElementFraction(),
                    magpie_featurizer,
                    pymatgen_featurizer,
                    deml_featurizer,
                    # matscholar_featurizer,
                    # megnet_featurizer,
                    Stoichiometry(p_list=[2, 3, 5, 7, 10]),
                    TMetalFraction(),
                    ValenceOrbital(props=["frac"]),
                    WenAlloys(),
                )

                self.oxid_composition_featurizers = (
                    IonProperty(fast=self.fast_oxid),
                    OxidationStates(stats=["mean"]),
                )

            else:
                # Get the initial presets from Matminer, without the duplicate features from Magpie
                pymatgen_featurizer_full = ElementProperty(
                    data_source=PymatgenData(),
                    stats=["minimum", "maximum", "range", "mean", "std_dev"],
                    features=pymatgen_features,
                )

                deml_featurizer_full = ElementProperty(
                    data_source=DemlData(),
                    stats=["minimum", "maximum", "range", "mean", "std_dev"],
                    features=deml_features,
                )

                self.composition_featurizers = (
                    AtomicOrbitals(),
                    # AtomicPackingEfficiency(),
                    BandCenter(),
                    ElementFraction(),
                    ElementProperty.from_preset("magpie"),
                    pymatgen_featurizer_full,
                    deml_featurizer_full,
                    # ElementProperty.from_preset("matscholar_el"),
                    # ElementProperty.from_preset("megnet_el"),
                    # Miedema(),
                    Stoichiometry(),
                    TMetalFraction(),
                    ValenceOrbital(props=["frac"]),
                    WenAlloys(),
                )

                self.oxid_composition_featurizers = (
                    CationProperty.from_preset("deml"),
                    ElectronAffinity(),
                    ElectronegativityDiff(),
                    IonProperty(fast=self.fast_oxid),
                    OxidationStates(),
                )

            self.structure_featurizers = (
                # BagofBonds(),  # > 24 000 features
                # BondFractions(),
                # ChemicalOrdering(),
                # CoulombMatrix(),  # Redundant with SineCoulombMatrix, which is better for periodic systems
                DensityFeatures(),
                # Dimensionality(),
                ElectronicRadialDistributionFunction(),
                EwaldEnergy(),
                GlobalSymmetryFeatures(),
                # JarvisCFID(),  # 1557 features, many redundant ones
                # MaximumPackingEfficiency(),
                # MinimumRelativeDistances(),
                # OrbitalFieldMatrix(),  # Buggy
                # PartialRadialDistributionFunction(),  # > 198 000 features
                # RadialDistributionFunction(),
                # SineCoulombMatrix(),  # not size extensive 
                StructuralComplexity(),
                # StructuralHeterogeneity(),
                # XRDPowderPattern(),
            )

            # Patch for matminer: see https://github.com/hackingmaterials/matminer/issues/864
            # self.structure_featurizers[2].desired_features = None
            # self.structure_featurizers[6].desired_features = None

            self.site_featurizers = (
                #AGNIFingerprints(),
                ## AngularFourierSeries.from_preset("gaussian"), # Redundant with GaussianSymmFunc
                #AverageBondAngle(VoronoiNN()),
                #AverageBondLength(VoronoiNN()),
                #BondOrientationalParameter(),
                #ChemEnvSiteFingerprint.from_preset("simple"),
                ## ChemicalSRO.from_preset("VoronoiNN"),  # Buggy
                #CoordinationNumber(),
                #CrystalNNFingerprint.from_preset("ops"),
                #EwaldSiteEnergy(),
                #GaussianSymmFunc(),
                #GeneralizedRadialDistributionFunction.from_preset("gaussian"),
                #IntersticeDistribution(),
                #LocalPropertyDifference(),
                #OPSiteFingerprint(),
                ## SOAP.from_preset("formation_energy"),  # Leads to >260 000 features...
                #VoronoiFingerprint(),
            )

    def featurize_composition(self, df):
        """Applies the preset composition featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """
        from pymatgen.core.periodic_table import Element

        df = super().featurize_composition(df)

        if self.composition_featurizers and not self.continuous_only:
            _orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}
            df["AtomicOrbitals|HOMO_character"] = df[
                "AtomicOrbitals|HOMO_character"
            ].map(_orbitals)
            df["AtomicOrbitals|LUMO_character"] = df[
                "AtomicOrbitals|LUMO_character"
            ].map(_orbitals)

            df["AtomicOrbitals|HOMO_element"] = df["AtomicOrbitals|HOMO_element"].apply(
                lambda x: -1 if not isinstance(x, str) else Element(x).Z
            )
            df["AtomicOrbitals|LUMO_element"] = df["AtomicOrbitals|LUMO_element"].apply(
                lambda x: -1 if not isinstance(x, str) else Element(x).Z
            )

        if self.continuous_only:
            # These are additional features that have shown discontinuities in my tests.
            # Hopefully, I got them all...
            df.drop(
                columns=[
                    "WenAlloys|Yang omega",
                    "WenAlloys|Yang delta",
                    "WenAlloys|Radii gamma",
                    "WenAlloys|Lambda entropy",
                    "WenAlloys|APE mean",
                    "WenAlloys|Interant electrons",
                    "WenAlloys|Interant s electrons",
                    "WenAlloys|Interant p electrons",
                    "WenAlloys|Interant d electrons",
                    "WenAlloys|Interant f electrons",
                    "WenAlloys|Atomic weight mean",
                    "WenAlloys|Total weight",
                    "ElementProperty|DemlData mean electric_pol",
                    "ElementProperty|DemlData mean FERE correction",
                    "ElementProperty|DemlData mean GGAU_Etot",
                    "ElementProperty|DemlData mean heat_fusion",
                    "ElementProperty|DemlData mean mus_fere",
                ],
                inplace=True,
            )

            if self.oxid_composition_featurizers:
                df.drop(columns=["IonProperty|max ionic char"], inplace=True)

        return modnet.featurizers.clean_df(df, drop_allnan=self.drop_allnan)

    def featurize_structure(self, df):
        """Applies the preset structural featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """

        if self.structure_featurizers:
            df = super().featurize_structure(df)

        _crystal_system = {
            "cubic": 1,
            "tetragonal": 2,
            "orthorombic": 3,
            "hexagonal": 4,
            "trigonal": 5,
            "monoclinic": 6,
            "triclinic": 7,
        }

        def _int_map(x):
            if x == np.nan:
                return 0
            elif x:
                return 1
            else:
                return 0

        df["GlobalSymmetryFeatures|crystal_system"] = df[
            "GlobalSymmetryFeatures|crystal_system"
        ].map(_crystal_system)
        df["GlobalSymmetryFeatures|is_centrosymmetric"] = df[
            "GlobalSymmetryFeatures|is_centrosymmetric"
        ].map(_int_map)

        return modnet.featurizers.clean_df(df, drop_allnan=self.drop_allnan)

    def featurize_site(self, df):
        """Applies the preset site featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """

        # rename some features for backwards compatibility with pretrained models
        aliases = {
            "GeneralizedRadialDistributionFunction": "GeneralizedRDF",
            "AGNIFingerprints": "AGNIFingerPrint",
            "BondOrientationalParameter": "BondOrientationParameter",
        }
        df = super().featurize_site(df, aliases=aliases)
        df = df.loc[:, (df != 0).any(axis=0)]

        return modnet.featurizers.clean_df(df, drop_allnan=self.drop_allnan)


if not data_path.exists():
    raise RuntimeError
else:
    with open(data_path, "r") as f:
        optimade_docs = json.load(f)


df_structures = pd.DataFrame.from_dict(
    {doc["id"]: {"structure": Structure(doc).as_pymatgen} for doc in optimade_docs}, orient="index"
)

featurizer = MatminerFastFeaturizer()
featurizer.featurizer_mode = "multi"
featurizer.set_n_jobs(6)

df_featurized = featurizer.featurize(df_structures)
df_featurized.to_pickle(data_path.parent / "mp_2024_df_featurized_multi.pkl")
