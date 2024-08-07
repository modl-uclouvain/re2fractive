import os
import random
import string
from pathlib import Path

import matplotlib.pyplot as plt
import modnet.featurizers
import numpy as np
from modnet.hyper_opt import FitGenetic
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
            from matminer.featurizers.composition import (  # AtomicPackingEfficiency,  # Slow; Meredig,  # Included in others; Miedema,; YangSolidSolution,  # Included in WenAlloys
                AtomicOrbitals, BandCenter, CationProperty, ElectronAffinity,
                ElectronegativityDiff, ElementFraction, ElementProperty,
                IonProperty, OxidationStates, Stoichiometry, TMetalFraction,
                ValenceOrbital, WenAlloys)
            from matminer.featurizers.site import (  # AngularFourierSeries,  # Redundant with GaussianSymmFunc; ChemicalSRO,  # Buggy; SiteElementalProperty,  # Already included in composition featurizers; SOAP, # Leads to >260 000 features...
                AGNIFingerprints, AverageBondAngle, AverageBondLength,
                BondOrientationalParameter, ChemEnvSiteFingerprint,
                CoordinationNumber, CrystalNNFingerprint, EwaldSiteEnergy,
                GaussianSymmFunc, GeneralizedRadialDistributionFunction,
                IntersticeDistribution, LocalPropertyDifference,
                OPSiteFingerprint, VoronoiFingerprint)
            from matminer.featurizers.structure import (  # BagofBonds,  # Leads to >24 000 features; BondFractions, # Slow; ChemicalOrdering,  # Slow; CoulombMatrix,  # Redundant with SineCoulombMatrix, which is better for periodic systems; Dimensionality,; GlobalInstabilityIndex,  # Still experimental?; JarvisCFID,; MaximumPackingEfficiency,; MinimumRelativeDistances,; OrbitalFieldMatrix,  # Buggy; PartialRadialDistributionFunction,  # Leads to >198 000 features; RadialDistributionFunction,  # Slow; SineCoulombMatrix,; SiteStatsFingerprint,  # Done in featurizers.py; StructuralHeterogeneity,  # Slow; XRDPowderPattern,
                DensityFeatures, ElectronicRadialDistributionFunction,
                EwaldEnergy, GlobalSymmetryFeatures, StructuralComplexity)
            # Get additional ElementProperty featurizer, but
            # get only the features that are not yet present with another featurizer.
            # For this reason, we cannot rely on the Matminer presets for those.
            # Also in the case of continuous features, use only the mean and avg_dev for the statistics.
            from matminer.utils.data import (  # MatscholarElementData,; MEGNetElementData,
                DemlData, PymatgenData)
            from pymatgen.analysis.local_env import VoronoiNN

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
                # AGNIFingerprints(),
                ## AngularFourierSeries.from_preset("gaussian"), # Redundant with GaussianSymmFunc
                # AverageBondAngle(VoronoiNN()),
                # AverageBondLength(VoronoiNN()),
                # BondOrientationalParameter(),
                # ChemEnvSiteFingerprint.from_preset("simple"),
                ## ChemicalSRO.from_preset("VoronoiNN"),  # Buggy
                # CoordinationNumber(),
                # CrystalNNFingerprint.from_preset("ops"),
                # EwaldSiteEnergy(),
                # GaussianSymmFunc(),
                # GeneralizedRadialDistributionFunction.from_preset("gaussian"),
                # IntersticeDistribution(),
                # LocalPropertyDifference(),
                # OPSiteFingerprint(),
                ## SOAP.from_preset("formation_energy"),  # Leads to >260 000 features...
                # VoronoiFingerprint(),
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


if __name__ == "__main__":
    import tensorflow as tf

    targets = ["refractive_index"]
    experiment_name = (
        "baseline_kfold_ensemble-"
        + "-".join(targets)
        + "-"
        + "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    )

    gpus = tf.config.list_physical_devices("GPU")
    gpus = None
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    from modnet.models import EnsembleMODNetModel
    from modnet.preprocessing import MODData

    data: MODData = MODData.load("mod-feature-selected-fast.data")

    # scrub now missing features
    bad_features = [
        "AtomicPackingEfficiency|dist from 1 clusters |APE| < 0.010",
        "ElectronegativityDiff|maximum EN difference",
        "ElectronegativityDiff|mean EN difference",
        "AtomicPackingEfficiency|dist from 3 clusters |APE| < 0.010",
        "AtomicPackingEfficiency|mean simul. packing efficiency",
        "ElectronegativityDiff|range EN difference",
        "ElectronegativityDiff|minimum EN difference",
    ]
    data.optimal_features = [
        col for col in data.optimal_features if col not in bad_features
    ]
    for k in data.optimal_features_by_target:
        data.optimal_features_by_target[k] = [
            col for col in data.optimal_features_by_target[k] if col not in bad_features
        ]

    md_naccarato = data

    kf = KFold(5, shuffle=True, random_state=42)
    scores = []
    for ind, (train, test) in enumerate(
        kf.split(md_naccarato.df_featurized, y=md_naccarato.df_targets)
    ):
        train_moddata, test_moddata = md_naccarato.split((train, test))
        model_path = Path(".") / "models" / "genetic" / f"{experiment_name}_{ind}.pkl"
        print(model_path)
        if model_path.exists():
            model = EnsembleMODNetModel.load(model_path)

        else:
            ga = FitGenetic(train_moddata)
            model = ga.run(
                size_pop=20,  # dflt
                num_generations=10,  # dflt
                nested=0,  # dflt = 5
                n_jobs=12,
                early_stopping=4,  # dflt
                refit=0,  # dflt = 5
                fast=False,
            )
            model.save(model_path)

            predictions, uncertainties = model.predict(test_moddata, return_unc=True)
            scores.append(np.mean(np.abs(predictions - test_moddata.df_targets)))

            plt.scatter(
                test_moddata.df_targets.values.ravel(),
                predictions.values.ravel(),
            )
            plt.errorbar(
                test_moddata.df_targets.values.ravel(),
                predictions.values.ravel(),
                yerr=uncertainties.values.ravel(),
                ls="none",
            )

    print("=" * 10 + f" {experiment_name} " + "=" * 10)
    print("Training complete.")
    print("Training complete.")
    print(f"Accuracy: {np.mean(scores):.3f}±{np.std(scores):.3f}")

    plt.plot(
        np.linspace(
            np.min(md_naccarato.df_targets.values),
            np.max(md_naccarato.df_targets.values),
            3,
        ),
        np.linspace(
            np.min(md_naccarato.df_targets.values),
            np.max(md_naccarato.df_targets.values),
            3,
        ),
        color="black",
        ls="--",
    )

    plt.ylabel("Predicted $n$ (dimensionless)")
    plt.xlabel("Computed $n$ (dimensionless)")
    plt.title(f"{experiment_name} - MAE: {np.mean(scores):.3f}±{np.std(scores):.3f}")
    plt.savefig(Path(__file__).parent / "figs" / f"{experiment_name}.pdf")
