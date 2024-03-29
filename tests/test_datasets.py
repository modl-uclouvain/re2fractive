import numpy as np
from re2fractive import CAMPAIGN_DIR, DATASETS_DIR


def test_naccarato():
    from re2fractive.datasets import NaccaratoDataset

    assert NaccaratoDataset.id == "Naccarato2019"
    dataset = NaccaratoDataset.load()
    assert (
        CAMPAIGN_DIR / "0001" / DATASETS_DIR / "Naccarato2019" / "Naccarato2019.jsonl"
    ).exists()
    assert (
        CAMPAIGN_DIR / "0001" / DATASETS_DIR / "Naccarato2019" / "Naccarato.csv"
    ).exists()
    assert (
        CAMPAIGN_DIR / "0001" / DATASETS_DIR / "Naccarato2019" / "meta.json"
    ).exists()

    # sanity checks on particular results
    df = dataset.as_df()

    assert len(df) == 3_688

    np.testing.assert_almost_equal(
        df.loc["mp-755116"]["_naccarato_refractive_index"], 2.79161332592
    )
    np.testing.assert_almost_equal(
        df.loc["mp-755116"]["_naccarato_gga_bandgap"], 2.0492
    )
    assert df.loc["mp-755116"]["chemical_formula_reduced"] == "In2O7Pt2"

    np.testing.assert_almost_equal(
        df.loc["mp-13190"]["_naccarato_refractive_index"], 1.43878703428
    )
    np.testing.assert_almost_equal(df.loc["mp-13190"]["_naccarato_gga_bandgap"], 5.8895)
    assert df.loc["mp-13190"]["chemical_formula_reduced"] == "F6GaKRb2"

    np.testing.assert_almost_equal(
        df.loc["mp-3444"]["_naccarato_refractive_index"], 2.15124715165
    )
    np.testing.assert_almost_equal(df.loc["mp-3444"]["_naccarato_gga_bandgap"], 4.3782)
    assert df.loc["mp-3444"]["chemical_formula_reduced"] == "ErO4Ta"


def test_mp2023():
    from re2fractive.datasets import MP2023Dataset

    assert MP2023Dataset.id == "MP2023"
    dataset = MP2023Dataset.load()
    assert (CAMPAIGN_DIR / "0001" / DATASETS_DIR / "MP2023" / "MP2023.jsonl").exists()
    assert (
        CAMPAIGN_DIR / "0001" / DATASETS_DIR / "Naccarato2019" / "meta.json"
    ).exists()

    # sanity checks on particular results
    df = dataset.as_df()

    assert len(df) == 33_087

    np.testing.assert_almost_equal(df.loc["mp-23558"]["_mp_band_gap"], 0.3872)
    np.testing.assert_almost_equal(df.loc["mp-23558"]["_mp_energy_above_hull"], 0.0)
    # values taken from tensor on the MP website
    np.testing.assert_almost_equal(
        df.loc["mp-23558"]["_mp_refractive_index"],
        np.sqrt(np.mean([14.64, 12.44, 11.76])),
        decimal=2,
    )
    assert df.loc["mp-23558"]["chemical_formula_reduced"] == "Ag2BiO3"

    np.testing.assert_almost_equal(df.loc["mp-8042"]["_mp_energy_above_hull"], 0.0)
    np.testing.assert_almost_equal(df.loc["mp-8042"]["_mp_band_gap"], 3.51)
    np.testing.assert_almost_equal(
        df.loc["mp-8042"]["_mp_refractive_index"],
        np.sqrt(np.mean([5.096, 5.096, 4.803])),
        decimal=2,
    )
    assert df.loc["mp-8042"]["chemical_formula_reduced"] == "GeO4Zr"
