import numpy as np
import pytest
from re2fractive import CAMPAIGN_ID, CAMPAIGNS_DIR, DATASETS_DIR


@pytest.mark.skipif(
    not (CAMPAIGNS_DIR / CAMPAIGN_ID).exists(),
    reason="Campaign directory does not exist for campaign {CAMPAIGN_ID}",
)
def test_naccarato():
    from re2fractive.datasets import NaccaratoDataset

    assert NaccaratoDataset.id == "Naccarato2019"
    dataset = NaccaratoDataset.load()
    assert (DATASETS_DIR / "Naccarato2019" / "Naccarato2019.jsonl").exists()
    assert (DATASETS_DIR / "Naccarato2019" / "Naccarato.csv").exists()
    assert (DATASETS_DIR / "Naccarato2019" / "meta.json").exists()

    # sanity checks on particular results
    df = dataset.as_df

    assert len(df) == 3_688

    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-755116"][
            "_naccarato_refractive_index"
        ],
        2.79161332592,
    )
    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-755116"][
            "_naccarato_gga_bandgap"
        ],
        2.0492,
    )
    assert (
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-755116"][
            "chemical_formula_reduced"
        ]
        == "In2O7Pt2"
    )

    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-13190"][
            "_naccarato_refractive_index"
        ],
        1.43878703428,
    )
    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-13190"][
            "_naccarato_gga_bandgap"
        ],
        5.8895,
    )
    assert (
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-13190"][
            "chemical_formula_reduced"
        ]
        == "F6GaKRb2"
    )

    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-3444"][
            "_naccarato_refractive_index"
        ],
        2.15124715165,
    )
    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-3444"][
            "_naccarato_gga_bandgap"
        ],
        4.3782,
    )
    assert (
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-3444"][
            "chemical_formula_reduced"
        ]
        == "ErO4Ta"
    )

    pmg_df = dataset.structure_df
    assert "structure" in pmg_df.columns

    property_df = dataset.property_df

    np.testing.assert_almost_equal(
        property_df.loc["https://optimade.materialsproject.org/v1/structures/mp-3444"][
            "refractive_index"
        ],
        2.15124715165,
    )


@pytest.mark.skipif(
    not (CAMPAIGNS_DIR / CAMPAIGN_ID).exists(),
    reason="Campaign directory does not exist for campaign {CAMPAIGN_ID}",
)
def test_mp2023():
    from re2fractive.datasets import MP2023Dataset

    assert MP2023Dataset.id == "MP2023"
    dataset = MP2023Dataset.load()
    assert (DATASETS_DIR / "MP2023" / "MP2023.jsonl").exists()
    assert (DATASETS_DIR / "Naccarato2019" / "meta.json").exists()

    # sanity checks on particular results
    df = dataset.as_df

    assert len(df) == 33_087

    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-23558"][
            "_mp_band_gap"
        ],
        0.3872,
    )
    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-23558"][
            "_mp_energy_above_hull"
        ],
        0.0,
    )
    # values taken from tensor on the MP website
    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-23558"][
            "_mp_refractive_index"
        ],
        np.sqrt(np.mean([14.64, 12.44, 11.76])),
        decimal=2,
    )
    assert (
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-23558"][
            "chemical_formula_reduced"
        ]
        == "Ag2BiO3"
    )

    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-8042"][
            "_mp_energy_above_hull"
        ],
        0.0,
    )
    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-8042"][
            "_mp_band_gap"
        ],
        3.51,
        decimal=2,
    )
    np.testing.assert_almost_equal(
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-8042"][
            "_mp_refractive_index"
        ],
        np.sqrt(np.mean([5.096, 5.096, 4.803])),
        decimal=2,
    )
    assert (
        df.loc["https://optimade.materialsproject.org/v1/structures/mp-8042"][
            "chemical_formula_reduced"
        ]
        == "GeO4Zr"
    )

    pmg_df = dataset.structure_df
    assert "structure" in pmg_df.columns

    property_df = dataset.property_df
    assert "hull_distance" in property_df.columns
    assert "band_gap" in property_df.columns


@pytest.mark.skipif(
    not (CAMPAIGNS_DIR / CAMPAIGN_ID).exists(),
    reason="Campaign directory does not exist for campaign {CAMPAIGN_ID}",
)
def test_alexandria():
    from re2fractive.datasets import Alexandria2024Dataset

    alexandria = Alexandria2024Dataset.load()
    assert (DATASETS_DIR / "Alexandria2024" / "Alexandria2024.jsonl").exists()
    assert (DATASETS_DIR / "Alexandria2024" / "meta.json").exists()

    assert len(alexandria) == 104_860
    df = alexandria.as_df
    assert df.loc["agm1000007964"]["chemical_formula_reduced"] == "Cl4Zr"
    assert df.loc["agm1000007964"]["_alexandria_band_gap"] == 3.752


@pytest.mark.skipif(
    not (CAMPAIGNS_DIR / CAMPAIGN_ID).exists(),
    reason="Campaign directory does not exist for campaign {CAMPAIGN_ID}",
)
def test_gnome():
    from re2fractive.datasets import GNome2024Dataset

    gnome = GNome2024Dataset.load()
    assert (DATASETS_DIR / "GNome2024" / "GNome2024.jsonl").exists()
    assert (DATASETS_DIR / "GNome2024" / "meta.json").exists()

    assert len(gnome) == 384_938
    df = gnome.as_df
    assert (
        df.loc[
            "https://optimade-gnome.odbx.science/v1/structures/data/gnome_data/by_id.zip/data/gnome_data/by_id/000006a8c4.CIF"
        ]["chemical_formula_reduced"]
        == "CsS6Zr3"
    )
