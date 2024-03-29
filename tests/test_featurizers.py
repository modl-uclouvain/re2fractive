import itertools

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Element, Lattice, Structure
from re2fractive.featurizers import MatminerFastFeaturizer


@pytest.fixture
def mp_df():
    structures = []
    # make binaries of first N elements and stick them in a featurizable dataframe
    N = 10
    for ind, (Z1, Z2) in enumerate(itertools.combinations(range(1, N), 2)):
        e1 = str(Element.from_Z(Z1))
        e2 = str(Element.from_Z(Z2))
        structures.append(
            {
                "id": f"id-{ind}",
                "structure": Structure(
                    Lattice.cubic(3), [e1, e2], [[0, 0, 0], [0.5, 0.5, 0.5]]
                ),
            }
        )
    return pd.DataFrame(structures)


def test_matminer_fast_featurizer(mp_df, tmpdir):
    num_expected_features = 599
    featurizer = MatminerFastFeaturizer()
    featurizer.batch_size = None
    featurizer.batch_dir = tmpdir

    featurized_df = featurizer.featurize(mp_df)
    assert len(featurized_df.columns) == num_expected_features
    assert len(featurized_df) == len(mp_df)

    featurizer = MatminerFastFeaturizer()
    featurizer.batch_size = 10
    featurizer.batch_dir = tmpdir

    batched_featurized_df = featurizer.featurize(mp_df)
    assert len(batched_featurized_df.columns) == num_expected_features
    assert len(batched_featurized_df) == len(mp_df)
    np.testing.assert_array_equal(batched_featurized_df.values, featurized_df.values)
