import numpy as np
import pandas as pd
import pytest
from data_analysis import DataAnalysis

"""
Test code for key logic used in data analysis matrix generation
"""

@pytest.fixture
def cpd_library():
    return pd.DataFrame(
        [
            ['m_c18n_0071', 'THIOACETIC ACID'],
            ['m_c18n_0072', 'DIACETYL']
        ],
        columns=['dname', 'Compound']
    )

def test_fold_change(cpd_library):
    analysis = DataAnalysis(pd.DataFrame(), pd.DataFrame(), '', cpd_library)

    metadata = pd.DataFrame([
        ['s0001', 'mono-colonization', 'feces', 'germ-free'],
        ['s0002', 'mono-colonization', 'feces', 'germ-free'],
        ['s0003', 'mono-colonization', 'feces', 'germ-free'],
        ['s0004', 'mono-colonization', 'feces', 'Bt'],
        ['s0005', 'mono-colonization', 'feces', 'Bt'],
        ['s0006', 'mono-colonization', 'feces', 'Bt'],
        ['s0007', 'mono-colonization', 'caecal', 'germ-free'],
        ['s0008', 'mono-colonization', 'caecal', 'germ-free'],
        ['s0009', 'mono-colonization', 'caecal', 'germ-free'],
        ['s0010', 'mono-colonization', 'caecal', 'Bt'],
        ['s0011', 'mono-colonization', 'caecal', 'Bt'],
        ['s0012', 'mono-colonization', 'caecal', 'Bt'],
    ], columns=['sample_id', 'experiment_type', 'sample_type', 'colonization']) \
        .set_index('sample_id')

    matrix = pd.DataFrame([
        ['s0001', 0],
        ['s0002', 1],
        ['s0003', 2],
        ['s0004', 1],
        ['s0005', 3],
        ['s0006', 7],

        ['s0007', 1],
        ['s0008', 3],
        ['s0009', 5],
        ['s0010', 15],
        ['s0011', 31],
        ['s0012', 63],

    ], columns=['sample_id', 'm_c18p_0001']) \
        .set_index('sample_id')

    expected = pd.DataFrame([
        ['s0001', -1],
        ['s0002', 0],
        ['s0003', np.log2(1.5)],
        ['s0004', 0],
        ['s0005', 1],
        ['s0006', 2],

        ['s0007', -1],
        ['s0008', 0],
        ['s0009', np.log2(1.5)],
        ['s0010', 2],
        ['s0011', 3],
        ['s0012', 4],
    ], columns=['sample_id', 'm_c18p_0001']) \
        .set_index('sample_id')

    pd.testing.assert_frame_equal(analysis.get_fold_change(matrix, metadata), expected)

def test_normalize_istd(cpd_library):
    analysis = DataAnalysis(pd.DataFrame(), pd.DataFrame(), '', cpd_library)

    cpd_dname_map = {
        **{(istd + '.c18positive'):istd for istd in analysis.ISTD_CHOICES['c18positive']},
        **{(istd + '.c18negative'):istd for istd in analysis.ISTD_CHOICES['c18negative']},
        **{(istd + '.hilicpositive'):istd for istd in analysis.ISTD_CHOICES['hilicpositive']},
        **{
            'IS_2-FLUROPHENYLGLYCINE.c18positive'   : 'm_c18p_0001',
            'IS_4-BROMO-PHENYLALANINE.c18positive'  : 'm_c18p_0002',
            '1-METHYLNICOTINAMIDE.c18positive'      : 'm_c18p_0003',
            'IS_4-BROMO-PHENYLALANINE.c18negative'  : 'm_c18n_0001',
            'IS_LEUCINE-5,5,5-D3.c18negative'       : 'm_c18n_0002',
            '2-DEOXY-GLUCOSE.c18negative'           : 'm_c18n_0003',
            'IS_4-BROMO-PHENYLALANINE.hilicpositive': 'm_hilicp_0001',
            'IS_LEUCINE-5,5,5-D3.hilicpositive'     : 'm_hilicp_0002',
            '1-OLEOYL-RAC-GLYCEROL.hilicpositive'   : 'm_hilicp_0003',
        }
    }

    dname_cpd_map = {value:key for key, value in cpd_dname_map.items()}

    metadata = pd.DataFrame([
        ['s0001', 'mono-colonization', 'feces'],
        ['s0002', 'mono-colonization', 'feces'],
        ['s0003', 'mono-colonization', 'feces'],
        ['s0004', 'mono-colonization', 'caecal'],
        ['s0005', 'mono-colonization', 'caecal'],
        ['s0006', 'mono-colonization', 'caecal'],
    ], columns=['sample_id', 'experiment_type', 'sample_type']) \
        .set_index('sample_id')

    matrix = pd.DataFrame([
        ['s0001', 2, 3,         np.nan, 10,     10, np.nan,     5, 100, 100],
        ['s0002', 2, 3,         1, 20,          20, 1,          10, 100, 100],
        ['s0003', 4, 6,         2, 30,          30, 2,          20, 90, 90],
        ['s0004', 4, 6,         7, 13,          13, 7,          30, 80, 80],
        ['s0005', 8, 12,        7, 13,          13, 7,          40, 50, 50],
        ['s0006', 8, 12,        3, 17,          17, 3,          80, 120, 120],
    ], columns=[
        'sample_id',

        # ISTD dnames
        'm_c18p_0001',
        'm_c18p_0002',
        'm_c18n_0001',
        'm_c18n_0002',
        'm_hilicp_0001',
        'm_hilicp_0002',

        # Actual compounds
        'm_c18p_0003',
        'm_c18n_0003',
        'm_hilicp_0003',
    ]) \
        .set_index('sample_id')

    expected = pd.DataFrame([
        ['s0001', 10., 200., 200.],
        ['s0002', 20., 100., 100.],
        ['s0003', 20., 60., 60.],
        ['s0004', 30., 80., 80.],
        ['s0005', 20., 50., 50.],
        ['s0006', 40., 120., 120.],
    ], columns=['sample_id', 'm_c18p_0003', 'm_c18n_0003', 'm_hilicp_0003']) \
        .set_index('sample_id')

    pd.testing.assert_frame_equal(
        analysis.normalize_by_istd(matrix, metadata, dname_cpd_map, cpd_dname_map)[['m_c18p_0003', 'm_c18n_0003', 'm_hilicp_0003']],
        expected
    )

def test_collapse_modes():
    fold_change_matrix = pd.DataFrame([
        [
            's0001',
            np.log2(3), # metabolite1
            np.log2(4),
            np.log2(5),
            np.log2(8), # metabolite2
            np.log2(10),
            np.log2(2), # metabolite3
            np.log2(14),
            np.log2(18),
        ],
    ], columns=[
        'sample_id',
        'metabolite1.c18positive',
        'metabolite1.c18negative',
        'metabolite1.hilicpositive',
        'metabolite2.c18positive',
        'metabolite2.c18negative',
        'metabolite3.c18positive',
        'metabolite3.c18negative',
        'metabolite3.hilicpositive',
    ]) \
        .set_index('sample_id')

    mode_picker = pd.DataFrame([
        ['metabolite1', 'c18p, c18n, hilicp', 'c18p, c18n, hilicp'],
        ['metabolite2', 'c18p, c18n', 'c18p'],
        ['metabolite3', 'c18p, c18n, hilicp', 'c18n, hilicp'],
    ], columns=['metabolite', 'mode_detected', 'mode_pref']).set_index('metabolite')

    analysis = DataAnalysis(pd.DataFrame(), pd.DataFrame(), '', cpd_library)

    expected = pd.DataFrame([
        ['s0001', 2., 3., 4.],
    ], columns=['sample_id', 'metabolite1', 'metabolite2', 'metabolite3']).set_index('sample_id')

    pd.testing.assert_frame_equal(
        analysis.collapse_modes(fold_change_matrix, mode_picker),
        expected
    )
