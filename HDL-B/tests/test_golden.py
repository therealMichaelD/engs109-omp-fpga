import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from omp_sim import omp_argmax_fixed

VECTOR_DIR = (Path.cwd().parent / 'goldenVectors')

import pytest

@pytest.mark.parametrize('vfile', sorted(VECTOR_DIR.glob('test_*.npz')))
def test_golden_vector(vfile):
    npz = np.load(vfile)
    idx, val = omp_argmax_fixed(npz['D_q15'], npz['r_q15'])
    assert idx == int(npz['expected_idx']), f"idx mismatch in {vfile.name}"
    assert val == int(npz['expected_val']), f"val mismatch in {vfile.name}"