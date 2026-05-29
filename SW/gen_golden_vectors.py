"""
gen_golden_vectors.py  —  SW deliverable, EOD Day 2
Generates the 5 mandatory test cases specified in §9 of the interface contract.

Each output is a .npz file at golden/test_NN_<description>.npz containing:
    D_float      float32 (M_LEN, N_ATOMS)  unquantised dictionary
    r_float      float32 (M_LEN,)           unquantised residual
    D_q15        int16   (M_LEN, N_ATOMS)  D quantised to Q1.15
    r_q15        int16   (M_LEN,)           r quantised to Q1.15
    expected_idx int32   scalar             argmax from bit-accurate correlator
    expected_val int64   scalar             accumulator at argmax (debug)

Run:
    python gen_golden_vectors.py
    python gen_golden_vectors.py --verify   # re-loads each file and checks it
"""

import argparse
import os
import numpy as np
from omp_params import M_LEN, N_ATOMS, Q15_SCALE, Q15_MAX, Q15_MIN
from omp_reference import quantise_q15, hw_correlate_q15

OUTDIR  = "golden"
RNG     = np.random.default_rng(seed=0xDEADBEEF)   # fixed seed → reproducible


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(name: str, D_f, r_f):
    """Quantise, run bit-accurate correlator, save .npz."""
    D_q15 = quantise_q15(D_f)
    r_q15 = quantise_q15(r_f)
    idx, val = hw_correlate_q15(D_q15, r_q15)

    path = os.path.join(OUTDIR, f"{name}.npz")
    np.savez(
        path,
        D_float      = D_f.astype(np.float32),
        r_float      = r_f.astype(np.float32),
        D_q15        = D_q15,
        r_q15        = r_q15,
        expected_idx = np.int32(idx),
        expected_val = np.int64(val),
    )

    # Quick sanity: recompute from the saved quantised arrays
    abs_accs = np.abs(D_q15.astype(np.int64).T @ r_q15.astype(np.int64))
    margin   = _margin(abs_accs, idx)
    print(f"  {name}.npz  →  idx={idx:3d}  val={val:+12d}  margin={margin:.3f}")
    return path


def _margin(abs_accs: np.ndarray, winner: int) -> float:
    """
    Fractional margin = (winner - runner-up) / winner.
    Returns 0 if winner accumulator is zero.
    """
    sorted_vals = np.sort(abs_accs)[::-1]
    if sorted_vals[0] == 0:
        return 0.0
    return float((sorted_vals[0] - sorted_vals[1]) / sorted_vals[0])


def _random_dict() -> np.ndarray:
    """Random Gaussian dictionary, columns normalised to unit L2 in float."""
    D = RNG.standard_normal((M_LEN, N_ATOMS)).astype(np.float32)
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    return D / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# The 5 mandatory cases (§9 of the contract)
# ---------------------------------------------------------------------------

def case_01_easy_margin():
    """
    (a) Easy case: argmax winner has >10% margin over runner-up.
    Strategy: build r as a scalar multiple of a random atom, add small noise.
    """
    D   = _random_dict()
    j   = int(RNG.integers(10, N_ATOMS - 10))   # winner somewhere in the middle
    r   = D[:, j] * 0.8 + RNG.standard_normal(M_LEN).astype(np.float32) * 0.02
    _save("test_01_easy_margin", D, r)


def case_02_tight_margin():
    """
    (b) Tight case: margin < 0.5%.
    Strategy: construct r as an equal mix of two atoms, then perturb one very
    slightly so there is a unique winner, but only just.
    """
    D  = _random_dict()
    j1 = int(RNG.integers(0, N_ATOMS // 2))
    j2 = int(RNG.integers(N_ATOMS // 2, N_ATOMS))
    # Make j1 the winner by an epsilon in Q1.15 space
    # Scale so quantisation lands with j1 just above j2
    r_f = D[:, j1] * 0.5 + D[:, j2] * 0.5
    # After quantisation |<D_q[:,j1], r_q>| ≈ |<D_q[:,j2], r_q>|
    # Nudge r_f toward j1 until margin < 0.5%
    for alpha in np.linspace(0.500, 0.510, 200):
        r_try = D[:, j1] * alpha + D[:, j2] * (1 - alpha)
        D_q   = quantise_q15(D)
        r_q   = quantise_q15(r_try)
        abs_accs = np.abs(D_q.astype(np.int64).T @ r_q.astype(np.int64))
        m = _margin(abs_accs, int(np.argmax(abs_accs)))
        if 0 < m < 0.005:
            r_f = r_try
            break
    _save("test_02_tight_margin", D, r_f)


def case_03_winner_near_zero():
    """
    (c) Winner near index 0.  Tests that the argmax tree doesn't accidentally
    skip the first few atoms due to off-by-one addressing.
    """
    D = _random_dict()
    j = int(RNG.integers(0, 4))   # winner in [0, 3]
    r = D[:, j] * 0.9 + RNG.standard_normal(M_LEN).astype(np.float32) * 0.01
    _save("test_03_winner_near_index_0", D, r)


def case_04_winner_near_end():
    """
    (d) Winner near index N_ATOMS-1.  Tests address wrap-around and the final
    pipeline drain stage of the FSM.
    """
    D = _random_dict()
    j = int(N_ATOMS - 1 - RNG.integers(0, 4))   # winner in [252, 255]
    r = D[:, j] * 0.9 + RNG.standard_normal(M_LEN).astype(np.float32) * 0.01
    _save("test_04_winner_near_index_N", D, r)


def case_05_engineered_tie():
    """
    (e) Engineered tie: two atoms produce EXACTLY the same |accumulator| in
    Q1.15 fixed-point arithmetic.  Hardware must resolve to the lower index.
    expected_idx should be the smaller of the two tied atoms.
    """
    # Build D so that column j1 == column j2 in Q1.15.
    # Then <D[:,j1], r> == <D[:,j2], r> always.
    D    = _random_dict()
    j1   = 10
    j2   = 200
    # Force quantised columns to be identical
    D_q  = quantise_q15(D)
    D_q[:, j2] = D_q[:, j1].copy()
    # Rebuild float D from the quantised version so D_float is consistent
    D_f  = D_q.astype(np.float32) / Q15_SCALE

    # Make j1 (and j2) the absolute winners
    r_f  = D_f[:, j1] * 0.85 + RNG.standard_normal(M_LEN).astype(np.float32) * 0.005
    r_q  = quantise_q15(r_f)

    # Verify the tie exists at the Q1.15 level
    abs_accs = np.abs(D_q.astype(np.int64).T @ r_q.astype(np.int64))
    assert abs_accs[j1] == abs_accs[j2], \
        f"Tie not achieved: {abs_accs[j1]} vs {abs_accs[j2]}"
    assert np.argmax(abs_accs) == j1, \
        "j1 must be max (lower index wins tie)"

    _save("test_05_engineered_tie", D_f, r_f)


# ---------------------------------------------------------------------------
# Verify: reload each .npz and re-run the correlator
# ---------------------------------------------------------------------------

def verify_all():
    print("\n--- Verification pass ---")
    for fname in sorted(os.listdir(OUTDIR)):
        if not fname.endswith(".npz"):
            continue
        path  = os.path.join(OUTDIR, fname)
        data  = np.load(path)
        idx_saved = int(data["expected_idx"])
        val_saved = int(data["expected_val"])
        idx_recomputed, val_recomputed = hw_correlate_q15(
            data["D_q15"], data["r_q15"]
        )
        ok = (idx_recomputed == idx_saved) and (val_recomputed == val_saved)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {fname}  saved=({idx_saved}, {val_saved})  "
              f"recomputed=({idx_recomputed}, {val_recomputed})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Re-load and re-check all saved vectors")
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    print("Generating golden vectors → ./golden/")
    case_01_easy_margin()
    case_02_tight_margin()
    case_03_winner_near_zero()
    case_04_winner_near_end()
    case_05_engineered_tie()

    if args.verify:
        verify_all()

    print("\nDone.  Hand the ./golden/ directory to HDL-A and HDL-B.")
