"""
Bit-accurate Python reference for the OMP correlation engine.
Mirrors the hardware described in OMP_Team_Interface_Contract

"""
import numpy as np
from omp_params import M_LEN, N_ATOMS, IO_TOTAL_BITS


def omp_argmax_fixed(D_q15, r_q15, mac_bits=IO_TOTAL_BITS):
    """
    Bit-accurate model of one COMPUTE-state execution of the hardware
    correlation engine.

    Given a Q1.15 dictionary D and Q1.15 residual r, compute
        accum[j] = sum_i D[i,j] * r[i]  for j=0..N_ATOMS-1
    using integer arithmetic, then return the argmax of [accum]

    Parameters
    -------------
    D_q15 : np.ndarray, dtype=int16, shape=(M_LEN, N_ATOMS)
        Dictionary in Q1.15 format. Atom-major: column j is atom j
    mac_bits : int, default=IO_TOTAL_BITS (16)
        Width of MAC inputs in bits. Sweep variable per contract.
        Values < 16 truncate the (16 - mac_bits) LSBs of D and r before the multiply. AXI interface always carries full Q1.15

    Final Call: Truncate before multiplying. Needs to line up with HDL-A

    Return
    ------------
    idx : int
        Argmax atom index, in [0, N_ATOMS-1]. Ties brown by lowest index (matches np.argmax)
    val : int
        Signed accumulator value at the winning atom. Matches what RESULT_VAL reports.
    """
    # Input validation
    # Catch the mostlikely upstream bugs early. These asserts pay for themselves the first time someone passes a transposed D or a float array by mistake.
    assert D_q15.dtype == np.int16, f"D_q15 must be int16, got {D_q15.dtype}"
    assert r_q15.dtype == np.int16, f"r_q15 must be int16, got {r_q15.dtype}"
    assert D_q15.shape == (M_LEN, N_ATOMS), \
        f"D_q15 must be ({M_LEN}, {N_ATOMS}), got {D_q15.shape}"
    assert r_q15.shape == (M_LEN,), \
        f"r_q15 must be ({M_LEN},), got {r_q15.shape}"
    assert 1 <= mac_bits <= IO_TOTAL_BITS, \
        f"mac_bits must be in [1, {IO_TOTAL_BITS}], got {mac_bits}"

    # --- promote to int64 for headroom ---
    # Q1.15 × Q1.15 products are Q2.30, summed M_LEN=128 times reaches
    # ~2^37 magnitude (contract §3 footnote). int64 is more than enough.
    # We promote BEFORE truncation so the shift operates on a wide type
    # and arithmetic-shift semantics are unambiguous.
    D = D_q15.astype(np.int64)
    r = r_q15.astype(np.int64)

    # --- apply MAC_BITS truncation (sweep knob) ---
    # Arithmetic right shift drops LSBs while preserving sign.
    # NumPy's >> on signed integers is arithmetic shift, matching what
    # a fixed-point hardware truncation does.
    if mac_bits < IO_TOTAL_BITS:
        shift = IO_TOTAL_BITS - mac_bits
        D = D >> shift
        r = r >> shift

    # --- integer inner product, all atoms at once ---
    # accum[j] = sum over i of D[i, j] * r[i]
    # D.T @ r computes exactly this in int64. No floating point anywhere.
    accum = D.T @ r                # shape (N_ATOMS,), dtype int64

    # --- argmax over absolute value ---
    # np.argmax returns the lowest index on ties. HDL-A's argmax tree
    # must match; pending §10 confirmation (see module docstring).
    idx = int(np.argmax(np.abs(accum)))
    val = int(accum[idx])

    return idx, val
