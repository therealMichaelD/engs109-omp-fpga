"""
omp_reference.py  —  SW deliverable, Day 2
Float32 and fixed-point OMP reference implementations.

Two entry points:
  omp_float(D, r, k_max)       — full floating-point OMP (accuracy ceiling)
  omp_fixedpoint(D_q15, r_q15) — single-iteration, bit-accurate inner-product
                                   + argmax that mirrors what the hardware does.

The hardware only accelerates the correlation step (one argmax per iteration).
Everything else — least-squares solve, residual update, support set — runs in
SW.  So omp_fixedpoint() is the function whose output must match RESULT_IDX.
"""

import numpy as np
from omp_params import (
    M_LEN, N_ATOMS, K_MAX,
    Q15_SCALE, Q15_MAX, Q15_MIN,
    ACC_FRAC_BITS,
)


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantise_q15(x: np.ndarray) -> np.ndarray:
    """
    Convert a float array to Q1.15 int16.
    Clips to [-1, 1) before scaling so no overflow is silent.
    """
    clipped = np.clip(x, -1.0, 1.0 - 2**-15)
    scaled  = np.round(clipped * Q15_SCALE).astype(np.int64)
    scaled  = np.clip(scaled, Q15_MIN, Q15_MAX)
    return scaled.astype(np.int16)


def dequantise_q15(x: np.ndarray) -> np.ndarray:
    """Q1.15 int16 → float64 (for residual comparison)."""
    return x.astype(np.float64) / Q15_SCALE


# ---------------------------------------------------------------------------
# Float32 OMP  (accuracy ceiling / golden reference for reconstruction quality)
# ---------------------------------------------------------------------------

def omp_float(
    D: np.ndarray,          # (M_LEN, N_ATOMS)  float32 dictionary
    y: np.ndarray,          # (M_LEN,)           float32 measurement vector
    k_max: int = K_MAX,
) -> dict:
    """
    Full OMP in float32.  Returns all intermediate values useful for debugging
    and for generating golden vectors.

    Returns a dict with:
        support     : list of selected atom indices, length k_max
        coeffs      : float32 coefficient vector at each iteration
        residuals   : list of residual vectors (length k_max + 1, r[0] = y)
        inner_prods : (k_max, N_ATOMS) array of |<D[:,j], r_i>| at each iter
    """
    assert D.shape == (M_LEN, N_ATOMS), f"D shape {D.shape} != ({M_LEN},{N_ATOMS})"
    assert y.shape == (M_LEN,),          f"y shape {y.shape} != ({M_LEN},)"

    D   = D.astype(np.float32)
    y   = y.astype(np.float32)
    r   = y.copy()

    support     = []
    residuals   = [r.copy()]
    inner_prods = []

    for _ in range(k_max):
        # --- correlation step (this is what the hardware accelerates) ---
        correlations = np.abs(D.T @ r)          # (N_ATOMS,)
        inner_prods.append(correlations.copy())
        idx = int(np.argmax(correlations))
        support.append(idx)

        # --- least-squares step (SW only, not on FPGA) ---
        D_sub  = D[:, support]                  # (M_LEN, len(support))
        coeffs, _, _, _ = np.linalg.lstsq(D_sub, y, rcond=None)

        # --- residual update ---
        r = y - D_sub @ coeffs
        residuals.append(r.copy())

        if np.linalg.norm(r) < 1e-6:
            break

    return {
        "support":     support,
        "coeffs":      coeffs,
        "residuals":   residuals,
        "inner_prods": np.array(inner_prods),
    }


# ---------------------------------------------------------------------------
# Fixed-point correlation engine  (mirrors hardware, one iteration)
# ---------------------------------------------------------------------------

def hw_correlate_q15(
    D_q15: np.ndarray,   # (M_LEN, N_ATOMS)  int16
    r_q15: np.ndarray,   # (M_LEN,)           int16
    mac_bits: int = 16,  # sweep knob — matches §3 MAC_BITS generic
) -> tuple[int, int]:
    """
    Bit-accurate model of the FPGA correlation engine for ONE iteration.

    Steps:
      1. Optionally truncate D and r to mac_bits (models MAC_BITS generic).
      2. For each atom j, compute acc_j = sum_i( D[i,j] * r[i] )  in int64.
      3. Return argmax of |acc_j|, and the accumulator value at that index.

    The internal accumulator is Q8.30 (38-bit), but we use int64 here
    because Python has no native 38-bit type and the extra bits don't affect
    the argmax result (we never overflow int64 for these dimensions).

    Returns:
        (argmax_index, accumulator_value_at_argmax)
    These map directly to RESULT_IDX and RESULT_VAL in the AXI register map.
    """
    assert D_q15.shape == (M_LEN, N_ATOMS)
    assert r_q15.shape == (M_LEN,)
    assert D_q15.dtype == np.int16
    assert r_q15.dtype == np.int16

    # --- MAC_BITS truncation (internal precision sweep knob) ---
    if mac_bits < 16:
        shift = 16 - mac_bits
        D_trunc = (D_q15.astype(np.int32) >> shift).astype(np.int16)
        r_trunc = (r_q15.astype(np.int32) >> shift).astype(np.int16)
    else:
        D_trunc = D_q15
        r_trunc = r_q15

    # --- inner products in int64 to prevent any overflow ---
    D64  = D_trunc.astype(np.int64)   # (M_LEN, N_ATOMS)
    r64  = r_trunc.astype(np.int64)   # (M_LEN,)
    accs = D64.T @ r64                 # (N_ATOMS,)  — exact integer arithmetic

    # --- argmax of absolute value (tie-break: lower index wins) ---
    abs_accs  = np.abs(accs)
    argmax_idx = int(np.argmax(abs_accs))
    argmax_val = int(accs[argmax_idx])

    return argmax_idx, argmax_val


# ---------------------------------------------------------------------------
# Full SW-side OMP loop using the fixed-point correlator
# (This is what runs on the Cortex-A9, calling the accelerator each iteration)
# ---------------------------------------------------------------------------

def omp_sw_with_hw_correlator(
    D_q15: np.ndarray,   # (M_LEN, N_ATOMS)  int16
    r_q15: np.ndarray,   # (M_LEN,)           int16 — initial residual (= y_q15)
    k_max: int = K_MAX,
    mac_bits: int = 16,
) -> dict:
    """
    SW OMP loop that calls hw_correlate_q15() for the correlation step
    and does the least-squares solve in float64 (SW-only).

    This is the reference for the full end-to-end accuracy evaluation.
    """
    D_f = dequantise_q15(D_q15)   # float64 for LS solve
    y_f = dequantise_q15(r_q15)
    r_f = y_f.copy()

    support   = []
    residuals = [r_q15.copy()]

    for _ in range(k_max):
        # --- hardware call (simulated) ---
        r_now_q15 = quantise_q15(r_f)
        idx, val  = hw_correlate_q15(D_q15, r_now_q15, mac_bits=mac_bits)
        support.append(idx)

        # --- SW least-squares in float64 ---
        D_sub  = D_f[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(D_sub, y_f, rcond=None)
        r_f    = y_f - D_sub @ coeffs

        residuals.append(quantise_q15(r_f))

        if np.linalg.norm(r_f) < 1e-8:
            break

    return {
        "support":   support,
        "coeffs":    coeffs,
        "residuals": residuals,
    }
