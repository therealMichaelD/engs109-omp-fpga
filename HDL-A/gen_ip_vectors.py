import numpy as np

import sys
sys.path.insert(0, "../HDL-B/src")   # folder that contains omp_params.py
from omp_params import IO_TOTAL_BITS, M_LEN, N_ATOMS   # 16, 128, 256

def single_atom_ip(D_q15, r_q15, j, mac_bits=16):
    D = D_q15[:, j].astype(np.int64)
    r = r_q15.astype(np.int64)
    if mac_bits < IO_TOTAL_BITS:
        shift = IO_TOTAL_BITS - mac_bits
        D = D >> shift
        r = r >> shift
    return int(D @ r)                       # signed accumulator value for atom j

# ---- pick where D_q15 / r_q15 come from ----
# OPTION A (real): load a golden vector from HDL-B's repo
data = np.load("../goldenVectors/test_01_easy_margin.npz")
D_q15, r_q15 = data["D_q15"], data["r_q15"]
#
# OPTION B (synthetic, no dependency on HDL-B): make your own and reuse the
# SAME arrays in your testbench .mem files.
rng = np.random.default_rng(0)              # fixed seed = reproducible
D_q15 = rng.integers(-32768, 32768, size=(M_LEN, N_ATOMS), dtype=np.int16)
r_q15 = rng.integers(-32768, 32768, size=(M_LEN,),         dtype=np.int16)

# ---- atom 0 ----
j = 0
print(f"atom {j:<4} MAC_BITS=16 -> {single_atom_ip(D_q15, r_q15, j, 16)}")
print(f"atom {j:<4} MAC_BITS= 8 -> {single_atom_ip(D_q15, r_q15, j, 8)}")

# ---- find an atom whose 16-bit result is negative (exercises the floor edge) ----
accum16 = np.array([single_atom_ip(D_q15, r_q15, k, 16) for k in range(N_ATOMS)])
neg_j = int(np.argmin(accum16))             # most-negative atom
print(f"atom {neg_j:<4} MAC_BITS=16 -> {single_atom_ip(D_q15, r_q15, neg_j, 16)}  (negative)")
print(f"atom {neg_j:<4} MAC_BITS= 8 -> {single_atom_ip(D_q15, r_q15, neg_j, 8)}  (negative)")