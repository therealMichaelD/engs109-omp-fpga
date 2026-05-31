#!/usr/bin/env python3
"""npz_to_mem.py -- generate the 15 testbench data files from the 5 golden .npz.

For each golden vector it emits, into VEC_DIR:
  <base>_D.mem   1024 lines, 128 hex chars (512-bit) each, linear chunk order
  <base>_r.mem      4 lines, 128 hex chars each
  <base>_exp.txt    1 line  "<IDX 2hex> <VAL 10hex>"
                    VAL = expected_val masked to 40-bit two's complement.

The 512-bit word packs 32 consecutive Q1.15 values, lane i (the i-th MAC) at
bits [16*i +: 16].  D and r use the SAME packer, so lane i always pairs D[k]
with r[k] for the same k -- the inner product is correct regardless of whether
corr_datapath reads lane 0 as the LSBs or the MSBs, and regardless of chunk
order (it's all one sum).  The only invariant that matters is that every entry
appears exactly once with D and r packed identically; this guarantees it.

Usage:  python npz_to_mem.py            (reads/writes ./goldenVectors)
        python npz_to_mem.py /path/dir  (reads/writes that dir)
"""
import numpy as np
import os
import sys

VEC_DIR = sys.argv[1] if len(sys.argv) > 1 else "../goldenVectors"
N_ATOMS, M_LEN, NUM_MACS, N_CHUNKS = 256, 128, 32, 4

BASES = [
    "test_01_easy_margin",
    "test_02_tight_margin",
    "test_03_winner_near_index_0",
    "test_04_winner_near_index_N",
    "test_05_engineered_tie",
]


def pack512(vals):
    assert len(vals) == NUM_MACS, f"expected {NUM_MACS} values, got {len(vals)}"
    word = 0
    for i, v in enumerate(vals):
        word |= (int(v) & 0xFFFF) << (16 * i)   # lane i -> bits [16i +: 16]
    return f"{word:0128X}"                       # 512 bits -> 128 hex chars, MSB first


def convert(base):
    path = os.path.join(VEC_DIR, base + ".npz")
    if not os.path.exists(path):
        print(f"  SKIP (missing): {path}")
        return False
    z = np.load(path)
    D = z["D_q15"].astype(np.int16)     # (M_LEN, N_ATOMS) = (128, 256)
    r = z["r_q15"].astype(np.int16)     # (M_LEN,)         = (128,)
    assert D.shape == (M_LEN, N_ATOMS), f"D_q15 shape {D.shape}"
    assert r.shape == (M_LEN,),         f"r_q15 shape {r.shape}"

    # D.mem -- linear chunk order  idx = atom*N_CHUNKS + chunk
    with open(os.path.join(VEC_DIR, base + "_D.mem"), "w") as f:
        for j in range(N_ATOMS):
            for c in range(N_CHUNKS):
                f.write(pack512(D[c * NUM_MACS:(c + 1) * NUM_MACS, j]) + "\n")

    # r.mem -- 4 chunks of the residual
    with open(os.path.join(VEC_DIR, base + "_r.mem"), "w") as f:
        for c in range(N_CHUNKS):
            f.write(pack512(r[c * NUM_MACS:(c + 1) * NUM_MACS]) + "\n")

    # exp.txt -- index (2 hex) + full accumulator sign-extended to 40 bits (10 hex)
    idx = int(z["expected_idx"])
    val = int(z["expected_val"])
    assert 0 <= idx < N_ATOMS, f"expected_idx out of range: {idx}"
    val40 = val & ((1 << 40) - 1)
    with open(os.path.join(VEC_DIR, base + "_exp.txt"), "w") as f:
        f.write(f"{idx:02X} {val40:010X}\n")

    print(f"  {base}: idx={idx}  val={val}  ->  _D.mem (1024)  _r.mem (4)  _exp.txt")
    return True


if __name__ == "__main__":
    print(f"writing into: {os.path.abspath(VEC_DIR)}")
    n = sum(convert(b) for b in BASES)
    print(f"done. {n}/5 vectors converted.")