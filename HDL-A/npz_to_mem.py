#!/usr/bin/env python3
"""
npz_to_mem.py -- Convert OMP golden-vector .npz files into .mem files for a
VHDL testbench (HDL-A).

For each input .npz (schema per Team Interface Contract §9) this emits three
files into the output directory, named after the input stem:

  <stem>_D.mem         Dictionary D_q15, ATOM-MAJOR, as hex words.
  <stem>_r.mem         Residual   r_q15,            as hex words.
  <stem>_expected.txt  expected_idx then expected_val, as hex (two lines).

Atom-major layout
-----------------
D_q15 has shape (M_LEN, N_ATOMS) == (row = measurement index i, col = atom j),
so "atom j" is column j. Atom-major == every entry of atom 0 first, then atom 1,
... == np.ravel(D_q15, order='F'). This matches §5 ("atom j's data is
contiguous").

Word grouping (re-targetable BRAM layout)
-----------------------------------------
--values-per-word G packs G consecutive Q1.15 samples into one hex word.
Within a word the LOWEST flat index occupies the LEAST-significant bits. With
G=2 this reproduces the §5 AXI packing exactly (even index -> low 16 bits, odd
index -> high 16 bits). Default G=1: one 16-bit sample per line, ready for a
16-bit-wide memory init (Verilog $readmemh or VHDL textio hread).

Expected-value file
-------------------
Two lines, no comments, so a TB hread loop can consume it directly:
  line 1: expected_idx, two's complement, --idx-bits wide (default 32)
  line 2: expected_val, two's complement, --val-bits wide (default 64)
expected_val is the full-precision int64 accumulator value; the TB can mask it
down to its 38-bit ACC_WIDTH if it wants a truncated compare (§4 RESULT_VAL).

Validation gate
---------------
By default the script re-reads every .mem it writes, reconstructs the int16
arrays, and asserts they are byte-exact against the .npz. Use --no-verify to
skip. A failed round-trip exits non-zero.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SAMPLE_BITS = 16          # Q1.15 -> signed 16-bit, fixed by §2.
SAMPLE_MASK = (1 << SAMPLE_BITS) - 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _i16_to_u16(arr: np.ndarray) -> np.ndarray:
    """Reinterpret an int16 array's bit pattern as uint16 (two's complement)."""
    return np.ascontiguousarray(arr.astype(np.int16)).view(np.uint16)


def pack_words(samples_u16: np.ndarray, g: int) -> np.ndarray:
    """Group a flat uint16 array into words of g samples, lowest index in LSBs.

    Pads the final word with zeros if the length is not a multiple of g.
    Returns an array of Python ints (word values).
    """
    n = samples_u16.size
    if n % g != 0:
        pad = g - (n % g)
        print(f"  note: padding {pad} zero sample(s) to fill the last "
              f"{g}-sample word", file=sys.stderr)
        samples_u16 = np.concatenate([samples_u16, np.zeros(pad, dtype=np.uint16)])
    grouped = samples_u16.reshape(-1, g).astype(object)  # object -> no overflow
    words = np.zeros(grouped.shape[0], dtype=object)
    for k in range(g):
        words = words | (grouped[:, k] << (SAMPLE_BITS * k))
    return words


def write_mem(path: Path, words, word_bits: int) -> None:
    hex_digits = (word_bits + 3) // 4
    with path.open("w") as fh:
        for w in words:
            fh.write(f"{int(w) & ((1 << word_bits) - 1):0{hex_digits}x}\n")


def write_expected(path: Path, idx: int, val: int,
                   idx_bits: int, val_bits: int) -> None:
    idx_hex = f"{idx & ((1 << idx_bits) - 1):0{(idx_bits + 3) // 4}x}"
    val_hex = f"{val & ((1 << val_bits) - 1):0{(val_bits + 3) // 4}x}"
    path.write_text(idx_hex + "\n" + val_hex + "\n")


# ---------------------------------------------------------------------------
# verify (the validation gate)
# ---------------------------------------------------------------------------
def _read_mem_to_i16(path: Path, g: int, n_expected: int) -> np.ndarray:
    """Inverse of write_mem: hex words -> flat int16 samples (lowest index = LSBs)."""
    samples = []
    for line in path.read_text().split():
        word = int(line, 16)
        for k in range(g):
            samples.append((word >> (SAMPLE_BITS * k)) & SAMPLE_MASK)
    u16 = np.array(samples[:n_expected], dtype=np.uint16)
    return u16.view(np.int16)


def verify(stem_dir: Path, stem: str, D_q15: np.ndarray, r_q15: np.ndarray,
           g: int) -> None:
    m_len, n_atoms = D_q15.shape

    d_flat = _read_mem_to_i16(stem_dir / f"{stem}_D.mem", g, D_q15.size)
    d_back = d_flat.reshape((m_len, n_atoms), order="F")   # invert atom-major
    if not np.array_equal(d_back, D_q15.astype(np.int16)):
        raise AssertionError(f"{stem}: D round-trip MISMATCH")

    r_back = _read_mem_to_i16(stem_dir / f"{stem}_r.mem", g, r_q15.size)
    if not np.array_equal(r_back, r_q15.astype(np.int16)):
        raise AssertionError(f"{stem}: r round-trip MISMATCH")

    print(f"  verify: D and r round-trip byte-exact ({m_len}x{n_atoms}, r={r_q15.size}) OK")


# ---------------------------------------------------------------------------
# per-file conversion
# ---------------------------------------------------------------------------
def convert(npz_path: Path, out_dir: Path, g: int,
            idx_bits: int, val_bits: int, do_verify: bool) -> None:
    print(f"{npz_path.name}:")
    z = np.load(npz_path)

    for key in ("D_q15", "r_q15", "expected_idx", "expected_val"):
        if key not in z:
            raise KeyError(f"{npz_path.name}: missing required array '{key}'")

    D_q15 = z["D_q15"]
    r_q15 = z["r_q15"]
    if D_q15.dtype != np.int16 or r_q15.dtype != np.int16:
        raise TypeError(f"{npz_path.name}: D_q15/r_q15 must be int16 "
                        f"(got {D_q15.dtype}, {r_q15.dtype})")

    m_len, n_atoms = D_q15.shape
    if r_q15.shape != (m_len,):
        raise ValueError(f"{npz_path.name}: r_q15 shape {r_q15.shape} "
                         f"!= ({m_len},) implied by D_q15")

    idx = int(z["expected_idx"])
    val = int(z["expected_val"])
    word_bits = g * SAMPLE_BITS
    stem = npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # D: atom-major (column-major) flatten, then pack.
    d_words = pack_words(_i16_to_u16(np.ravel(D_q15, order="F")), g)
    write_mem(out_dir / f"{stem}_D.mem", d_words, word_bits)

    # r: natural order, same packing.
    r_words = pack_words(_i16_to_u16(np.ravel(r_q15)), g)
    write_mem(out_dir / f"{stem}_r.mem", r_words, word_bits)

    write_expected(out_dir / f"{stem}_expected.txt", idx, val, idx_bits, val_bits)

    print(f"  wrote {stem}_D.mem ({len(d_words)} words), "
          f"{stem}_r.mem ({len(r_words)} words), {stem}_expected.txt "
          f"[{word_bits}-bit words, idx={idx}, val={val}]")

    if do_verify:
        verify(out_dir, stem, D_q15, r_q15, g)


def main() -> int:
    p = argparse.ArgumentParser(description="OMP golden-vector .npz -> .mem converter")
    p.add_argument("inputs", nargs="+", type=Path, help="one or more .npz files")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: next to each input)")
    p.add_argument("--values-per-word", type=int, default=1, metavar="G",
                   help="Q1.15 samples packed per hex word, LSB=lowest index "
                        "(default 1; G=2 == §5 AXI packing)")
    p.add_argument("--idx-bits", type=int, default=32,
                   help="bit width of expected_idx field (default 32)")
    p.add_argument("--val-bits", type=int, default=64,
                   help="bit width of expected_val field (default 64)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the byte-exact round-trip check")
    args = p.parse_args()

    if args.values_per_word < 1:
        p.error("--values-per-word must be >= 1")

    ok = True
    for npz_path in args.inputs:
        out_dir = args.out if args.out is not None else npz_path.parent
        try:
            convert(npz_path, out_dir, args.values_per_word,
                    args.idx_bits, args.val_bits, not args.no_verify)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
