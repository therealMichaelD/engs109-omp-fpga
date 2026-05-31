import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


M_LEN, N_ATOMS, WINDOW_N = 128, 256, 256
Q15 = 1 << 15

def quantise_q15(x):
    c = np.clip(x, -1.0, 1.0 - 2**-15)
    return np.clip(np.round(c * Q15), -32768, 32767).astype(np.int16)
def dequant(x): return x.astype(np.float64) / Q15

# --- load the committed db4 artifacts ---
D_atom_major = np.fromfile("D.bin", dtype="<f4").reshape(N_ATOMS, M_LEN)
D_float      = D_atom_major.T                      # (128, 256), unit-norm atoms
col_norms    = np.fromfile("col_norms.bin", dtype="<f4").astype(np.float64)
Psi          = np.fromfile("Psi.bin", dtype="<f4").reshape(WINDOW_N, WINDOW_N).astype(np.float64)
X            = np.fromfile("X.bin", dtype="<f4").reshape(-1, WINDOW_N)   # original windows
Y            = np.fromfile("Y.bin", dtype="<f4").reshape(-1, M_LEN)      # measurements
D_q15        = quantise_q15(D_float)

def omp_argmax_fixed(D_q15, r_q15, mac_bits):
    shift = 16 - mac_bits
    Dt = (D_q15.astype(np.int64) >> shift) if shift > 0 else D_q15.astype(np.int64)
    rt = (r_q15.astype(np.int64) >> shift) if shift > 0 else r_q15.astype(np.int64)
    corr = np.abs(Dt.T @ rt)
    return int(np.argmax(corr)), corr

def reconstruct_window(idx_window, mac_bits=8, k_max=32):
    s = X[idx_window].astype(np.float64)
    y = Y[idx_window].astype(np.float64)
    y_scale = np.abs(y).max() + 1e-12
    y_q15 = quantise_q15(y / y_scale)
    D_f, y_f = D_float, y / y_scale
    r_f = dequant(y_q15).copy()
    support, coeffs = [], np.zeros(0)
    for _ in range(k_max):
        r_q15 = quantise_q15(r_f)
        idx, _ = omp_argmax_fixed(D_q15, r_q15, mac_bits)
        if idx in support: break
        support.append(idx)
        Ds = D_f[:, support]
        coeffs, *_ = np.linalg.lstsq(Ds, dequant(y_q15), rcond=None)
        r_f = dequant(y_q15) - Ds @ coeffs
    a = np.zeros(N_ATOMS)
    a[support] = (coeffs / col_norms[support]) * y_scale
    return s, Psi @ a, support

def prd(s, sh): return 100*np.linalg.norm(s-sh)/np.linalg.norm(s)

# pick a clean window — scan a few, print PRD, eyeball
import matplotlib.pyplot as plt
for w in [0, 5, 10, 20, 40]:
    s, sh, S = reconstruct_window(w, mac_bits=8)
    print(f"Window {w:3d}: |Support|={len(S):2d}  PRD={prd(s,sh):.1f}%")

# choose one and export
W = 0    # <- set to the window index you liked from the scan
s, sh, S = reconstruct_window(W, mac_bits=8)
print(f"Chosen Window {W}: PRD={prd(s,sh):.2f}%")
np.save("beat_original.npy", s)
np.save("beat_recon.npy",    sh)
plt.figure(figsize=(9,3)); plt.plot(s,label="original"); plt.plot(sh,"--",label="recon 8-bit")
plt.legend(); plt.title(f"Window {W}, PRD={prd(s,sh):.1f}%"); plt.tight_layout(); plt.show()