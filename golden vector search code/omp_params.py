# omp_params.py
# Single source of truth for all frozen system parameters.
# Matches §2 of OMP_Team_Interface_Contract exactly.
# If this file and the contract table diverge, THIS FILE IS WRONG.

WINDOW_N   = 256   # signal window length (not used by correlation engine directly)
M_LEN      = 128   # measurement vector length / atom length
N_ATOMS    = 256   # number of dictionary atoms
K_MAX      = 32    # max OMP iterations (SW-enforced; hardware is stateless)
CLK_HZ     = 100e6 # 100 MHz

# Q1.15 fixed-point: 1 sign bit, 15 fractional bits
IO_INT_BITS  = 1
IO_FRAC_BITS = 15
Q15_SCALE    = 2 ** IO_FRAC_BITS          # 32768
Q15_MAX      =  (2 ** 15) - 1             #  32767
Q15_MIN      = -(2 ** 15)                 # -32768

# Internal accumulator — Q8.30 (38-bit).
# Informational only for SW; hardware owns the implementation.
ACC_INT_BITS  = 8
ACC_FRAC_BITS = 30
