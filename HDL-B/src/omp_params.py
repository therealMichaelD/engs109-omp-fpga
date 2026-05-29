"""
Canonical Python copy of the frozen system parameters from
OMP_Team_Interface_Contract.docx

"""

# Window size - defines D shape and PRD evaluation
WINDOW_N = 256

# Atom length / number of measurements
M_LEN = 128

# Number of dictionary atoms
N_ATOMS = 256

# Max OMP iterations (software-enforced; hardware is stateless)
MAX_ITERS = 32

# I/O Fixed-point format: Q1.15
IO_INT_BITS = 1
IO_FRAC_BITS = 15

IO_TOTAL_BITS = IO_INT_BITS + IO_FRAC_BITS

# System clock
CLK_HZ = 100_000_000
