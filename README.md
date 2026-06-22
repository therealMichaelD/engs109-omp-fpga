# Fixed-Point OMP for ECG Compressed Sensing — A Precision–Resource Study on the Zynq-7020

A fixed-point **Orthogonal Matching Pursuit (OMP)** signal-recovery engine for compressed-sensing
ECG, targeting the **Xilinx Zynq-7020** SoC (Digilent Zybo Z7-20). The project partitions OMP so the
parallel **correlation/argmax** step runs in programmable logic (PL) while the sequential
**least-squares solve** runs in software on the ARM **Cortex-A9** (PS). Its central result is a
**precision–resource Pareto front**: how reconstruction quality and FPGA resource cost trade off as a
function of the multiply-accumulate operand bit-width.

**Authors:** Brandon Carido, Michael Dang, Connor Machado — Thayer School of Engineering, Dartmouth College.

---

## Motivation

Compressed sensing (CS) lets a wearable ECG sensor acquire and transmit far fewer samples than the
Nyquist rate, cutting the dominant energy costs (analog front end + radio). The catch is that CS moves
the computational burden to the **recovery** side: the receiver must solve an underdetermined inverse
problem. OMP is a standard greedy recovery algorithm, but its per-iteration correlation against every
dictionary atom is expensive on a general-purpose CPU. That structure — a wide, parallel correlation
step plus a small sequential solve — maps naturally onto an FPGA, which is what this project explores.

## What we set out to do

1. Implement a fixed-point OMP recovery engine for the Zynq-7020 with a hardware/software split that
   plays to each side's strengths.
2. Sweep the MAC operand bit-width and characterize the **PRD (reconstruction error) vs. resource cost**
   trade-off, identifying an efficient operating point.
3. Benchmark the hardware correlation engine against a tuned Cortex-A9 software implementation on the
   same SoC.

## Approach

**Compressed-sensing model.** Each 256-sample ECG window is sampled as `y = Φs` using a fixed
**Bernoulli ±1** measurement matrix at a **0.5 compression ratio** (M = 128 measurements, N = 256
samples). Recovery uses the effective dictionary `D = Φψ`.

**Sparsifying basis.** A plain **DCT** basis represented the QRS transient poorly (~57% PRD). We
switched to a **Daubechies-4 (db4) orthonormal wavelet** basis with unit-norm atoms, which more than
halved the error (~20% PRD) and avoided the dual-basis complications of biorthogonal wavelets.

**Hardware/software partition.**
- **PL (FPGA fabric):** the correlation engine — 256 atoms × 128-tap inner products with **32 parallel
  multiply-accumulate (MAC) units**, a pipelined argmax, Q1.15 I/O and a wide internal accumulator.
- **PS (Cortex-A9):** the sequential least-squares stage (Cholesky-style update) and the OMP outer loop.

**Parameterized precision.** A single `MAC_BITS` generic lets one design synthesize at every operating
point, which is what makes the bit-width sweep possible without rewriting the engine.

## Methodology

- **PRD (accuracy axis):** measured in a **bit-accurate fixed-point simulation** over real MIT-BIH ECG
  windows, swept across `MAC_BITS ∈ {3 … 16}`.
- **Resources (cost axis):** DSP / LUT / BRAM pulled from **Vivado synthesis** at each sweep point.
- **Latency:** computed analytically from the FSM cycle count × clock period (100 MHz).
- **Software baseline:** a tuned C correlation pass (`-O3`, NEON, fast-math) on the Cortex-A9 at
  667 MHz, timed on-chip over 1000 iterations.
- **Dataset:** MIT-BIH ECG Compression Test Database (PhysioNet), segmented into 256-sample windows.

The fixed-point simulator was validated **bit-exact** against five golden test vectors (matching both
the selected atom index and the accumulator value), covering an easy margin, a near-tie (~0.002%
margin), winners near both ends of the index range, and an engineered exact tie. The agreed numerical
conventions were **truncate-before-multiply** and **lowest-index-wins** tie-breaking. The same vectors
are converted to `.mem`/`.txt` and consumed by the RTL testbench (`HDL-A/`), so the hardware
correlation engine is checked against the identical bit-exact references in simulation.

## Key results

Reconstruction error (PRD) is essentially **flat from 16 bits down to ~8 bits**, then degrades sharply
below 6 bits. The efficient knee is **11-bit arithmetic**: it matches full 16-bit reconstruction quality
*and* is the point at which the multipliers map onto the otherwise-idle **DSP48 array** instead of
consuming LUT fabric — freeing logic for the AXI interface and control at no accuracy cost.

| MAC_BITS | PRD (%) | DSP | LUT  | BRAM | Latency (cyc) |
|---------:|--------:|----:|-----:|-----:|--------------:|
| 3        | 82.9    | 0   | 867  | 16   | 1033          |
| 4        | 74.6    | 0   | 1473 | 16   | 1033          |
| 5        | 39.3    | 0   | 1760 | 16   | 1033          |
| 6        | 22.5    | 0   | 2078 | 16   | 1033          |
| 8        | 20.4    | 0   | 3034 | 16   | 1033          |
| **11**   | **20.1**| **47** | **610** | **16** | **1033** |
| 16       | 20.1    | 47  | 610  | 16   | 1033          |

*(11-bit highlighted as the recommended operating point; lower PRD is better.)*

**Speedup (estimated, analytical).** At the 11-bit operating point the correlation pass takes 1033
cycles ≈ **10.33 µs** at 100 MHz, versus **187 µs** for the tuned Cortex-A9 baseline — an estimated
**~18× speedup on the correlation step** (compute-only, single pass).

**Recommendation.** For resource-constrained deployment of this class of recovery problem, an **11-bit
fixed-point correlation engine with a software least-squares solve** is the efficient design point.

## Scope and limitations (read this before judging the numbers)

- **No end-to-end run on hardware.** The PS↔PL integration over AXI was out of scope for the project
  timeframe. The correlation engine and the software baseline were each validated **independently**.
- **FPGA-path latencies are analytical**, derived from cycle counts × clock period and Vivado synthesis
  estimates — **not measured on a live board**. The ~18× speedup is therefore an **estimate**, and it
  covers the **correlation step only**, not full end-to-end OMP recovery.
- **Absolute reconstruction quality is bounded by the compression ratio, not by precision.** Mean PRD
  is ~20% at M/N = 0.5, above the conventional <9% "very good" threshold. This is a consequence of
  aggressive undersampling on morphologically complex ECG. The contribution here is the
  **precision/resource characterization at a usable operating point**, not meeting an absolute quality target.

The repository is organized by the three workstreams defined in the team interface contract
(HDL-A = the FPGA IP, HDL-B = the fixed-point simulator, SW = the driver and software baseline), with
the shared golden test vectors at the root so both the RTL testbench and the Python suite point at the
same files.

```
.
├── HDL-A/          # Synthesizable correlation-engine RTL (Verilog) + the per-MAC_BITS
│                   #   Vivado synthesis runs (DSP/LUT/BRAM). Includes a converter that
│                   #   turns the .npz golden vectors into .mem/.txt for RTL simulation.
├── HDL-B/          # Bit-accurate fixed-point Python reference (the accuracy axis):
│                   #   the OMP sim, the MAC_BITS sweep, the PRD curve, and figures.
├── SW/             # ARM Cortex-A9 software: AXI driver and the tuned C correlation
│                   #   baseline used for the speedup comparison.
├── goldenVectors/  # The 5 shared test cases as .npz, plus generated .mem/.txt copies
│                   #   consumed by the RTL testbench.
├── FinalReport/    # Final poster and write-up.
├── .vscode/        # Editor config.
└── .gitignore
```

## Reproducing the headline figure

The headline Pareto front is reproducible **without hardware**:

1. Run the fixed-point simulator (`HDL-B/`) over the MIT-BIH windows for each `MAC_BITS` value → PRD per setting.
2. Synthesize the RTL (`HDL-A/`) at the same `MAC_BITS` points in Vivado → DSP / LUT / BRAM per setting.
3. Combine the two into the PRD-vs-resource Pareto plot; latency follows from the FSM cycle count.

## Tech stack

Verilog / RTL · Xilinx Vivado synthesis · Zynq-7020 SoC · AXI-Lite · DSP48 mapping · fixed-point
(Q1.15 I/O) arithmetic · Python / NumPy bit-accurate modeling · pytest · embedded C on ARM Cortex-A9
(NEON) · compressed sensing · OMP · Daubechies wavelets.

## References

1. J. A. Tropp and A. C. Gilbert, "Signal recovery from random measurements via orthogonal matching
   pursuit," *IEEE Trans. Inf. Theory*, vol. 53, no. 12, pp. 4655–4666, Dec. 2007.
2. H. Rabah, A. Amira, B. K. Mohanty, S. Almaadeed, and P. K. Meher, "FPGA implementation of orthogonal
   matching pursuit for compressive sensing reconstruction," *IEEE Trans. VLSI Syst.*, vol. 23, no. 10,
   pp. 2209–2220, Oct. 2015.
3. A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23,
   pp. e215–e220, Jun. 2000.
