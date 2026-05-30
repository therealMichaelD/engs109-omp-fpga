/*
 * omp_baseline.c  -  Cortex-A9 float OMP software baseline  (SW role)
 *
 * Purpose
 *   1. TIMED kernel: correlation_pass() is the only timed region. It mirrors
 *      one hardware COMPUTE pass (Interface Contract section 8): correlate the
 *      residual against all N_ATOMS atoms, return argmax of |inner product|.
 *      Per-pass software time / (HDL-A per-pass cycles * 10 ns) = speedup.
 *   2. UNTIMED scaffold: a full float OMP loop (support set, least-squares,
 *      residual update) so this baseline produces a reconstructed signal and
 *      PRD per window. PRD here should match HDL-B's *float* reference on the
 *      same data -- that agreement is the cross-check.
 *
 * Precision policy (state this on the poster):
 *   - correlation_pass: float32   <- the thing compared to the FPGA
 *   - LS solve + residual: double  <- "high precision", the accuracy ceiling
 *   If HDL-B's reference uses float32 LS, change the double's below to float.
 *
 * Build (host, for correctness + PRD cross-reference):
 *   cc -O3 -o omp_baseline omp_baseline.c -lm
 * Build (Zynq bare-metal, for timing -- see note in main()):
 *   add -DZYNQ_BAREMETAL and link against the standalone BSP (xtime_l.h)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- frozen params: mirror omp_params.h / Contract section 2 ---- */
#define M_LEN     128   /* measurements per window = atom length          */
#define N_ATOMS   256   /* dictionary atoms = DCT coefficients to recover */
#define WINDOW_N  256   /* signal window length                           */
#define MAX_ITERS 32    /* sparsity cap (Contract: max OMP iterations)    */

/* ---- portable timer shim ----
 * Bare-metal Zynq uses XTime_GetTime (the AXI/global timer named in the
 * proposal). On a host PC we fall back to CLOCK_MONOTONIC so the SAME source
 * compiles in both places. Host timing is not meaningful -- it just lets you
 * validate PRD on your laptop before moving the timing run to the board.     */
#if defined(ZYNQ_BAREMETAL)
  #include "xtime_l.h"
  typedef XTime tick_t;
  static inline void get_ticks(tick_t *t) { XTime_GetTime(t); }
  #define TICKS_PER_SEC ((double)COUNTS_PER_SECOND)
#else
  #include <time.h>
  typedef unsigned long long tick_t;
  static inline void get_ticks(tick_t *t) {
      struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
      *t = (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
  }
  #define TICKS_PER_SEC (1.0e9)
#endif

/* ============================================================= *
 *  TIMED KERNEL  -  one correlation pass (one COMPUTE state)     *
 *  D is atom-major: D[j*M_LEN + i] is entry i of atom j.         *
 *  This is the ONLY region you wrap in the timer.               *
 * ============================================================= */
int correlation_pass(const float *D, const float *r, float *out_val)
{
    int   best_idx = 0;
    float best_abs = -1.0f, best_val = 0.0f;
    for (int j = 0; j < N_ATOMS; j++) {
        const float *atom = D + (size_t)j * M_LEN;
        float acc = 0.0f;
        for (int i = 0; i < M_LEN; i++)
            acc += atom[i] * r[i];
        float a = fabsf(acc);
        if (a > best_abs) {            /* strict '>' = lowest index wins (np.argmax) */
            best_abs = a;
            best_idx = j;
            best_val = acc;
        }
    }
    *out_val = best_val;
    return best_idx;
}

/* ---- SPD solve: C a = b, C is k x k row-major. Cholesky. ----
 * Returns 0 on success, -1 if C is not positive definite (atom set went
 * rank-deficient -- we stop OMP there).                                  */
static int cholesky_solve(const double *C, const double *b, double *a, int k)
{
    double L[MAX_ITERS * MAX_ITERS];
    for (int i = 0; i < k; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = C[i * k + j];
            for (int p = 0; p < j; p++) sum -= L[i * k + p] * L[j * k + p];
            if (i == j) {
                if (sum <= 0.0) return -1;       /* not positive definite */
                L[i * k + j] = sqrt(sum);
            } else {
                L[i * k + j] = sum / L[j * k + j];
            }
        }
    }
    double z[MAX_ITERS];
    for (int i = 0; i < k; i++) {                 /* forward: L z = b */
        double sum = b[i];
        for (int p = 0; p < i; p++) sum -= L[i * k + p] * z[p];
        z[i] = sum / L[i * k + i];
    }
    for (int i = k - 1; i >= 0; i--) {            /* back: L^T a = z */
        double sum = z[i];
        for (int p = i + 1; p < k; p++) sum -= L[p * k + i] * a[p];
        a[i] = sum / L[i * k + i];
    }
    return 0;
}

/* ---- full float OMP for one window (untimed scaffold) ----
 * y          : measurement vector, length M_LEN
 * D          : atom-major dictionary, M_LEN x N_ATOMS
 * coeffs_out : length N_ATOMS, zeroed except on the support set
 * support_out: length MAX_ITERS, the selected atom indices in order (or NULL)
 * acc_ticks/acc_calls: correlation-pass timing accumulators
 * returns    : support size (number of iterations actually run)
 *
 * Note: this re-solves the LS from scratch each iteration via Cholesky on the
 * Gram matrix. A rank-1 update would be faster, but LS is NOT timed, so a
 * clean correct re-solve is the right call for a baseline.                    */
int omp_recover(const float *D, const float *y,
                float *coeffs_out, int *support_out, int max_iters,
                tick_t *acc_ticks, long *acc_calls)
{
    float r[M_LEN];
    memcpy(r, y, sizeof(float) * M_LEN);          /* r_0 = y */
    memset(coeffs_out, 0, sizeof(float) * N_ATOMS);

    int support[MAX_ITERS];
    int nsup = 0;

    for (int it = 0; it < max_iters; it++) {
        /* -------- TIMED: one correlation pass -------- */
        float wval;
        tick_t t0, t1;
        get_ticks(&t0);
        int lambda = correlation_pass(D, r, &wval);
        get_ticks(&t1);
        *acc_ticks += (t1 - t0);
        (*acc_calls)++;
        /* --------------------------------------------- */

        int dup = 0;                              /* refuse to re-pick an atom */
        for (int s = 0; s < nsup; s++) if (support[s] == lambda) { dup = 1; break; }
        if (dup) break;
        support[nsup++] = lambda;

        /* C = D_S^T D_S  (nsup x nsup),  b = D_S^T y */
        double C[MAX_ITERS * MAX_ITERS], b[MAX_ITERS], a[MAX_ITERS];
        for (int p = 0; p < nsup; p++) {
            const float *ap = D + (size_t)support[p] * M_LEN;
            double bp = 0.0;
            for (int i = 0; i < M_LEN; i++) bp += (double)ap[i] * (double)y[i];
            b[p] = bp;
            for (int q = 0; q <= p; q++) {
                const float *aq = D + (size_t)support[q] * M_LEN;
                double c = 0.0;
                for (int i = 0; i < M_LEN; i++) c += (double)ap[i] * (double)aq[i];
                C[p * nsup + q] = c;
                C[q * nsup + p] = c;
            }
        }
        if (cholesky_solve(C, b, a, nsup) != 0) break;   /* ill-conditioned: stop */

        /* residual r = y - D_S a */
        for (int i = 0; i < M_LEN; i++) {
            double s = 0.0;
            for (int p = 0; p < nsup; p++)
                s += (double)D[(size_t)support[p] * M_LEN + i] * a[p];
            r[i] = (float)((double)y[i] - s);
        }
        for (int p = 0; p < nsup; p++) coeffs_out[support[p]] = (float)a[p];
    }
    if (support_out) memcpy(support_out, support, sizeof(int) * nsup);
    return nsup;
}

/* ---- reconstruct signal: xhat = Psi * (coeffs / col_norms) ----
 * Psi is row-major WINDOW_N x N_ATOMS (db4 synthesis basis: x = Psi a).
 *
 * The dictionary atoms are UNIT-NORMALIZED (D_float = (Phi@Psi) / col_norms),
 * so the LS coefficients are in normalized-atom space. To get the true DCT/
 * wavelet coefficients we must divide each coeff by its atom's col_norm before
 * applying Psi -- exactly as the Python reference does:
 *     a_hat[support] = coeffs / col_norms[support]
 * Skipping this divide silently mis-scales every recovered coefficient and
 * corrupts PRD. col_norms has length N_ATOMS; entries off the support set
 * multiply zero coeffs, so dividing the whole vector is safe.                 */
void reconstruct(const float *Psi, const float *coeffs,
                 const float *col_norms, float *xhat)
{
    for (int n = 0; n < WINDOW_N; n++) {
        double s = 0.0;
        for (int j = 0; j < N_ATOMS; j++) {
            double a = (coeffs[j] != 0.0f)
                     ? (double)coeffs[j] / (double)col_norms[j]
                     : 0.0;
            s += (double)Psi[(size_t)n * N_ATOMS + j] * a;
        }
        xhat[n] = (float)s;
    }
}

/* ---- PRD = 100 * ||x - xhat|| / ||x|| ----
 * WARNING: some references subtract the signal mean first
 * (||x - xhat|| / ||x - mean(x)||). Use whatever HDL-B uses, exactly, or the
 * two PRD curves will not line up on the poster.                            */
double compute_prd(const float *x, const float *xhat)
{
    double num = 0.0, den = 0.0;
    for (int n = 0; n < WINDOW_N; n++) {
        double d = (double)x[n] - (double)xhat[n];
        num += d * d;
        den += (double)x[n] * (double)x[n];
    }
    return 100.0 * sqrt(num / den);
}

/* ---- load a raw float32 binary file (numpy: arr.astype('<f4').tofile) ---- */
static float *load_f32(const char *path, size_t count)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    float *buf = (float *)malloc(count * sizeof(float));
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    if (fread(buf, sizeof(float), count, f) != count) {
        fprintf(stderr, "short read on %s (expected %zu floats)\n", path, count);
        exit(1);
    }
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    /* HOST build: load everything from files exported by the Python pipeline.
     * Basis is db4 wavelet (pinned). Expected files (little-endian float32):
     *   D.bin        : M_LEN * N_ATOMS    (atom-major: atom j contiguous; = D_float.T)
     *   Psi.bin      : WINDOW_N * N_ATOMS (row-major db4 synthesis matrix)
     *   col_norms.bin: N_ATOMS            (per-atom norms of Phi@Psi, for un-scaling)
     *   X.bin        : num_windows * WINDOW_N  (ground-truth signal windows)
     *   Y.bin        : num_windows * M_LEN     (measurements  y = Phi x)
     *
     * ZYNQ BARE-METAL build: there is no filesystem, so fopen() won't work.
     * For the timing run you do NOT need all the windows or PRD -- correctness
     * was already validated here on the host. Replace the loads below with one
     * or two windows embedded as static const arrays, call omp_recover on them,
     * and report acc_ticks/acc_calls. That is the only number the board run
     * needs to produce.                                                         */
    int num_windows = (argc > 1) ? atoi(argv[1]) : 1;

    float *D   = load_f32("D.bin",   (size_t)M_LEN * N_ATOMS);
    float *Psi = load_f32("Psi.bin", (size_t)WINDOW_N * N_ATOMS);
    float *X   = load_f32("X.bin",   (size_t)num_windows * WINDOW_N);
    float *Y   = load_f32("Y.bin",   (size_t)num_windows * M_LEN);
    float *col_norms = load_f32("col_norms.bin", (size_t)N_ATOMS);

    float coeffs[N_ATOMS], xhat[WINDOW_N];
    int   support[MAX_ITERS];

    tick_t acc_ticks = 0;
    long   acc_calls = 0;
    double prd_sum = 0.0;

    FILE *out = fopen("baseline_results.csv", "w");
    fprintf(out, "window,nsup,prd,support...\n");

    for (int w = 0; w < num_windows; w++) {
        const float *y = Y + (size_t)w * M_LEN;
        const float *x = X + (size_t)w * WINDOW_N;

        int nsup = omp_recover(D, y, coeffs, support, MAX_ITERS,
                               &acc_ticks, &acc_calls);
        reconstruct(Psi, coeffs, col_norms, xhat);
        double prd = compute_prd(x, xhat);
        prd_sum += prd;

        fprintf(out, "%d,%d,%.6f", w, nsup, prd);
        for (int s = 0; s < nsup; s++) fprintf(out, ",%d", support[s]);
        fprintf(out, "\n");
    }
    fclose(out);

    double per_pass_us = (acc_calls > 0)
        ? ((double)acc_ticks / (double)acc_calls) / TICKS_PER_SEC * 1e6
        : 0.0;

    printf("windows           : %d\n", num_windows);
    printf("correlation passes: %ld\n", acc_calls);
    printf("mean PRD          : %.4f %%\n", prd_sum / num_windows);
    printf("mean per-pass time: %.4f us  <-- divide HDL-A cycles*10ns into this\n",
           per_pass_us);
    printf("per-window results written to baseline_results.csv\n");

    free(D); free(Psi); free(X); free(Y); free(col_norms);
    return 0;
}