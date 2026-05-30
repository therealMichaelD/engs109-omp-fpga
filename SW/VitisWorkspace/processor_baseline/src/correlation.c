#include <math.h>
#define M_LEN 128
#define N_ATOMS 256

int correlation_pass(const float *D, const float *r, float *out_val) {
    int   best_idx = 0;
    float best_abs = -1.0f, best_val = 0.0f;
    for (int j = 0; j < N_ATOMS; j++) {
        const float *atom = D + (size_t)j * M_LEN;
        float acc = 0.0f;
        for (int i = 0; i < M_LEN; i++)
            acc += atom[i] * r[i];
        float a = fabsf(acc);
        if (a > best_abs) { best_abs = a; best_idx = j; best_val = acc; }
    }
    *out_val = best_val;
    return best_idx;
}
