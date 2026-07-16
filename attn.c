/* Attention kernel for MHA and GQA using ARM BF16 (BFDOT / BFMMLA).
 *
 * One kernel handles both MHA and GQA: the only difference is the query-head ->
 * kv-head mapping  kv = h / group,  group = num_q_heads / num_kv_heads.
 *   - MHA:  num_kv_heads == num_q_heads  -> group == 1
 *   - GQA:  num_kv_heads <  num_q_heads  -> group  > 1
 *
 * Two compute paths, selected by use_mmla:
 *   - BFDOT  path: each score / output element is a bf16 dot-product reduction
 *                  (GEMV-friendly; works for any shape incl. decode Sq==1).
 *   - BFMMLA path: 2x2 output tiles (query-pair x key-pair for QK^T,
 *                  query-pair x dim-pair for P.V); tails handled by zero-padding.
 *
 * bf16 tensors are stored as uint16_t bit patterns (== top 16 bits of fp32) and
 * reinterpreted as bfloat16_t for the intrinsics, so the SAME bits feed both the
 * hardware kernel and the fp32 reference.
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* ---------- bf16 <-> f32 ---------- */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    if (((x >> 23) & 0xff) == 0xff) {
        return (uint16_t)(x >> 16);
    } /* inf/nan: pass through */
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb; /* round to nearest even */
    return (uint16_t)(x >> 16);
}
static inline float bf16_to_f32(uint16_t h) {
    uint32_t x = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

#define RUP(a, b) (((a) + (b) - 1) / (b) * (b)) /* round a up to multiple of b */

/* ================= fp32 reference (ground truth) ================= */
/* ref_rows>0 limits the reference to the first ref_rows query rows per head
 * (keeps the O(Sq*Sk*D) scalar reference affordable at long sequence lengths). */
static void attn_ref(const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O, int Hq,
                     int Hkv, int Sq, int Sk, int D, int causal, int ref_rows) {
    int group = Hq / Hkv;
    int Rq = (ref_rows > 0 && ref_rows < Sq) ? ref_rows : Sq;
    float scale = 1.0f / sqrtf((float)D);
    float *s = malloc(sizeof(float) * Sk);
    for (int h = 0; h < Hq; h++) {
        int kv = h / group;
        const uint16_t *Qh = Q + (size_t)h * Sq * D;
        const uint16_t *Kh = K + (size_t)kv * Sk * D;
        const uint16_t *Vh = V + (size_t)kv * Sk * D;
        for (int i = 0; i < Rq; i++) {
            int lim = causal ? (Sk - Sq + i) : (Sk - 1); /* last key this query may see */
            float mx = -INFINITY;
            for (int j = 0; j < Sk; j++) {
                if (j > lim) {
                    s[j] = -INFINITY;
                    continue;
                }
                float acc = 0.f;
                for (int d = 0; d < D; d++)
                    acc += bf16_to_f32(Qh[i * D + d]) * bf16_to_f32(Kh[j * D + d]);
                acc *= scale;
                s[j] = acc;
                if (acc > mx)
                    mx = acc;
            }
            float sum = 0.f;
            for (int j = 0; j < Sk; j++) {
                if (s[j] == -INFINITY) {
                    s[j] = 0.f;
                } else {
                    s[j] = expf(s[j] - mx);
                    sum += s[j];
                }
            }
            for (int j = 0; j < Sk; j++)
                s[j] /= sum;
            for (int d = 0; d < D; d++) {
                float acc = 0.f;
                for (int j = 0; j < Sk; j++)
                    acc += s[j] * bf16_to_f32(Vh[j * D + d]);
                O[((size_t)h * Sq + i) * D + d] = acc;
            }
        }
    }
    free(s);
}

/* ================= bf16 kernel (BFDOT or BFMMLA) ================= */
static void attn_bf16(const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O, int Hq,
                      int Hkv, int Sq, int Sk, int D, int causal, int use_mmla) {
    int group = Hq / Hkv;
    float scale = 1.0f / sqrtf((float)D);
    /* padded dims for SIMD tiles: Dp = head_dim->mult8, Skp = Sk->mult8, Sq2 = Sq->even */
    int Dp = RUP(D, 8), Skp = RUP(Sk, 8), Sq2 = RUP(Sq, 2);

    uint16_t *Qp = calloc((size_t)Sq2 * Dp, 2);  /* [Sq2][Dp] bf16, zero padded */
    uint16_t *Kp = calloc((size_t)Skp * Dp, 2);  /* [Skp][Dp] */
    uint16_t *Vt = calloc((size_t)Dp * Skp, 2);  /* [Dp][Skp] = V^T, zero padded */
    uint16_t *Pb = calloc((size_t)Sq2 * Skp, 2); /* [Sq2][Skp] softmax probs, bf16 */
    float *S = malloc(sizeof(float) * Sq2 * Skp);

    for (int h = 0; h < Hq; h++) {
        int kv = h / group;
        const uint16_t *Qh = Q + (size_t)h * Sq * D;
        const uint16_t *Kh = K + (size_t)kv * Sk * D;
        const uint16_t *Vh = V + (size_t)kv * Sk * D;

        /* pack padded Q, K, and V^T */
        memset(Qp, 0, (size_t)Sq2 * Dp * 2);
        memset(Kp, 0, (size_t)Skp * Dp * 2);
        memset(Vt, 0, (size_t)Dp * Skp * 2);
        for (int i = 0; i < Sq; i++)
            for (int d = 0; d < D; d++)
                Qp[i * Dp + d] = Qh[i * D + d];
        for (int j = 0; j < Sk; j++)
            for (int d = 0; d < D; d++)
                Kp[j * Dp + d] = Kh[j * D + d];
        for (int j = 0; j < Sk; j++)
            for (int d = 0; d < D; d++)
                Vt[d * Skp + j] = Vh[j * D + d];

        /* -------- S = scale * Q . K^T -------- */
        if (!use_mmla) {
            for (int i = 0; i < Sq; i++)
                for (int j = 0; j < Sk; j++) {
                    float32x4_t a = vdupq_n_f32(0.f);
                    for (int d = 0; d < Dp; d += 8) {
                        a = vbfdotq_f32(a, vld1q_bf16((const bfloat16_t *)(Qp + i * Dp + d)),
                                        vld1q_bf16((const bfloat16_t *)(Kp + j * Dp + d)));
                    }
                    S[i * Skp + j] = vaddvq_f32(a) * scale;
                }
        } else {
            for (int i = 0; i < Sq2; i += 2)
                for (int j = 0; j < Skp; j += 2) {
                    float32x4_t r = vdupq_n_f32(0.f);
                    for (int d = 0; d < Dp; d += 4) {
                        bfloat16x8_t a =
                            vcombine_bf16(vld1_bf16((const bfloat16_t *)(Qp + i * Dp + d)),
                                          vld1_bf16((const bfloat16_t *)(Qp + (i + 1) * Dp + d)));
                        bfloat16x8_t b =
                            vcombine_bf16(vld1_bf16((const bfloat16_t *)(Kp + j * Dp + d)),
                                          vld1_bf16((const bfloat16_t *)(Kp + (j + 1) * Dp + d)));
                        r = vbfmmlaq_f32(r, a, b);
                    }
                    float t[4];
                    vst1q_f32(t, r);
                    S[i * Skp + j] = t[0] * scale;             /* Qi   . Kj   */
                    S[i * Skp + (j + 1)] = t[1] * scale;       /* Qi   . Kj+1 */
                    S[(i + 1) * Skp + j] = t[2] * scale;       /* Qi+1 . Kj   */
                    S[(i + 1) * Skp + (j + 1)] = t[3] * scale; /* Qi+1 . Kj+1 */
                }
        }

        /* -------- mask + softmax (rows i<Sq, cols j<Sk) -------- */
        for (int i = 0; i < Sq; i++) {
            int lim = causal ? (Sk - Sq + i) : (Sk - 1);
            float mx = -INFINITY;
            for (int j = 0; j < Skp; j++) {
                if (j >= Sk || j > lim)
                    S[i * Skp + j] = -INFINITY;
                else if (S[i * Skp + j] > mx)
                    mx = S[i * Skp + j];
            }
            float sum = 0.f;
            for (int j = 0; j < Skp; j++) {
                float e = (S[i * Skp + j] == -INFINITY) ? 0.f : expf(S[i * Skp + j] - mx);
                S[i * Skp + j] = e;
                sum += e;
            }
            for (int j = 0; j < Skp; j++)
                Pb[i * Skp + j] = f32_to_bf16(S[i * Skp + j] / sum);
        }
        for (int i = Sq; i < Sq2; i++)
            for (int j = 0; j < Skp; j++)
                Pb[i * Skp + j] = 0;

        /* -------- O = P . V  (contract over keys) -------- */
        if (!use_mmla) {
            for (int i = 0; i < Sq; i++)
                for (int d = 0; d < D; d++) {
                    float32x4_t a = vdupq_n_f32(0.f);
                    for (int k = 0; k < Skp; k += 8) {
                        a = vbfdotq_f32(a, vld1q_bf16((const bfloat16_t *)(Pb + i * Skp + k)),
                                        vld1q_bf16((const bfloat16_t *)(Vt + d * Skp + k)));
                    }
                    O[((size_t)h * Sq + i) * D + d] = vaddvq_f32(a);
                }
        } else {
            for (int i = 0; i < Sq2; i += 2)
                for (int d = 0; d < Dp; d += 2) {
                    float32x4_t r = vdupq_n_f32(0.f);
                    for (int k = 0; k < Skp; k += 4) {
                        bfloat16x8_t a =
                            vcombine_bf16(vld1_bf16((const bfloat16_t *)(Pb + i * Skp + k)),
                                          vld1_bf16((const bfloat16_t *)(Pb + (i + 1) * Skp + k)));
                        bfloat16x8_t b =
                            vcombine_bf16(vld1_bf16((const bfloat16_t *)(Vt + d * Skp + k)),
                                          vld1_bf16((const bfloat16_t *)(Vt + (d + 1) * Skp + k)));
                        r = vbfmmlaq_f32(r, a, b);
                    }
                    float t[4];
                    vst1q_f32(t, r);
                    if (i < Sq) {
                        if (d < D)
                            O[((size_t)h * Sq + i) * D + d] = t[0];
                        if (d + 1 < D)
                            O[((size_t)h * Sq + i) * D + d + 1] = t[1];
                    }
                    if (i + 1 < Sq) {
                        if (d < D)
                            O[((size_t)h * Sq + i + 1) * D + d] = t[2];
                        if (d + 1 < D)
                            O[((size_t)h * Sq + i + 1) * D + d + 1] = t[3];
                    }
                }
        }
    }
    free(Qp);
    free(Kp);
    free(Vt);
    free(Pb);
    free(S);
}

/* ================= test harness ================= */
static uint32_t rng = 0x12345678u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xffff) / 32768.0f - 1.0f;
}
static double now_s(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

/* ref_cap>0 : only reference-check the first ref_cap query rows per head
 * (BFDOT vs BFMMLA is still cross-checked on ALL rows). */
static int run_case(const char *name, int Hq, int Hkv, int Sq, int Sk, int D, int causal,
                    int ref_cap) {
    rng = 0x12345678u;
    size_t nq = (size_t)Hq * Sq * D, nk = (size_t)Hkv * Sk * D, no = (size_t)Hq * Sq * D;
    uint16_t *Q = malloc(nq * 2), *K = malloc(nk * 2), *V = malloc(nk * 2);
    for (size_t i = 0; i < nq; i++)
        Q[i] = f32_to_bf16(frand());
    for (size_t i = 0; i < nk; i++)
        K[i] = f32_to_bf16(frand());
    for (size_t i = 0; i < nk; i++)
        V[i] = f32_to_bf16(frand());
    float *Oref = malloc(no * 4), *Odot = malloc(no * 4), *Omma = malloc(no * 4);

    int Rq = (ref_cap > 0 && ref_cap < Sq) ? ref_cap : Sq;
    attn_ref(Q, K, V, Oref, Hq, Hkv, Sq, Sk, D, causal, Rq);
    double t0 = now_s();
    attn_bf16(Q, K, V, Odot, Hq, Hkv, Sq, Sk, D, causal, 0);
    double tdot = now_s() - t0;
    t0 = now_s();
    attn_bf16(Q, K, V, Omma, Hq, Hkv, Sq, Sk, D, causal, 1);
    double tmma = now_s() - t0;

    /* kernel vs reference over the first Rq rows of each head */
    float mad_dot = 0, mad_mma = 0, maxref = 0;
    int ok = 1;
    float atol = 2e-2f, rtol = 2e-2f;
    for (int h = 0; h < Hq; h++)
        for (int i = 0; i < Rq; i++)
            for (int d = 0; d < D; d++) {
                size_t idx = ((size_t)h * Sq + i) * D + d;
                float r = Oref[idx];
                float ad = fabsf(Odot[idx] - r), am = fabsf(Omma[idx] - r);
                if (ad > mad_dot)
                    mad_dot = ad;
                if (am > mad_mma)
                    mad_mma = am;
                if (fabsf(r) > maxref)
                    maxref = fabsf(r);
                if (ad > atol + rtol * fabsf(r))
                    ok = 0;
                if (am > atol + rtol * fabsf(r))
                    ok = 0;
            }
    /* BFDOT vs BFMMLA on every output element */
    float dvsm = 0;
    for (size_t i = 0; i < no; i++) {
        float d = fabsf(Odot[i] - Omma[i]);
        if (d > dvsm)
            dvsm = d;
    }

    int group = Hq / Hkv;
    double gflop = 4.0 * (double)Hq * Sq * Sk * D / 1e9; /* 2 matmuls x 2 flop/madd */
    printf("%-16s H=%d/%-2d(g%d) Sq=%-4d Sk=%-4d D=%3d %s | err/scale dot=%.1e mma=%.1e "
           "d-vs-m=%.0e | %.2f GFLOP  mma %.2fs dot %.2fs%s -> %s\n",
           name, Hq, Hkv, group, Sq, Sk, D, causal ? "causal" : "full ",
           maxref > 0 ? mad_dot / maxref : 0.f, maxref > 0 ? mad_mma / maxref : 0.f, dvsm, gflop,
           tmma, tdot, (Rq < Sq) ? " [ref:subset]" : "", ok ? "PASS" : "*** FAIL ***");
    free(Q);
    free(K);
    free(V);
    free(Oref);
    free(Odot);
    free(Omma);
    return ok;
}

int main(int argc, char **argv) {
    int longrun = (argc > 1 && strcmp(argv[1], "--long") == 0);
    int all = 1;
    printf("=== bf16 attention kernel verification (BFDOT vs BFMMLA vs fp32 ref) ===\n");
    /* --- small functional suite (MHA/GQA/MQA, prefill/decode, causal, tails) --- */
    all &= run_case("MHA prefill", 4, 4, 8, 8, 64, 0, 0);
    all &= run_case("MHA prefil.causal", 4, 4, 8, 8, 64, 1, 0);
    all &= run_case("GQA prefill", 8, 2, 8, 8, 64, 0, 0);
    all &= run_case("GQA prefil.causal", 8, 2, 8, 8, 64, 1, 0);
    all &= run_case("MHA decode(Sq=1)", 4, 4, 1, 16, 64, 0, 0);
    all &= run_case("GQA decode(Sq=1)", 8, 2, 1, 16, 64, 0, 0);
    all &= run_case("MQA prefill", 8, 1, 8, 8, 64, 0, 0);
    all &= run_case("GQA odd shapes", 6, 3, 5, 7, 40, 1, 0);
    all &= run_case("GQA big head_dim", 8, 2, 4, 12, 128, 0, 0);
    if (longrun) {
        printf("--- long-sequence prefill (target 1k/2k/4k, D=128, causal) ---\n");
        all &= run_case("prefill 1k MHA", 1, 1, 1024, 1024, 128, 1, 16);
        all &= run_case("prefill 1k GQA", 4, 1, 1024, 1024, 128, 1, 16);
        all &= run_case("prefill 2k MHA", 1, 1, 2048, 2048, 128, 1, 16);
        all &= run_case("prefill 4k MHA", 1, 1, 4096, 4096, 128, 1, 16);
    } else {
        printf("(pass --long to also run 1k/2k/4k prefill cases)\n");
    }
    printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
