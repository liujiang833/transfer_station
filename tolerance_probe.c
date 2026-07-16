/* tolerance_probe.c -- MEASUREMENT harness behind flash.c's TOL=8e-3 (EXEC_LOG F10/F11).
 * NOT part of the shipped kernel; a frozen 2026-07-16 snapshot of flash.c's compute +
 * reference with a probe harness bolted on. It MEASURES err/scale and never judges:
 * for each shape it prints the same raw error under three candidate normalisations --
 *   case = mad / max|ref| over the whole case   (what flash.c ships)
 *   head = max_h  mad_h / maxref_h
 *   row  = max_ih mad_ih / maxref_ih
 * -- plus maxref and the count of rows the fp32 ref returned NaN for (see F9: causal
 * rows with zero valid keys divide 0/0, and NaN silently escapes every comparison).
 *
 * Findings it produced (129 shapes, VL=512 and VL=2048, identical to 4 digits):
 *   case-norm worst 2.76e-03 (D=8 non-causal)  <- the number TOL is set from
 *   head-norm worst 5.60e-03 (D=1 non-causal, degenerate maxref=0.116)
 *   row-norm  worst 1.159    (D=1 causal, single output element cancels to ~0)
 * i.e. per-row and per-head normalisation are degenerate at small D; per-case is not.
 *
 * Build & run (short sweep ~70s, --long qwen3 spread-row sweep ~34min):
 *   aarch64-linux-gnu-gcc -O2 -static -march=armv8.6-a+sve+bf16 -Wall -Wextra \
 *       tolerance_probe.c -lm -o /tmp/tolerance_probe
 *   ./qemu_pkg/extracted/usr/bin/qemu-aarch64-static -cpu max /tmp/tolerance_probe
 *   ./qemu_pkg/extracted/usr/bin/qemu-aarch64-static \
 *       -cpu max,sve2048=on,sve-default-vector-length=-1 /tmp/tolerance_probe
 */
/* Flash-attention-style bf16 kernel (MHA + GQA + MQA), SVE-only, BFDOT compute.
 *
 * Blocked over keys with ONLINE softmax: the [Sq x Sk] score matrix is never
 * materialised. Per query-block we keep only running max m[Bq], running
 * denominator l[Bq], and running output acc[Bq x D]; each key-block updates them
 * with the standard rescale  acc <- alpha*acc + P@V ,  l <- alpha*l + rowsum(P) ,
 * where alpha = exp(m_old - m_new). Memory is O(Bq*D + Bq*Bk), not O(Sq*Sk).
 * Key-blocks wholly in the causal future are skipped (and end the k-loop).
 *
 * SVE ONLY (arm_sve.h): no NEON, no SME/ZA. svbfdot_f32 is the single compute
 * primitive for BOTH QK^T and P.V. VL-agnostic: every loop is driven by
 * svcntw()/svcnth()/svwhilelt at runtime; nothing is hardcoded to a vector width.
 *
 * PACKING: Q and K are NOT packed, V is. QK^T contracts over head_dim, which is
 * contiguous in both Q rows and K rows, so svld1_bf16 feeds BFDOT straight from
 * the tensors -- and a D-tail costs nothing, since a zeroing predicated load
 * makes the padding lanes contribute 0 to the dot. P.V contracts over KEYS:
 * contiguous in P but stride-D in V, so V alone is transposed to Vt[d][k].
 * The price of an unpacked K is that every score ends in one svaddv horizontal
 * reduction; UNR independent accumulators per pass overlap that latency.
 *
 * The softmax is SVE too -- fast_exp (svexpa-based exp2, ported from kutacc
 * src/math/fast_exp.h) -- so there is no expf() and no scalar j-loop anywhere in
 * the kernel. Scalar code lives only in the fp32 reference and the harness.
 *
 * *** GCC 13.3 MISCOMPILES C float->bfloat16_t CASTS: it applies IEEE fp16
 * semantics, so (bfloat16_t)1.0f yields 0x3c00 (fp16 1.0) where bf16 1.0 is
 * 0x3f80. Confirmed at -O0 and -O2. It hides well because fp16 and bf16 share
 * the encoding 0x4000 for 2.0. Hence: bf16 tensors are stored as uint16_t and
 * reinterpreted to bfloat16_t* for intrinsics, and f32->bf16 goes through the
 * f32_to_bf16() bit helper (scalar, RNE) or svcvt_bf16_x + svuzp1_bf16 (vector,
 * hot path). NEVER use a C float->bf16 cast here. ***
 *
 * Verified against a scalar fp32 reference that consumes the same bf16 bits.
 */
#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static inline uint16_t f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    if (((x >> 23) & 0xff) == 0xff)
        return (uint16_t)(x >> 16);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;
    return (uint16_t)(x >> 16);
}
static inline float bf16_to_f32(uint16_t h) {
    uint32_t x = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &x, 4);
    return f;
}
#define BQ 64  /* query block */
#define BK 64  /* key block   */
#define UNR 4  /* independent BFDOT accumulators per pass (hides svaddv latency) */
#define BF(p) ((const bfloat16_t *)(p)) /* uint16_t bf16 storage -> intrinsic type */
#define BFW(p) ((bfloat16_t *)(p))

/* ---------------- fast_exp: SVE exp, ported from kutacc/src/math/fast_exp.h ----------------
 * svexpa-based exp2: add a magic bias so the integer part of z0 lands in the
 * mantissa (z4), recover the fraction (z1), evaluate a degree-2 poly on it, and
 * let svexpa build 2^int from z4's low bits. Max rel err ~2e-6 over [-40, 0],
 * which is the only range flash-attention exercises (scores minus the row max).
 */
static inline svfloat32_t fast_exp2_sve(svbool_t pg, svfloat32_t z0) {
    svfloat32_t z1 = svreinterpret_f32_u32(svdup_n_u32(1212161984u)); /* magic bias   */
    svfloat32_t z2 = svreinterpret_f32_u32(svdup_n_u32(1060205250u)); /* 0.693157315  */
    svfloat32_t z3 = svreinterpret_f32_u32(svdup_n_u32(1047920148u)); /* 0.240227044  */
    svfloat32_t z5 = svreinterpret_f32_u32(svdup_n_u32(1123811328u)); /* 127.0 (ovf)  */
    svfloat32_t z4 = svadd_f32_x(pg, z1, z0);
    z1 = svsub_f32_x(pg, z4, z1);
    z1 = svsub_f32_x(pg, z0, z1);
    z2 = svmla_f32_x(pg, z2, z1, z3);
    z3 = svreinterpret_f32_u32(svdup_n_u32(1065353216u)); /* 1.0 */
    z1 = svmad_f32_x(pg, z1, z2, z3);
    z2 = svexpa_f32(svreinterpret_u32_f32(z4));
    z1 = svmul_f32_x(pg, z2, z1);
    svbool_t poverflow = svacge_f32(pg, z0, z5);
    if (__builtin_expect(svptest_any(pg, poverflow), 0)) {
        svbool_t pgt = svcmpgt_f32(pg, z0, z5);
        z1 = svsel_f32(pgt, svdup_n_f32(INFINITY), z1);
        svbool_t plt = svcmplt_f32(pg, z0, svneg_f32_x(pg, z5));
        z1 = svsel_f32(plt, svdup_n_f32(0.0f), z1);
    }
    return z1;
}
static inline svfloat32_t fast_exp_sve(svbool_t pg, svfloat32_t z0) {
    return fast_exp2_sve(pg, svmul_f32_x(pg, z0, svdup_n_f32(1.442695041f)));
}
/* Scalar-valued exp, still SVE: broadcast, exp, read lane 0 (svlasta with an
 * all-false predicate returns element 0). Used once per row per key-block for
 * alpha, so it is off the hot path -- but it keeps expf() out of the kernel. */
static inline float fast_exp_lane0(float x) {
    return svlasta_f32(svpfalse_b(), fast_exp_sve(svptrue_b32(), svdup_n_f32(x)));
}

/* ---------------- fp32 reference (scalar by design) ---------------- */
static void attn_ref(const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O, int Hq,
                     int Hkv, int Sq, int Sk, int D, int causal, const int *rows, int nrows) {
    int group = Hq / Hkv;
    float scale = 1.0f / sqrtf((float)D);
    float *s = malloc(sizeof(float) * Sk);
    for (int h = 0; h < Hq; h++) {
        int kv = h / group;
        const uint16_t *Qh = Q + (size_t)h * Sq * D, *Kh = K + (size_t)kv * Sk * D,
                       *Vh = V + (size_t)kv * Sk * D;
        for (int t = 0; t < nrows; t++) {
            int i = rows[t];
            int lim = causal ? (Sk - Sq + i) : (Sk - 1);
            float mx = -INFINITY;
            for (int j = 0; j < Sk; j++) {
                if (j > lim) {
                    s[j] = -INFINITY;
                    continue;
                }
                float a = 0;
                for (int d = 0; d < D; d++)
                    a += bf16_to_f32(Qh[i * D + d]) * bf16_to_f32(Kh[j * D + d]);
                a *= scale;
                s[j] = a;
                if (a > mx)
                    mx = a;
            }
            float sum = 0;
            for (int j = 0; j < Sk; j++) {
                if (s[j] == -INFINITY)
                    s[j] = 0;
                else {
                    s[j] = expf(s[j] - mx);
                    sum += s[j];
                }
            }
            for (int j = 0; j < Sk; j++)
                s[j] /= sum;
            for (int d = 0; d < D; d++) {
                float a = 0;
                for (int j = 0; j < Sk; j++)
                    a += s[j] * bf16_to_f32(Vh[j * D + d]);
                O[((size_t)h * Sq + i) * D + d] = a;
            }
        }
    }
    free(s);
}

/* ---------------- SVE BFDOT tiles ----------------
 * All three take REAL (unpadded) extents; SVE predication handles every tail, so
 * no tile is zero-padded and no Q/K pack buffer exists.
 */

/* V^T pack: Vt[d][k] = Vb[k][d]. The only transpose in the kernel -- P.V needs V
 * key-major. Vectorised with a 16-bit gather (index k*D, in bf16 elements) and a
 * truncating 32->16 store; cost is O(rk*D) per key-block, amortised over rq rows. */
static void pack_vt(const uint16_t *Vb, uint16_t *Vt, int rk, int D, int Vstride) {
    int VLw = (int)svcntw();
    for (int d = 0; d < D; d++)
        for (int k = 0; k < rk; k += VLw) {
            svbool_t pg = svwhilelt_b32_s32(k, rk);
            svuint32_t g = svld1uh_gather_s32index_u32(pg, Vb + d, svindex_s32(k * D, D));
            svst1h_u32(pg, Vt + (size_t)d * Vstride + k, g);
        }
}

/* QK^T: S[i][j] = scale * dot(Qb[i][:], Kb[j][:]) over D. Both operands are read
 * straight from the tensors (contiguous in d); the whilelt_b16 load zeroes the
 * D-tail so BFDOT's pairwise lanes contribute 0 there. UNR keys per pass. */
static void qk_tile(const uint16_t *Qb, const uint16_t *Kb, float *S, int rq, int rk, int D,
                    int Sstride, float scale) {
    int VLh = (int)svcnth();
    svbool_t pg32 = svptrue_b32();
    for (int i = 0; i < rq; i++) {
        const uint16_t *qr = Qb + (size_t)i * D;
        int j = 0;
        for (; j + UNR <= rk; j += UNR) {
            const uint16_t *k0 = Kb + (size_t)j * D;
            svfloat32_t a0 = svdup_n_f32(0), a1 = svdup_n_f32(0), a2 = svdup_n_f32(0),
                        a3 = svdup_n_f32(0);
            for (int d = 0; d < D; d += VLh) {
                svbool_t pg = svwhilelt_b16_s32(d, D);
                svbfloat16_t qv = svld1_bf16(pg, BF(qr + d));
                a0 = svbfdot_f32(a0, qv, svld1_bf16(pg, BF(k0 + d)));
                a1 = svbfdot_f32(a1, qv, svld1_bf16(pg, BF(k0 + D + d)));
                a2 = svbfdot_f32(a2, qv, svld1_bf16(pg, BF(k0 + 2 * D + d)));
                a3 = svbfdot_f32(a3, qv, svld1_bf16(pg, BF(k0 + 3 * D + d)));
            }
            float *Sr = S + (size_t)i * Sstride + j;
            Sr[0] = svaddv_f32(pg32, a0) * scale;
            Sr[1] = svaddv_f32(pg32, a1) * scale;
            Sr[2] = svaddv_f32(pg32, a2) * scale;
            Sr[3] = svaddv_f32(pg32, a3) * scale;
        }
        for (; j < rk; j++) { /* key tail: same dot, one accumulator */
            svfloat32_t a0 = svdup_n_f32(0);
            for (int d = 0; d < D; d += VLh) {
                svbool_t pg = svwhilelt_b16_s32(d, D);
                a0 = svbfdot_f32(a0, svld1_bf16(pg, BF(qr + d)),
                                 svld1_bf16(pg, BF(Kb + (size_t)j * D + d)));
            }
            S[(size_t)i * Sstride + j] = svaddv_f32(pg32, a0) * scale;
        }
    }
}

/* P.V + online rescale, fused: acc[i][d] = al[i]*acc[i][d] + sum_k P[i][k]*Vt[d][k].
 * Contraction is over k, contiguous in both P and Vt. UNR dims d per pass. */
static void pv_acc(const uint16_t *Pb, const uint16_t *Vt, float *acc, const float *al, float *pv,
                   int rq, int rk, int D, int Pstride, int Vstride) {
    int VLh = (int)svcnth(), VLw = (int)svcntw();
    svbool_t pg32 = svptrue_b32();
    for (int i = 0; i < rq; i++) {
        const uint16_t *pr = Pb + (size_t)i * Pstride;
        int d = 0;
        for (; d + UNR <= D; d += UNR) {
            const uint16_t *v0 = Vt + (size_t)d * Vstride;
            svfloat32_t a0 = svdup_n_f32(0), a1 = svdup_n_f32(0), a2 = svdup_n_f32(0),
                        a3 = svdup_n_f32(0);
            for (int k = 0; k < rk; k += VLh) {
                svbool_t pg = svwhilelt_b16_s32(k, rk);
                svbfloat16_t pvec = svld1_bf16(pg, BF(pr + k));
                a0 = svbfdot_f32(a0, pvec, svld1_bf16(pg, BF(v0 + k)));
                a1 = svbfdot_f32(a1, pvec, svld1_bf16(pg, BF(v0 + Vstride + k)));
                a2 = svbfdot_f32(a2, pvec, svld1_bf16(pg, BF(v0 + 2 * Vstride + k)));
                a3 = svbfdot_f32(a3, pvec, svld1_bf16(pg, BF(v0 + 3 * Vstride + k)));
            }
            pv[d] = svaddv_f32(pg32, a0);
            pv[d + 1] = svaddv_f32(pg32, a1);
            pv[d + 2] = svaddv_f32(pg32, a2);
            pv[d + 3] = svaddv_f32(pg32, a3);
        }
        for (; d < D; d++) { /* dim tail */
            svfloat32_t a0 = svdup_n_f32(0);
            for (int k = 0; k < rk; k += VLh) {
                svbool_t pg = svwhilelt_b16_s32(k, rk);
                a0 = svbfdot_f32(a0, svld1_bf16(pg, BF(pr + k)),
                                 svld1_bf16(pg, BF(Vt + (size_t)d * Vstride + k)));
            }
            pv[d] = svaddv_f32(pg32, a0);
        }
        /* acc <- alpha*acc + pv, one FMA per lane */
        svfloat32_t av = svdup_n_f32(al[i]);
        float *ar = acc + (size_t)i * D;
        for (int e = 0; e < D; e += VLw) {
            svbool_t pg = svwhilelt_b32_s32(e, D);
            svst1_f32(pg, ar + e,
                      svmla_f32_x(pg, svld1_f32(pg, pv + e), svld1_f32(pg, ar + e), av));
        }
    }
}

/* Mask + online-softmax update for one key-block; writes P (bf16) and al[]. */
static void softmax_block(float *S, uint16_t *Pb, float *m, float *l, float *al, int rq, int rk,
                          int Sstride, int Pstride, int kj, int qbase, int causal) {
    int VLw = (int)svcntw(), VLh = (int)svcnth();
    svbool_t pg32 = svptrue_b32();
    svfloat32_t zero = svdup_n_f32(0);
    for (int i = 0; i < rq; i++) {
        /* Both mask conditions (j >= rk, and key kj+j beyond row i's causal limit)
         * are prefixes of j, so the whole mask is one whilelt bound: j < jmax. */
        int jmax = rk;
        if (causal) {
            int lim = qbase + i - kj; /* last LOCAL key index row i may attend */
            jmax = (lim + 1 < rk) ? lim + 1 : rk;
            if (jmax < 0)
                jmax = 0;
        }
        /* pass 1: this block's row max. Merging predication leaves masked lanes
         * at -inf, so no explicit -inf fill is needed. */
        svfloat32_t mv = svdup_n_f32(-INFINITY);
        for (int j = 0; j < jmax; j += VLw) {
            svbool_t pg = svwhilelt_b32_s32(j, jmax);
            mv = svmax_f32_m(pg, mv, svld1_f32(pg, S + (size_t)i * Sstride + j));
        }
        float mij = svmaxv_f32(pg32, mv);
        if (mij == -INFINITY) { /* row entirely masked in this block: P=0, state held */
            for (int j = 0; j < rk; j += VLh)
                svst1_u16(svwhilelt_b16_s32(j, rk), Pb + (size_t)i * Pstride + j, svdup_n_u16(0));
            al[i] = 1.f;
            continue;
        }
        float mnew = (m[i] > mij) ? m[i] : mij;
        float alpha = (m[i] == -INFINITY) ? 0.f : fast_exp_lane0(m[i] - mnew);
        /* pass 2: P = exp(S - mnew) (0 where masked), rowsum, and f32->bf16.
         * Two f32 vectors -> one contiguous bf16 vector via svcvt_bf16 + svuzp1. */
        svfloat32_t mnv = svdup_n_f32(mnew), s0 = zero, s1 = zero;
        for (int j = 0; j < rk; j += VLh) {
            svbool_t p0 = svwhilelt_b32_s32(j, jmax), p1 = svwhilelt_b32_s32(j + VLw, jmax);
            const float *Sr = S + (size_t)i * Sstride + j;
            svfloat32_t e0 = svsel_f32(
                p0, fast_exp_sve(p0, svsub_f32_x(p0, svld1_f32(p0, Sr), mnv)), zero);
            svfloat32_t e1 = svsel_f32(
                p1, fast_exp_sve(p1, svsub_f32_x(p1, svld1_f32(p1, Sr + VLw), mnv)), zero);
            s0 = svadd_f32_x(pg32, s0, e0);
            s1 = svadd_f32_x(pg32, s1, e1);
            svst1_u16(svwhilelt_b16_s32(j, rk), Pb + (size_t)i * Pstride + j,
                      svreinterpret_u16_bf16(svuzp1_bf16(svcvt_bf16_f32_x(pg32, e0),
                                                         svcvt_bf16_f32_x(pg32, e1))));
        }
        l[i] = alpha * l[i] + svaddv_f32(pg32, svadd_f32_x(pg32, s0, s1));
        m[i] = mnew;
        al[i] = alpha;
    }
}

/* ---------------- flash attention forward ---------------- */
static void attn_flash(const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O, int Hq,
                       int Hkv, int Sq, int Sk, int D, int causal) {
    /* Naming legend:
     *   Hq/Hkv     - number of query / key-value heads; group = Hq/Hkv (q-heads per kv-head)
     *   Sq/Sk      - query / key sequence lengths;  D = head_dim
     *   h,kv       - current query-head and its kv-head (kv = h/group)
     *   qi,kj      - start row/col of the current query-block / key-block
     *   rq,rk      - REAL rows/cols in the current block (<= BQ/BK; shrink at the tail)
     *   off,qbase  - causal offset Sk-Sq; qbase = off+qi (block row i attends keys <= qbase+i)
     *   jmax       - per-row count of unmasked keys in this block (mask == one whilelt bound)
     *   VLw,VLh    - RUNTIME lanes per SVE vector: svcntw() f32 / svcnth() bf16. Never assumed.
     * Per-query-block online-softmax state, one entry per query row i:
     *   m[i]       - running row max of scores seen so far
     *   l[i]       - running softmax denominator (running sum of exp)
     *   acc[i*D+.] - running UNnormalised output (running sum of P*V)
     *   al[i]      - this block's rescale factor alpha = exp(m_old - m_new)
     * Scratch tiles (reused every block); "*2" is bytes-per-bf16:
     *   Vt         - packed V^T block (feature-major), row stride Vstride=BK. Only pack.
     *   S          - score tile [BQ x Sstride] fp32
     *   Pb         - softmax probs P for this block (bf16), row stride Pstride
     *   pv         - one row's P*V partial [D] fp32, folded into acc immediately
     * Q and K are read in place -- no pack buffers (see header).
     */
    int group = Hq / Hkv; /* query heads sharing one kv head */
    float scale = 1.0f / sqrtf((float)D);
    int Sstride = BK, Pstride = BK, Vstride = BK; /* real extents + predication: no padding */
    uint16_t *Vt = malloc((size_t)D * Vstride * 2), *Pb = malloc((size_t)BQ * Pstride * 2);
    float *S = malloc(sizeof(float) * BQ * Sstride), *pv = malloc(sizeof(float) * D);
    float *acc = malloc(sizeof(float) * BQ * D), *m = malloc(sizeof(float) * BQ);
    float *l = malloc(sizeof(float) * BQ), *al = malloc(sizeof(float) * BQ);

    for (int h = 0; h < Hq; h++) {
        int kv = h / group; /* kv head feeding this query head */
        const uint16_t *Qh = Q + (size_t)h * Sq * D, *Kh = K + (size_t)kv * Sk * D,
                       *Vh = V + (size_t)kv * Sk * D; /* per-head base pointers */
        int off = Sk - Sq;                            /* causal offset */
        for (int qi = 0; qi < Sq; qi += BQ) {         /* qi = query-block start */
            int rq = Sq - qi;                         /* rq = real query rows in this block */
            if (rq > BQ)
                rq = BQ;
            const uint16_t *Qb = Qh + (size_t)qi * D;
            for (int i = 0; i < rq; i++) {
                m[i] = -INFINITY;
                l[i] = 0;
            }
            memset(acc, 0, sizeof(float) * rq * D);

            for (int kj = 0; kj < Sk; kj += BK) { /* kj = key-block start */
                if (causal && kj > off + qi + rq - 1)
                    break;        /* whole block is in the future -> skip it and the rest */
                int rk = Sk - kj; /* rk = real keys in this block */
                if (rk > BK)
                    rk = BK;
                pack_vt(Vh + (size_t)kj * D, Vt, rk, D, Vstride);
                qk_tile(Qb, Kh + (size_t)kj * D, S, rq, rk, D, Sstride, scale);
                softmax_block(S, Pb, m, l, al, rq, rk, Sstride, Pstride, kj, off + qi, causal);
                pv_acc(Pb, Vt, acc, al, pv, rq, rk, D, Pstride, Vstride);
            }
            int VLw = (int)svcntw();
            for (int i = 0; i < rq; i++) { /* O = acc / l (l==0 only if the row saw no key) */
                svfloat32_t iv = svdup_n_f32((l[i] > 0) ? 1.f / l[i] : 0.f);
                const float *ar = acc + (size_t)i * D;
                float *Or = O + ((size_t)h * Sq + qi + i) * D;
                for (int d = 0; d < D; d += VLw) {
                    svbool_t pg = svwhilelt_b32_s32(d, D);
                    svst1_f32(pg, Or + d, svmul_f32_x(pg, svld1_f32(pg, ar + d), iv));
                }
            }
        }
    }
    free(Vt);
    free(Pb);
    free(S);
    free(pv);
    free(acc);
    free(m);
    free(l);
    free(al);
}


/* ---------------- PROBE harness: measure err/scale, do NOT judge ---------------- */
static uint32_t rng;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xffff) / 32768.0f - 1.0f;
}
static int ref_rowset(int Sq, int cap, int *rows) {
    if (cap <= 0 || cap >= Sq) {
        for (int i = 0; i < Sq; i++) rows[i] = i;
        return Sq;
    }
    if (cap == 1) { rows[0] = Sq - 1; return 1; }
    for (int t = 0; t < cap; t++)
        rows[t] = (int)(((int64_t)t * (Sq - 1)) / (cap - 1));
    return cap;
}

/* worst-case trackers, global across the whole sweep */
static double W_case = 0, W_head = 0, W_row = 0;
static char W_case_name[128], W_head_name[128], W_row_name[128];

static void probe(const char *name, int Hq, int Hkv, int Sq, int Sk, int D, int causal,
                  int ref_cap) {
    rng = 0x12345678u;
    size_t nq = (size_t)Hq * Sq * D, nk = (size_t)Hkv * Sk * D, no = (size_t)Hq * Sq * D;
    uint16_t *Q = malloc(nq * 2), *K = malloc(nk * 2), *V = malloc(nk * 2);
    for (size_t i = 0; i < nq; i++) Q[i] = f32_to_bf16(frand());
    for (size_t i = 0; i < nk; i++) K[i] = f32_to_bf16(frand());
    for (size_t i = 0; i < nk; i++) V[i] = f32_to_bf16(frand());
    float *Oref = calloc(no, 4), *Odot = calloc(no, 4);
    int *rows = malloc(sizeof(int) * Sq);
    int nrows = ref_rowset(Sq, ref_cap, rows);
    attn_ref(Q, K, V, Oref, Hq, Hkv, Sq, Sk, D, causal, rows, nrows);
    attn_flash(Q, K, V, Odot, Hq, Hkv, Sq, Sk, D, causal);

    /* three candidate normalizations of the same raw error */
    double g_mad = 0, g_maxref = 0;   /* per-CASE  : mad / max|ref| over the case */
    double worst_head = 0;            /* per-HEAD  : max_h  mad_h / maxref_h      */
    double worst_row = 0;             /* per-ROW   : max_ih mad_ih / maxref_ih    */
    long nan_rows = 0, live_rows = 0;
    for (int h = 0; h < Hq; h++) {
        double h_mad = 0, h_maxref = 0;
        for (int t = 0; t < nrows; t++) {
            int i = rows[t];
            double r_mad = 0, r_maxref = 0;
            int isnan_row = 0;
            for (int d = 0; d < D; d++) {
                size_t idx = ((size_t)h * Sq + i) * D + d;
                float r = Oref[idx], ad = fabsf(Odot[idx] - r);
                if (isnan(r) || isnan(Odot[idx])) { isnan_row = 1; continue; }
                if (ad > r_mad) r_mad = ad;
                if (fabsf(r) > r_maxref) r_maxref = fabsf(r);
            }
            if (isnan_row) { nan_rows++; continue; }
            live_rows++;
            if (r_mad > h_mad) h_mad = r_mad;
            if (r_maxref > h_maxref) h_maxref = r_maxref;
            if (r_maxref > 0 && r_mad / r_maxref > worst_row) worst_row = r_mad / r_maxref;
        }
        if (h_mad > g_mad) g_mad = h_mad;
        if (h_maxref > g_maxref) g_maxref = h_maxref;
        if (h_maxref > 0 && h_mad / h_maxref > worst_head) worst_head = h_mad / h_maxref;
    }
    double g = g_maxref > 0 ? g_mad / g_maxref : 0.0;
    printf("%-26s H=%2d/%-2d Sq=%-5d Sk=%-5d D=%3d %s cap=%-4d nrows=%-3d lastrow=%-5d | "
           "case=%.2e head=%.2e row=%.2e | maxref=%.3f nanrows=%ld/%ld\n",
           name, Hq, Hkv, Sq, Sk, D, causal ? "caus" : "full", ref_cap, nrows,
           rows[nrows - 1], g, worst_head, worst_row, g_maxref, nan_rows,
           nan_rows + live_rows);
    fflush(stdout);
    if (g > W_case) { W_case = g; snprintf(W_case_name, sizeof W_case_name, "%s", name); }
    if (worst_head > W_head) { W_head = worst_head; snprintf(W_head_name, sizeof W_head_name, "%s", name); }
    if (worst_row > W_row) { W_row = worst_row; snprintf(W_row_name, sizeof W_row_name, "%s", name); }
    free(Q); free(K); free(V); free(Oref); free(Odot); free(rows);
}

int main(int argc, char **argv) {
    int longrun = (argc > 1 && strcmp(argv[1], "--long") == 0);
    char nm[128];
    printf("=== PROBE: err/scale sweep. svcntw()=%d svcnth()=%d ===\n", (int)svcntw(),
           (int)svcnth());
    if (!longrun) {
        /* the 9 shipped cases, unchanged shapes */
        probe("ship:MHA prefill", 4, 4, 8, 8, 64, 0, 0);
        probe("ship:MHA prefil.causal", 4, 4, 8, 8, 64, 1, 0);
        probe("ship:GQA prefill", 8, 2, 8, 8, 64, 0, 0);
        probe("ship:GQA prefil.causal", 8, 2, 8, 8, 64, 1, 0);
        probe("ship:MHA decode", 4, 4, 1, 16, 64, 0, 0);
        probe("ship:GQA decode", 8, 2, 1, 16, 64, 0, 0);
        probe("ship:MQA prefill", 8, 1, 8, 8, 64, 0, 0);
        probe("ship:odd + multiblock", 6, 3, 130, 70, 40, 1, 0);
        probe("ship:GQA big head_dim", 8, 2, 4, 12, 128, 0, 0);
        /* D sweep incl. tiny/odd/large, both causal and full */
        int Ds[] = {1, 2, 3, 7, 8, 16, 33, 40, 64, 65, 96, 128};
        for (unsigned x = 0; x < sizeof Ds / sizeof *Ds; x++)
            for (int c = 0; c < 2; c++) {
                snprintf(nm, sizeof nm, "D=%d %s", Ds[x], c ? "caus" : "full");
                probe(nm, 4, 2, 96, 96, Ds[x], c, 0);
            }
        /* Sq/Sk block edges 63/64/65 and 127/128/129, both causal and full */
        int Ss[] = {63, 64, 65, 127, 128, 129};
        for (unsigned a = 0; a < sizeof Ss / sizeof *Ss; a++)
            for (unsigned b = 0; b < sizeof Ss / sizeof *Ss; b++)
                for (int c = 0; c < 2; c++) {
                    snprintf(nm, sizeof nm, "Sq=%d Sk=%d %s", Ss[a], Ss[b], c ? "caus" : "full");
                    probe(nm, 4, 2, Ss[a], Ss[b], 64, c, 0);
                }
        /* Sk < Sq causal (fully-masked early rows: F9 NaN territory) */
        probe("Sk<Sq caus 130x70", 6, 3, 130, 70, 40, 1, 0);
        probe("Sk<Sq caus 128x64", 4, 2, 128, 64, 64, 1, 0);
        probe("Sk<Sq caus 200x33", 4, 2, 200, 33, 32, 1, 0);
        /* GQA groups 1/2/4/16 */
        int Hqs[] = {16, 16, 16, 16}, Hkvs[] = {16, 8, 4, 1};
        for (int x = 0; x < 4; x++)
            for (int c = 0; c < 2; c++) {
                snprintf(nm, sizeof nm, "g=%d %s", Hqs[x] / Hkvs[x], c ? "caus" : "full");
                probe(nm, Hqs[x], Hkvs[x], 96, 96, 64, c, 0);
            }
        /* decode Sq=1 over growing Sk (single row attends everything) */
        int Sks[] = {1, 16, 64, 65, 128, 257, 512, 1024};
        for (unsigned x = 0; x < sizeof Sks / sizeof *Sks; x++) {
            snprintf(nm, sizeof nm, "decode Sq=1 Sk=%d", Sks[x]);
            probe(nm, 8, 2, 1, Sks[x], 128, 0, 0);
        }
        /* multi-block causal, FULL reference (every row) -- the honest baseline */
        probe("full-ref caus 256", 4, 2, 256, 256, 128, 1, 0);
        probe("full-ref caus 384", 4, 2, 384, 384, 64, 1, 0);
        probe("full-ref caus 512", 2, 1, 512, 512, 128, 1, 0);
        /* qwen3 shape, small S, spread vs prefix rows */
        probe("qwen3-shape 512 cap8", 16, 8, 512, 512, 128, 1, 8);
        probe("qwen3-shape 512 full", 16, 8, 512, 512, 128, 1, 0);
    } else {
        probe("qwen3 1k cap8-spread", 16, 8, 1024, 1024, 128, 1, 8);
        probe("qwen3 2k cap8-spread", 16, 8, 2048, 2048, 128, 1, 8);
        probe("qwen3 4k cap8-spread", 16, 8, 4096, 4096, 128, 1, 8);
        probe("qwen3 4k cap16-spread", 16, 8, 4096, 4096, 128, 1, 16);
    }
    printf("=== WORST case-norm=%.3e (%s)  head-norm=%.3e (%s)  row-norm=%.3e (%s) ===\n", W_case,
           W_case_name, W_head, W_head_name, W_row, W_row_name);
    return 0;
}
