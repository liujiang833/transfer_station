/* Flash-attention-style bf16 kernel (MHA + GQA + MQA), SVE-only, BFDOT compute.
 *
 * Blocked over keys with ONLINE softmax: the [Sq x Sk] score matrix is never
 * materialised. Per query-block we keep only running max m[Bq], running
 * denominator l[Bq], and running output acc[Bq x D]; each key-block updates them
 * with the standard rescale  acc <- alpha*acc + P@V ,  l <- alpha*l + rowsum(P) ,
 * where alpha = exp(m_old - m_new). Memory is O(Bq*D + Bq*Bk), not O(Sq*Sk).
 * Key-blocks wholly in the causal future are skipped (and end the k-loop).
 *
 * ALLOCATION-FREE: attn_flash() mallocs nothing. The caller sizes ONE block with
 * attn_flash_scratch_bytes(D), aligns it to ATTN_SCRATCH_ALIGN, and passes it in.
 * That size depends only on head_dim -- it is O(Bq*D + Bq*Bk) and never O(Sq*Sk),
 * which is exactly the property above -- so a driver allocates one buffer per thread
 * and reuses it for every sequence length, prefill and decode alike. The size query
 * IS attn_flash_layout()'s measuring pass, so the reported size and the carve the
 * kernel performs cannot drift apart. See the scratch section above attn_flash().
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
 * Verified against a scalar fp32 reference that consumes the same bf16 bits. The
 * reference is O(Sk*D) per checked query row, so the big cases check a SPREAD
 * sample of rows (always including Sq-1, which sees every key block and every
 * rescale) rather than a cheap-but-vacuous prefix -- same cost, real coverage.
 * The pass bound is scale-relative (mad <= TOL*maxref, TOL=8e-3): bf16 P entries
 * put the irreducible noise at ~1e-3 of the output scale, measured worst 2.76e-3.
 * See ref_rowset() / run_case() for the measurements behind both choices.
 *
 * Both halves of the harness are NaN-SAFE, and that is load-bearing rather than
 * decorative: a fully-masked causal row (Sk<Sq) has an undefined softmax, and a
 * reference that answered 0/0=NaN there would make the bound test pass VACUOUSLY
 * on those rows -- NaN loses every ordered comparison. attn_ref emits a zero row
 * (matching the kernel's l=0 -> inv=0), and run_case counts non-finite elements
 * out-of-band so no NaN can ever be silent. This was a real blind spot: it hid
 * 360 of the 780 rows (46%) of the "odd + multiblock" case -- the ONLY shipped
 * case spanning >1 query block AND >1 key block. See attn_ref / run_case.
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

/* ---------------- fp32 reference (scalar by design) ----------------
 * Checks only the query rows listed in rows[0..nrows), and writes O for those
 * rows ONLY -- every other row of O is left untouched, so the caller must both
 * zero-init O and iterate the same rows[] (see ref_rowset / run_case).
 *
 * A causal row that attends ZERO keys (possible whenever Sk < Sq: row i's limit
 * is Sk-Sq+i, negative for i < Sq-Sk) has a mathematically undefined softmax.
 * This reference emits a ZERO output row there -- matching both the kernel
 * (l=0 -> inv=0, see attn_flash) and standard flash-attention implementations.
 * Emitting NaN instead (0/0) would be worse than wrong: NaN loses every ordered
 * float comparison, so the harness's bound test would pass VACUOUSLY on exactly
 * the rows it failed to define. See the sum==0 guard below and run_case's
 * non-finite counter, which are the two halves of that fix.
 */
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
            int i = rows[t]; /* checked rows are SPREAD over [0,Sq), not a prefix */
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
            /* sum==0 <=> the row attends no keys at all: every s[j] took the
             * -INFINITY branch above and is already 0. (With >=1 valid key,
             * sum >= 1 exactly -- the argmax term contributes expf(0)=1 -- so
             * this test cannot misfire on a real row.) Mirror the kernel's
             * l==0 -> inv=0 and emit a zero row; s[j] /= 0 would give 0/0 = NaN. */
            float inv = (sum > 0) ? 1.0f / sum : 0.0f;
            for (int j = 0; j < Sk; j++)
                s[j] *= inv;
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

/* ---------------- scratch: the kernel allocates NOTHING ----------------
 * Every buffer attn_flash() touches is carved out of one caller-owned block. The
 * driver sizes it with attn_flash_scratch_bytes(D) and hands the same block back on
 * every call; the kernel never mallocs, never frees, and keeps no state across calls
 * (it initialises everything it reads).
 *
 * The extents depend ONLY on D: BQ/BK are compile-time constants, and the working set
 * is O(BQ*D + BQ*BK), never O(Sq*Sk) -- that IS flash attention's memory property, so
 * the scratch is INDEPENDENT of Sq and Sk. One buffer per (thread, head_dim) serves
 * every sequence length: allocate at driver start-up, reuse for 1-token decode and 4k
 * prefill alike, free at shutdown.
 *
 * ONE SOURCE OF TRUTH, deliberately: attn_flash_layout() both measures and carves, and
 * attn_flash_scratch_bytes() is literally its measuring pass. A hand-written size sum
 * kept next to a separate carve is how you get a silent heap overflow -- the two drift,
 * nothing complains, and the kernel scribbles past the end. There is no second copy of
 * these sizes anywhere; do not add one.
 */
#define ATTN_SCRATCH_ALIGN 64 /* cache line. EVERY sub-buffer starts on a multiple of it:
                               * the carve mixes uint16_t and float tiles whose natural
                               * sizes depend on D, so without padding a sub-buffer's
                               * start would wander with D and SVE loads would straddle
                               * lines. The caller's block must be this aligned too. */

/* Sub-buffers, in carve order. "*2" is bytes-per-bf16; strides are the kernel's
 * (Sstride == Pstride == Vstride == BK):
 *   Vt [D x BK] bf16   packed V^T      Pb [BQ x BK] bf16  softmax probs P
 *   S  [BQ x BK] f32   score tile      pv [D] f32         one row's P*V partial
 *   acc [BQ x D] f32   running output  m, l, al [BQ] f32  online-softmax state
 */
struct attn_scratch {
    uint16_t *Vt, *Pb;
    float *S, *pv, *acc, *m, *l, *al;
};

/* Carve one sub-buffer: round the running offset up to ATTN_SCRATCH_ALIGN, hand back
 * that slot (NULL when only measuring), then advance past nbytes. */
static inline void *attn_carve(void *base, size_t *off, size_t nbytes) {
    size_t a = ATTN_SCRATCH_ALIGN;
    *off = (*off + a - 1) & ~(a - 1);
    void *p = base ? (void *)((char *)base + *off) : NULL;
    *off += nbytes;
    return p;
}

/* THE layout -- the only place these sizes exist. base==NULL measures; base!=NULL also
 * carves into base and fills *s. Returns total bytes, INCLUDING the inter-buffer
 * alignment padding, rounded up to ATTN_SCRATCH_ALIGN: that makes the result a legal
 * aligned_alloc() size (C11 wants size % alignment == 0) and keeps an array of
 * scratches aligned. Both callers -- the size query and attn_flash -- come through
 * here, so the reported size always covers the exact carve the kernel performs. */
static size_t attn_flash_layout(void *base, int D, struct attn_scratch *s) {
    struct attn_scratch discard; /* measure-only callers pass s == NULL */
    if (!s)
        s = &discard;
    size_t off = 0, a = ATTN_SCRATCH_ALIGN;
    s->Vt = attn_carve(base, &off, (size_t)D * BK * 2);
    s->Pb = attn_carve(base, &off, (size_t)BQ * BK * 2);
    s->S = attn_carve(base, &off, sizeof(float) * BQ * BK);
    s->pv = attn_carve(base, &off, sizeof(float) * D);
    s->acc = attn_carve(base, &off, sizeof(float) * BQ * D);
    s->m = attn_carve(base, &off, sizeof(float) * BQ);
    s->l = attn_carve(base, &off, sizeof(float) * BQ);
    s->al = attn_carve(base, &off, sizeof(float) * BQ);
    return (off + a - 1) & ~(a - 1);
}

/* Bytes of scratch attn_flash() needs for this head_dim. Independent of Sq/Sk -- see
 * the scratch note above. The caller's buffer must be ATTN_SCRATCH_ALIGN-aligned
 * (aligned_alloc / posix_memalign); run_case() shows the intended driver pattern. */
size_t attn_flash_scratch_bytes(int D) {
    return attn_flash_layout(NULL, D, NULL);
}

/* ---------------- flash attention forward ---------------- */
static void attn_flash(const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O, int Hq,
                       int Hkv, int Sq, int Sk, int D, int causal, void *scratch) {
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
     *   scratch    - CALLER-OWNED block, >= attn_flash_scratch_bytes(D) bytes and
     *                ATTN_SCRATCH_ALIGN-aligned. Every tile below is carved from it by
     *                attn_flash_layout(); this kernel allocates nothing. Its size needs
     *                only D, so one block per thread serves every Sq/Sk (see header).
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
    struct attn_scratch s;
    attn_flash_layout(scratch, D, &s); /* the very carve attn_flash_scratch_bytes() sized */
    uint16_t *Vt = s.Vt, *Pb = s.Pb;
    float *S = s.S, *pv = s.pv;
    float *acc = s.acc, *m = s.m, *l = s.l, *al = s.al;

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
    /* No frees: every buffer above is the caller's. */
}

/* ---------------- harness ---------------- */
static uint32_t rng;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xffff) / 32768.0f - 1.0f;
}
static double now_s(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}
/* Useful (i,j) score pairs actually needed. For causal this is ~half of Sq*Sk,
 * so the common 4*Hq*Sq*Sk*D formula (non-causal) would overstate the work. */
static double attn_pairs(int Sq, int Sk, int causal) {
    if (!causal)
        return (double)Sq * Sk;
    double n = 0;
    for (int i = 0; i < Sq; i++) {
        int cnt = Sk - Sq + i + 1; /* keys row i may attend */
        if (cnt > Sk)
            cnt = Sk;
        if (cnt < 0)
            cnt = 0;
        n += cnt;
    }
    return n;
}

/* Which query rows the fp32 reference checks, into rows[] (capacity Sq); returns
 * the count. ref_cap<=0 or >=Sq means "every row".
 *
 * A PREFIX would be near-worthless for the causal Sq==Sk cases: rows 0..cap-1
 * attend only keys 0..cap-1, i.e. a cap x cap corner of the FIRST key block, so
 * the multi-block online-softmax rescale -- the whole point of this kernel, and
 * the only hard part at 4k -- would never be checked at all. So SPREAD the sample
 * over [0,Sq) and always include row Sq-1, which attends all Sk keys and hence
 * walks every key block and every rescale step. This is FREE coverage: attn_ref
 * costs O(Sk*D) per checked row wherever that row sits, so a spread sample and a
 * prefix sample of the same size do the same reference work. */
static int ref_rowset(int Sq, int ref_cap, int *rows) {
    if (ref_cap <= 0 || ref_cap >= Sq) {
        for (int i = 0; i < Sq; i++)
            rows[i] = i;
        return Sq; /* full reference: every row */
    }
    if (ref_cap == 1) {
        rows[0] = Sq - 1; /* only room for one: take the row that sees every key */
        return 1;
    }
    for (int t = 0; t < ref_cap; t++) /* rows[0]=0 .. rows[cap-1]=Sq-1, evenly spread */
        rows[t] = (int)(((int64_t)t * (Sq - 1)) / (ref_cap - 1));
    return ref_cap;
}

/* Pass bound: mad <= TOL * maxref, i.e. the printed err/scale IS the pass metric.
 * Rationale for normalising by the per-case output scale rather than a per-element
 * |k-r| <= atol + rtol*|r|: the P entries are bf16, so each output element carries
 * ~2^-9 relative noise of the ROW's magnitude (err <= 2^-9 * max_j|V[j][d]|); the
 * error simply does not scale with each element's own |r|, and the old atol term
 * dominated it entirely (2e-2 against outputs of O(0.1-1.0) -> an effective ~3e-2
 * bound, ~30x the real ~1e-3 noise, so a uniform 1% output error passed all 9).
 * Per-ROW normalisation was measured and REJECTED: at D=1 a row whose single output
 * element cancels to ~0 gives a degenerate denominator (err/scale hit 1.16 on a
 * correct kernel). Per-HEAD was rejected for the same reason, more mildly (5.6e-3,
 * from one D=1 head whose maxref was 0.116 while its absolute error was a normal
 * 6.5e-4). Pooling the whole case makes the scale estimate robust.
 * TOL=8e-3 is set from measurement, not guesswork -- 129 shapes x VL 512/2048
 * (D=1..128, Sq/Sk block edges 63/64/65/127/128/129, Sk<Sq, GQA g=1/2/4/16,
 * causal+full) put the worst err/scale at 2.76e-3 (D=8 non-causal), the shipped 9
 * at <=1.70e-3, and qwen3 1k/2k/4k at <=2.9e-4. So TOL is 2.9x the global worst and
 * 4.7x the shipped worst, while staying under 1e-2 so that a uniform x1.01 output
 * error (which lands err/scale at ~=e, since the max-|ref| element carries the max
 * absolute error) always FAILS. Bisected detection threshold: x1.0072 passes,
 * x1.0074 fails -> ~0.73%, vs ~3.5-4% for the old bound (A/B measured). */
#ifndef TOL
#define TOL 8e-3f
#endif

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
    /* Oref is zero-init AND only rows[] is ever read back: the reference leaves
     * every unchecked row untouched, so neither loop may wander outside rows[]. */
    float *Oref = calloc(no, 4), *Odot = calloc(no, 4);
    int *rows = malloc(sizeof(int) * Sq);
    int nrows = ref_rowset(Sq, ref_cap, rows);
    attn_ref(Q, K, V, Oref, Hq, Hkv, Sq, Sk, D, causal, rows, nrows);
    /* Intended DRIVER pattern for the allocation-free kernel: ask for the size, get one
     * ATTN_SCRATCH_ALIGN-aligned block, hand it to every attn_flash() call, free it at
     * the end. The size needs only D -- not Sq/Sk -- so a real driver hoists this out of
     * the sequence loop and keeps one block per thread for the process's lifetime. The
     * alignment is a contract, not a nicety: aligned_alloc(), never plain malloc(). */
    size_t sbytes = attn_flash_scratch_bytes(D);
    void *scratch = aligned_alloc(ATTN_SCRATCH_ALIGN, sbytes);
    double t0 = now_s();
    attn_flash(Q, K, V, Odot, Hq, Hkv, Sq, Sk, D, causal, scratch);
    double tdot = now_s() - t0;

    /* NaN-SAFETY, deliberate: a NaN loses EVERY ordered comparison, so it neither
     * trips `ad > mad` nor `fabsf(r) > maxref` -- left to the maxima alone it would
     * slip through silently and the row would never be compared at all. Nor can it
     * be carried IN mad (`!(ad <= mad)` would latch NaN, then the next finite ad
     * would overwrite it -- poison, then un-poison). So non-finite elements are
     * counted OUT-OF-BAND in nbad and fail the case on their own; mad/maxref stay
     * maxima over finite data, which keeps the printed err/scale meaningful. */
    float mad = 0, maxref = 0; /* max abs deviation, and the case's output scale */
    long nbad = 0;             /* kernel or ref element not finite -> unconditional FAIL */
    for (int h = 0; h < Hq; h++)
        for (int t = 0; t < nrows; t++)
            for (int d = 0; d < D; d++) {
                size_t idx = ((size_t)h * Sq + rows[t]) * D + d; /* same rows as the ref */
                float r = Oref[idx], k = Odot[idx], ad = fabsf(k - r);
                if (!isfinite(k) || !isfinite(r)) {
                    nbad++;
                    continue;
                }
                if (ad > mad)
                    mad = ad;
                if (fabsf(r) > maxref)
                    maxref = fabsf(r);
            }
    /* One bound on the whole case, judged after mad/maxref are final. maxref==0
     * (no reference signal at all) degrades to demanding an exact match. Written
     * `mad <= bound` and not `!(mad > bound)`: the <= form fails closed, so even a
     * mad that somehow arrived NaN cannot pass. */
    int ok = (nbad == 0) && (mad <= TOL * maxref);
    char reflbl[48];
    if (nrows < Sq)
        snprintf(reflbl, sizeof reflbl, " [ref:%d rows spread 0..%d]", nrows, Sq - 1);
    else
        reflbl[0] = 0;
    /* A non-finite element is named, not just counted into a bare FAIL: err/scale
     * is computed from the finite rest and so looks innocent when NaN is the bug. */
    char verdict[40];
    if (nbad)
        snprintf(verdict, sizeof verdict, "*** FAIL (%ld non-finite) ***", nbad);
    else
        snprintf(verdict, sizeof verdict, "%s", ok ? "PASS" : "*** FAIL ***");
    double gflop = 4.0 * (double)Hq * D * attn_pairs(Sq, Sk, causal) / 1e9;
    printf("%-17s H=%2d/%-2d(g%d) Sq=%-4d Sk=%-4d D=%3d %s | err/scale=%.1e | %7.2f GFLOP "
           "%7.2fs %5.2f GFLOP/s%s -> %s\n",
           name, Hq, Hkv, Hq / Hkv, Sq, Sk, D, causal ? "causal" : "full  ",
           maxref > 0 ? mad / maxref : 0.f, gflop, tdot, gflop / tdot, reflbl, verdict);
    free(Q);
    free(K);
    free(V);
    free(Oref);
    free(Odot);
    free(rows);
    free(scratch);
    return ok;
}

int main(int argc, char **argv) {
    int longrun = (argc > 1 && strcmp(argv[1], "--long") == 0);
    int all = 1;
    printf("=== FLASH bf16 attention (online softmax, SVE BFDOT vs fp32 ref) ===\n");
    printf("SVE VL: svcntw()=%d f32 lanes, svcnth()=%d bf16 lanes (runtime, not hardcoded)\n",
           (int)svcntw(), (int)svcnth());
    all &= run_case("MHA prefill", 4, 4, 8, 8, 64, 0, 0);
    all &= run_case("MHA prefil.causal", 4, 4, 8, 8, 64, 1, 0);
    all &= run_case("GQA prefill", 8, 2, 8, 8, 64, 0, 0);
    all &= run_case("GQA prefil.causal", 8, 2, 8, 8, 64, 1, 0);
    all &= run_case("MHA decode(Sq=1)", 4, 4, 1, 16, 64, 0, 0);
    all &= run_case("GQA decode(Sq=1)", 8, 2, 1, 16, 64, 0, 0);
    all &= run_case("MQA prefill", 8, 1, 8, 8, 64, 0, 0);
    all &= run_case("odd + multiblock", 6, 3, 130, 70, 40, 1, 0); /* spans >1 Bq and Bk block */
    all &= run_case("GQA big head_dim", 8, 2, 4, 12, 128, 0, 0);
    if (longrun) {
        printf("--- Qwen3 prefill sweep: Hq=16 Hkv=8 (GQA group=2) D=128 causal ---\n");
        printf("  Covers BOTH Qwen3-0.6B AND Qwen3-1.7B: their attention shapes are identical\n");
        printf("  (16 Q heads / 8 KV heads / head_dim 128 / 28 layers). Only hidden_size differs\n");
        printf("  (1024 vs 2048), which sizes the QKV projections and never reaches this kernel.\n");
        printf("  GFLOP = causal-correct USEFUL flops = 4*Hq*D*#{(i,j): j<=i}, i.e. ~half of the\n");
        printf("  non-causal 4*Hq*Sq*Sk*D. Timings are QEMU FUNCTIONAL EMULATION of one core --\n");
        printf("  they measure work done, NOT hardware speed. Do not read them as ARM perf.\n");
        printf("  [ref:N rows spread 0..Sq-1]: the fp32 ref checks N rows SPREAD over the whole\n");
        printf("  sequence, incl. row Sq-1 -- which attends all Sk keys and so walks every key\n");
        printf("  block and every online-softmax rescale. (A prefix of N rows would be causal-\n");
        printf("  masked down to an NxN corner of the FIRST key block: same cost, no coverage.)\n");
        all &= run_case("qwen3 1k", 16, 8, 1024, 1024, 128, 1, 8);
        all &= run_case("qwen3 2k", 16, 8, 2048, 2048, 128, 1, 8);
        all &= run_case("qwen3 4k", 16, 8, 4096, 4096, 128, 1, 8);
    } else
        printf("(pass --long to also run the Qwen3 1k/2k/4k prefill sweep)\n");
    printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
