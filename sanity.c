/* Sanity check: BFDOT and BFMMLA operand layout + toolchain/qemu bf16 support. */
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline uint16_t f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb; /* round to nearest even */
    return (uint16_t)(x >> 16);
}

int main(void) {
    /* ---- BFDOT: 4 lanes, each lane = dot of 2 bf16 pairs ---- */
    float af[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float bf[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    uint16_t ab[8], bb[8];
    for (int i = 0; i < 8; i++) {
        ab[i] = f32_to_bf16(af[i]);
        bb[i] = f32_to_bf16(bf[i]);
    }
    bfloat16x8_t va = vld1q_bf16((const bfloat16_t *)ab);
    bfloat16x8_t vb = vld1q_bf16((const bfloat16_t *)bb);
    float32x4_t acc = vdupq_n_f32(0.0f);
    acc = vbfdotq_f32(acc, va, vb);
    float o[4];
    vst1q_f32(o, acc);
    printf("BFDOT lanes : %.3f %.3f %.3f %.3f  (expect 22 38 38 22)\n", o[0], o[1], o[2], o[3]);
    printf("BFDOT total : %.3f  (expect 120)\n", o[0] + o[1] + o[2] + o[3]);

    /* ---- BFMMLA: C(2x2) += A(2x4 row-major) * B(4x2 col-major) ----
       A row0=[1,2,3,4] row1=[5,6,7,8]; B col0=[1,0,0,1] col1=[0,1,1,0] */
    float Af[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float Bf[8] = {1, 0, 0, 1, 0, 1, 1, 0};
    uint16_t Ab[8], Bb[8];
    for (int i = 0; i < 8; i++) {
        Ab[i] = f32_to_bf16(Af[i]);
        Bb[i] = f32_to_bf16(Bf[i]);
    }
    bfloat16x8_t vA = vld1q_bf16((const bfloat16_t *)Ab);
    bfloat16x8_t vB = vld1q_bf16((const bfloat16_t *)Bb);
    float32x4_t r = vdupq_n_f32(0.0f);
    r = vbfmmlaq_f32(r, vA, vB);
    float rr[4];
    vst1q_f32(rr, r);
    printf("BFMMLA C    : [%.3f %.3f ; %.3f %.3f]  (expect [5 5 ; 13 13])\n", rr[0], rr[1], rr[2],
           rr[3]);
    return 0;
}
