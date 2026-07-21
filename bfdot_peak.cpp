// BFDOT peak-throughput microbenchmark (SVE, bf16 -> f32).
//
// Measures the SUSTAINED throughput of svbfdot_f32 with zero memory traffic: two operand
// vectors are held in registers and NACC independent accumulator chains run back-to-back, so
// the loop is bound only by the BFDOT execution units (not loads, not latency). This is the
// compute roofline peak -- the peak FLOP/s that anchors the machine-balance / arithmetic-
// intensity discussion (ridge point = this peak / DRAM bandwidth).
//
// Why NACC independent chains: each chain  c = svbfdot(c, a, b)  is a self-recurrence, so a
// single chain is LATENCY-bound. Running NACC of them in parallel exposes enough independent
// work to fill the pipeline and expose THROUGHPUT. NACC=16 hides up to ~8-cycle FMA latency at
// 2 issues/cycle; if a core needs more, raise NACC (watch the 32 Z-register budget).
//
// Build + run (native ARM -- this is the real number; do it on the target machine):
//   g++ -O3 -march=native bfdot_peak.cpp -o bfdot_peak && ./bfdot_peak [freq_GHz]
//   (clang++ works too:  clang++ -O3 -march=native ...)
// Pass the core clock in GHz as arg 1 to also get svbfdot/cycle and FLOPs/cycle.
//
// Cross-building for QEMU needs a C++ AArch64 cross-compiler (aarch64-linux-gnu-g++); the numbers
// under emulation are MEANINGLESS anyway (softfloat), so run it natively. Only C-style headers and
// SVE intrinsics are used, so a C++ compiler is not strictly required -- the same body compiles as
// C after swapping the <cXXX> headers for <XXX.h> and writing `struct timespec`.

#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define NACC 16 // independent accumulator chains (ILP); see header

static double now_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t); // monotonic: never steps backward mid-measurement
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// One pass = NACC independent svbfdot_f32, each folding a*b into its own accumulator.
#define DOT()                                                                                      \
    do {                                                                                           \
        c0 = svbfdot_f32(c0, a, b);                                                                 \
        c1 = svbfdot_f32(c1, a, b);                                                                 \
        c2 = svbfdot_f32(c2, a, b);                                                                 \
        c3 = svbfdot_f32(c3, a, b);                                                                 \
        c4 = svbfdot_f32(c4, a, b);                                                                 \
        c5 = svbfdot_f32(c5, a, b);                                                                 \
        c6 = svbfdot_f32(c6, a, b);                                                                 \
        c7 = svbfdot_f32(c7, a, b);                                                                 \
        c8 = svbfdot_f32(c8, a, b);                                                                 \
        c9 = svbfdot_f32(c9, a, b);                                                                 \
        c10 = svbfdot_f32(c10, a, b);                                                               \
        c11 = svbfdot_f32(c11, a, b);                                                               \
        c12 = svbfdot_f32(c12, a, b);                                                               \
        c13 = svbfdot_f32(c13, a, b);                                                               \
        c14 = svbfdot_f32(c14, a, b);                                                               \
        c15 = svbfdot_f32(c15, a, b);                                                               \
    } while (0)

int main(int argc, char **argv) {
    double freq_ghz = (argc > 1) ? atof(argv[1]) : 0.0; // optional: core clock for FLOPs/cycle

    // Operands seeded through a volatile so the compiler can't constant-fold the whole loop.
    // 0x3f80 is 1.0f in bf16; the exact value is irrelevant to throughput.
    volatile uint16_t bseed = 0x3f80;
    svbfloat16_t a = svreinterpret_bf16_u16(svdup_n_u16(bseed));
    svbfloat16_t b = svreinterpret_bf16_u16(svdup_n_u16(bseed));

    // Distinct starting values (volatile-seeded) so the NACC chains are provably NOT equal --
    // otherwise value-numbering could merge them and we'd time a single chain.
    volatile float fseed = 1.0f;
    float s = fseed;
    svfloat32_t c0 = svdup_n_f32(s + 0), c1 = svdup_n_f32(s + 1), c2 = svdup_n_f32(s + 2),
                c3 = svdup_n_f32(s + 3), c4 = svdup_n_f32(s + 4), c5 = svdup_n_f32(s + 5),
                c6 = svdup_n_f32(s + 6), c7 = svdup_n_f32(s + 7), c8 = svdup_n_f32(s + 8),
                c9 = svdup_n_f32(s + 9), c10 = svdup_n_f32(s + 10), c11 = svdup_n_f32(s + 11),
                c12 = svdup_n_f32(s + 12), c13 = svdup_n_f32(s + 13), c14 = svdup_n_f32(s + 14),
                c15 = svdup_n_f32(s + 15);

    const int VLh = (int)svcnth();          // bf16 lanes per vector (runtime, VL-agnostic)
    const double flops_per_bfdot = 2.0 * VLh; // VLh multiplies + VLh adds

    // Adaptive: grow the iteration count until a trial takes >= 0.5 s, so it self-scales across
    // the huge QEMU-vs-hardware speed gap. Each trial resets the accumulators first.
    long iters = 1L << 20;
    double secs = 0;
    for (;;) {
        s = fseed; // re-read volatile
        c0 = svdup_n_f32(s + 0), c1 = svdup_n_f32(s + 1), c2 = svdup_n_f32(s + 2),
        c3 = svdup_n_f32(s + 3), c4 = svdup_n_f32(s + 4), c5 = svdup_n_f32(s + 5),
        c6 = svdup_n_f32(s + 6), c7 = svdup_n_f32(s + 7), c8 = svdup_n_f32(s + 8),
        c9 = svdup_n_f32(s + 9), c10 = svdup_n_f32(s + 10), c11 = svdup_n_f32(s + 11),
        c12 = svdup_n_f32(s + 12), c13 = svdup_n_f32(s + 13), c14 = svdup_n_f32(s + 14),
        c15 = svdup_n_f32(s + 15);

        double t0 = now_s();
        for (long i = 0; i < iters; i++)
            DOT();
        secs = now_s() - t0;
        if (secs >= 0.5 || iters >= (1L << 34))
            break;
        iters <<= 1;
    }

    // Consume the accumulators so nothing is dead-code-eliminated.
    svfloat32_t acc = svadd_f32_x(svptrue_b32(), c0, c1);
    acc = svadd_f32_x(svptrue_b32(), acc, svadd_f32_x(svptrue_b32(), c2, c3));
    acc = svadd_f32_x(svptrue_b32(), acc, svadd_f32_x(svptrue_b32(), c4, c5));
    acc = svadd_f32_x(svptrue_b32(), acc, svadd_f32_x(svptrue_b32(), c6, c7));
    acc = svadd_f32_x(svptrue_b32(), acc, svadd_f32_x(svptrue_b32(), c8, c9));
    acc = svadd_f32_x(svptrue_b32(), acc, svadd_f32_x(svptrue_b32(), c10, c11));
    acc = svadd_f32_x(svptrue_b32(), acc, svadd_f32_x(svptrue_b32(), c12, c13));
    acc = svadd_f32_x(svptrue_b32(), acc, svadd_f32_x(svptrue_b32(), c14, c15));
    volatile float sink = svaddv_f32(svptrue_b32(), acc);
    (void)sink;

    const double nbfdot = (double)iters * NACC;
    const double gflops = nbfdot * flops_per_bfdot / secs / 1e9;

    printf("SVE VL       : %d bf16 lanes/vec (%d-bit), %.0f FLOP per svbfdot\n", VLh, VLh * 16,
           flops_per_bfdot);
    printf("chains(NACC) : %d independent accumulators\n", NACC);
    printf("work         : %.3g svbfdot in %.4f s (iters=%ld x %d)\n", nbfdot, secs, iters, NACC);
    printf("THROUGHPUT   : %.2f GFLOP/s   (%.2f G svbfdot/s)\n", gflops, nbfdot / secs / 1e9);
    if (freq_ghz > 0) {
        double cyc = secs * freq_ghz * 1e9;
        printf("per cycle    : %.3f svbfdot/cycle, %.2f bf16-FLOP/cycle  (@ %.3f GHz)\n",
               nbfdot / cyc, nbfdot * flops_per_bfdot / cyc, freq_ghz);
    } else {
        printf("per cycle    : pass core clock in GHz as arg 1 for svbfdot/cycle + FLOP/cycle\n");
    }
    return 0;
}
