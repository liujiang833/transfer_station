# bf16 Attention Kernel (MHA + GQA) with BFDOT / BFMMLA

A self-contained, **QEMU-verified** attention kernel for AArch64 that computes
scaled-dot-product attention in **bfloat16** using the Arm BF16 instructions
`BFDOT` and `BFMMLA`, with an fp32 reference for correctness checking.

Two kernels live here, and they use *different* instruction sets:

- **`attn.c` — NEON.** Materialises the `[Sq×Sk]` score matrix; selectable
  BFDOT / BFMMLA compute paths.
- **`flash.c` — SVE only.** Flash attention with online softmax; `svbfdot_f32`
  is its single compute primitive. This is the one you'd ship.

Built and tested on an **x86_64** host (no ARM hardware, no root) via an AArch64
cross-compiler + QEMU user-mode emulation.

## Files

| File | Purpose |
|---|---|
| `sanity.c` | **NEON.** Tiny test that pins down BFDOT/BFMMLA operand layout against hand-computed values |
| `attn.c` | **NEON.** Materialising kernel: fp32 reference + BFDOT path + BFMMLA path + harness |
| `flash.c` | **SVE-only BFDOT.** Flash-attention kernel: online softmax, key-blocked, O(Bq·D) memory + harness |
| `tolerance_probe.c` | Measurement harness behind `flash.c`'s `TOL=8e-3`: prints `err/scale` under per-case / per-head / per-row normalisation and counts NaN reference rows. **Not part of the shipped kernel**; frozen snapshot, build/run lines in its header |
| `run.sh` | Build all three (AArch64, `-march=armv8.6-a+sve+bf16`) and run under `qemu-aarch64 -cpu max` |
| `qemu_pkg/` | Locally-extracted `qemu-aarch64-static` (no root needed) |

## Build & run

```bash
./run.sh
```

If `qemu-aarch64-static` is missing, fetch it **without root**:
```bash
cd qemu_pkg && apt-get download qemu-user-static && dpkg-deb -x qemu-user-static_*.deb ./extracted
```
(Or use a Docker container with root and `apt-get install qemu-user gcc-aarch64-linux-gnu`.)

## Trap: GCC 13 miscompiles `float` → `bfloat16_t` casts

**Never write `(bfloat16_t)some_float` in this tree.** `aarch64-linux-gnu-gcc
13.3.0` with `-march=armv8.6-a+sve+bf16` applies **IEEE fp16** conversion
semantics to the cast:

```c
bfloat16_t a[64];
for (int i = 0; i < 64; i++) a[i] = (bfloat16_t)1.0f;   /* -> bits 0x3c00 */
```

`0x3c00` is fp16 `1.0`. bf16 `1.0` is **`0x3f80`**. It reproduces at both `-O0`
and `-O2` (at `-O2` via IPA-CP folding the constant to `fmov h0, #1.0`, an fp16
immediate), so it is not an optimizer artifact you can flag your way out of.

It is nasty because it is **quiet**: results are wrong but never NaN, never a
crash, never a warning. And it hides from casual probing — fp16 and bf16 share
the encoding `0x4000` for `2.0`, so a probe written with only `2.0` looks
perfectly correct.

This is **why** both kernels store bf16 tensors as `uint16_t` bit patterns and
reinterpret them to `bfloat16_t*` for the intrinsics. Convert fp32 → bf16 only
via:

- `f32_to_bf16()` — the explicit bit helper, round-to-nearest-even (scalar), or
- `svcvt_bf16_x` + `svuzp1_bf16` — the vector path in `flash.c`'s hot loop.

Both are verified correct against a scalar RNE reference (all lanes, 0
mismatches; round-trip through `svbfdot` exact). Only the C cast is broken.

## The two instructions (NEON / 128-bit forms)

- **`BFDOT`** `vbfdotq_f32(f32x4 acc, bf16x8 a, bf16x8 b)` — 4 output lanes, each a
  2-element bf16 dot: `acc[k] += a[2k]*b[2k] + a[2k+1]*b[2k+1]`. **8 MACs/instr.**
  GEMV-friendly (one score / one output element per reduction).
- **`BFMMLA`** `vbfmmlaq_f32(f32x4 acc, bf16x8 a, bf16x8 b)` — a `[2×4]·[4×2] → [2×2]`
  matmul-accumulate. `a` holds a **2×4 row-major** tile, `b` holds a **4×2
  column-major** tile (i.e. `b[0..3]`=col0, `b[4..7]`=col1). **16 MACs/instr.**
  GEMM-friendly, but needs ≥2 independent rows on *both* operands to be full-rate.

`sanity.c` verifies both layouts (`BFDOT → 22 38 38 22`, `BFMMLA → [5 5; 13 13]`).

BFMMLA appears only in `attn.c` and `sanity.c`. `flash.c` dropped its BFMMLA path
(and its `use_mmla` flag) when it moved to SVE, and uses the VL-agnostic
`svbfdot_f32` for everything — see below.

## Kernel design (`attn.c`, NEON)

**One kernel handles MHA and GQA.** The only difference is the head→kv mapping
`kv = h / group`, `group = num_q_heads / num_kv_heads`:
- MHA: `num_kv_heads == num_q_heads` → `group == 1`
- GQA: `num_kv_heads  < num_q_heads` → `group  > 1`
- MQA: `num_kv_heads == 1`

(`flash.c` uses the identical mapping.)

Per head it does `S = scale · Q·Kᵀ` → mask (causal optional) → softmax (fp32) →
`O = P·V`. Two selectable compute paths:

- **BFDOT path** — every score `S[i,j]` and every output `O[i,d]` is a bf16
  dot-product reduction. Works for any shape, including decode `Sq=1`.
  `V` is transposed to `Vᵀ` so the `P·V` contraction is contiguous.
- **BFMMLA path** — `Q·Kᵀ` is tiled as **query-pair × key-pair** (contract over
  `head_dim` in steps of 4); `P·V` as **query-pair × dim-pair** (contract over
  keys). Odd `Sq`/`Sk`/`D` are handled by **zero-padding** the packed buffers, and
  padded key columns are masked to `-inf` so they don't leak into softmax.

bf16 tensors are stored as `uint16_t` bit patterns (the top 16 bits of fp32) and
reinterpreted as `bfloat16_t`, so the **exact same bits** feed the hardware kernel
and the fp32 reference — the only divergence is bf16 rounding of the softmax
probabilities `P` and fp32 accumulation order.

> Note on decode: with `Sq=1`, the BFMMLA path pads the query tile to 2 rows, so
> its second row is wasted — this is the concrete "50% / double-compute" cost of
> forcing BFMMLA onto a single-query MHA/GQA decode step. See *Enhancements*.
> (`flash.c` is BFDOT-only and has no such waste.)

## Verification results — `attn.c` (QEMU 8.2.2, `-cpu max`)

```
MHA prefill            grp1 Sq=8 Sk= 8 D= 64 full   err/scale: BFDOT=1.8e-03 BFMMLA=1.8e-03  dot-vs-mmla=6e-08  PASS
MHA prefill causal     grp1 Sq=8 Sk= 8 D= 64 causal err/scale: BFDOT=1.6e-03 BFMMLA=1.6e-03  dot-vs-mmla=6e-08  PASS
GQA prefill            grp4 Sq=8 Sk= 8 D= 64 full   err/scale: BFDOT=2.0e-03 BFMMLA=2.0e-03  dot-vs-mmla=6e-08  PASS
GQA prefill causal     grp4 Sq=8 Sk= 8 D= 64 causal err/scale: BFDOT=2.5e-03 BFMMLA=2.5e-03  dot-vs-mmla=6e-08  PASS
MHA decode(Sq=1)       grp1 Sq=1 Sk=16 D= 64 full   err/scale: BFDOT=1.8e-03 BFMMLA=1.8e-03  dot-vs-mmla=3e-08  PASS
GQA decode(Sq=1)       grp4 Sq=1 Sk=16 D= 64 full   err/scale: BFDOT=2.5e-03 BFMMLA=2.5e-03  dot-vs-mmla=1e-08  PASS
MQA prefill            grp8 Sq=8 Sk= 8 D= 64 full   err/scale: BFDOT=2.1e-03 BFMMLA=2.1e-03  dot-vs-mmla=3e-08  PASS
GQA odd shapes         grp2 Sq=5 Sk= 7 D= 40 causal err/scale: BFDOT=2.1e-03 BFMMLA=2.1e-03  dot-vs-mmla=0      PASS
GQA big head_dim       grp4 Sq=4 Sk=12 D=128 full   err/scale: BFDOT=2.0e-03 BFMMLA=2.0e-03  dot-vs-mmla=3e-08  PASS
=== ALL PASS ===
```

- `err/scale` = max |kernel − fp32ref| normalised by output signal scale.
  ~`2e-3` matches the ~0.4% bf16 mantissa rounding of `P`.
- `dot-vs-mmla` ~`1e-8` ⇒ BFDOT and BFMMLA agree to floating-point noise (both do
  exact bf16 products with fp32 accumulation).
- Pass criterion **for `attn.c`**: `|kernel − ref| ≤ 2e-2 + 2e-2·|ref|` for every
  element. `flash.c` no longer uses this bound — it was measured to be ~30× looser
  than the real bf16 noise; see *Tolerance* below.

### Long-sequence prefill (`./attn --long`, D=128, causal)

```
prefill 1k MHA   Sq=1024 Sk=1024 D=128  err/scale=1.9e-3  0.54 GFLOP  PASS
prefill 1k GQA   Sq=1024 Sk=1024 D=128  err/scale=1.9e-3  2.15 GFLOP  PASS   (4 q-heads / 1 kv)
prefill 2k MHA   Sq=2048 Sk=2048 D=128  err/scale=1.9e-3  2.15 GFLOP  PASS
prefill 4k MHA   Sq=4096 Sk=4096 D=128  err/scale=1.3e-3  8.59 GFLOP  PASS
```

Accuracy holds at 4k (softmax over 4096 keys stays stable via max-subtraction).
The `--long` cases reference-check only a subset of query rows (the scalar fp32
reference is O(Sq*Sk*D) and slow under emulation) but cross-check BFDOT vs BFMMLA
on **every** row. QEMU timings are functional-emulation only (~0.1 GFLOP/s), not
representative of hardware.

> `attn.c`'s subset is a **prefix** of query rows. For causal `Sq==Sk` that is
> weak coverage for the same reason it was in `flash.c` — see *Reference rows*
> below. `flash.c` was fixed; `attn.c` was not.

**Memory note for 1k-4k prefill:** this kernel *materializes* the score matrix
`S` (Sq*Sk*4 bytes = 4 / 16 / **64** MB at 1k / 2k / 4k per head). Fine for
verification, but it blows past cache at 4k, so a production kernel should use
flash-attention-style key-blocking with online softmax (memory O(Sq*D)), keeping
the `S`/`P` tiles resident. See below.

## Flash-attention kernel (`flash.c`) — SVE-only, BFDOT

The materialising kernel above keeps the full `S` matrix (64 MB/head at 4k).
`flash.c` is the version you'd actually ship for 1k–4k prefill: it blocks over
keys with **online softmax** so `S` is never materialised.

It is **SVE only** — `arm_sve.h` is the sole ARM header, no NEON, no SME/ZA — and
`svbfdot_f32` is the single compute primitive for **both** `Q·Kᵀ` and `P·V`.

Per query-block (`BQ=64`) it keeps only running state — max `m[Bq]`, denominator
`l[Bq]`, output accumulator `acc[Bq×D]`. Each key-block (`BK=64`):

1. `S = scale · Q_blk·K_blkᵀ`  (BFDOT, block-local)
2. per row: `m_new = max(m, rowmax S)`, `α = exp(m−m_new)`, `P = exp(S−m_new)`
3. `l ← α·l + rowsum(P)`,  `acc ← α·acc + P·V_blk`  (BFDOT again)

Final `O = acc / l`. Causal key-blocks wholly in the future are skipped (and end
the k-loop). **Memory is O(Bq·D + Bq·Bk)** (a few tens of KB/head) regardless of
sequence length — the right shape for long-context prefill.

### Why only V is packed

This is the interesting part of the port. kutacc's original
(`src/attention/flash_mha/flash_attention.cpp`) is an **SME** kernel:
`svmopa_za32_bf16_m` accumulating outer products into the ZA tile under
`__arm_new("za") __arm_streaming`. **MOPA dictates the ZA operand layout**, so
that kernel must pre-pack Q, K **and** V. BFDOT has no ZA-layout constraint, so
none of that rationale carries over — the contraction axes decide instead:

- `Q·Kᵀ` contracts over **head_dim**, which is contiguous in **both** Q rows and
  K rows → `svld1_bf16` feeds BFDOT straight from the tensors. **No Q/K pack.**
- `P·V` contracts over **keys**: contiguous in `P`, but stride-`D` in `V` → `V`
  alone is transposed to `Vt[d][k]`. **Pack V only.**

The price of an unpacked K, accepted deliberately: BFDOT cannot use the
`svbfdot_lane` outer-product form, so every score ends in one `svaddv` horizontal
reduction. `UNR=4` independent accumulators per pass overlap that latency.

### Predication replaced all padding

The NEON code zero-pads its tiles (`Bqp` even, `Bkp`/`Dp` multiple of 8) and
`-inf`-masks the padded key columns. SVE needs none of it:

- **D-tail:** a `svwhilelt_b16` *zeroing* load makes BFDOT's pair lanes contribute
  0 to the dot. A short row costs nothing and needs no padded buffer.
- **Key-tail + causal mask:** both are *prefixes* of `j`, so the entire mask
  collapses to a single `whilelt` bound `jmax = min(rk, lim−kj+1)`. No `-inf`
  fill, no padded tile.

The `Qp`/`Kp` pack buffers and every `RUP()` are gone.

### Softmax

Fully SVE: `fast_exp` (`svexpa`-based exp2, ported from kutacc
`src/math/fast_exp.h` — pure SVE already, no SME). Max rel err ~2e-6 over
`[−40, 0]`, which is the only range flash attention exercises (scores minus the
row max). There is no `expf()` and no scalar `j`-loop anywhere in the kernel;
scalar code lives only in the fp32 reference and the harness.

### VL-agnostic

Every loop is driven by `svcntw()` / `svcnth()` / `svwhilelt` at runtime; nothing
is hardcoded to a vector width. Verified 9/9 PASS at **VL = 128 / 256 / 512 /
1024 / 2048** (`qemu -cpu max,sve<N>=on,sve-default-vector-length=-1`), each
reporting the expected lane counts, worst `err/scale` = 1.7e-3 at *every* width.
VL=2048 is the strong case: `svcnth()`=128 exceeds `BK`=64 — exactly what a
padded design would break on.

### Verification (`./flash`)

```
=== FLASH bf16 attention (online softmax, SVE BFDOT vs fp32 ref) ===
SVE VL: svcntw()=16 f32 lanes, svcnth()=32 bf16 lanes (runtime, not hardcoded)
MHA prefill       H= 4/4 (g1) Sq=8    Sk=8    D= 64 full   | err/scale=1.4e-03 |    0.00 GFLOP    0.00s  0.02 GFLOP/s -> PASS
MHA prefil.causal H= 4/4 (g1) Sq=8    Sk=8    D= 64 causal | err/scale=9.4e-04 |    0.00 GFLOP    0.00s  0.03 GFLOP/s -> PASS
GQA prefill       H= 8/2 (g4) Sq=8    Sk=8    D= 64 full   | err/scale=1.5e-03 |    0.00 GFLOP    0.00s  0.04 GFLOP/s -> PASS
GQA prefil.causal H= 8/2 (g4) Sq=8    Sk=8    D= 64 causal | err/scale=1.3e-03 |    0.00 GFLOP    0.00s  0.03 GFLOP/s -> PASS
MHA decode(Sq=1)  H= 4/4 (g1) Sq=1    Sk=16   D= 64 full   | err/scale=1.1e-03 |    0.00 GFLOP    0.00s  0.04 GFLOP/s -> PASS
GQA decode(Sq=1)  H= 8/2 (g4) Sq=1    Sk=16   D= 64 full   | err/scale=1.7e-03 |    0.00 GFLOP    0.00s  0.05 GFLOP/s -> PASS
MQA prefill       H= 8/1 (g8) Sq=8    Sk=8    D= 64 full   | err/scale=1.6e-03 |    0.00 GFLOP    0.00s  0.05 GFLOP/s -> PASS
odd + multiblock  H= 6/3 (g2) Sq=130  Sk=70   D= 40 causal | err/scale=1.3e-03 |    0.00 GFLOP    0.11s  0.02 GFLOP/s -> PASS
GQA big head_dim  H= 8/2 (g4) Sq=4    Sk=12   D=128 full   | err/scale=1.5e-03 |    0.00 GFLOP    0.01s  0.04 GFLOP/s -> PASS
=== ALL PASS ===
```

9/9 PASS in ~0.2 s, worst `err/scale` = 1.7e-3. `odd + multiblock`
(`Sq=130, Sk=70`) spans >1 query block and >1 key block, exercising the
cross-block rescale — and since the NaN fix (*Known limits* 2) all **780** of its
rows are genuinely compared, where 360 of them used to pass vacuously. Beyond the shipped suite, 13 extra stress shapes (odd
`D` = 1/3/7/33/65, block edges `Sq`/`Sk` = 63/64/65, `Sq=1`, `Sk<Sq` causal, MQA
g16) × 3 VLs = 39 combos all PASS.

### Tolerance: `mad ≤ TOL·maxref`, `TOL = 8e-3`

The pass bound is **measured, not guessed**. `tolerance_probe.c` swept 129 shapes
(`D` = 1…128, `Sq`/`Sk` block edges 63/64/65/127/128/129, `Sk<Sq`, GQA
g = 1/2/4/16, decode, causal + non-causal) at **VL=512 and VL=2048**:

| statistic | `err/scale` on a correct kernel |
|---|---|
| median | 1.57e-03 |
| p90 | 2.17e-03 |
| **max** | **2.76e-03** (`D=8`, non-causal) |
| the shipped 9 | ≤ 1.70e-03 |

VL=512 and VL=2048 were numerically identical except one case in the 3rd digit, so
the noise is VL-invariant and the bound needs no VL margin. Non-causal dominates
because the error is bounded by `2^-9 · max_j|V[j][d]|` (bf16 `P`) regardless of
shape, while non-causal softmax averages ~96 random `V`s so `maxref` falls to
0.12–0.24 — the denominator shrinks, the numerator doesn't.

`TOL=8e-3` is **2.9×** the global measured worst and **4.7×** the shipped suite's
worst. The margin is capped from above by sensitivity: a uniform scale error `e`
lands `err/scale ≈ e`, so any bound ≥1e-2 would fail to catch a clean ×1.01. The
two requirements are in tension and 8e-3 is the balance.

- **Sensitive:** bisected detection threshold **×1.0072 PASS / ×1.0073 FAIL →
  ~0.73%**, symmetric (×0.9925 also caught). The old `atol+rtol·|r| = 2e-2` bound
  detects only **~3.5–4%** (×1.035 PASS, ×1.04 FAIL) — a **~4.7× sensitivity
  gain**. That old form mixed in an absolute term that *dominated*: against
  outputs of O(0.1–1.0) it evaluated to an effective ~3e-2, ~30× the real ~1e-3
  noise, so a uniform 1% output error passed all 9 cases.
- **Stable:** unmodified kernel PASSes 9/9 at every VL, zero flakiness.
- **Normalised per case, deliberately.** Per-**row** normalisation was measured
  and rejected: at `D=1` a row's scale is a *single* element, and when it cancels
  to ~0 the relative error explodes to **1.159 on a correct kernel**. Per-**head**
  fails the same way more mildly (5.60e-03, from a `D=1` head whose `maxref` was
  0.116 while its absolute error was a normal 6.5e-4 — the denominator was the
  anomaly). Pooling the whole case makes the scale estimate robust.

Because the bound is scale-relative, **the printed `err/scale` *is* the pass
metric** — headroom reads straight off the output line.

### Reference rows: spread, not a prefix

The fp32 reference is O(Sk·D) per checked row, so big cases check a sample. That
sample is **spread evenly over `[0,Sq)` and always includes row `Sq-1`** — the row
that attends all `Sk` keys and therefore walks every key block and every rescale.

A prefix would be near-worthless here: for causal `Sq==Sk`, rows `0..cap-1` attend
only keys `0..cap-1` — a `cap×cap` corner of the **first** key block — so the
multi-block online rescale, the whole point of this kernel, would never be checked.
This is not theory. Same data, same tolerance, kernel deliberately broken with
`if (kj == BK) continue;` (silently drop the 2nd key block), qwen3 `Sq=Sk=512`
causal:

| reference rows | `err/scale` | verdict |
|---|---|---|
| prefix 0..7 (old) | 1.3e-03 | **PASS** — indistinguishable from a correct kernel |
| spread 0..511 (new) | **1.5e-01** | **FAIL** — 19× over the bound |

No tolerance, however tight, could have caught that: it was a *coverage* bug. And
the spread sample is **free** — the reference costs the same per row wherever the
row sits.

### Benchmark: Qwen3 prefill sweep (`./flash --long`)

`Hq=16, Hkv=8` (GQA group=2), `D=128`, causal.

**One sweep covers both Qwen3-0.6B and Qwen3-1.7B**: their attention shapes are
*identical* — 16 Q heads / 8 KV heads / head_dim 128 / 28 layers. Only
`hidden_size` differs (1024 vs 2048), and that only sizes the QKV projections; it
never reaches the attention kernel.

| Sq=Sk | useful GFLOP (causal) | QEMU s | `err/scale` | headroom vs TOL | key blocks row `Sq-1` walks | verdict |
|---|---|---|---|---|---|---|
| 1024 | 4.30 | 56.70 | 2.9e-04 | 27× | 16 | PASS |
| 2048 | 17.19 | 216.03 | 2.1e-04 | 38× | 32 | PASS |
| 4096 | 68.74 | 856.38 | 1.5e-04 | 53× | **64** | PASS |

- **GFLOP is the causal-correct *useful* count** — `4·Hq·D·#{(i,j): j≤i}`, which is
  **exactly half** (ratio 2.000, verified) of the non-causal `4·Hq·Sq·Sk·D` the old
  harness printed.
- **Timings are QEMU *functional emulation* of one core. They measure work done,
  NOT ARM hardware speed** — do not read them as ARM perf. Throughput came out at
  ~0.079 GFLOP/s across all three, and time scales 3.99× / 4.00× per doubling —
  clean O(S²), i.e. the causal block-skip holds. The benchmark binary was proven
  identical to a build of the shipped `flash.c`, so the numbers apply to the
  delivered kernel exactly.
- **Do not read the low `err/scale` as better accuracy** — it is the dilution
  described under *Known limits*. The absolute error is unchanged; the denominator
  is not.

## Known limits (`flash.c` harness)

Limit 1 is real and unfixed. Limit 2 (the NaN blind spot) **has since been fixed** — it is kept
below, with its resolution, because it is the reason the harness is now NaN-safe. Both are/were
harness limits, not kernel bugs.

**1. `err/scale` is diluted ~7× on causal spread-row checks.** For causal
`Sq==Sk`, `maxref ≈ 1.0` comes from row 0 (`O = V[0]` exactly, zero error), while
the error comes from late rows whose own scale is only ~0.13. Measured at qwen3 1k
with spread rows: per-case `err/scale = 2.92e-04` but per-row `= 2.20e-03` — a
~7.5× gap. So a *subtle* (~1%) drift confined to late rows is under-reported by
~7× at the long shapes, and would need to reach ~5–6% of the row's own magnitude
to trip `TOL`. **Structural** multi-block bugs are far above that (1.5e-01
measured above) and are caught loudly. Per-row normalisation would close the gap
but is unusable (it hits 1.159 on a *correct* kernel at `D=1`).

**2. ~~The fp32 reference NaNs on fully-masked rows, which then pass vacuously.~~
FIXED 2026-07-16 — see below.**
If `Sk < Sq` and causal, early rows attend zero keys, so `attn_ref` computed `0/0`
→ NaN. The comparison `NaN > tol` is **false**, and NaN never updates
`mad`/`maxref` either — so those rows were silently skipped, not checked. The
kernel itself is correct there (it emits 0.0, via `l=0 → inv=0`, which is
standard); it was the *reference* that was wrong.

> **This did affect the shipped suite.** `odd + multiblock`
> (`Hq=6/Hkv=3, Sq=130, Sk=70, D=40, causal`) has `Sk < Sq`, and **360 of its 780
> checked rows — 46% — were vacuous NaN passes** (rows 0..59 of each head attend no
> keys at all). Its reported `err/scale=1.3e-03` was computed from the other 54%.
> The `EXEC_LOG` recorded this limit as "absent from the shipped suite"; that was
> incorrect (now corrected in place there), and `tolerance_probe.c` reports the
> `360/780` count against the case it labels `ship:odd + multiblock`. The bug was
> pre-existing and unrelated to the SVE port, but the case was not the clean
> coverage its PASS implied.

> **RESOLVED 2026-07-16.** Two changes, both harness-side (kernel compute proved
> untouched — the `fast_exp`/`pack_vt`/`qk_tile`/`pv_acc`/`softmax_block`/`attn_flash`
> region is byte-identical, md5 `35c8e05b…`):
> 1. **`attn_ref` emits a zero row** for a fully-masked row —
>    `inv = (sum > 0) ? 1.0f/sum : 0.0f`, mirroring the kernel's `l=0 → inv=0`.
>    Cannot misfire on a real row: with ≥1 valid key the argmax term gives
>    `expf(0)=1`, so `sum ≥ 1` strictly.
> 2. **`run_case` counts non-finite elements out-of-band** (`nbad`) and fails on
>    them unconditionally: `ok = (nbad == 0) && (mad <= TOL*maxref)`. Out-of-band
>    because a NaN cannot be carried *in* `mad` — `!(ad <= mad)` would latch NaN and
>    the next finite `ad` would overwrite it. The verdict names the count
>    (`*** FAIL (N non-finite) ***`), since `err/scale` looks innocent when NaN is
>    the bug.
>
> Verified: NaN reference rows over the short suite **360/1080 → 0/1080**; the 360
> rows of `odd + multiblock` are now a real kernel-`0.0` vs ref-`0.0` check
> (**14400/14400 elements exactly zero on both sides**). `err/scale` is
> **unchanged at 1.3e-03** — masked rows contribute `|dev|=0` *and* `|ref|=0`, so
> they raise neither `mad` nor the `maxref` denominator (still `0.9883`, from the
> normal rows). Teeth proven by fault injection: one NaN forced into the kernel
> output makes the **old** harness print `ALL PASS` (exit 0) and the **new** one
> fail all 9 cases (exit 1).

## Enhancements (not yet implemented)

- ~~**Fix the NaN blind spot**: have the fp32 reference emit `0.0` for fully-masked
  rows (matching the kernel) instead of `0/0`, so `Sk<Sq` causal rows are actually
  checked. This is the highest-value item — a shipped case is currently 46%
  vacuous.~~ **Done — see *Known limits* 2.** The reference emits a zero row and the
  comparison is NaN-safe; `odd + multiblock` now checks all 780 rows (was 46%
  vacuous), `err/scale` unchanged at 1.3e-03.
- **GQA-decode full-rate BFMMLA** — applies to `attn.c` / NEON, *not* the SVE flash
  kernel: for `Sq=1`, instead of padding the query tile, pack **two query heads of
  the same group** into the 2×2 tile (they share the KV head). This fills both tile
  rows with useful work → no 50% waste. MHA (`group=1`) can't do this and falls
  back to BFDOT.
- ~~Online/streaming softmax (flash-style) to avoid materialising `S`.~~ **Done — `flash.c`.**
- ~~Fully vectorise the flash kernel (QK^T, softmax and P·V all in SVE).~~ **Done — `flash.c`.**
- Keep `P` in fp32 for the `P·V` step to cut the dominant error source — the whole
  ~1e-3 noise floor is `2^-9 · max_j|V[j][d]|` from bf16 `P`.
- Close the dilution gap: a **per-row bound with a floor tied to the case scale**
  (plain per-row normalisation is degenerate — see *Known limits*).
- Give `attn.c` the spread-row reference sample too; its `--long` prefix has the
  same coverage hole `flash.c` just fixed.
- Cache-blocking / multithreading the flash kernel; tune `BQ`/`BK` per core.
