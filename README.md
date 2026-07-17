# bf16 Attention Kernel (MHA + GQA) — SVE BFDOT flash attention

A self-contained, **QEMU-verified** flash-attention kernel for AArch64 that computes
scaled-dot-product attention in **bfloat16** using the Arm BF16 instruction `BFDOT`,
with an fp32 reference for correctness checking.

**`flash.c` — SVE only.** Flash attention with online softmax; `svbfdot_f32` is its
single compute primitive. `arm_sve.h` is the sole ARM header: no NEON, no SME/ZA.

Built and tested on an **x86_64** host (no ARM hardware, no root) via an AArch64
cross-compiler + QEMU user-mode emulation.

## Files

| File | Purpose |
|---|---|
| `flash.c` | **SVE-only BFDOT.** Flash-attention kernel: online softmax, key-blocked with a **runtime** key-block size `bk`, O(Bq·D + Bq·bk) memory, **allocation-free** (caller-owned scratch) + harness |
| `tolerance_probe.c` | Measurement harness behind `flash.c`'s `TOL=8e-3`: prints `err/scale` under per-case / per-head / per-row normalisation and counts NaN reference rows. **Not part of the shipped kernel**; a **frozen snapshot** with its own private copy of the kernel (still `BK=64`, compile-time) — it is deliberately *not* kept in step with `flash.c`; build/run lines in its header |
| `run.sh` | Build `flash.c` (AArch64, `-march=armv8.6-a+sve+bf16`) and run it under `qemu-aarch64 -cpu max` |
| `flash_long.log` | Recorded `./flash --long` output; header cites the md5 of the `flash.c` it came from |
| `qemu_pkg/` | Locally-extracted `qemu-aarch64-static` (no root needed) |

### Harness modes

```bash
./flash                  # 9-case short suite at the default bk (128)
./flash --bk 256         # ...the same suite at any legal bk — NO REBUILD
./flash --sweep-bk       # tune bk: one shape across bk=32..1024 (see the QEMU caveat)
./flash --check-pick-bk  # self-test attn_flash_pick_bk's analytical cache model
./flash --long           # + the Qwen3 1k/2k/4k prefill sweep
```

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

This is **why** the kernel stores bf16 tensors as `uint16_t` bit patterns and
reinterprets them to `bfloat16_t*` for the intrinsics. Convert fp32 → bf16 only
via:

- `f32_to_bf16()` — the explicit bit helper, round-to-nearest-even (scalar), or
- `svcvt_bf16_x` + `svuzp1_bf16` — the vector path in `flash.c`'s hot loop.

Both are verified correct against a scalar RNE reference (all lanes, 0
mismatches; round-trip through `svbfdot` exact). Only the C cast is broken.

## Why BFDOT, not BFMMLA

Arm's two BF16 matmul instructions differ in the layout they *demand* of their
operands, and that is what picks the design here:

- **`BFDOT`** — 4 output lanes, each a 2-element bf16 dot:
  `acc[k] += a[2k]*b[2k] + a[2k+1]*b[2k+1]`. **8 MACs/instr** (128-bit form).
  GEMV-friendly: one score / one output element per reduction, and it imposes **no
  layout constraint** on either operand — it dots whatever two vectors it is handed.
- **`BFMMLA`** — a `[2×4]·[4×2] → [2×2]` matmul-accumulate: `a` holds a **2×4
  row-major** tile, `b` a **4×2 column-major** tile. **16 MACs/instr**, twice BFDOT's
  rate — but it is only full-rate given ≥2 independent rows on *both* operands, and
  it **dictates the operand layout**, so both inputs must be pre-packed into its tile
  shape.

`flash.c` takes BFDOT. The consequence is the whole reason the kernel packs almost
nothing: because BFDOT constrains no layout, the contraction axes alone decide what
must be packed — and they say **Q and K need no pack at all** (see *Why only V is
packed*). BFMMLA's 2× MAC rate would have to pay for a Q pack, a K pack, and, at
decode (`Sq=1`), a padded 2-row query tile whose second row is pure waste — the
concrete "50% double-compute" cost of forcing a GEMM instruction onto a GEMV shape.

## Flash-attention kernel (`flash.c`) — SVE-only, BFDOT

A *materialising* attention kernel keeps the full `[Sq×Sk]` score matrix — `Sq*Sk*4`
bytes, i.e. 4 / 16 / **64** MB per head at 1k / 2k / 4k — which blows past cache at
long context. `flash.c` blocks over keys with **online softmax** so `S` is never
materialised, which is the right shape for 1k–4k prefill.

**One kernel handles MHA and GQA.** The only difference is the head→kv mapping
`kv = h / group`, `group = num_q_heads / num_kv_heads`:
- MHA: `num_kv_heads == num_q_heads` → `group == 1`
- GQA: `num_kv_heads  < num_q_heads` → `group  > 1`
- MQA: `num_kv_heads == 1`

bf16 tensors are stored as `uint16_t` bit patterns (the top 16 bits of fp32) and
reinterpreted as `bfloat16_t`, so the **exact same bits** feed the hardware kernel
and the fp32 reference — the only divergence is bf16 rounding of the softmax
probabilities `P` and fp32 accumulation order.

It is **SVE only** — `arm_sve.h` is the sole ARM header, no NEON, no SME/ZA — and
`svbfdot_f32` is the single compute primitive for **both** `Q·Kᵀ` and `P·V`.

Per query-block (`BQ=64`, compile-time) it keeps only running state — max `m[Bq]`,
denominator `l[Bq]`, output accumulator `acc[Bq×D]`. Each key-block (`bk`, a **runtime**
argument, default 128 — see *Tuning `bk`*):

1. `S = scale · Q_blk·K_blkᵀ`  (BFDOT, block-local)
2. per row: `m_new = max(m, rowmax S)`, `α = exp(m−m_new)`, `P = exp(S−m_new)`
3. `l ← α·l + rowsum(P)`,  `acc ← α·acc + P·V_blk`  (BFDOT again)

Final `O = acc / l`. Causal key-blocks wholly in the future are skipped (and end
the k-loop). **Memory is O(Bq·D + Bq·Bk)** (~50–180 KB/head over `D`=1..256; see the
table below) regardless of sequence length — the right shape for long-context prefill.

### Allocation-free: the caller owns the scratch

`attn_flash()` **allocates nothing**. No `malloc`/`free`, no VLA, no large stack
array (its frame is a compile-time constant) — so it does no heap traffic per call,
cannot fail for want of memory, and works under an arena/bump allocator or a
no-malloc-in-the-hot-path policy. The caller passes one scratch block:

```c
/* Largest power-of-2 bk whose per-key-block cache footprint double-buffers into
 * l2_bytes. An ANALYTICAL model, not a measurement — see "Tuning bk" below. */
int    attn_flash_pick_bk(int D, size_t l2_bytes);

/* Scratch bytes for this head_dim AND key-block size. */
size_t attn_flash_scratch_bytes(int D, int bk);

/* bk: power of two in [ATTN_BK_MIN, ATTN_BK_MAX] = [16, 4096].
 * scratch: caller-owned, >= attn_flash_scratch_bytes(D, bk), ATTN_SCRATCH_ALIGN-aligned. */
static void attn_flash(const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O,
                       int Hq, int Hkv, int Sq, int Sk, int D, int causal, int bk,
                       void *scratch);
```

**The size depends only on `D` and `bk` — never on `Sq`/`Sk`.** That is not a
convenience, it *is* the O(Bq·D + Bq·bk) property above, so a driver allocates **one
block per thread at start-up and reuses it for every sequence length** — 1-token decode
and 4k prefill alike — and frees it at shutdown. The kernel keeps no state across calls;
it initialises everything it reads.

**`bk` is validated, loudly.** A power of two in `[16, 4096]`; anything else — including
`0` and negatives — prints the contract to `stderr` and `abort()`s, at *both* doors
(`attn_flash` and `attn_flash_scratch_bytes`). This is not pedantry: `bk` is
simultaneously the k-loop step **and** the `Vt`/`Pb`/`S` row stride, so a `bk` the caller
did not also use to *size* the block mis-strides every tile and runs off the end —
numerically invisible, exactly the silent heap overflow the single-source-of-truth layout
exists to prevent. The power-of-two rule is a **contract requirement, not an
implementation need**: every loop is `whilelt`-predicated and would take `bk=100` happily.
It is enforced anyway so a tuning loop cannot wander onto a value the contract does not
cover. The bounds' rationale: `16` is `svcnth()` on the 256-bit target (one whole bf16
vector — below it the `bfdot`/`svaddv` ratio that *motivates* `bk` drops under 1), and
`4096` is where `S[BQ×bk]` alone reaches 1 MiB and the O(Bq·bk) memory property has
stopped meaning anything.

**Alignment is a contract, not a nicety.** The block must be aligned to
`ATTN_SCRATCH_ALIGN` (**64 bytes**, a cache line). The carve mixes `uint16_t` and
`float` tiles whose extents depend on `D`, so each sub-buffer's offset is padded up
to `ATTN_SCRATCH_ALIGN` — every sub-buffer is 64-byte aligned for **any** `D`, and
SVE loads never straddle a line. Use `aligned_alloc` / `posix_memalign`, not plain
`malloc`. The returned size already includes that padding *and* is itself rounded up
to 64, so it is a legal C11 `aligned_alloc` size.

Driver usage — a **fixed** `bk`, which is what `run_case()` does:

```c
int    bk      = attn_flash_pick_bk(D, 1u << 20);   /* or your own tuned value */
size_t sbytes  = attn_flash_scratch_bytes(D, bk);   /* depends on D and bk only */
void  *scratch = aligned_alloc(ATTN_SCRATCH_ALIGN, sbytes);   /* once, per thread */

for (each request) /* any Sq, any Sk, any causal — same block */
    attn_flash(Q, K, V, O, Hq, Hkv, Sq, Sk, D, causal, bk, scratch);

free(scratch);                                      /* at shutdown */
```

#### A driver that **sweeps** `bk` must size for the LARGEST `bk` it will try

This is the one obligation runtime `bk` adds, and getting it wrong is a silent heap
overflow. Three of the eight sub-buffers (`Vt`, `Pb`, `S`) scale with `bk`, so a block
cut for `bk=64` handed to a `bk=512` call scribbles past the end — and nothing in the
carve re-checks the caller's size. **Size once, for the max, then reuse that one block:**

```c
static const int bks[] = {64, 128, 256, 512, 1024};
int bk_max = bks[sizeof bks / sizeof bks[0] - 1];             /* the LARGEST, not the first */

size_t sbytes  = attn_flash_scratch_bytes(D, bk_max);         /* ONE allocation ... */
void  *scratch = aligned_alloc(ATTN_SCRATCH_ALIGN, sbytes);

for (size_t i = 0; i < sizeof bks / sizeof bks[0]; i++)       /* ... reused for every bk */
    attn_flash(Q, K, V, O, Hq, Hkv, Sq, Sk, D, causal, bks[i], scratch);

free(scratch);
```

A larger-than-needed block is always safe: `scratch_bytes` is monotone non-decreasing in
`bk` (checked over `D`=1…1024 × `bk`=16…4096) and the carve is a prefix, so a smaller
`bk` simply uses less of it. `sweep_bk()` in `flash.c` (`./flash --sweep-bk`) is this
pattern, verbatim.

Both halves of that rule are **tested, not just asserted** (`Hq=4/Hkv=2, Sq=130, Sk=300,
D=128, causal`, one block, canary verified after *every* call, `bk` ascending **and**
descending):

| | result |
|---|---|
| size for the **largest** `bk`, reuse for all (the rule) | **10/10 PASS, 0 canary breaches**, `err/scale` 1.5e-3…1.6e-3 across 10→1 key blocks |
| size for the **smallest** `bk`, reuse for all (the bug) | **SIGSEGV** |

Note *how* the bug failed: at `D=128` the block would be 54 528 B while `bk=512` needs
361 728 — a 300 KB overrun, far past any canary and off the end of the heap, so it
segfaulted rather than corrupting quietly. A smaller mismatch would not be so kind; it
would just silently scribble. Hence the rule.

**One source of truth.** `attn_flash_layout(base, D, bk, s)` both measures
(`base == NULL`) and carves; `attn_flash_scratch_bytes()` is literally its measuring
pass, and `attn_flash()` calls it to carve. The size query and the carve therefore
cannot drift apart — a hand-written size sum kept beside a separate carve is how you get
a silent heap overflow. Do not add a second copy of those sizes. Making `bk` runtime does
not weaken this: `bk` is a *parameter* of the one layout function, so measure and carve
still agree by construction **for a given `bk`** — it is now the caller's job to measure
with the same `bk` it later passes in.

Sizes grow linearly in `D` and in `bk` (measured, `BQ=64`):

| `attn_flash_scratch_bytes(D, bk)` | `D`=1 | 40 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| `bk`=32 | 13 440 | 26 048 | 33 792 | 54 528 | 96 000 |
| `bk`=64 | 25 792 | 40 896 | 50 176 | 75 008 | 124 672 |
| **`bk`=128** *(default)* | **50 496** | **70 592** | **82 944** | **115 968** | **182 016** |
| `bk`=256 | 99 904 | 129 984 | 148 480 | 197 888 | 296 704 |
| `bk`=512 | 198 720 | 248 768 | 279 552 | 361 728 | 526 080 |

Only the three `bk`-sized tiles move — `Vt[D×bk]`, `Pb[BQ×bk]`, `S[BQ×bk]`; `acc[BQ×D]`,
`pv[D]` and `m`/`l`/`al[BQ]` are constant in `bk`. Ignoring alignment padding the whole
carve is

```
scratch_bytes(D, bk) = 2·bk·(D + 3·BQ)  +  (4·BQ·D + 4·D + 12·BQ)
                       \___bk-dep____/     \________fixed_______/
```

which reproduces every column above exactly (only `D=1` differs, by the ≤60 B of padding).
So each doubling of `bk` costs `2·bk·(D+3·BQ)` more: at `D=128`, 64→128 is `+16 KiB`
(`Vt`) `+8 KiB` (`Pb`) `+16 KiB` (`S`) = **exactly +40 KiB**, 75 008 → **115 968 B**.

> Note this is **not** `attn_flash_pick_bk`'s formula, and the two are not meant to
> agree — see *Tuning `bk`* below. This one counts the scratch the kernel actually
> allocates; that one counts a cache footprint including streamed `K`/`V`/`Q` blocks
> that are never copied into scratch at all.

> **Analytical note, not a measurement.** At `D=128` the scratch crosses a 64 KiB L1D
> between `bk=32` and `bk=64`:
>
> | `bk` at `D=128` | 16 | 32 | 64 | 128 | 256 |
> |---|---|---|---|---|---|
> | scratch | 43.25 KiB | 53.25 KiB | 73.25 KiB | 113.25 KiB | 193.25 KiB |
> | fits 64 KiB L1D? | yes | yes | **no** | **no** | **no** |
>
> So the shipped default (`bk=128`) does not fit an L1D, and neither did the old `BK=64`
> — that line was already crossed before `bk` became tunable. And this is scratch
> *alone*, before the streamed K/V/Q blocks. This is why `pick_bk` targets **L2, not
> L1D**: at any `bk` big enough to be worth using, L1D residency is not on the table, so
> the interesting question is whether the working set stays in L2. All arithmetic:
> **QEMU models no cache**, so nothing in this repo measures the consequence, and no
> performance claim either way is made from it.

Alignment padding costs ≤ 60 bytes total (worst case `D=1`, 0.12%); at `D`=64/128/256
every extent is already a multiple of 64 and the padding is exactly **0**.

**Re-verified for runtime `bk`** over `D` ∈ {1,2,3,7,8,33,40,63,64,65,127,128,129,256} ×
`bk` ∈ {32,64,128,256,512} — **70/70 OK**: all 8 sub-buffers in bounds, 64-byte aligned,
pairwise non-overlapping, measure == carve, and the reported size equals the carve's
high-water mark exactly (**zero slack at every (D, bk)**). Each of the 70 also ran a real
`attn_flash` call (`Hq=6/Hkv=3, Sq=130, Sk=70, causal` — 3 query blocks, `Sk<Sq`) with the
scratch **pre-poisoned to `0xFF`** (NaN in every f32/bf16 lane) and a **256-byte `0x5A`
canary** past the declared size: **70/70 PASS, 0 breaches**, and `err/scale` is
**bk-invariant per D** (e.g. 1.0e-3 at `D=1` and 1.4e-3 at `D=256` at *all five* `bk`).

Both guards were **fault-injected to prove they have teeth**:

| injected fault | canary | NaN-poison counter | suite verdict |
|---|---|---|---|
| 1-byte write at `scratch_bytes(D,bk)` | **70/70 BREACH** | quiet | still `err/scale=1.4e-3` — *numerically invisible* |
| `acc` zero-init deleted | quiet | **70/70 caught** (`nbad=780`) | `*** FAIL (non-finite) ***` |
| `acc` zero-init deleted **and poison disabled** | quiet | quiet | **ALL OK** ← the negative control |

The first row is the entire reason the canary exists: a heap overflow does not move the
numbers. The third is the control that proves it is the *poison* catching the second row
and not some unrelated check.

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

**And that price is what makes `bk` worth tuning.** With no packed K, each score/output
element ends in exactly one `svaddv`, so the useful ratio is

```
bfdots per svaddv = (contraction length) / svcnth()
```

`P·V` contracts over **`bk`** — so `bk` *is* that contraction length. Targeting
**256-bit SVE** (Neoverse V1 / Graviton3 class), `svcnth()` = 16, giving

| `bk` | bfdot per `svaddv` at VL=256 |
|---|---|
| 64 | 4 |
| **128** | **8** |
| 256 | 16 |

`BQ` does not enter the ratio — it contracts over `D`, not `bk` — so it stays a
compile-time **64**.

### Tuning `bk` — it is a **runtime** parameter

`bk` is the one knob whose best value is a property of the **machine**, not of the code:
it trades against `svcnth()` (a runtime VL) and against L2 residency (a per-part cache
size). Neither is knowable at compile time, so `bk` is an **argument to `attn_flash()`**
and a driver retunes it per deployment **without rebuilding**. `BQ` stays compile-time
for exactly the inverse reason.

Correctness is independent of `bk`: every extent is a real extent and every tail is
predicated. Verified **9/9 × `bk` ∈ {32,64,128,256,512} × VL ∈ {128,256,512,1024,2048}
= 25/25 ALL PASS**, plus a multi-key-block shape (`Sk=1024`, i.e. 32 → 1 key blocks) at
**6 `bk` × 5 VL = 30/30 PASS**, `err/scale` 1.4e-3…1.7e-3 throughout.

#### `attn_flash_pick_bk(D, l2_bytes)` — the sizing model

**Criterion: the whole per-key-block working set, double-buffered, must fit in L2** —
`2 · working_set(D, bk) ≤ l2_bytes` — take the largest power-of-2 `bk` satisfying it. The
×2 is the double-buffering allowance: while the loop computes on key-block `kj`, the next
block's K/V are expected to be arriving, so budget two of everything.

The working set counts **both** the scratch **and the streamed tensor blocks the loop
touches**. K/V/Q are read straight from the caller's tensors — no pack buffer — but they
still occupy cache, so a scratch-only budget would understate the `bk`-dependent term and
pick a `bk` that thrashes. The streamed K+V blocks alone are `4·D·bk` of the `6·bk·(D+BQ)`
term — **44% at `D=128`**, rising toward ⅔ as `D` grows (33% at `D`=64, 53% at `D`=256):

| term | bytes | `bk`-dependent? |
|---|---|---|
| `Vt` packed Vᵀ (scratch) | `2·D·bk` | yes |
| `Pb` probs bf16 (scratch) | `2·BQ·bk` | yes |
| `S` scores fp32 (scratch) | `4·BQ·bk` | yes |
| K block (streamed in place) | `2·D·bk` | yes |
| V block (streamed in place) | `2·D·bk` | yes |
| `acc` (scratch) | `4·BQ·D` | no |
| Q block (streamed in place) | `2·BQ·D` | no |
| `pv` (scratch) | `4·D` | no |
| `m`,`l`,`al` (scratch) | `12·BQ` | no |

```
working_set(D, bk) = 6·bk·(D + BQ)  +  (6·BQ·D + 4·D + 12·BQ)
                     \__bk-dep____/     \________fixed_______/
```

Worked example — `D=128`, `BQ=64`, `l2` = 1 MiB:

```
bk-dep = 6·(128+64) = 1152 B per unit of bk      fixed = 6·64·128 + 4·128 + 12·64 = 50 432 B
2·(1152·bk + 50 432) ≤ 1 048 576  →  bk ≤ 411.3  →  largest power of two = 256
```

`./flash --check-pick-bk` reproduces this numerically and asserts that `512` would violate
the bound (**PICK_BK OK, 0 checks failed**). What it picks:

| | 1 MiB L2 | 2 MiB | 4 MiB | 8 MiB |
|---|---|---|---|---|
| `pick_bk(128, ·)` | **256** | 512 | 1024 | 2048 |

| | `D`=64 | 128 | 256 | 512 |
|---|---|---|---|---|
| `pick_bk(·, 1 MiB)` | 512 | **256** | 128 | 64 |

#### `pick_bk` and `scratch_bytes` are **not** the same model — deliberately

They answer different questions and **will not agree**. Do not derive one from the other:

- **`scratch_bytes(D, bk)`** — the **exact allocation** the kernel needs. Scratch
  sub-buffers only, plus 64 B inter-buffer padding. **Size with this one, always.**
- **`pick_bk(D, l2)`** — a **cache-footprint heuristic**. Counts streamed K/V/Q blocks
  that are never allocated here at all. It is an **analytical model, not a measurement**.

#### To change the model

Edit **only `attn_flash_pick_bk()`** — the formula is two lines and nothing else in the
kernel reads them. A different L2? Pass a different `l2_bytes` (it is the caller's number,
not baked in). Driver pre-packs K/V elsewhere? Drop their `4·D·bk` from `per_bk`. Single-
buffering? Change the `2` in the bound.

Edge cases, all deliberate and tested by `--check-pick-bk`: the working set is monotone in
`bk`, so the first failing `bk` ends the scan; if even `ATTN_BK_MIN` busts the bound (tiny
L2, huge `D`) it returns **`ATTN_BK_MIN`** — never `0`, which no caller could use, and
never a loop that cannot terminate; the result is always a power of two in `[16, 4096]`;
`l2_bytes` = `0` yields the floor and `SIZE_MAX` the cap; `D<1` is clamped, not trusted.

> **A bug found here by review, and worth the warning.** `pick_bk` originally wrote
> `(size_t)(D + BQ)`, which computes `D + BQ` in **`int`** and only then widens — signed
> overflow (UB) for `D ≥ INT_MAX−64`. It did not just trip a sanitizer: the wrap made the
> per-`bk` coefficient negative, **inverting the bound test**, so `pick_bk` returned a
> *too-large* `bk` (`pick_bk(INT_MAX, 2e12)` → 64 where the model says 16). Fixed by
> widening before the add; now verified against an exact 128-bit model over
> `D` ∈ {1…INT_MAX} × `l2` ∈ {0…SIZE_MAX} — **60 combos, 0 mismatches**. If you edit the
> formula, keep every `D` cast to `size_t` *before* it enters an arithmetic expression.

**Known gap (pre-existing, not fixed):** `attn_flash_scratch_bytes` and the carve do
**not** validate `D`. `attn_flash_scratch_bytes(-1, 128)` returns a plausible-looking
49 408 while the carve places sub-buffers *below* the block — silent corruption. `bk` is
validated; `D` is not. This predates the runtime-`bk` change (`(size_t)D * BK * 2` wrapped
identically) and is left as a deliberate follow-up decision, since adding a `D` contract
would change the semantics of a shipped size query.

#### `./flash --sweep-bk` — the tuning tool

Runs one fixed shape across `bk` = 32…1024 and reports correctness + timing per `bk`,
**from one binary, no rebuild**. It is also the reference *size-for-the-largest-`bk`*
driver pattern.

> ### ⚠ QEMU cannot validate `bk`, and this repo therefore does not
>
> `pick_bk`'s entire rationale is **cache residency**. QEMU is a **functional emulator**:
> **no cache model, no memory latency, no pipeline**, and this build has **no TCG
> plugin**. Its wall-clock is at best a proxy for **instructions executed**. So **nothing
> in this repo is evidence that one `bk` beats another on silicon**, and no such claim is
> made anywhere in it. `--sweep-bk` is built to be run **on real hardware** (Neoverse V1 /
> Graviton3 class, 1 MiB L2). Under QEMU, read the PASS column only. The sweep prints this
> caveat in its own output header.

QEMU sweep at VL=512 (`Hq=4/Hkv=2, Sq=128, Sk=1024, D=128`, non-causal) —
**an instruction-count observation only, NOT a hardware result**:

| `bk` | key blocks | scratch B | `err/scale` | QEMU s |
|---|---|---|---|---|
| 32 | 32 | 54 528 | 1.5e-03 | 3.64 |
| 64 | 16 | 75 008 | 1.4e-03 | 3.59 |
| 128 | 8 | 115 968 | 1.5e-03 | 3.67 |
| 256 | 4 | 197 888 | 1.5e-03 | 3.70 |
| 512 | 2 | 361 728 | 1.7e-03 | 3.56 |
| 1024 | 1 | 689 408 | 1.5e-03 | 3.47 |

All PASS. The times span ~6% with **no interpretable trend** — which is the expected
result, not a disappointment: the shape does identical arithmetic at every `bk` (that is
why the sweep shape is **non-causal** — under a causal mask the block-skip granularity is
`bk` itself, so total work would change with `bk` and confound the timing), and the one
effect `bk` is *supposed* to have is invisible to an emulator with no cache. **The
cache-driven part of `pick_bk` is unmeasurable here. Full stop.**

### Predication replaced all padding

A NEON/BFMMLA design must zero-pad its tiles (query rows to a multiple of 2, key and
dim extents to a multiple of 8 — the tile shapes BFMMLA dictates) and `-inf`-mask the
padded key columns so they cannot leak into softmax. SVE needs none of it:

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
reporting the expected lane counts, worst `err/scale` = 1.7e-3 at *every* width —
byte-for-byte the same nine values at all five widths.

VL=2048 remains the strong case at the default `bk=128`: `svcnth()`=128 means a **whole
key block is one predicated vector** (and it still exceeds the shipped `D`=64), so the
`P·V` k-loop runs a single, fully-predicated iteration. A padded design would break here;
this one needs no padding because the `whilelt` bound *is* the block extent. Runtime `bk`
widens this: at VL=2048, `bk` ∈ {32,64} makes the k-loop a *partly-masked* single vector
(`bk` < `svcnth()`), which is the `ATTN_BK_MIN` rationale made concrete — still correct
(verified 9/9 at every combination), just wasteful of lanes.

### Verification (`./flash`)

```
=== FLASH bf16 attention (online softmax, SVE BFDOT vs fp32 ref) ===
SVE VL: svcntw()=16 f32 lanes, svcnth()=32 bf16 lanes (runtime, not hardcoded)
bk=128 (RUNTIME, --bk N to change; scratch 115968 B at D=128). pick_bk(128, 1 MiB L2)=256
MHA prefill       H= 4/4 (g1) Sq=8    Sk=8    D= 64 bk=128  full   | err/scale=1.4e-03 |    0.00 GFLOP    0.00s  0.03 GFLOP/s -> PASS
MHA prefil.causal H= 4/4 (g1) Sq=8    Sk=8    D= 64 bk=128  causal | err/scale=9.4e-04 |    0.00 GFLOP    0.00s  0.03 GFLOP/s -> PASS
GQA prefill       H= 8/2 (g4) Sq=8    Sk=8    D= 64 bk=128  full   | err/scale=1.5e-03 |    0.00 GFLOP    0.00s  0.05 GFLOP/s -> PASS
GQA prefil.causal H= 8/2 (g4) Sq=8    Sk=8    D= 64 bk=128  causal | err/scale=1.3e-03 |    0.00 GFLOP    0.00s  0.02 GFLOP/s -> PASS
MHA decode(Sq=1)  H= 4/4 (g1) Sq=1    Sk=16   D= 64 bk=128  full   | err/scale=1.1e-03 |    0.00 GFLOP    0.00s  0.04 GFLOP/s -> PASS
GQA decode(Sq=1)  H= 8/2 (g4) Sq=1    Sk=16   D= 64 bk=128  full   | err/scale=1.7e-03 |    0.00 GFLOP    0.00s  0.05 GFLOP/s -> PASS
MQA prefill       H= 8/1 (g8) Sq=8    Sk=8    D= 64 bk=128  full   | err/scale=1.6e-03 |    0.00 GFLOP    0.00s  0.04 GFLOP/s -> PASS
odd + multiblock  H= 6/3 (g2) Sq=130  Sk=70   D= 40 bk=128  causal | err/scale=1.3e-03 |    0.00 GFLOP    0.12s  0.02 GFLOP/s -> PASS
GQA big head_dim  H= 8/2 (g4) Sq=4    Sk=12   D=128 bk=128  full   | err/scale=1.5e-03 |    0.00 GFLOP    0.00s  0.05 GFLOP/s -> PASS
=== ALL PASS ===
```

9/9 PASS in ~0.2 s, worst `err/scale` = 1.7e-3. `./flash --bk N` reruns the same suite at
any legal `bk`, no rebuild: **25/25 ALL PASS** over `bk` ∈ {32,64,128,256,512} × VL ∈
{128,256,512,1024,2048}, with the nine `err/scale` values **identical in all 25**.

> **Read that matrix honestly — it is weaker than it looks.** The shipped suite's `Sk` are
> 8, 8, 8, 8, 16, 16, 8, **70**, 12 — so for **8 of the 9 cases every `bk` ≥ 32 is a single
> key block**, and sweeping `bk` re-blocks *nothing*. Only `odd + multiblock` (`Sk=70`)
> genuinely re-blocks, and only at `bk` ∈ {32,64} (3 and 2 key blocks); at `bk` ≥ 128 it is
> one partial block. That the 25 runs agree bit-for-bit is therefore mostly a statement
> that identical work gives identical answers.
>
> The real bk-invariance evidence is elsewhere, on shapes where `bk` actually re-blocks:
> - **`--sweep-bk`** (`Sk=1024` → **32/16/8/4/2/1** key blocks) × 5 VLs = **30/30 PASS**,
>   `err/scale` 1.4e-3…1.7e-3 — bk-invariant across a 32× change in blocking.
> - **Tail/boundary probe**: for each `bk`, `Sk` ∈ {1, `bk`−1, `bk`, `bk`+1, 2`bk`,
>   2`bk`+1, 70} × {causal, full} = **70/70 PASS, 0 canary breaches**, worst `err/scale`
>   3.1e-3 vs `TOL`=8e-3. This covers `Sk = bk` exactly (no tail at all — where an
>   off-by-one in a `whilelt` bound hides) and **`bk` > `Sk`** (e.g. `bk=512, Sk=70`), the
>   pure partial-block path.
> - **`--long`** — qwen3 1k/2k/4k = 8/16/32 key blocks at `bk=128`.
>
> Restoring multi-key-block coverage to the *short* suite would need a case with `Sk > 128`;
> still deliberately not added, since the shipped case list remains out of scope.

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

At the default `bk=128` (`./flash --long --bk N` reruns it at any other):

| Sq=Sk | useful GFLOP (causal) | QEMU s | `err/scale` | headroom vs TOL | key blocks row `Sq-1` walks | verdict |
|---|---|---|---|---|---|---|
| 1024 | 4.30 | 59.07 | 2.9e-04 | 28× | 8 | PASS |
| 2048 | 17.19 | 229.26 | 1.8e-04 | 44× | 16 | PASS |
| 4096 | 68.74 | 871.55 | 1.5e-04 | 53× | **32** | PASS |

- **GFLOP is the causal-correct *useful* count** — `4·Hq·D·#{(i,j): j≤i}`, which is
  **exactly half** (ratio 2.000, verified) of the non-causal `4·Hq·Sq·Sk·D` the old
  harness printed.
- **Timings are QEMU *functional emulation* of one core. They measure work done,
  NOT ARM hardware speed** — do not read them as ARM perf. Throughput is
  0.073 / 0.075 / 0.079 GFLOP/s, and **useful GFLOP scales 4.00× / 4.00×** per doubling —
  clean O(S²), i.e. the causal block-skip holds. Wall-clock scales slightly *sub*-quadratically
  (3.88× / 3.80×), which is a QEMU/host artifact (per-run fixed overhead amortising, host
  noise), not a kernel property — and precisely the sort of thing not to read anything
  into. `flash_long.log` records this run; its header cites the **md5 of the `flash.c` it
  was generated from**, verified to match the shipped file.
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
- ~~Online/streaming softmax (flash-style) to avoid materialising `S`.~~ **Done — `flash.c`.**
- ~~Fully vectorise the flash kernel (QK^T, softmax and P·V all in SVE).~~ **Done — `flash.c`.**
- Keep `P` in fp32 for the `P·V` step to cut the dominant error source — the whole
  ~1e-3 noise floor is `2^-9 · max_j|V[j][d]|` from bf16 `P`.
- Close the dilution gap: a **per-row bound with a floor tied to the case scale**
  (plain per-row normalisation is degenerate — see *Known limits*).
- ~~Make `BK` tunable without a rebuild.~~ **Done** — `bk` is a runtime argument, with
  `attn_flash_pick_bk()` and `./flash --sweep-bk`. See *Tuning `bk`*.
- **Run `--sweep-bk` on real hardware and pick `bk` from data.** This is the obvious next
  step and the one thing this repo *cannot* do: `pick_bk`'s model is analytical and QEMU
  models no cache, so `bk`'s real optimum is still unmeasured. Needs a Neoverse V1 /
  Graviton3-class part.
- Multithreading the flash kernel; tune `bk` per core. The scratch contract is already
  the right shape for this — the kernel allocates nothing and touches only the caller's
  block, so per-thread parallelism needs one `attn_flash_scratch_bytes(D, bk)` block per
  thread and no other change. `bk` being runtime means threads on heterogeneous cores
  (big.LITTLE, differing L2) can each use their own without a second build.
- Add a short-suite case with `Sk > 128` to restore multi-key-block coverage to `./flash`
  itself (see the coverage note under *Verification*).
