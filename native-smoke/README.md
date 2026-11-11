# Exact wide polynomial implementation

Complete coefficient-to-coefficient negacyclic multiplication at both FHERMA challenge points:

| N | W | L | RNS primes | CRT byte stride |
|---:|---:|---:|---:|---:|
| 65,536 | 1,747 | 55 | 114 | 480 |
| 32,768 | 868 | 28 | 57 | 256 |

`solve.cu` prepares prime/root/Montgomery/CRT constants from the fixed point during setup. Every measured call packs both inputs into fixed pinned staging with at most 12 CPU workers, copies them once to the GPU, encodes all limbs to transposed RNS planes with unsigned MMA, performs cuPQC staged NTT multiplication with a Montgomery forward twist, untwists and weights residues directly into coalesced `[id][primeWord][i]` ByteCRT storage, runs exact unsigned-MMA CRT with centered correction, copies all canonical coefficient limbs to fixed pinned D2H staging, then constructs the complete returned vector directly from that range. Large disjoint host copies use at most twelve CPU workers; standard vector allocation and ownership remain intact. Input/output transfers and output construction remain inside the call.

## Arithmetic

- `rns-plan.hpp`: C++17 standard-library-only fixed setup; prime product exceeds `4*N*(q-1)^2`. Both point moduli use the specified `q=2^W-gap` form with `0<gap<2^32`.
- `rns-encode.cuh`: the validated scalar RNS encoder and shared exact Montgomery helper.
- `rns-encode-mma.cuh`: 32 coefficients / seven full warps in the current MT32 candidate, fixed byte-position powers modulo each prime, four exact unsigned byte-dot sums and one 64-bit Barrett reduction. Fourteen primes per group; padded shared rows make MMA A gathers bank-distinct.
- `io-copy.cpp`: twelve disjoint host-copy tasks, at most twelve OpenMP workers, a completion barrier before the one bulk pinned H2D transfer.
- `vector-copy.cpp`: GNU host-link wrappers for `memcpy` and `memmove`; disjoint copies of at least 8 MiB use twelve chunks, overlapping moves and small copies use libc. All bytes and the completion barrier stay inside the standard vector call.
- `ntt-pipeline.cuh`: coalesced first-stage input, disjoint normal intermediate stores; cuPQC 0.6 `uint32`, M256/Block128 runtime-prime NTT. The middle pass fuses two forward second stages, pointwise multiplication and the first inverse stage.
- `ntt-weighted.cuh`: final inverse with ordinary `N^-1`, followed by one Montgomery product with the ordinary `psi^-i * inv(P/p)` factor. It writes canonical residues in coalesced `[id][primeWord][i]` order; G1 restores each shared coefficient row and writes the natural output ABI.
- `g1-crt.cuh`: MT16/seven warps, original unsigned `m16n8k32` layout, FP32 centered alpha with the retained CRT guard, seven-byte-sum carry, split high×gap reduction and coalesced canonical output.

Both warm-ups print per-kernel CUDA event times, pinned H2D event time and host pack/transfer/output construction times. Scored calls use the same arithmetic without phase events.

## Build

Inside the FHERMA `cupqc` image, with the target GPU visible:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/solution /path/to/official/cases
```

CMake uses the image’s GCC/OpenMP host runtime and links the external cuPQC 0.6 NTT static library with device LTO and native architecture. SDK files stay external.

## Exact gates

- [C1](native-c1-result.json): actual Blackwell unsigned MMA, all 128 integer cells.
- [C2](native-c2-result.json): full-width G1, 36 signed coefficients + 60 zero rows / 3,984 words, including cross-64-bit fold products.
- [C3](native-c3-result.json): actual sm120 cuPQC staged NTT, both full sizes × two real primes / 196,608 canonical residues, checked against an independent ordinary CPU radix-2 oracle. N16 schoolbook validates that oracle.
- [Fixed plan CPU](rns-plan-cpu-result.json): all prime/root/inverse/power words, FP64 reciprocals and B bytes agree with the independent Python model and original G1 fixtures at both widths/gap boundaries.
- [C6 MMA encoding CPU](rns-encode-mma-cpu-result.json): the earlier seven-column word-product encoding passes its original source-extracted/mock check.
- [C10 byte-power encoding CPU](rns-byte-power-cpu-result.json): four-column setup/reducer matches independent whole-coefficient BigInt modulo at all 171 actual primes; locked unsigned-MMA fragments and full per-CTA selector coverage pass.
- [C10 weighted-boundary CPU](weighted-boundary-cpu-result.json): full-size inverse/G1 layout and output permutation, source-extracted original arithmetic and full-width fixture outputs pass.
- [C11 FP32 alpha](fp32-alpha-cpu-result.json): retained `|c/P|<1/4` plus `P*gamma_(P+4)<0.001` proves exact nearest-integer alpha at both prime counts; source-extracted RN checks pass 64 valid full-wide cases.
- [C11 mapped input](mapped-input-source-result.json): fixed alias/owner semantics passed; C11 performance selected explicit bulk H2D.
- [C12 MT32](mt32-cpu-result.json): source-extracted top/bottom m16 gathers and shared B/C layout, natural/tail/transposed input indexes and valid full-wide CRT outputs pass; original MT16 fixture path remains available.
- [Host pack CPU](io-copy-cpu-result.json): complete A/B copy, untouched sources, disjoint tails and destination guards.
- [Vector copy CPU](vector-copy-cpu-result.json): all 3,604,480 output words and source/vector lifetimes, natural standard-library memcpy/memmove entry paths, byte tails and overlapping-move fallback. C8 passes both full outputs at three fresh seeds per point, with complete-call medians 2.748 ms / 1.168 ms.
- [Scalar encoding CPU](rns-encode-cpu-result.json): source-extracted arithmetic/kernel agrees with independent Python BigInt modulo, including full-width words and coefficient/prime tails.

C1–C3 diagnostic commits used the attributed upstream schoolbook for their small timed polynomial answer. The full pipeline passes [C4](native-c4-result.json) at both complete official-size outputs, one measured seed per point: 6.598 ms / 1.892 ms. The pinned-output/Montgomery-forward [C5](native-c5-result.json) also passes both complete outputs and improves whole-call latency. The tensor-RNS/pinned-input [C6](native-c6-result.json) passes 3/3 fresh seeds at each point and measures 3.797 ms / 1.267 ms. Coalesced first-NTT [C7](native-c7-result.json) passes 3/3 fresh seeds per full point: 3.733 ms / 1.146 ms complete-call medians. Forward-stage GPU time improves from 372/96 to 150/39 µs in second warm-ups. [C8 parallel vector-copy](native-c8-result.json) also passes 3/3 fresh seeds per full point: 2.748 ms / 1.168 ms complete-call medians. Primary second-warm-up output construction drops from 985 to 188 µs, while pinned H2D also changes from 876 to 647 µs. Whole-call and warm-up phases are recorded with their own scopes. [C9 write-combined input](native-c9-result.json) passes both full outputs at 3/3 seeds, with medians 2.775 ms / 1.184 ms. [C10](native-c10-result.json) passes 3/3 full seeds per point and measures 2.594 ms / 1.070 ms. Primary second-warm-up encode/inverse drop to 265/82 µs; G1 remains 291 µs. [C11](native-c11-result.json) passes both full outputs but mapped demand input reads measure 5.135 ms / 1.601 ms whole-call; explicit bulk H2D is retained for C12. FP32 alpha remains exact and reduces G1 phase to 180/45 µs. C12 tests fixed MT32 dual-MMA tiles for both encode and G1, reusing each B load across two m16 accumulator sets.

`ntt-smoke.cu` supports standalone native oracle execution with `BYTECRT_NTT_SMOKE_STANDALONE`; `g1-smoke.cu`/`mma-smoke.cu` preserve earlier setup checks. These gates stay out of the current complete-call implementation.

## Source credit

The generated `fherma.h`/`main.cpp` and the original CMake device-LTO pattern originate from [fherma-ai/polynomial-mult-cupqc-example](https://github.com/fherma-ai/polynomial-mult-cupqc-example/tree/e157fecf6066dac35c883a0c2e29e6d47d23e4bd), fixed commit e157fecf6066dac35c883a0c2e29e6d47d23e4bd. The official example declares Apache-2.0. ByteCRT arithmetic and integration are Apache-2.0 under the repository licence. The platform rewrites the generated harness at measurement.
