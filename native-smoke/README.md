# Exact wide polynomial implementation

Complete coefficient-to-coefficient negacyclic multiplication at both FHERMA challenge points:

| N | W | L | RNS primes | CRT byte stride |
|---:|---:|---:|---:|---:|
| 65,536 | 1,747 | 55 | 114 | 480 |
| 32,768 | 868 | 28 | 57 | 256 |

`solve.cu` prepares prime/root/Montgomery/CRT constants from the fixed point during setup. Every measured call packs A and B separately into fixed pinned staging with at most 12 CPU workers each. Two nonblocking streams overlap the two pinned H2D transfers, unchanged unsigned-MMA RNS encoding and first forward NTT; B joins A-forwarded before the common spectral middle. It then performs cuPQC staged NTT multiplication with a Montgomery forward twist, untwists and weights residues directly into coalesced `[id][primeWord][i]` ByteCRT storage, runs exact unsigned-MMA CRT with centered correction, copies all canonical coefficient limbs to fixed pinned D2H staging, then constructs the complete returned vector directly from that range. Large disjoint host copies use at most twelve CPU workers; standard vector allocation and ownership remain intact. Input/output transfers and output construction remain inside the call.

## Arithmetic

- `rns-plan.hpp`: C++17 standard-library-only fixed setup; prime product exceeds `4*N*(q-1)^2`. Both point moduli use the specified `q=2^W-gap` form with `0<gap<2^32`.
- `rns-encode.cuh`: the validated scalar RNS encoder and shared exact Montgomery helper.
- `rns-encode-mma.cuh`: 32 coefficients / seven full warps in the current MT32 candidate, fixed byte-position powers modulo each prime, four exact unsigned byte-dot sums and one 64-bit Barrett reduction. Fourteen primes per group; padded shared rows make MMA A gathers bank-distinct.
- `io-copy.cpp`: production packs each operand in twelve disjoint libc chunks with at most twelve OpenMP workers and a completion barrier before its H2D. Original two-input pack remains available for its existing fixtures. C13 non-temporal input and C14 bulk-input measurements remain historical gates.
- `vector-copy.cpp`: GNU host-link wrappers for `memcpy` and `memmove`; disjoint copies of at least 8 MiB use twelve chunks, overlapping moves and small copies use libc. All bytes and the completion barrier stay inside the standard vector call.
- `ntt-pipeline.cuh`: coalesced first-stage input, disjoint normal intermediate stores; cuPQC 0.6 `uint32`, M256/Block128 runtime-prime NTT. The middle pass fuses two forward second stages, pointwise multiplication and the first inverse stage.
- `ntt-weighted.cuh`: final inverse with ordinary `N^-1`, followed by one Montgomery product with the ordinary `psi^-i * inv(P/p)` factor. It writes canonical residues in coalesced `[id][primeWord][i]` order; G1 restores each shared coefficient row and writes the natural output ABI.
- `g1-crt.cuh`: production MT32/seven warps with paired unsigned `m16n8k32` fragments, exact FP32 centered alpha, four q-reduced byte-weight columns per limb, 64-bit carry and the retained two-fold canonical reducer. Original MT16/B7 fixture kernels remain available.

Both warm-ups print per-stream H2D/encode/forward spans, their encode/forward sums, A-H2D-start to B-H2D-end coverage, common-kernel times, the complete GPU pipeline wall span and join wait. Host logs include A/B pack and a four-phase wall decomposition: the first field is front A pack, while H2D+product includes overlapped B pack. These spans overlap and carry their own scopes. Scored calls use the same arithmetic and A-forwarded dependency without diagnostic events.

## Four-column byte-weighted CRT

For each nonnegative term `t_j` (including alpha), prepare solely from the fixed point

```text
W[j,a] = (256^a * weight_j) mod q, a=0..3
B[4*j+a, 4*l+d] = byte_d(word_l(W[j,a])), d=0..3
z = sum(byte_a(t_j) * W[j,a])
```

Thus `z mod q` is the original CRT result and `z < 4*(P+1)*255*q < 2^(W+17)`. Each dot remains below `2^31`; `lo = C0 + 256*C1 + 256^2*C2 + 256^3*C3`, `total = carry+lo` fits u64, and `carry=total>>32`. The original two-fold reducer retains its wider W+38 envelope. MT32 handles fourteen limbs per 56-column group: primary 7→4 groups, auxiliary 4→2. This is fixed parameter preparation; every term, alpha, dot, carry, fold and output belongs to the measured call.

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
- [C13 input streaming CPU](input-stream-cpu-result.json): fallback full-word/source/guard checks and AVX512 compile/instruction gates pass. Target setup reports the selected backend; the native whole-call gate validates its actual output and performance.
- [C14 B4 byte CRT](byte-crt-cpu-result.json): all eight width/gap plans, 256 valid signed coefficients, every actual prime boundary, 537,088 B4 bytes, 42,496 exact dots and 11,136 source epilogue words agree with independent integer modulo. Original MT16/B7 fixtures remain unchanged.
- [C15 operand pipeline](operand-pipeline-cpu-result.json): complete A/B pack and source-extracted input/RNS/forward offsets match both original layouts; lifted host/stream orchestration passes both full owned output shapes and the A-before-middle dependency with zero diagnostic events in scored calls. Fixed setup locality helper passes present/key/missing branches. CUDA compile and whole-call exactness stay with the target run.
- [Host pack CPU](io-copy-cpu-result.json): complete A/B copy, untouched sources, disjoint tails and destination guards.
- [Vector copy CPU](vector-copy-cpu-result.json): all 3,604,480 output words and source/vector lifetimes, natural standard-library memcpy/memmove entry paths, byte tails and overlapping-move fallback. C8 passes both full outputs at three fresh seeds per point, with complete-call medians 2.748 ms / 1.168 ms.
- [Scalar encoding CPU](rns-encode-cpu-result.json): source-extracted arithmetic/kernel agrees with independent Python BigInt modulo, including full-width words and coefficient/prime tails.

C1–C3 diagnostic commits used the attributed upstream schoolbook for their small timed polynomial answer. The full pipeline passes [C4](native-c4-result.json) at both complete official-size outputs, one measured seed per point: 6.598 ms / 1.892 ms. The pinned-output/Montgomery-forward [C5](native-c5-result.json) also passes both complete outputs and improves whole-call latency. The tensor-RNS/pinned-input [C6](native-c6-result.json) passes 3/3 fresh seeds at each point and measures 3.797 ms / 1.267 ms. Coalesced first-NTT [C7](native-c7-result.json) passes 3/3 fresh seeds per full point: 3.733 ms / 1.146 ms complete-call medians. Forward-stage GPU time improves from 372/96 to 150/39 µs in second warm-ups. [C8 parallel vector-copy](native-c8-result.json) also passes 3/3 fresh seeds per full point: 2.748 ms / 1.168 ms complete-call medians. Primary second-warm-up output construction drops from 985 to 188 µs, while pinned H2D also changes from 876 to 647 µs. Whole-call and warm-up phases are recorded with their own scopes. [C9 write-combined input](native-c9-result.json) passes both full outputs at 3/3 seeds, with medians 2.775 ms / 1.184 ms. [C10](native-c10-result.json) passes 3/3 full seeds per point and measures 2.594 ms / 1.070 ms. Primary second-warm-up encode/inverse drop to 265/82 µs; G1 remains 291 µs. [C11](native-c11-result.json) passes both full outputs but mapped demand input reads measure 5.135 ms / 1.601 ms whole-call; explicit bulk H2D is retained for C12. FP32 alpha remains exact and reduces G1 phase to 180/45 µs. [C12](native-c12-result.json) passes 3/3 full seeds per point and measures 2.099 ms / 0.963 ms. Fixed MT32 dual-MMA tiles reuse each B load across two m16 accumulator sets. [C13](native-c13-result.json) passes 3/3 full seeds per point on the actual AVX512 NT backend, at 2.185 ms / 0.894 ms. Primary latency rises 4.1% from C12; auxiliary improves 7.2%. C14 retains the primary-winning libc input path. [Its complete B4 run](native-c14-result.json) passes 3/3 full seeds per point at 2.223 ms / 0.997 ms. Primary G1 drops from 170 to 115 µs in the second warm-up; identical libc packing changes from 254 to 632 µs and H2D from 808 to 935 µs in those two runs. Complete-call and warm-up measurements retain separate scopes.

`ntt-smoke.cu` supports standalone native oracle execution with `BYTECRT_NTT_SMOKE_STANDALONE`; `g1-smoke.cu`/`mma-smoke.cu` preserve earlier setup checks. These gates stay out of the current complete-call implementation.

## Source credit

The generated `fherma.h`/`main.cpp` and the original CMake device-LTO pattern originate from [fherma-ai/polynomial-mult-cupqc-example](https://github.com/fherma-ai/polynomial-mult-cupqc-example/tree/e157fecf6066dac35c883a0c2e29e6d47d23e4bd), fixed commit e157fecf6066dac35c883a0c2e29e6d47d23e4bd. The official example declares Apache-2.0. ByteCRT arithmetic and integration are Apache-2.0 under the repository licence. The platform rewrites the generated harness at measurement.

## Two-operand complete native result

[C15](native-c15-result.json) passes 3/3 fresh seeds per full point plus two warm-ups: **1.937 ms / 1.055 ms** complete-call medians. Primary latency improves 12.9% from C14 and 7.7% from C12; auxiliary rises 5.8% from C14. Primary second-warm-up GPU pipeline-wall is 1536.288 µs; G1/D2H/output construction are 114.688/264.276/177.618 µs. Per-stream spans overlap and encode/forward sums carry their stated scope. Fixed locality reads GPU NUMA node 0, GPU-local CPUs 0–23, and allowed CPUs 0–23; affinity and allocation policy remain unchanged.
