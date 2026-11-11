# ByteCRT

An exact unsigned-byte matrix model for a fused, multi-limb GPU CRT epilogue.

**Official complete result:** the first linked FHERMA challenge run passes **40/40 exact**, at **20 fresh seeds per point**, on RTX PRO 6000 Blackwell. Complete-call medians are **1.927 ms / 0.938 ms** ([official C15 result](native-smoke/native-official-c15-result.json)); source is pinned at [the first-official-20seed tag](https://github.com/blucca/fherma-bytecrt/tree/first-official-20seed). Every wide-coefficient/RNS conversion, transfer, dependency and standard-owned output construction remains inside the call. The current candidate adds chunked output overlap, with the same exact arithmetic; its CPU/source checks are complete and its native target gate follows.

```sh
python3 -B reference.py
```

Python standard library; the command prints the checked dimensions and evidence scope.

## The construction

For distinct RNS primes, let `P = product(p_j)`, `P_j = P/p_j`, and `t_j = (c mod p_j) * inverse(P_j mod p_j) mod p_j`. The retained product guard gives `|c/P| < 1/4`, so the centered correction is `alpha = round(sum(t_j/p_j))`. The output modulo `q` is

```text
z = sum(t_j * (P_j mod q)) + alpha * (q - (P mod q))
c_mod_q = z mod q
```

Treat the correction as one extra matrix row. Expand each nonnegative 32-bit term and each 32-bit weight word into four unsigned bytes. For output limb `l` and byte position `d` in 0..6:

```text
A[coefficient, 4*j+a] = byte_a(term_j)
B[4*j+a, 7*l+d] = byte_(d-a)(weight_word[j,l])  when 0 <= d-a < 4
C = A * B
```

The inner axis is padded to 32 for an intended `u8 × u8 → s32` MMA. Every dot product is bounded by `4*(prime_count+1)*255² < 2³¹`. The CPU checker verifies the wide integer product reconstructed from all seven byte sums against an independent Python integer sum.

For each output limb:

```text
lo = C0 + 2^8*C1 + 2^16*C2 + 2^24*C3
hi = C4 + 2^8*C5 + 2^16*C6
digit = (carry + lo) mod 2^32
next_carry = floor((carry + lo)/2^32) + hi
```

The epilogue can retain these values in registers/shared memory, reduce the near-power-of-two modulus, and write only canonical output limbs. The intended kernel fuses the epilogue with the MMA; a full global `C` allocation belongs to the CPU reference.

## Checked scope

| Geometry | RNS primes with retained guard | Padded inner axis | Byte columns | Coefficient fixtures |
|---|---:|---:|---:|---:|
| N=65,536, W=1,747, L=55 | 114 | 480 | 385 | 18 |
| N=32,768, W=868, L=28 | 57 | 256 | 196 | 18 |

Fixtures cover zero, ±1, both signed coefficient bounds and seeded random coefficients, with three synthetic moduli per geometry. Modulus primality is immaterial for this coefficient-wise CRT/modular identity. The full convolution and official prime moduli have a separate native validation step. `reference-result.json` records the run.

The prime product preserves two centered-CRT guard bits. Relative to a conservative 118/59-prime plan, this removes 3.39% of prime rows while retaining the same padded MMA inner dimensions. Measured savings need the full implementation.

## Native work

The complete implementation combines one fused MMA/CRT kernel with a coalesced output handoff and an exact RNS NTT under the [FHERMA challenge's](https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025) complete timing boundary. Its competitive family already includes RNS NTT and tensor-core CRT; improvement must be measured on the target runner.

The benchmark reference is the [official cuPQC example](https://github.com/fherma-ai/polynomial-mult-cupqc-example), with the generated host interface. Original reference code: Apache-2.0.

### Locked unsigned MMA fragments

[`fragment_check.py`](fragment_check.py) implements the unsigned `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32` register contract from [CUDA 12.8.1 / PTX ISA 8.7, §9.7.14.5.10](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832).

For `g=lane/4`, `k=4*(lane%4)`, the four A registers start at `(g,k)`, `(g+8,k)`, `(g,k+16)`, `(g+8,k+16)`. Each packs four consecutive unsigned bytes. B uses two registers in column `g`, starting at rows `k` and `k+16`; C/D hold two columns in row `g`, then the same columns in row `g+8`.

```sh
python3 -B fragment_check.py
```

The CPU check covers every A/B/C cell exactly once (512/256/128), independently inverts the register coordinates, reconstructs byte-packed operands and checks all 128 integer `A·B+C` outputs against a direct matrix oracle. Fixtures include bytes above 127. [`fragment-result.json`](fragment-result.json) records the result; native GPU execution passes all 128 cells in the C1 result below.

The C1 diagnostic commit executed one full warp on fixed matrices during setup, with a host oracle checking all 128 cells. The attributed upstream cuPQC schoolbook supplied the small-polynomial interface answer for that gate. On the target RTX PRO 6000 Blackwell Server GPU (CUDA 12.8.61 / GCC 13.3), all 128 native integer cells passed; the official N1024/W128 upstream polynomial answer passed 1/1 measured cases after two warm-ups. [`native-c1-result.json`](native-smoke/native-c1-result.json) records this stage. The current full pipeline replaces schoolbook with RNS encoding, staged NTT and fused ByteCRT.

## Host output construction probe

The generated official interface returns `std::vector<uint32_t>`, and only `fherma_run` is timed. A CPU-only probe compares `resize` followed by an opaque overwrite with direct range construction from an already available staging buffer. Both produce identical complete output vectors.

```sh
mkdir -p temp
g++ -std=c++17 -O3 -fno-lto host-output-check.cpp -o temp/host-check
./temp/host-check
```

GCC 16.2.1, Linux x86_64: 20 interleaved samples per method after 3 warm-ups. Local medians were 1.468 ms / 0.864 ms for N65536/L55 and 0.160 ms / 0.096 ms for N32768/L28. The overwrite is CPU memcpy; these measurements cover vector construction. GPU DMA, pinned staging and official whole-call measurements have their own target-run validation.

A concrete native experiment is fixed pinned D2H staging plus range construction, with the entire DMA/copy/construction inside the timer. Its acceptance is unchanged exact output and measured complete-call improvement.

## Native full-width fused CRT

G1 passes all six fixed setup fixtures on the RTX PRO 6000 Blackwell Server GPU: 36 signed full-width coefficients plus 60 zero rows, 3,984 canonical output words checked. Both FHERMA widths are covered, with FP64 centered alpha, unsigned MMA, limb carry and near-power reduction; UINT32_MAX gap exercises products beyond 64 bits. [Native G1 result](native-smoke/native-c2-result.json).

The complete polynomial pipeline now uses cuPQC staged RNS NTT, tiled wide-coefficient encoding and fused inverse/CRT-weighted packing. Its first whole-call native measurement passes both complete outputs; the next measured iteration uses pinned output staging and a Montgomery forward twist.

### Runtime-prime staged NTT gate

[Native NTT integration](native-smoke/ntt-pipeline.cuh) uses cuPQC 0.6 M256/Block128 at both challenge sizes. A fused middle stage keeps both forward spectral tiles and the product in shared memory through the first inverse stage. [The fixed setup oracle](native-smoke/ntt-smoke.cu) compares all 196,608 residues at two real RNS primes against an independent CPU radix-2 product, whose small-size result is checked against schoolbook. Actual sm120 compilation/execution passes all 196,608 residues on the target. [C3 native result](native-smoke/native-c3-result.json). The G1 gate remains coefficient-level, and C3 covers RNS negacyclic convolution before complete wide-coefficient handoff. The SDK remains an external dependency.

## Complete-call iteration

The initial [C4 native run](native-smoke/native-c4-result.json) passes both complete polynomial outputs at the actual prime moduli, including all 3,604,480 / 917,504 output words. The single measured seeds cost 6.598 ms / 1.892 ms. The first warm-up separates GPU kernels from host transfers and output allocation; its phase values have that warm-up scope.

The current iteration replaces a runtime division in the forward twist with an exact Montgomery product. It also uses fixed pinned D2H staging and direct output-vector range construction. Complete transfers, construction and arithmetic remain timed. C5 passes both complete outputs; pinned D2H takes 265 µs / 76 µs in the first warm-up, while forward-stage timing stays about 376 µs / 101 µs. The next native method applies exact unsigned-MMA input RNS encoding and fixed pinned input packing with bounded CPU parallel copies.

## Tensor RNS input encoding

[`rns-encode-mma.cuh`](native-smoke/rns-encode-mma.cuh) reuses each 16-coefficient byte tile across all primes. Seven unsigned byte-dot sums reconstruct the full ≤69-bit ordinary power-weighted sum. A fixed 64-bit Barrett reciprocal plus one Montgomery high-word product gives the exact canonical residue. The [C6 native run](native-smoke/native-c6-result.json) passes both full outputs at three fresh seeds per point, with complete-call medians 3.797 ms / 1.267 ms. Source-extracted CPU checks match independent whole-coefficient Python modulo, including full-u32 words and tails. Input/output copies remain within the complete-call timer.

## Coalesced first NTT

The next complete native method writes RNS inputs directly in staged `[prime][id][i]` order and loads the first NTT’s input/forward-twist powers contiguously. SDK stage-local ids, arithmetic and normal intermediate stores stay unchanged. Input and normal-intermediate buffers are disjoint, adding 57 / 14.25 MiB of GPU workspace. [CPU layout proof](native-smoke/coalesced-ntt-cpu-result.json) checks every index/shared-value identity at both sizes and a small independent full convolution. [C7 native](native-smoke/native-c7-result.json) passes 3/3 fresh seeds at both full points: 3.733 ms / 1.146 ms whole-call. Second warm-up forward NTT drops from 372 to 150 µs at the primary point and from 96 to 39 µs at the auxiliary point. Primary pinned H2D rose in this run; phase and whole-call effects stay separately recorded. The next measured bottleneck is the primary returned-vector range copy, about 985 µs in the second warm-up.

## Bounded parallel standard-vector copy

[C8](native-smoke/native-c8-result.json) preserves standard vector allocation and ownership, and uses GNU host-link wrappers to copy disjoint ranges of at least 8 MiB in twelve OpenMP chunks. Overlapping moves and smaller ranges stay with libc. Both full outputs pass 3/3 fresh seeds; primary whole-call median improves 26.4% from C7. The auxiliary remains on the small-copy path and measures 1.168 ms, versus C7’s 1.146 ms. [C9](native-smoke/native-c9-result.json) changes the fixed write-only input allocation to CUDA write-combined memory, passes both full outputs and measures 2.775 ms / 1.184 ms. The next method retains cached pinned input.

## Four-column byte powers and coalesced inverse/CRT

The next exact input method uses `w_k = 2^(8*k) mod p` for each input byte position. Four unsigned byte-dot sums reconstruct `x = sum(inputByte[k]*w_k) < 2^47`; one 64-bit Barrett reduction returns the canonical wide coefficient modulo p. This reduces MMA groups from 15 to 9 at the primary point and from 8 to 5 at the auxiliary point. [Source-extracted CPU proof](native-smoke/rns-byte-power-cpu-result.json) covers all actual primes and independent full-wide modulo.

The final NTT now writes weighted residues contiguously in `[id][primeWord][i]` order. G1 restores the shared coefficient rows and writes the original natural output. Sixteen extra shared bytes per row make the unsigned-MMA A gathers bank-distinct at both widths. [Full-size CPU boundary/fixture proof](native-smoke/weighted-boundary-cpu-result.json) records the exact source scope. [C10 target](native-smoke/native-c10-result.json) passes both full outputs at 3/3 seeds and measures 2.594 ms / 1.070 ms. Primary encode/inverse phase times are 265/82 µs, versus C8’s 389/195 µs.

## Guarded FP32 alpha and mapped input

The retained centered-CRT bound gives `|c/product| < 1/4`. At most 114 terms make the FP32 round-to-nearest error less than `P*gamma_(P+4) < 0.001`, so the nearest integer alpha is exact. C11 uses fixed float reciprocals and explicit RN arithmetic; [the proof and source check](native-smoke/fp32-alpha-cpu-result.json) include the two widths, signed bounds and four gap choices.

C11 also packs inputs once into fixed write-combined mapped pinned memory and gives its explicit CUDA device alias to encode. Each input tile is read once to shared memory; all demand PCIe reads belong to encode and the complete-call timer. Standard output ownership and all per-input work remain unchanged. [C11](native-smoke/native-c11-result.json) passes both full outputs but measures 5.135 ms / 1.601 ms, with primary demand-read encoding at 4.027 ms. The current method retains explicit bulk H2D. FP32 alpha reduces G1 phase from 291/76 to 180/45 µs while preserving all exact outputs.

## Fixed MT32 dual MMA

C12 doubles each encoder/CRT tile from 16 to 32 coefficients. Two m16 unsigned-MMA accumulator sets reuse every B0/B1 load; total arithmetic and global byte extents stay fixed, while CTA count and group barriers halve. The original MT16 fixture route remains available. [Source-extracted CPU checks](native-smoke/mt32-cpu-result.json) cover second-row reducers, shared layouts and natural/transposed output indexes. [C12](native-smoke/native-c12-result.json) passes both complete outputs at three fresh seeds per point, with medians 2.099 ms / 0.963 ms. C12 retains cached pinned input with one bulk H2D and the proven FP32 alpha.

## Non-temporal CPU input packing

C13 keeps the twelve disjoint input tasks and uses AVX512 aligned streaming stores on supporting CPUs. Prefix/tail byte copies preserve arbitrary alignment and lengths; each worker drains its stores before the completion barrier and the one bulk H2D. The fixed setup reports the actual CPU backend. [CPU preflight](native-smoke/input-stream-cpu-result.json) passes the complete fallback outputs and target-ISA compilation; [C13](native-smoke/native-c13-result.json) passes both full outputs, with complete-call medians 2.185 ms / 0.894 ms. Primary input-pack phase rises from 254 to 519 µs in the second warm-up; the primary call rises 4.1% from C12. The current source restores libc packing. The auxiliary improvement of 7.2% remains recorded with its three-seed scope.

## Four-column q-reduced CRT weights

C14 replaces production B7 columns with four columns per limb: `W[j,a]=(256^a*weight_j) mod q`, and `z=sum(byte_a(term_j)*W[j,a])`. This preserves the CRT result modulo q and bounds `z<2^(W+17)`. Four dot sums per limb use an exact u64 carry; the original two-fold reducer and FP32 alpha guard remain intact. MT32 uses fourteen limbs per group, reducing primary groups from seven to four and auxiliary groups from four to two. The original B7 Python model and MT16 fixtures stay available. [Independent integer/source CPU checks](native-smoke/byte-crt-cpu-result.json) pass all eight width/gap plans and 11,136 epilogue words. [C14](native-smoke/native-c14-result.json) passes all complete outputs at 3/3 fresh seeds per point. Primary G1 falls from 170 to 115 µs in the second warm-up; the complete-call median is 2.223 ms versus C12's 2.099 ms, alongside changes in the identical libc packing/transfer path. Whole-call samples and warm-up phases retain their separate scopes.

## Two-operand transfer/compute pipeline

The current candidate makes each operand DMA-ready using twelve disjoint libc CPU chunks. A and B use separate nonblocking streams for pinned H2D, unchanged MT32 RNS encoding and first forward NTT. The B stream joins the A-forwarded dependency before the common spectral middle, inverse and B4 CRT. All input packing, dependencies, transfers, full product and owned output construction stay inside the measured call. [Source-extracted CPU dependency/offset checks](native-smoke/operand-pipeline-cpu-result.json) pass both full layouts and complete host orchestration. [C15](native-smoke/native-c15-result.json) passes both complete outputs at 3/3 fresh seeds per point. Primary whole-call latency falls 12.9% from C14 to 1.937 ms; auxiliary is 1.055 ms. The second warm-up reports a 1.536 ms GPU pipeline wall span and summed/overlapping per-stream timings. Formal challenge entry uses its own 20-seed run. Fixed setup reads PCIe/NUMA/allowed-CPU locality; CPU affinity and allocation policy remain unchanged.

## Chunked CRT, D2H and owned output

The current output candidate uses fixed 512-CTA slices, each covering 16,384 contiguous natural coefficients: four chunks at N65536, two at N32768. The common B stream produces each CRT chunk; the existing A stream waits for it and returns its complete range to pinned staging. A standard vector reserves its full capacity, then appends each copied range after its completion event. This overlaps CRT, DMA and CPU construction while keeping all bytes, synchronization and standard ownership inside the call. Disjoint GNU host copies at least 1.5 MiB use the existing capped parallel wrapper; input worker copies stay below that threshold. [CPU/source checks](native-smoke/output-pipeline-cpu-result.json) cover complete indexes, pending-copy/insert order, returned words and unchanged arithmetic. Native full-call exactness and latency are the next gate.
