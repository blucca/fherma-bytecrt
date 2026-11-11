# ByteCRT

An exact unsigned-byte matrix model for a fused, multi-limb GPU CRT epilogue.

**Current result:** the complete exact pipeline passes both official-size outputs on RTX PRO 6000 Blackwell, at three fresh measured seeds per point. Latest complete-call medians are **3.733 ms / 1.146 ms** ([C7 result](native-smoke/native-c7-result.json)), including all input/output conversion and transfers. The primary call improves 43.4% from the initial complete implementation. The next measured bottleneck is the returned-vector range copy.

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
