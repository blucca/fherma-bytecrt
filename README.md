# ByteCRT

An exact unsigned-byte matrix model for a fused, multi-limb GPU CRT epilogue.

**Current result:** the independent CPU model passes at both FHERMA polynomial-multiplication geometries, using three synthetic full-width moduli and 18 coefficient fixtures per geometry. Native GPU compilation, complete official convolutions and target latency are pending.

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

The next artifact is one fused MMA/CRT kernel with a coalesced output handoff, integrated with an exact RNS NTT under the [FHERMA challenge's](https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025) complete timing boundary. Its competitive family already includes RNS NTT and tensor-core CRT; improvement must be measured on the target runner.

The benchmark reference is the [official cuPQC example](https://github.com/fherma-ai/polynomial-mult-cupqc-example), with the generated host interface. Original reference code: Apache-2.0.

### Locked unsigned MMA fragments

[`fragment_check.py`](fragment_check.py) implements the unsigned `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32` register contract from [CUDA 12.8.1 / PTX ISA 8.7, §9.7.14.5.10](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832).

For `g=lane/4`, `k=4*(lane%4)`, the four A registers start at `(g,k)`, `(g+8,k)`, `(g,k+16)`, `(g+8,k+16)`. Each packs four consecutive unsigned bytes. B uses two registers in column `g`, starting at rows `k` and `k+16`; C/D hold two columns in row `g`, then the same columns in row `g+8`.

```sh
python3 -B fragment_check.py
```

The CPU check covers every A/B/C cell exactly once (512/256/128), independently inverts the register coordinates, reconstructs byte-packed operands and checks all 128 integer `A·B+C` outputs against a direct matrix oracle. Fixtures include bytes above 127. [`fragment-result.json`](fragment-result.json) records the result; native GPU execution follows as a separate step.

[`native-smoke/`](native-smoke/) is the runnable next step: one full warp executes the unsigned MMA on fixed matrices during setup, and a host oracle checks all 128 cells. An attributed upstream cuPQC schoolbook implementation supplies the complete polynomial answer to the official harness. Target compilation and this new native check are pending; the CPU model and the full competitive pipeline have separate evidence scopes.

## Host output construction probe

The generated official interface returns `std::vector<uint32_t>`, and only `fherma_run` is timed. A CPU-only probe compares `resize` followed by an opaque overwrite with direct range construction from an already available staging buffer. Both produce identical complete output vectors.

```sh
mkdir -p temp
g++ -std=c++17 -O3 -fno-lto host-output-check.cpp -o temp/host-check
./temp/host-check
```

GCC 16.2.1, Linux x86_64: 20 interleaved samples per method after 3 warm-ups. Local medians were 1.468 ms / 0.864 ms for N65536/L55 and 0.160 ms / 0.096 ms for N32768/L28. The overwrite is CPU memcpy; these measurements cover vector construction. GPU DMA, pinned staging and official whole-call measurements have their own target-run validation.

A concrete native experiment is fixed pinned D2H staging plus range construction, with the entire DMA/copy/construction inside the timer. Its acceptance is unchanged exact output and measured complete-call improvement.
