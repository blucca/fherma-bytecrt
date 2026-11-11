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
