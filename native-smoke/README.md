# Exact wide polynomial implementation

Complete coefficient-to-coefficient negacyclic multiplication at both FHERMA challenge points:

| N | W | L | RNS primes | CRT byte stride |
|---:|---:|---:|---:|---:|
| 65,536 | 1,747 | 55 | 114 | 480 |
| 32,768 | 868 | 28 | 57 | 256 |

`solve.cu` prepares prime/root/Montgomery/CRT constants from the fixed point during setup. Every measured call copies the two complete coefficient inputs, encodes all limbs to RNS, performs cuPQC staged NTT multiplication with a Montgomery forward twist, untwists and weights residues directly into coefficient-major ByteCRT storage, runs exact unsigned-MMA CRT with centered correction, copies all canonical coefficient limbs to fixed pinned D2H staging, then constructs the complete returned vector directly from that range. Input/output transfers and output construction remain inside the call.

## Arithmetic

- `rns-plan.hpp`: C++17 standard-library-only fixed setup; prime product exceeds `4*N*(q-1)^2`. Both point moduli use the specified `q=2^W-gap` form with `0<gap<2^32`.
- `rns-encode.cuh`: 16 coefficients × 8 primes per CTA; full-u32 input limbs, Montgomery power dot products, canonical residues.
- `ntt-pipeline.cuh`: cuPQC 0.6 `uint32`, M256/Block128 runtime-prime NTT. The middle pass fuses two forward second stages, pointwise multiplication and the first inverse stage.
- `ntt-weighted.cuh`: final inverse with ordinary `N^-1`, followed by one Montgomery product with the ordinary `psi^-i * inv(P/p)` factor. It writes weighted canonical residues directly into G1 rows.
- `g1-crt.cuh`: MT16/seven warps, original unsigned `m16n8k32` layout, FP64 centered alpha, seven-byte-sum carry, split high×gap reduction and coalesced canonical output.

The first warm-up prints per-kernel CUDA event times and host transfer/allocation times. Scored calls use the same arithmetic without phase events.

## Build

Inside the FHERMA `cupqc` image, with the target GPU visible:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/solution /path/to/official/cases
```

CMake links the external cuPQC 0.6 NTT static library with device LTO and native architecture. SDK files stay external.

## Exact gates

- [C1](native-c1-result.json): actual Blackwell unsigned MMA, all 128 integer cells.
- [C2](native-c2-result.json): full-width G1, 36 signed coefficients + 60 zero rows / 3,984 words, including cross-64-bit fold products.
- [C3](native-c3-result.json): actual sm120 cuPQC staged NTT, both full sizes × two real primes / 196,608 canonical residues, checked against an independent ordinary CPU radix-2 oracle. N16 schoolbook validates that oracle.
- [Fixed plan CPU](rns-plan-cpu-result.json): all prime/root/inverse/power words, FP64 reciprocals and B bytes agree with the independent Python model and original G1 fixtures at both widths/gap boundaries.
- [Encoding CPU](rns-encode-cpu-result.json): source-extracted arithmetic/kernel agrees with independent Python BigInt modulo, including full-width words and coefficient/prime tails.

C1–C3 diagnostic commits used the attributed upstream schoolbook for their small timed polynomial answer. The full pipeline passes [C4](native-c4-result.json) at both complete official-size outputs, one measured seed per point: 6.598 ms / 1.892 ms. This iteration now checks pinned-output and Montgomery-forward improvements in the next complete native run.

`ntt-smoke.cu` supports standalone native oracle execution with `BYTECRT_NTT_SMOKE_STANDALONE`; `g1-smoke.cu`/`mma-smoke.cu` preserve earlier setup checks. These gates stay out of the current complete-call implementation.

## Source credit

The generated `fherma.h`/`main.cpp` and the original CMake device-LTO pattern originate from [fherma-ai/polynomial-mult-cupqc-example](https://github.com/fherma-ai/polynomial-mult-cupqc-example/tree/e157fecf6066dac35c883a0c2e29e6d47d23e4bd), fixed commit e157fecf6066dac35c883a0c2e29e6d47d23e4bd. The official example declares Apache-2.0. ByteCRT arithmetic and integration are Apache-2.0 under the repository licence. The platform rewrites the generated harness at measurement.
