# Native unsigned-MMA setup check

This fixture executes `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32` on a single full warp. It checks all 128 integer outputs against an independent host matrix product. A packs four registers in NVIDIA PTX order; B is column-major so each register loads four contiguous bytes. Operand fixtures include unsigned bytes above 127 and signed C accumulators.

The check runs once during `fherma_init`, on fixed synthetic matrices. The timed `fherma_run` uses the unchanged cuPQC schoolbook arithmetic to give the official harness a complete, independently verifiable polynomial answer. Current scope is native MMA fragment execution plus the baseline polynomial interface. Fused CRT, RNS NTT and the primary challenge workload follow in the complete implementation.

## Build and test

Inside FHERMA's `cupqc` image, with the target GPU visible:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/solution /path/to/official/cases
```

Use N1024/W128 for the first bring-up run. A successful setup prints `ByteCRT unsigned MMA: 128 integer cells PASS (fixed setup fixture)` to stderr. The harness then judges the full polynomial answer, with setup time kept separate.

## Source credit

`solve.cu` schoolbook arithmetic, `CMakeLists.txt`, and the generated `fherma.h`/`main.cpp` originate from [fherma-ai/polynomial-mult-cupqc-example](https://github.com/fherma-ai/polynomial-mult-cupqc-example/tree/e157fecf6066dac35c883a0c2e29e6d47d23e4bd), fixed commit e157fecf6066dac35c883a0c2e29e6d47d23e4bd. The official FHERMA baseline declares Apache-2.0. This version adds `mma-smoke.cu`, a single setup call, and its build entry; upstream arithmetic is retained with attribution. The platform rewrites the generated harness at measurement. ByteCRT's new code is Apache-2.0 under the repository licence.

## Actual target result

CUDA 12.8.61 / GCC 13.3 on RTX PRO 6000 Blackwell Server: native unsigned-MMA fixed setup checked all 128 cells and passed; the official N1024/W128 polynomial answer passed 1/1 measured cases after two warm-ups. [Result](native-c1-result.json). The timed polynomial arithmetic remains the attributed schoolbook baseline.

## Full-width G1 fixture data

`python3 make_g1_fixtures.py` reproduces the six files in `fixtures/`. Both official geometries have three synthetic full-width moduli and six signed coefficient cases each, padded to 16 rows. The third gap is 2^32−1 and forces `(wide >> W) * gap` across 64 bits, exercising split multiplication. A contains weighted residues with alpha initially zero; native G1 computes centered alpha. B is column-major; the golden canonical limbs come independently from signed `x % q`. Native G1 execution is the next check.

The integrated G1 check is `g1-smoke.cu`, using `g1-crt.cuh`: MT16, seven full warps, unsigned MMA, FP64 centered alpha, limb carry and near-power reduction with split high×gap. Source-extracted CPU reducer and fixture checks pass all 3,984 words; native compilation and all six fixtures pass on the target (CUDA12.8.61/GCC13.3). The G1 setup prints one PASS per fixture and a 36-coefficient summary. This fixed setup check has its own scope; the timed polynomial answer is still the upstream schoolbook.

[Actual G1 target result](native-c2-result.json): 3,984 words PASS, with both full widths and cross-64-bit high-product cases.

## Staged NTT native gate

`ntt-pipeline.cuh` binds cuPQC 0.6 runtime-prime uint32 staged NTT (M256, 128 threads), with a shared-memory middle pass that fuses two forward second stages, pointwise multiplication and the first inverse stage. `ntt-smoke.cu` checks N65536 and N32768 at two real 31-bit RNS primes against an independent ordinary radix-2 CPU negacyclic product. All 196,608 residues are compared; an N16 schoolbook check validates the CPU oracle. This new sm120 compile/exact gate is pending the actual target run. Wide coefficient encoding, weighted-residue packing and complete two-point latency follow in the full implementation.
