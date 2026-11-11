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
