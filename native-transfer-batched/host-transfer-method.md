# Byte-copy transfer host method proof

Result: PASS; real CUDA measurements: **0** at this source checkpoint.

Run with a temporary output directory:

```sh
python3 products/fherma-bytecrt/native-transfer-batched/host-transfer-check.py \
  --out temp/round603/fherma-timing/host-transfer-check
```

This reuses the round542 lifted-helper technique. It extracts the actual new wrapper after the frozen core include, rewrites its three CUDA launch expressions as ordinary C++ mock calls, and shrinks the array dimensions. The one B1 N-dispatch predicate uses a separate production-N tag so both original `(65536,114,480)` and `(32768,57,256)` choices are checked while arrays stay small. Mathematical kernel bodies are replaced by deterministic tags; the real frozen `core/io-copy.cpp` and `core/vector-copy.cpp` are compiled and linked with OpenMP and the original libc copy wrapping.

The deferred event/DMA queue checks pinned ranges before reuse. B1, B2 and B128 each execute two complete h2d/run/d2h calls; the first owned output remains alive through the second call and state destruction. Full small-array A/B byte equality, direct raw-C product order, output shape/count, non-aliasing, all-device completion on return, stream/event ownership and malformed-shape rejection pass. Production `[B,N,L]` ranges are separately counted using the actual dimensions.

Each phase is instrumented: callbacks launch zero mathematical operations and allocate zero device buffers; run submits every encoder/forward/middle/inverse/G1 operation, completes the mock device, and returns an empty result vector with full reserved capacity. A source audit checks callback helper closure and run for kernels/copies/resize/zero-fill in their assigned phases. Every one of the nine core files is compared byte-for-byte with frozen ACA. The exact C++ callback function types are compile-time assertions. The unmodified official generated header/main are checked for exact declarations, h2d→run→d2h ordering, moving the reserved owned vector and separate transfer timer fields.

Scope: scheduling, copy contract, shapes and ownership. Native CUDA compilation and all-coefficient mathematical exactness are the root-owned fresh native gate. `host-transfer-cpu-result.json` records source/header hashes and all assertions.
