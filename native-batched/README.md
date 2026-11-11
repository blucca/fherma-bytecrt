# ByteCRT: independent batched products

This artifact implements `negacyclic-batched@1.0.0` at **N=65536, W=1747, B=1..128**. Every pair uses the same exact RNS / cuPQC NTT / ByteCRT kernels as the original C18 product. The initial batch adapter is a serial baseline: each pair reads a contiguous pointer range, then appends its completed canonical coefficients into one standard caller-owned vector. No intermediate input tensors are copied.

The generated `Point` carries B, N, W, L and the common q. Inputs and output have shape `{B,N,L}`. Fixed setup comes solely from Point. Coefficient encoding, all transfers, and the completed owned output remain inside `fherma_run`. The baseline holds one product's temporary GPU / pinned memory, so residency stays bounded as B grows.

The independent original implementation and its frozen `official-c18-20seed` source/result remain in place. Original single-product measurements and new batched measurements have their own specification and seeded data.

## Build

Build from this directory with the official `fherma/cupqc` image (cuPQC 0.6.0, CUDA 12.8). The CMake target shares the original mathematical sources in `../native-smoke`. `main.cpp` and `fherma.h` are generated from the published batched signature; the platform replaces these with the same generated contract at measurement.

Native B=1 and B=128 measurements follow the first new target job. The original measurements are [published separately](../native-smoke/native-official-c18-result.json).
