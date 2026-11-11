# ByteCRT: independent batched products

This artifact implements `negacyclic-batched@1.0.0` at **N=65536, W=1747, B=1..128**. Every pair uses the exact RNS / cuPQC NTT / ByteCRT kernels from C18. **B=128 streams through four product slots and three independent H2D / compute / D2H queues**; B=1 preserves the original complete product path. Each pair reads its own contiguous caller-owned input ranges.

The generated `Point` carries B, N, W, L and the common q. Inputs and output have shape `{B,N,L}`. Fixed setup comes solely from Point. Coefficient encoding, all transfers, and the completed owned output remain inside `fherma_run`. Device residency is bounded by four products as B grows.

## Complete streaming operation

The host packs four 16,384-coefficient chunks per operand into cached pinned staging. A dedicated H2D queue moves each chunk. The compute queue waits for an operand's final copy, runs one full MT32 encoder and the original full forward stage, then processes the other operand, fused spectral middle, weighted inverse, and four 512-CTA CRT chunks. The independent D2H queue waits for each CRT chunk and copies it to pinned output staging while later compute and input transfers proceed.

The output vector reserves its full capacity inside the timed call. The host appends copied chunks through standard vector `insert`, in product order, using the existing OpenMP12 copy wrapper. The same host thread alternates input pack and owned output construction, so one OpenMP team is active at a time. A slot is recycled after its final D2H completion and all four owned inserts; the return completes every product and the entire standard allocator-owned result.

For B=128, explicit device arrays total **1,375,529,584 bytes**, cached pinned staging totals **173,015,040 bytes**, and the complete returned vector holds **1,845,493,760 bytes**. The four slots retain independent copies of Point-derived fixed tables. Input buffers, output buffers, RNS scratch, and events remain independent; three global streams have one owner. Host inputs plus owned output plus pinned staging total about **5.71 GB**, within the published 32 GB budget.

Setup prints the device's asynchronous copy engine count. One unscored warmup reports host pack / owned-copy sums and the three GPU queue coverage spans; coverage includes host-supply gaps and overlapping work. Scored calls record the required operand-ready / CRT-ready / copied dependencies.

The independent original implementation and its frozen `official-c18-20seed` source/result remain in place. Original single-product measurements and new batched measurements have their own specification and seeded data.

## Build

Build from this directory with the official `fherma/cupqc` image (cuPQC 0.6.0, CUDA 12.8). The CMake target shares the original mathematical sources in `../native-smoke`. `main.cpp` and `fherma.h` are generated from the published batched signature; the platform replaces these with the same generated contract at measurement.

The serial B0 adapter is fixed at commit `a311487` and its B=1 / B=128 target benchmark is queued. This streaming method has passed a deferred-event / mock-DMA interface check using two B=128 calls, including slot reuse and full owned output order. Native CUDA compilation, all-coefficient exactness and whole-batch timing follow the target benchmark. The original single-product measurements are [published separately](../native-smoke/native-official-c18-result.json).
