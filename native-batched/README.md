# ByteCRT: independent batched products

This artifact implements `negacyclic-batched@1.0.0` at **N=65536, W=1747, B=1..128**. Every pair uses the exact RNS / cuPQC NTT / ByteCRT kernels from C18. **B=128 streams through four product slots and three independent H2D / compute / D2H queues**; B=1 preserves the original complete product path. Each pair reads its own contiguous caller-owned input ranges.

The generated `Point` carries B, N, W, L and the common q. Inputs and output have shape `{B,N,L}`. Fixed setup comes solely from Point. Coefficient encoding, all transfers, and the completed owned output remain inside `fherma_run`. Device residency is bounded by four products as B grows.

## Native complete-call result

The self-contained serial baseline `35d3956` passes B1/B128 exact (one fresh scored seed each): B1 **1.600642 ms**, B128 **291.519151 ms**. The four-slot retained-heap source `aca09d6` passes the same complete native point coverage: B1 **1.815500 ms**, B128 **89.808734 ms**, a **69.19285%** reduction at B128. These are distinct fresh native measurements. B1 keeps the original mathematical/execution path; only the new B128 point determines the updated challenge rank.

The B128 run reports **9177.2 MB process memory / 1915 MB GPU memory**. Its first unscored warmup costs 104.421 ms, including 58.295 ms total host packing and 28.639 ms total owned inserts. Queue coverage spans include host-supply gaps and overlap one another. The device reports two async copy engines. [Serial](native-b0p-result.json) and [streaming](native-s1p-result.json) results pin their exact sources. The official one-point three-fresh-seed measurement passed **3/3 exact**, median **89.648935 ms**, min/max **89.608885 / 89.716234 ms**, ranked **#3** at observation. Its reported process memory is **12801.4 MB** and GPU memory **1915 MB**. [Full official receipt](native-official-s1-3seed-result.json) pins source/run/three-seed evidence. Participation is submitted and the official row is ranked. The platform has fixed the per-spec query: this same S1 run qualifies with three seeds. Its older receipt preserves the earlier query mismatch as history.

## Complete streaming operation

The host packs four 16,384-coefficient chunks per operand into cached pinned staging. A dedicated H2D queue moves each chunk. The compute queue waits for an operand's final copy, runs one full MT32 encoder and the original full forward stage, then processes the other operand, fused spectral middle, weighted inverse, and four 512-CTA CRT chunks. The independent D2H queue waits for each CRT chunk and copies it to pinned output staging while later compute and input transfers proceed.

The output vector reserves its full capacity inside the timed call. The host appends copied chunks through standard vector `insert`, in product order, using the existing OpenMP12 copy wrapper. The same host thread alternates input pack and owned output construction, so one OpenMP team is active at a time. A slot is recycled after its final D2H completion and all four owned inserts; the return completes every product and the entire standard allocator-owned result.

For B=128, explicit device arrays total **1,375,529,584 bytes**, cached pinned staging totals **173,015,040 bytes**, and the complete returned vector holds **1,845,493,760 bytes**. The four slots retain independent copies of Point-derived fixed tables. Input buffers, output buffers, RNS scratch, and events remain independent; three global streams have one owner. Host inputs plus owned output plus pinned staging total about **5.71 GB**, within the published 32 GB budget.

Setup prints the device's asynchronous copy engine count. One unscored warmup reports host pack / owned-copy sums and the three GPU queue coverage spans; coverage includes host-supply gaps and overlapping work. Scored calls record the required operand-ready / CRT-ready / copied dependencies.

The independent original implementation and its frozen `official-c18-20seed` source/result remain in place. Original single-product measurements and new batched measurements have their own specification and seeded data.

## Large-output allocation

B>1 setup retains the ordinary glibc heap for allocations below `INT_MAX` and disables automatic trimming. This addresses the full standard output vector's repeated first-touch page allocation; all vector ownership and ordered construction remain unchanged. A [CPU-only full-size experiment](owned-output-cpu.md) measured 450,560 faults in every default fresh 1.845 GB output and zero faults after warmup with retained heap. The complete native two-point and official three-seed latency / memory results above cover the retained-heap streaming artifact.

## Build

Build from this directory with the official `fherma/cupqc` image (cuPQC 0.6.0, CUDA 12.8). The artifact is self-contained: `core/` carries the nine required exact C18 dependency files, and CMake resolves every production source locally. The mathematical bodies match `../native-smoke`; the generated Point header is resolved at the artifact root. `main.cpp` and `fherma.h` are generated from the published batched signature; the platform replaces these with the same generated contract at measurement.

The first B0 job at `a311487` stopped during CMake configuration because the platform packages only the selected artifact directory. Its cases and timings are zero. The corrected serial B0 adapter is fixed at `35d3956` (`batch-serial-b0-self-contained`), with local exact C18 dependencies. That corrected snapshot and the subsequent streaming snapshot both passed native B1/B128 all-coefficient exactness. The official three-seed entry follows its own measurement and ranking receipt. This streaming method has passed a deferred-event / mock-DMA interface check using two B=128 calls, including slot reuse and full owned output order. Native CUDA compilation, all-coefficient exactness and whole-batch timing have passed the two-point target benchmark linked above. The original single-product measurements are [published separately](../native-smoke/native-official-c18-result.json).

## Reading pipeline timings

[Three queues, one timed return](queue-timing-notes.md) aligns the existing warmup host-copy counters, overlapping queue spans and subsequent fresh whole-call measurements. The standard-library checker reproduces the table and the PCIe payload scale from the published receipts.

## Completed host-service experiments

The [standalone NT](native-nt-result.json) and [whole-service bulk](native-bulk-result.json) candidates each pass exact native B1/B128 output. Their B128 latency is 89.068534/89.033466 ms; both stop at their respective 5% improvement gates and add zero official measurements. Their immutable sources are tagged `batch-nt-host-candidate` and `batch-bulk-host-candidate`. The selected production adapter and copy backend here retain the qualified S1 source. CPU benchmarks preserve their separate ordinary-buffer verification scopes.

The platform has fixed the dialog and per-spec qualifying query. The original S1 run now reports `qualifies:true`, `already_entered:true`, with the intended B128 three-seed condition. The official row is currently #4 at observation; its same-run median remains 89.648935 ms.

## Completed shared-compute memory method

The [single-compute/Point sharing method](gpu-workspace-method.md) passes native B1/B128 exact, with B128 **91.995788 ms** (one fresh scored seed), versus the fixed bulk baseline **89.033466 ms**. Its 5% throughput gate fails and official follow-up stops. Explicit device arrays decrease **65.566%**; target-reported GPU memory is **1323 MB**, process memory **9176.9 MB**, with **128 MiB L2**. [Complete native result](native-workspace-result.json). The source and its actual-helper ownership/deferred-DMA check remain at immutable tag `batch-shared-compute-candidate`. Production source and API metadata here retain the qualified S1 baseline.
