# ByteCRT: independent batched products

This artifact implements `negacyclic-batched@1.0.0` at **N=65536, W=1747, B=1..128**. Every pair uses the exact RNS / cuPQC NTT / ByteCRT kernels from C18. **B=128 streams through four product slots and three independent H2D / compute / D2H queues**; B=1 preserves the original complete product path. Each pair reads its own contiguous caller-owned input ranges.

The generated `Point` carries B, N, W, L and the common q. Inputs and output have shape `{B,N,L}`. Fixed setup comes solely from Point. Coefficient encoding, all transfers, and the completed owned output remain inside `fherma_run`. The current shared-compute candidate bounds residency to one compute workspace and at most four private DMA input/output slots.

## Native complete-call result

The self-contained serial baseline `35d3956` passes B1/B128 exact (one fresh scored seed each): B1 **1.600642 ms**, B128 **291.519151 ms**. The four-slot retained-heap source `aca09d6` passes the same complete native point coverage: B1 **1.815500 ms**, B128 **89.808734 ms**, a **69.19285%** reduction at B128. These are distinct fresh native measurements. B1 keeps the original mathematical/execution path; only the new B128 point determines the updated challenge rank.

The B128 run reports **9177.2 MB process memory / 1915 MB GPU memory**. Its first unscored warmup costs 104.421 ms, including 58.295 ms total host packing and 28.639 ms total owned inserts. Queue coverage spans include host-supply gaps and overlap one another. The device reports two async copy engines. [Serial](native-b0p-result.json) and [streaming](native-s1p-result.json) results pin their exact sources. The official one-point three-fresh-seed measurement passed **3/3 exact**, median **89.648935 ms**, min/max **89.608885 / 89.716234 ms**, ranked **#3** at observation. Its reported process memory is **12801.4 MB** and GPU memory **1915 MB**. [Full official receipt](native-official-s1-3seed-result.json) pins source/run/three-seed evidence. Participation is submitted and the official row is ranked. The platform has fixed the per-spec query: this same S1 run qualifies with three seeds. Its older receipt preserves the earlier query mismatch as history.

## Complete streaming operation

The current shared-compute candidate starts from the immutable bulk host-service source. The host packs each full 14.4 MB operand into private pinned staging, and a dedicated H2D queue transfers it once. The measured S1 source uses four input chunks. The compute queue waits for an operand's final copy, runs one full MT32 encoder and the original full forward stage, then processes the other operand, fused spectral middle, weighted inverse, and four 512-CTA CRT chunks. The independent D2H queue waits for each CRT chunk and copies it to pinned output staging while later compute and input transfers proceed.

The output vector reserves its full capacity inside the timed call. The current candidate appends each fully copied product through one standard vector `insert`, in product order, using the existing OpenMP12 NT copy wrapper. The measured S1 source uses four owned inserts. The same host thread alternates input pack and owned output construction, so one OpenMP team is active at a time. A slot is recycled after its final D2H completion and the complete product insert; the return completes every product and the entire standard allocator-owned result.

For B=128, the candidate shares twelve Point-derived device tables and four compute-only workspaces through the single compute FIFO. Each of three borrower slots removes those sixteen former allocations from its existing owner inventory, retaining only independent device inputs/output. Pinned staging and dependency events stay private. Explicit device arrays fall from **1,375,529,584 to 473,643,676 bytes**; pinned staging stays **173,015,040 bytes** and the timed owned result **1,845,493,760 bytes**. [The complete sharing contract and measurement gate](gpu-workspace-method.md) preserve serial calls and the unchanged DMA/compute order. Host inputs plus owned output plus pinned staging remain about **5.71 GB**. Native peak memory follows its target receipt.

Setup prints the device's asynchronous copy engine count and L2 cache capacity. One unscored warmup reports host pack / owned-copy sums and the three GPU queue coverage spans; coverage includes host-supply gaps and overlapping work. Scored calls record the required operand-ready / CRT-ready / copied dependencies.

The independent original implementation and its frozen `official-c18-20seed` source/result remain in place. Original single-product measurements and new batched measurements have their own specification and seeded data.

## Large-output allocation

B>1 setup retains the ordinary glibc heap for allocations below `INT_MAX` and disables automatic trimming. This addresses the full standard output vector's repeated first-touch page allocation; all vector ownership and ordered construction remain unchanged. A [CPU-only full-size experiment](owned-output-cpu.md) measured 450,560 faults in every default fresh 1.845 GB output and zero faults after warmup with retained heap. The original S1 target measurement above includes complete latency and peak memory; the shared-compute candidate has its own native gate.

## Build

Build from this directory with the official `fherma/cupqc` image (cuPQC 0.6.0, CUDA 12.8). The artifact is self-contained: `core/` carries the required exact C18 dependencies and the NT copy header, and CMake resolves every production source locally. The mathematical bodies match `../native-smoke`; the generated Point header is resolved at the artifact root. `main.cpp` and `fherma.h` are generated from the published batched signature; the platform replaces these with the same generated contract at measurement.

The first B0 job at `a311487` stopped during CMake configuration because the platform packages only the selected artifact directory. Its cases and timings are zero. The corrected serial B0 adapter is fixed at `35d3956` (`batch-serial-b0-self-contained`), with local exact C18 dependencies. That corrected snapshot and the subsequent streaming snapshot both passed native B1/B128 all-coefficient exactness. The official three-seed entry follows its own measurement and ranking receipt. This streaming method has passed a deferred-event / mock-DMA interface check using two B=128 calls, including slot reuse and full owned output order. Native CUDA compilation, all-coefficient exactness and whole-batch timing have passed the two-point target benchmark linked above. The original single-product measurements are [published separately](../native-smoke/native-official-c18-result.json).

## Completed host-service experiments

The [standalone NT](native-nt-result.json) and [whole-service bulk](native-bulk-result.json) candidates each pass exact native B1/B128 output. Their B128 latency is 89.068534/89.033466 ms; both stop at their respective 5% improvement gates and add zero official measurements. Their immutable sources are tagged `batch-nt-host-candidate` and `batch-bulk-host-candidate`. The current shared-compute candidate reuses the bulk host service as a fixed origin. The original S1 official result stays pinned at its immutable measured source. CPU benchmarks preserve their separate ordinary-buffer verification scopes.

The platform has fixed the dialog and per-spec qualifying query. The original S1 run now reports `qualifies:true`, `already_entered:true`, with the intended B128 three-seed condition. The official row is currently #4 at observation; its same-run median remains 89.648935 ms.
