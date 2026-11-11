# One compute workspace with four private DMA slots

The four-product streaming method uses a single FIFO compute queue. Every product's encode, forward stages, spectral middle, weighted inverse and all four CRT chunks finish in queue order before the next product's compute. H2D and D2H run on their own queues and continue to use separate per-slot arrays.

This complete candidate reuses one set of Point-derived tables and compute-only workspaces at stable device addresses. Its host service is fixed at the immutable `batch-bulk-host-candidate` source; the NT and bulk host-only experiments already have complete negative results at their own whole-call improvement gates.

## Lifetime and ownership

- **Shared Point fields (12):** `tw`, `inv_tw`, `psi`, `untwist_crt`, `encode_b`, `encode_mu`, `q`, `primes`, `n_inverse`, `b_matrix`, `invp`, `constants`.
- **Shared compute fields (4):** `ab`, `ab_t`, `c`, `weighted`. All their readers/writers execute on the single compute queue.
- **Private DMA fields:** device `inputs`/`out`, pinned `hostinput`/`hostout`, and each slot's dependency events.

`borrow_device_allocation` finds each child's original allocation in the existing `State::allocations` inventory, frees it, erases the ownership entry and rebinds the pointer to slot 0. Slot 0 keeps its 18 original device allocations; each borrower owns only two. State destruction frees its own inventory once. Weighted padding is initialized once in the owner's setup; subsequent kernels overwrite their existing valid ranges.

The sharing contract is **serial `fherma_run` calls and one FIFO compute queue through CRT**. A future parallel-compute implementation would provision one workspace per compute queue. B1 delegates its original serial product path and retains all its own storage.

For B128, explicit device arrays change from **1,375,529,584 to 473,643,676 bytes** (65.566% lower). Per-slot pinned staging remains **43,253,760 bytes**, four slots total **173,015,040 bytes**. The timed standard-owned output remains **1,845,493,760 bytes**. Input/output transfer bytes and every mathematical kernel stay unchanged. Target-reported process/GPU peaks include additional runner/runtime allocations.

## Complete gates

The [actual-source lifted check](gpu-workspace-orchestration-result.json) passes deferred DMA/event dependencies, two B128 generations with distinct inputs, all 256 complete scaled owned products, unchanged input data/shapes, prior returned-output preservation, private DMA buffers/events, twelve shared tables remaining unchanged, and the actual allocation inventory. The lifted production helper, `State`, `allocate` and destructor verify parent 18/borrower 2 allocations and single frees in both destruction orders. Its scope is host orchestration and ownership; official native CUDA verifies every real coefficient.

One native target measurement covers B1 and B128, each with one fresh scored seed. The complete method proceeds to one official three-fresh-seed measurement after both exact points pass and B128 is at most **84.5817927 ms**, a 5% reduction from the measured bulk baseline **89.033466 ms**. Failure stops this sharing method. The qualified S1 official result stays independently pinned at `aca09d6`.

Setup also reports L2 capacity. The expected mechanism is a smaller stable-address working set between adjacent products; actual complete-call timing decides the result.

## Reproduce the orchestration and ownership check

From the repository root, choose a regenerable output directory:

```sh
python3 native-batched/gpu-workspace-check.py /path/to/workspace/temp/bytecrt-check
```

Python standard library and `g++` suffice. The checker extracts the current production wrapper, helper, State and allocator, then compiles the scaled deferred-DMA harness. Its real mathematical kernels and target latency are covered by the separate native measurement.
