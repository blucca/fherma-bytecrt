# Generated-harness heap lifecycle: CPU check

A small real-harness experiment: the generated `main.cpp` and `fherma.h` are byte-identical to the batched contract. The custom CPU solver returns a complete ordered allocator-owned vector using per-product XOR and standard vector inserts. Each tensor is 33 MiB (B128/N2048/L33); the first of four cases is treated as warmup. Both allocator modes read both inputs, run, write the entire output and release output before inputs, exactly through the generated main. The fixture reuses deterministic input bytes through hard links.

All 8 complete outputs passed a 69,206,016-word check. The run finished in 23.50 s with one permitted CPU, a 256 MiB child address-space limit and a 60 s script budget.

| Metric | glibc default | init-time retained heap |
|---|---:|---:|
| First case output-copy minor faults | 8448 | 0 |
| Three successor copy minor faults | 8448 / 6915 / 6915 | 0 / 6 / 0 |
| Successor copy median | 9.375 ms | 2.699 ms |
| Successor generated-call median, diagnostics included | 12.997 ms | 7.193 ms |
| Peak RSS | 105940 KiB | 138460 KiB |

The retained-heap mode calls `mallopt(M_MMAP_THRESHOLD, INT_MAX)` and `mallopt(M_TRIM_THRESHOLD, -1)` during setup. All a/b/c allocations were observed in the ordinary heap; c's address was reused in all successor cases. The input reads' stream/string/vector allocations prepare heap pages before the timed call. Thus this real harness's first output copy already had zero faults. Under default allocation, the last two c addresses also repeated, with 6915 copy faults each: virtual-address reuse and resident-page reuse are separate measurements.

Scope: this is a 33 MiB CPU allocator/copy lifecycle experiment with fake XOR arithmetic, one worker, and per-product inserts. The native target uses 1760 MiB tensors, the exact CUDA mathematical body, four bounded slots, and 3.4375 MiB output chunks copied by OpenMP12. Its full whole-call and memory results are measured separately. At full size the `stringstream` growth can cross INT_MAX, so its largest temporary buffer follows a different allocation class; full-target peak memory and latency remain the native gate. The previous independent full-size CPU experiment measured output-only retained-heap behavior.

Temporary fixtures and binaries are regenerable. `harness-heap-cpu-result.json` retains the summarized result. Reproduce from the repository root with `python3 native-batched/cpu-harness-check/run.py`; it requires g++ and Python standard libraries and writes fixtures, outputs and receipts under `temp/bytecrt-harness-heap/`.
