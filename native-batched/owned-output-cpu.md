# Large owned output: allocate pages once, keep vector ownership

CPU-only experiment, 2026-11-12. B=128 produces a single standard `std::vector<uint32_t>` of **1,845,493,760 bytes**. It reserves once and constructs 512 ordered ranges from one reused 14,417,920-byte source through the existing copy wrapper.

## Complete measured sequence

Linux Ubuntu 24.04 / glibc 2.39 / GCC 13.3, AMD EPYC 9254, affinity restricted to two CPU workers. The wrapper keeps its twelve disjoint copy tasks and chooses two workers from the available CPUs. Page size is 4,096 bytes. Both probes use the same data and standard vector operations; each records three fresh create/destruct calls and three retained-capacity calls. Four words per copied range, wrapper counts, output size/capacity and complete construction are checked. All 24 observations passed.

| allocation policy | fresh call 1 | fresh call 2 | fresh call 3 | insert minor faults per call |
|---|---:|---:|---:|---|
| default | 2,675.626 ms | 986.859 ms | 918.653 ms | 450,560 / 450,560 / 450,560 |
| ordinary heap retained | 1,521.155 ms | 102.787 ms | 99.814 ms | 447,039 / 0 / 0 |

Time includes reserve and all inserts; destruction is measured separately. Default retained-capacity calls 2/3 take 90.433 / 89.145 ms, with zero insert faults. The retained-heap fresh calls 2/3 average **101.301 ms**, compared with **952.756 ms** for default fresh calls 2/3. Both processes peak near 1.75 GiB RSS.

The 14,417,920-byte default output moves from an anonymous mapping on its first allocation to the ordinary heap; its third fresh call has zero insert faults and takes 0.537 ms. The batch result remains a fresh anonymous mapping in all three default calls. Thus the previous small-output copy measurement and the large standard owned output use different page-allocation lifecycles.

## Two point-local process settings

```cpp
mallopt(M_MMAP_THRESHOLD, INT_MAX);
mallopt(M_TRIM_THRESHOLD, -1);
```

The batched artifact selects these during setup for B>1 at the declared fixed point. The allocator owns every returned standard vector, callers release it normally, and every result word is still constructed after its own D2H completion inside the timed call. Retaining allocator pages removes repeated first-touch work after the first allocation. Input values and GPU arithmetic stay unchanged. The process releases its retained heap on exit.

This measurement covers CPU memory construction on the machine above. The complete target-native B=128 operation, including input arrays, CUDA, transfer overlap and the harness allocator lifecycle, supplies the final latency and peak-memory gate.

## Real generated-harness lifecycle

A [33 MiB real-harness CPU check](harness-heap-cpu.md) passed all 8 complete outputs. The retained-heap mode had zero first-case output-copy faults and 0/6/0 successor faults, with the generated input reads supplying prefaulted heap storage before each timed call. The full native target remains the complete throughput and memory gate.
