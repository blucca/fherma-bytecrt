# Non-temporal host traffic: paired CPU experiment

B128 consumes two 1.845 GB operands and returns one standard-owned 1.845 GB vector. The four-slot native baseline spends 58.295 ms packing inputs and 28.639 ms constructing owned output during its first unscored warmup. Each 3.4375 MiB chunk has twelve fixed tasks of approximately 300 KiB.

The candidate uses AVX2 unaligned loads and aligned streaming stores. Libc handles the alignment head and short tail. Each OpenMP worker performs one `sfence` after all its assigned disjoint tasks, before the parallel-region barrier. Large disjoint `memmove` uses the same path; overlapping ranges keep libc. The four-slot CUDA algorithm and exact arithmetic are unchanged. The legacy artifact remains frozen.

## Measured CPU scope

EPYC 9254, two permitted CPU workers, ordinary heap buffers, 4 GiB address-space cap. Five paired AB/BA comparisons per case use exactly the same task partition for libc and NT; every destination byte is checked after each timing. Streaming source/destination rings exceed 32 MiB. The owned case rotates a 14.4 MB source across streaming destinations.

| Case | Median paired latency reduction |
|---|---:|
| Worker, hot | -17.97% |
| Worker, streaming | 63.16% |
| Full chunk, hot owned-byte tasks | 9.27% |
| Full chunk, hot pack-word tasks | -8.07% |
| Full chunk, streaming pack | 60.31% |
| Full chunk, streaming owned | 66.79% |

The preregistered streaming gate weights pack/owned ratios by the measured baseline host-service times above and requires at least 10% reduction. **62.44% passes.** Hot costs remain in the dataset.

The same bounded run then tests the complete **1,845,493,760-byte** B128 output with standard `std::vector` reserve/clear/insert, retained ordinary heap pages, one first-touch warmup and three AB/BA/AB pairs. All seven calls pass bytewise verification for every product. The post-warm paired median latency reduction is **62.86%**. Total CPU experiment wall time including compilation is **8.21 seconds**.

[Full raw paired data](host-copy-cpu-result.json). These CPU measurements cover ordinary buffers and two workers. The candidate's actual pinned-buffer/DMA, twelve-worker and CUDA whole-call result is a separate native measurement. The officially measured baseline remains [`aca09d6`](https://github.com/blucca/fherma-bytecrt/tree/aca09d65fe30e6c9497f7a0f038e9a9f4b7384df/native-batched), three fresh seeds, 89.648935 ms.

## Reproduce

The CPU-only runner selects two available CPUs, caps address space at 4 GiB and total wall time at 180 seconds, writes every row/checkpoint locally, and runs the full-size output comparison only after the streaming gate:

```sh
cd native-batched
python3 cpu-nt-probe.py
```

A small integration check compiles the actual production copy source and covers pack guards, standard-owned bytes, short/empty operands and overlapping-move fallback:

```sh
g++ -O3 -mavx2 -std=c++17 -fopenmp -Wl,--wrap=memcpy -Wl,--wrap=memmove   host-copy-check.cpp core/io-copy.cpp core/vector-copy.cpp -o host-copy-check
./host-copy-check
```

The original exhaustive primitive suite remains available as `./cpu-nt-probe --semantics` after compilation; it covers offsets 0–63, worker/chunk lengths and guards. The production integration check passes 27 cases.
