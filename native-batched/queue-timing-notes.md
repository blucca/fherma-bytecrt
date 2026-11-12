# Three queues, one timed return

The four-slot B128 artifact brings the complete native call from **291.519 ms to 89.809 ms**, with exact output at both B1 and B128. Its separate official three-fresh-seed result is **89.648935 ms**, 3/3 exact. Those complete calls include wide-coefficient encoding, all transfers, and the final standard-owned vector.

The follow-up host and memory experiments give a useful lesson in reading GPU pipeline measurements: preserve the **source, call and timing boundary** of every number.

## The return is part of the workload

At B128 / N65536 / L55, each input is 1,845,493,760 bytes, and the output has the same size. Four bounded slots carry pinned staging and device scratch. Three FIFO streams handle input DMA, computation and output DMA. The host copies input chunks into pinned memory and appends completed output chunks into a reserved `std::vector<uint32_t>`.

`fherma_run` returns after the last owned insert. That final return is the end-to-end contract. The [allocation experiment](owned-output-cpu.md) separately explains the large result's repeated first-touch cost and retained-heap policy.

## One table, two kinds of call

All values below are milliseconds. The phase counters and queue spans belong to the **first unscored warmup**. Each fresh whole-call value comes from the subsequent **one-fresh-seed B128 native measurement**. Each candidate also passes the separate B1 point, giving 2/2 exact native outputs.

| Method | Warm whole | Warm host copy sum | H2D span | Compute span | D2H span | Fresh whole |
|---|---:|---:|---:|---:|---:|---:|
| [NT](native-nt-result.json) | 100.367 | 89.275 | 98.720 | 99.621 | 98.840 | 89.068534 |
| [Whole-service bulk](native-bulk-result.json) | 89.317 | 65.013 | 87.040 | 88.254 | 87.549 | 89.033466 |
| [Shared compute / Point](native-workspace-result.json) | 89.528 | 64.210 | 87.167 | 88.557 | 87.899 | 91.995788 |

The host copy sum adds the timed pack calls and owned inserts. Each includes its existing OpenMP12 copy/barrier. CUDA submissions, waits for copied data, reserve and other host orchestration sit outside those two counters.

A queue span runs from its first recorded event to its last. Within that interval, the queue can execute work, wait on another stream's ready event, or catch up with host submissions. The three spans share elapsed wall time: H2D for a later product overlaps computation and D2H for earlier products. End-to-end latency follows the dependency graph through the final owned insert.

From NT to bulk, the first-warmup host copy sum falls **24.262005 ms**. The subsequent fresh whole-call values differ by **0.035068 ms**. The measurements remain adjacent to their own calls. The native whole improvement gates stop both host candidates, while the original official S1 result stays selected.

The shared-workspace method reduces explicit device arrays by **65.566%**, while its fresh whole call rises to **91.995788 ms**. Memory footprint and complete-call throughput are separate measured outcomes.

## Read the submission path

The [selected adapter](solve.cu) uses one host thread for this sequence:

```text
wait for the old slot's copied output
append that output into the owned vector
pack A, submitting each H2D chunk
submit A's input-ready event, full encoder and forward stage
pack B, submitting each H2D chunk while A's GPU work can proceed
submit B's input-ready event, full encoder and forward stage
submit middle, inverse, CRT and D2H
```

Other slots can keep their GPU work in flight during those host operations. Each slot's final D2H completion and final owned insert precede its reuse. The diagram explains where to look for submission gaps; active DMA or kernel time needs its own interval measurement, aligned with the call being studied.

The selected S1 artifact prints phase diagnostics in its first unscored warmup. Its official receipt records three fresh whole calls: **89.608885 / 89.648935 / 89.716234 ms**. These are the production measurement references for that source.

## A physical scale check

The saved target link is PCIe **32 GT/s ×16**. At its 128/130 coding rate, the ideal one-direction payload rate is about **63.015 GB/s**. The B128 payload therefore requires:

- H2D, both inputs: **3,690,987,520 B / 58.5728 ms** at the coding-only rate.
- D2H, complete output: **1,845,493,760 B / 29.2864 ms** at the same rate.

With ideal full-duplex transfer overlap, the larger input time supplies a **58.5728 ms** lower bound. Protocol costs, actual DMA/memory bandwidth, preparation, computation and the final output tail consume additional time. This bound uses this dense payload and the observed link; source changes that alter transferred bytes have their own traffic model.

## Reproduce the table

The checker reads the published JSON receipts and uses Python's standard library:

```sh
python3 -B native-batched/check-timing-scopes.py
```

It verifies the two-point exact flags, prints each warmup next to its own fresh whole value, checks the NT→bulk differences, and calculates the link/payload scale. The complete native experiments and the official three-seed result retain their original immutable sources and measurement receipts.
