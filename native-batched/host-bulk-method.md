# Whole-operand and whole-owned-product service

The four-slot batch implementation previously packed each operand in four 3.4375 MB calls and submitted four H2D transfers. Encoding already waits for the complete operand. This candidate packs/copies the **full 14.4 MB operand once**, retaining the operand-ready event and all cross-product dependencies. B128 pack/H2D submissions drop **1024 → 256**.

CRT and D2H keep their four original chunks. The host waits for the final D2H event and inserts the **full standard-owned 14.4 MB product once**, in the same product order. Owned copy/barrier regions drop **512 → 128**. The final event orders all earlier copies on that stream, and the owned-copy barrier completes every byte before staging reuse. Four slots, exact arithmetic, total bytes, fixed setup and ownership stay unchanged.

## Evidence used to select the complete native method

Standalone NT native passes B1/B128 exact but reduces B128 whole-call latency by **0.824%** to **89.068534 ms**; it stops at the 5% improvement gate. Its first warmup spends **67.732 ms** packing and **21.543 ms** constructing owned output, **88.95%** of the 100.367 ms whole call. Locality is the full CPU list 0–23, PCIe 32 GT/s ×16 and two async engines. [Actual NT result](native-nt-result.json).

The CPU-only bulk comparison uses two separate full B128 operands, four two-operand staging slots, and the complete 1,845,493,760-byte standard-owned output. NT is the common backend. One warmup and three AB/BA/AB pairs per phase give **14 rows**, with bytewise verification of all eight final operand staging buffers after each pack timing and all 128 owned products after each owned timing. Two-worker pack reduction is **5.285%**, owned **4.774%**, aggregate measured pack+owned service reduction **6.133%**; the CPU copy 10% gate fails. [All raw rows](host-bulk-cpu-result.json). This comparison measures ordinary-buffer copying with two workers. The complete native method measures twelve-worker fork/join, driver/DMA submission and the final-D2H wait tradeoff at the real target.

The candidate's actual native gate is **B1/B128 exact** and **at least 5% B128 whole-call reduction** from the measured NT native baseline, at most **84.6151073 ms**. Passing proceeds to the challenge's official three-fresh-seed measurement; failure stops this bulk formula. The official S1 source/result remains pinned independently.

## Reproduce CPU protocol

```sh
cd native-batched
python3 cpu-bulk-probe.py
```

The runner uses two available CPUs, 4 GiB address space and 180 seconds total wall time, including compilation. Both full operands are allocated together (about 3.8 GB including staging), released before the owned phase, and retained heap blocks are reused. All checkpoints, pair rows and summaries are written locally. Actual production NT copying also passes a 27-case full-product guard/owned/overlap check at 14.4 MB.

The [actual-source lifted orchestration check](host-bulk-orchestration-result.json) passes B1 delegation and two B128 generations with distinct inputs, deferred CUDA event/DMA dependencies, full owned product order, input preservation, slot reuse, and resource cleanup. Its arithmetic is deliberately scaled/mocked. The measured production-copy check is reproducible with:

```sh
g++ -O3 -mavx2 -std=c++17 -fopenmp -Wl,--wrap=memcpy -Wl,--wrap=memmove \
  host-bulk-copy-check.cpp core/io-copy.cpp core/vector-copy.cpp -o host-bulk-copy-check
./host-bulk-copy-check
```

## Complete native terminal result

The immutable `batch-bulk-host-candidate` source passes B1/B128 exact, one fresh seed per point: **1.660239 / 89.033466 ms**. B128 improves **0.03937%** over the standalone NT native baseline. The 5% gate fails and the bulk formula stops before an official measurement. [Native result](native-bulk-result.json). Its first warmup reports whole 89.317 ms, pack 50.655 ms, owned 14.358 ms, and overlapping H2D/compute/D2H coverage 87.040/88.254/87.549 ms. The selected S1 source remains qualified independently.

For the production NT/bulk copy checks above, use the corresponding immutable experiment tag; main retains the selected S1 libc backend. The two CPU protocol files are standalone reproducers.
