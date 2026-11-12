# B128 active-service diagnostic (round548)

Independent ACA source artifact. Build target `solution` retains the generated harness, nine byte-identical core files (seven mathematics/solve plus two copy files), original GPU state/streams/events/buffers, B1 branch, and caller-owned result operations. The immutable baseline is `aca09d65fe30e6c9497f7a0f038e9a9f4b7384df`; `instrumentation-only.patch` and direct byte comparisons in `check-closure.py` expose the source boundary.

## Fixed collection contract

- Enable only `CONCURRENT_KERNEL`, `MEMCPY`, `EXTERNAL_CORRELATION` after original B128 initialization. Runtime API tracing is off. Added CUDA timing event count: zero.
- Four actual `fherma_run` scopes: call 1 warm; calls 2–4 scored fresh inputs. External ID is `(uint64_t(128) << 32) | call`. CUSTOM0 push at entry, pop at return/unwind. Original harness timing stays around the original call.
- Init allocates eight aligned 4 MiB CUPTI delivery buffers, four host trace arrays, GPU record arrays, and a fixed external-map array. Buffer callbacks retain delivered bytes and store fixed counters; collection code starts zero application threads. Successful timed host hooks perform timestamps/assignments and the CUPTI scope calls.
- `fherma_free` flushes completed activities, disables the three kinds, flushes remaining buffers, then parses/aggregates/prints. Sorting, base64, string construction, and all diagnostic trace printing occur there.
- Expected per call: 1,024 H2D / 3,690,987,520 B; 512 D2H / 1,845,493,760 B; 1,280 kernels. Every copy is 3,604,480 B. Kernel FIFO per product: encode-A, forward1-A, encode-B, forward1-B, middle, inverse-weighted, CRT chunks 0–3. Single-stream FIFO assigns the fixed product coordinates only after true external records map activities to the call.
- All GPU intervals lie in that call's CUPTI begin/end; zero dropped/unmapped/conflicting records, exact counts/bytes, three consistent resource streams, expected kernel names/FIFO, and satisfied H2D→encode / CRT→D2H dependencies gate each call's `valid` field. Failed timestamps emit an empty lane payload with `valid:false`.

## Actual image build gate

Use the existing CUDA 12.8 cuPQC challenge image and original native build invocation. CMake searches the finite `/usr/local/cuda[-12.8]/{extras/CUPTI,include,lib64,targets/x86_64-linux/lib}` paths. Missing library/headers or failed record ABI stops the build. No downloads or alternative platform mode are present.

Record structs: `CUpti_ActivityKernel9`, `CUpti_ActivityMemcpy5`, `CUpti_ActivityExternalCorrelation`. Both compile-time `CUPTI_API_VERSION` and actual `cuptiGetVersion()` are printed and checked for equality. `CUDA_VERSION` is checked as 12080. Memcpy mapping tries the actual `correlationId` and `runtimeCorrelationId`; matching uses delivered CUSTOM0 external records. The synthetic stubs deliberately use API version 999 and a private layout: the actual image supplies the ABI decision.

## Stderr and decoder

Original setup/coarse warmup lines remain. Free adds one `active_session` JSON line and four `active_trace` JSON lines. Codec is standard base64 over unsigned LEB128 integers, with no external dependency. Per GPU lane each row is **start delta from preceding start, duration, actual matched correlation ID**. The first delta starts at that call's CUPTI begin. Per host lane each row is **start delta, duration**, relative to `host.origin_ns`. All values use ns.

`lanes` carry count/stream/context/device/bytes plus active duration sum, first→last coverage, inter-interval idle sum, head/tail idle, and max gap. `stage_active_ns` order is encode, forward1, middle, inverse-weighted, CRT. `gpu_union_ns` merges all three lanes.

Host `pack` index → `(product=i//8, operand=(i//4)%2, chunk=i%4)`; `wait` and `owned` index → `(product=i//4, chunk=i%4)`. A drain supplies wait-begin, wait-return/owned-begin, owned-end timestamps. `reserve` brackets the original vector reserve. Clock calibration at both call endpoints prints `[host_before, CUPTI_timestamp, host_after]`; decoder fits their host midpoints and reports bracket widths and fit scale.

```sh
python3 -B check-closure.py
python3 -B cpu-check/run.py /your/workspace/temp/b128-cpu-check
python3 -B decode-trace.py /path/to/native-stderr.txt > /path/to/summary.json
```

Run `check-closure.py` from a repository checkout containing `../../native-batched`; its optional argument sets another ACA baseline directory. Keep generated check outputs under your workspace temp directory. The decoder also accepts the native log's timestamp/OUT prefix before the two exact active JSON markers. `decode_log(path)` returns every decoded interval in each call's `_decoded` dictionary and all compute/H2D-supply gaps. The CLI prints active/coverage/host sums and ten largest gaps per type. `compute_gaps` separates observed H2D-readiness overlap from post-readiness/non-encode idle. These are measured dependency coordinates; root performs critical-path attribution against host pack/owned/wait and cross-lane overlap. Critical time available to a complete host-lookahead mechanism is a separate analysis result.

## Decision gate

Root owns one native B128 point: 1 warm + 3 fresh, formal count 0. Use that native's original three scored times for the perturbation median. ACA qualified reference is 89.648935 ms; the symmetric ≤3% interval is **86.95946695–92.33840305 ms**, using `abs(native_scored_median / 89.648935 - 1) <= 0.03`. CUPTI window times in the decoder are diagnostic coordinates. Build/ABI, exact, record-contract, and timing gates precede any next optimization investment. A failed gate terminates this diagnostic round and records its terminal state.

Local checks cover source closure, fixed host callback protocol, late external mapping, both memcpy keys, interval codec/aggregation/dependencies, and payload size. GPU compilation, GPU exact, real CUPTI record completeness, and real timing perturbation belong to the single root native run.
