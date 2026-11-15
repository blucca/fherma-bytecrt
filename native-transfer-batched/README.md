# ByteCRT: run-only byte-copy transfer port

This isolated artifact implements `polymul/negacyclic-batched@1.0.0` at **B=1..128, N=65536, W=1747, L=55**. B1 also retains the original auxiliary N32768/W868/L28 dispatch. It has its own `core/`; CMake packages local sources only. All nine mathematical/core files retain frozen qualified ACA bytes from `aca09d65fe30e6c9497f7a0f038e9a9f4b7384df`. Existing ACA and single-product C18 artifacts stay in place.

## Transfer contract

- **init** prepares the original Point tables and four independent legacy States (one for B1), original pinned input/output staging and streams/events. It additionally allocates complete original-layout device raw A/B/C arrays.
- **h2d** uses the frozen twelve-worker disjoint memcpy helper and ordinary H2D copies to send input bytes unchanged. Pinned-slot reuse waits for its previous DMA; the final input stream synchronization completes every copy.
- **run** reads per-product raw A/B offsets. B>1 uses the original full encoder and one compute stream through four scratch slots; B1 keeps the original range encoder, two operand streams and A-forward join. All RNS encoding, padding, transposition, NTT, inverse weighting, CRT and canonical reduction remain GPU run work. G1 writes the original natural `[N,L]` bytes directly to each raw C product range. The final `cudaDeviceSynchronize` completes all device work before return. The returned vector carries shape `[B,N,L]` and full reserved capacity with size zero; it performs no bulk host zero-fill.
- **d2h** waits for the completed result event, copies raw C chunks into the original pinned output slots and constructs a complete independent owned vector using standard inserts in product order. These are result-byte copies only; no output transformation runs here.

At B128, the raw A/B/C device buffers add **5,536,481,280 B**. With the retained legacy arrays, explicit device memory is **6,912,010,864 B**; input A/B, owned output C and pinned staging total **5,709,496,320 host B**. Runtime context/allocator overhead and harness transients are checked at the fresh native gate. See `transfer-memory.json`.

## Validation and timing scope

The [small lifted host method proof](host-transfer-method.md) passes B1/B2/B128 and auxiliary B1 dispatch, deferred staging reuse, original bytes, raw product order, complete run synchronization and two-call independent output ownership. Mathematical exactness and performance use the root-owned fresh native run.

This source checkpoint has **0 new CUDA measurements** and no new ranked score. The prior ACA 89.648935 ms is a no-hook whole-call measurement; the updated challenge ranks only `run`. New results will report `h2d_s` and `d2h_s` beside the completed device run time.

## Build

Use the official `fherma/cupqc` image (cuPQC 0.6.0/CUDA 12.8), from this directory. `CMakeLists.txt` retains the qualified ACA build and local core sources. The platform regenerates `fherma.h` and `main.cpp` before each fresh commit measurement. The checked-in `fherma.h` and `main.cpp` are unmodified official fherma-lang **0.20.0** output for the published signature. They declare and call the exact hooks, move the reserved answer into d2h, and report `h2d_s`/`d2h_s` separately from `seconds`. All three authored phases complete their device work explicitly. Generator provenance and file hashes are in `harness-provenance.json`.
