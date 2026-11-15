# ByteCRT byte-copy transfer port

Work in progress, isolated from the qualified ACA and C18 entries. This artifact contains its own complete frozen ACA core; no sibling sources are required.

Baseline: `aca09d65fe30e6c9497f7a0f038e9a9f4b7384df` (`native-batched`). Mathematical CUDA bodies are retained byte for byte. The new wrapper separates pure input/result copies into the exact FHERMA h2d/d2h callbacks and keeps all transformations and GPU completion inside run.

The initial copied legacy generated header/main are pending canonical updated harness replacement. This checkpoint is not a transfer-aware native measurement.
