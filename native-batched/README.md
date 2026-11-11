# ByteCRT: independent serial batch baseline

This self-contained artifact implements `negacyclic-batched@1.0.0`, B=1..128 at N=65536/W=1747. It applies the exact C18 RNS / cuPQC NTT / ByteCRT path to each independent coefficient pair in order, appending a complete standard-vector-owned output inside `fherma_run`. B=1 preserves the original product path.

The platform packages only the selected artifact directory. `core/` carries the nine required C18 dependency files; the mathematical bodies are identical to `../native-smoke`, with the generated Point header resolved locally. `CMakeLists.txt` uses these local files. The generated `fherma.h` and `main.cpp` match the published batch signature.

The first B0 native job at a311487 reached the target and stopped at CMake configuration because sibling `../native-smoke/io-copy.cpp` was outside the selected artifact. Cases and timings from that job are zero. This corrected serial snapshot supplies the complete selected directory. Native full B1/B128 exactness and timing follow the official target job.

The original C18 single-product implementation and its formal two-point result remain frozen separately.

## Build

Build inside this directory with the official `fherma/cupqc` image (cuPQC 0.6, CUDA 12.8). `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`. The runner supplies its generated harness at this directory root.
