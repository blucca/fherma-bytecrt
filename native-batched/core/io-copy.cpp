// ByteCRT fixed-input host packing, Apache-2.0.
// The caller supplies separate source operands and 2*words destination words.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include "nt-copy.hpp"

void bytecrt_pack_inputs(uint32_t* dest, const uint32_t* a, const uint32_t* b,
                         std::size_t words) {
    const int workers = std::min(12, omp_get_num_procs());
    const std::size_t chunk_words = words / 6, extra = words % 6;
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int task = 0; task < 12; ++task) {
        const std::size_t chunk = static_cast<std::size_t>(task % 6);
        const std::size_t begin = chunk_words * chunk + std::min(extra, chunk);
        const std::size_t count = chunk_words + (chunk < extra);
        if (count) {
            const uint32_t* src = task < 6 ? a : b;
            uint32_t* dst = dest + (task < 6 ? 0 : words);
            std::memcpy(dst + begin, src + begin, count * sizeof(uint32_t));
        }
    }
    // The parallel-for's implicit barrier completes all copies before return.
}

// One operand is made DMA-ready before the caller starts packing the other.
// Twelve disjoint chunks use the same capped worker count. B128 input traffic
// streams past the CPU caches; each worker fences before the return barrier.
void bytecrt_pack_operand(uint32_t* dest, const uint32_t* source,
                          std::size_t words) {
    const int workers = std::min(12, omp_get_num_procs());
    const std::size_t chunk_words = words / 12, extra = words % 12;
#pragma omp parallel num_threads(workers)
    {
#pragma omp for schedule(static) nowait
        for (int task = 0; task < 12; ++task) {
            const std::size_t chunk = static_cast<std::size_t>(task);
            const std::size_t begin = chunk_words * chunk + std::min(extra, chunk);
            const std::size_t count = chunk_words + (chunk < extra);
            if (count)
                bytecrt_nt_copy_unfenced(reinterpret_cast<unsigned char*>(dest + begin),
                    reinterpret_cast<const unsigned char*>(source + begin), count * sizeof(uint32_t));
        }
        _mm_sfence();
    }
    // This barrier completes the operand before its pinned H2D submission.
}
