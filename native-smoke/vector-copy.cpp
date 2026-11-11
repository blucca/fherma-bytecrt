// ByteCRT large host vector-copy interception, Apache-2.0.
// GNU ld wraps only this answer's undefined memcpy/memmove references.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <omp.h>

extern "C" void* __real_memcpy(void*, const void*, std::size_t);
extern "C" void* __real_memmove(void*, const void*, std::size_t);

#ifdef BYTECRT_VECTOR_COPY_CHECK
extern "C" void bytecrt_vector_copy_observe(int, bool, std::size_t);
#endif

namespace {
constexpr std::size_t kParallelBytes = 8 * 1024 * 1024;

void* parallel_copy(void* dest, const void* source, std::size_t bytes) {
    const int workers = std::min(12, omp_get_num_procs());
    const std::size_t chunk_bytes = bytes / 12, extra = bytes % 12;
    auto* dst = static_cast<unsigned char*>(dest);
    const auto* src = static_cast<const unsigned char*>(source);
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int task = 0; task < 12; ++task) {
        const std::size_t chunk = static_cast<std::size_t>(task);
        const std::size_t begin = chunk_bytes * chunk + std::min(extra, chunk);
        const std::size_t count = chunk_bytes + (chunk < extra);
        // --wrap resolves __real_memcpy directly to libc, terminating interception.
        __real_memcpy(dst + begin, src + begin, count);
    }
    return dest;  // Implicit OpenMP barrier has completed every byte chunk.
}
}  // namespace

extern "C" void* __wrap_memcpy(void* dest, const void* source, std::size_t bytes) {
    // Standard memcpy callers supply disjoint source/destination ranges.
    const bool parallel = bytes >= kParallelBytes;
#ifdef BYTECRT_VECTOR_COPY_CHECK
    bytecrt_vector_copy_observe(1, parallel, bytes);
#endif
    return parallel ? parallel_copy(dest, source, bytes) : __real_memcpy(dest, source, bytes);
}

extern "C" void* __wrap_memmove(void* dest, const void* source, std::size_t bytes) {
    const uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
    const uintptr_t src = reinterpret_cast<uintptr_t>(source);
    const bool disjoint = dst >= src ? dst - src >= bytes : src - dst >= bytes;
    const bool parallel = bytes >= kParallelBytes && disjoint;
#ifdef BYTECRT_VECTOR_COPY_CHECK
    bytecrt_vector_copy_observe(2, parallel, bytes);
#endif
    return parallel ? parallel_copy(dest, source, bytes) : __real_memmove(dest, source, bytes);
}
