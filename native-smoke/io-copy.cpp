// ByteCRT fixed-input host packing, Apache-2.0.
// The caller supplies separate source operands and 2*words destination words.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <omp.h>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define BYTECRT_INPUT_X86_STREAM 1

__attribute__((target("avx512f")))
static void bytecrt_stream_copy(void* dest, const void* source, std::size_t bytes) {
    auto* dst = static_cast<unsigned char*>(dest);
    auto* src = static_cast<const unsigned char*>(source);
    const std::size_t prefix = std::min(
        bytes, (64 - (reinterpret_cast<std::uintptr_t>(dst) & 63)) & 63);
    std::memcpy(dst, src, prefix);
    dst += prefix;
    src += prefix;
    bytes -= prefix;
    while (bytes >= 64) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                           _mm512_loadu_si512(src));
        dst += 64;
        src += 64;
        bytes -= 64;
    }
    std::memcpy(dst, src, bytes);
}
#else
#define BYTECRT_INPUT_X86_STREAM 0
#endif

bool bytecrt_input_stream_enabled() {
#if BYTECRT_INPUT_X86_STREAM
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void bytecrt_pack_inputs(uint32_t* dest, const uint32_t* a, const uint32_t* b,
                         std::size_t words) {
    const int workers = std::min(12, omp_get_num_procs());
    const std::size_t chunk_words = words / 6, extra = words % 6;
#if BYTECRT_INPUT_X86_STREAM
    // This runtime feature gate includes the OS AVX-512 state support.
    const bool use_stream = bytecrt_input_stream_enabled();
#endif
#pragma omp parallel num_threads(workers)
    {
#pragma omp for schedule(static) nowait
        for (int task = 0; task < 12; ++task) {
            const std::size_t chunk = static_cast<std::size_t>(task % 6);
            const std::size_t begin = chunk_words * chunk + std::min(extra, chunk);
            const std::size_t count = chunk_words + (chunk < extra);
            if (count) {
                const uint32_t* src = task < 6 ? a : b;
                uint32_t* dst = dest + (task < 6 ? 0 : words);
#if BYTECRT_INPUT_X86_STREAM
                if (use_stream)
                    bytecrt_stream_copy(dst + begin, src + begin,
                                        count * sizeof(uint32_t));
                else
#endif
                    std::memcpy(dst + begin, src + begin, count * sizeof(uint32_t));
            }
        }
#if BYTECRT_INPUT_X86_STREAM
        // Each worker drains its streaming stores before the region's barrier.
        if (use_stream) __asm__ __volatile__("sfence" ::: "memory");
#endif
    }
    // The parallel region's implicit barrier completes all copies before return.
}
