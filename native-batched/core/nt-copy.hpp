// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

extern "C" void* __real_memcpy(void*, const void*, size_t);

// AVX2 target: each caller fences on its own worker after its disjoint tasks.
// Short heads/tails use the existing libc backend; the streaming body is the
// CPU-paired, full-byte-verified primitive from the B128 traffic experiment.
static inline void bytecrt_nt_copy_unfenced(unsigned char* dst, const unsigned char* src,
                                    size_t bytes) {
    const size_t head = std::min(bytes, size_t(-reinterpret_cast<uintptr_t>(dst)) & 63);
    if (head) __real_memcpy(dst, src, head);
    dst += head; src += head; bytes -= head;
    for (; bytes >= 128; dst += 128, src += 128, bytes -= 128) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 64));
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 96));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst), a);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 32), b);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 64), c);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 96), d);
    }
    for (; bytes >= 32; dst += 32, src += 32, bytes -= 32)
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src)));
    if (bytes) __real_memcpy(dst, src, bytes);
}

