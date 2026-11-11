// ByteCRT: unsigned m16n8k32 native register contract, Apache-2.0.
// CUDA 12.8.1 / PTX 8.7, §9.7.14.5.10. One full warp, fixed setup data.
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
__global__ void matrix_fixture(const uint8_t* a, const uint8_t* b, int32_t* out) {
    const unsigned lane = threadIdx.x;
    const unsigned g = lane / 4, t = lane % 4, k = 4 * t;
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a + g * 32 + k);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(a + (g + 8) * 32 + k);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(a + g * 32 + k + 16);
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(a + (g + 8) * 32 + k + 16);
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + g * 32 + k);
    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + g * 32 + k + 16);
    const int32_t c0 = static_cast<int32_t>(g * 1009) - static_cast<int32_t>(2 * t * 37);
    const int32_t c1 = c0 - 37;
    const int32_t c2 = c0 + 8 * 1009;
    const int32_t c3 = c2 - 37;
    int32_t d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    out[g * 8 + 2 * t] = d0;
    out[g * 8 + 2 * t + 1] = d1;
    out[(g + 8) * 8 + 2 * t] = d2;
    out[(g + 8) * 8 + 2 * t + 1] = d3;
}

void require(cudaError_t e, const char* stage) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ByteCRT MMA ") + stage + ": " + cudaGetErrorString(e));
}
}  // namespace

void bytecrt_mma_smoke() {
    std::array<uint8_t, 512> a{};
    std::array<uint8_t, 256> b{};  // column-major, K contiguous
    std::array<int32_t, 128> got{};
    for (unsigned r = 0; r < 16; ++r)
        for (unsigned k = 0; k < 32; ++k)
            a[r * 32 + k] = (r * 71 + k * 29) % 256;
    for (unsigned col = 0; col < 8; ++col)
        for (unsigned k = 0; k < 32; ++k)
            b[col * 32 + k] = (k * 97 + col * 31 + 255) % 256;
    uint8_t* device = nullptr;
    // A+B occupy 768 bytes, so the s32 output begins on a 256-byte boundary.
    require(cudaMalloc(&device, a.size() + b.size() + sizeof(got)), "allocate");
    auto* out = reinterpret_cast<int32_t*>(device + a.size() + b.size());
    require(cudaMemcpy(device, a.data(), a.size(), cudaMemcpyHostToDevice), "copy A");
    require(cudaMemcpy(device + a.size(), b.data(), b.size(), cudaMemcpyHostToDevice), "copy B");
    matrix_fixture<<<1, 32>>>(device, device + a.size(), out);
    require(cudaGetLastError(), "launch");
    require(cudaMemcpy(got.data(), out, sizeof(got), cudaMemcpyDeviceToHost), "copy D");
    require(cudaFree(device), "free");
    for (unsigned r = 0; r < 16; ++r) {
        for (unsigned col = 0; col < 8; ++col) {
            int32_t want = static_cast<int32_t>(r * 1009) - static_cast<int32_t>(col * 37);
            for (unsigned k = 0; k < 32; ++k)
                want += static_cast<int32_t>(a[r * 32 + k]) * b[col * 32 + k];
            if (got[r * 8 + col] != want)
                throw std::runtime_error("ByteCRT MMA cell " + std::to_string(r * 8 + col) +
                                         " got=" + std::to_string(got[r * 8 + col]) +
                                         " want=" + std::to_string(want));
        }
    }
    std::fprintf(stderr, "ByteCRT unsigned MMA: 128 integer cells PASS (fixed setup fixture)\n");
}
