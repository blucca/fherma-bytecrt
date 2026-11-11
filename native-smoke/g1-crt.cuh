// ByteCRT G1: exact unsigned-byte tensor CRT, Apache-2.0.
// The two full-width points use MT=16, seven full warps and NT=56 byte columns.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace bytecrt {
namespace detail {

// q=2^W-gap. The input is the nonnegative CRT weighted sum, below 2^(W+38).
// W>70 makes two folds sufficient. The first high*gap may span 70 bits.
template <unsigned W, unsigned L>
__device__ __forceinline__ void fold_near_power(uint32_t* digits, uint32_t gap) {
    constexpr unsigned r = W - 32 * (L - 1);
    constexpr uint32_t mask = (uint32_t{1} << r) - 1;
    static_assert(W > 70 && r > 0 && r < 32, "G1 partial top limb");

    const uint64_t high = (uint64_t{digits[L - 1]} >> r) |
                          (uint64_t{digits[L]} << (32 - r)) |
                          (uint64_t{digits[L + 1]} << (64 - r));
    digits[L - 1] &= mask;
    digits[L] = 0;
    digits[L + 1] = 0;

    // Construct all three words of high*gap without a truncated u64 product.
    const uint64_t product0 = uint64_t{static_cast<uint32_t>(high)} * gap;
    const uint64_t product1 = uint64_t{static_cast<uint32_t>(high >> 32)} * gap +
                              (product0 >> 32);
    uint64_t total = uint64_t{digits[0]} + static_cast<uint32_t>(product0);
    digits[0] = static_cast<uint32_t>(total);
    total = uint64_t{digits[1]} + static_cast<uint32_t>(product1) + (total >> 32);
    digits[1] = static_cast<uint32_t>(total);
    total = uint64_t{digits[2]} + static_cast<uint32_t>(product1 >> 32) + (total >> 32);
    digits[2] = static_cast<uint32_t>(total);
    for (unsigned limb = 3; limb < L; ++limb) {
        total = uint64_t{digits[limb]} + (total >> 32);
        digits[limb] = static_cast<uint32_t>(total);
    }
    digits[L] = static_cast<uint32_t>(total >> 32);
}

template <unsigned W, unsigned L>
__device__ __forceinline__ void reduce_near_power(uint32_t* digits,
                                                const uint32_t* q,
                                                uint32_t gap) {
    fold_near_power<W, L>(digits, gap);
    fold_near_power<W, L>(digits, gap);

    bool ge = true;
    for (int limb = static_cast<int>(L) - 1; limb >= 0; --limb) {
        if (digits[limb] != q[limb]) {
            ge = digits[limb] > q[limb];
            break;
        }
    }
    if (ge) {
        uint64_t borrow = 0;
        for (unsigned limb = 0; limb < L; ++limb) {
            const uint64_t subtrahend = uint64_t{q[limb]} + borrow;
            const uint64_t word = digits[limb];
            digits[limb] = static_cast<uint32_t>(word - subtrahend);
            borrow = word < subtrahend;
        }
    }
}

}  // namespace detail

// A: row-major M*Kpad bytes; each row starts with P weighted canonical u32 terms t_j = c_j * inv(P_j) mod p_j,
// followed by the overwritten alpha u32 and zero padding. B: 7L column-major
// byte columns, each Kpad bytes. M is a multiple of 16; grid.x=M/16.
// Transposed=true selects full-pipeline A_t[id][prime_word][i] instead.
// Fixed invp[j]=1/p_j and q are setup data. All input-dependent work stays here.
template <unsigned W, unsigned L, unsigned P, unsigned Kpad, bool Transposed = false>
__global__ void g1_crt16(const uint8_t* __restrict__ a,
                        const uint8_t* __restrict__ b,
                        const double* __restrict__ invp,
                        const uint32_t* __restrict__ q,
                        uint32_t gap,
                        uint32_t* __restrict__ out) {
    static_assert((W == 1747 && L == 55 && P == 114 && Kpad == 480) ||
                  (W == 868 && L == 28 && P == 57 && Kpad == 256),
                  "G1 full-width geometries");
    constexpr unsigned MT = 16, NT = 56, threads = 224;
    constexpr unsigned digit_stride = (L + 2) | 1;
    constexpr unsigned groups = (L + 7) / 8;
    // Four padding words make the m16n8k32 A gather bank-bijective:
    // stride_words mod 32 is 28 (main) or 4 (aux), bank = g*stride+t.
    constexpr unsigned row_words = Kpad / 4;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned shared_word_stride = shared_stride / 4;
    constexpr unsigned transpose_m = 256;
    constexpr unsigned transpose_k = W == 1747 ? 256 : 128;
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];
    __shared__ int32_t shared_c[MT][NT + 1];
    __shared__ uint32_t digits[MT][digit_stride];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    const unsigned id = Transposed ? blockIdx.x % transpose_m : 0;
    const unsigned first_row = (Transposed ? blockIdx.x / transpose_m : blockIdx.x) * MT;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    const auto* a_words = reinterpret_cast<const uint32_t*>(a);
    for (unsigned word = tid; word < MT * row_words; word += threads) {
        if constexpr (Transposed) {
            // A_t[id][prime_word][i]. A half-warp reads 16 consecutive i.
            const unsigned row = word % MT, prime_word = word / MT;
            shared_words[row * shared_word_stride + prime_word] =
                a_words[(size_t(id) * row_words + prime_word) * transpose_k + first_row + row];
        } else {
            const unsigned row = word / row_words, prime_word = word % row_words;
            shared_words[row * shared_word_stride + prime_word] =
                a_words[(size_t(first_row) + row) * row_words + prime_word];
        }
    }
    __syncthreads();

    if (tid < MT) {
        uint32_t* terms = shared_words + tid * shared_word_stride;
        double alpha_sum = 0.0;
        for (unsigned j = 0; j < P; ++j)
            alpha_sum = __dadd_rn(alpha_sum, __dmul_rn(static_cast<double>(terms[j]), invp[j]));
        // Centered CRT has |x/product|<1/4, with FP64 error inside that margin.
        terms[P] = __double2uint_rn(alpha_sum);
    }
    __syncthreads();

    uint64_t carry = 0;
    for (unsigned group = 0; group < groups; ++group) {
        const unsigned b_col = group * NT + warp * 8 + g;
        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (unsigned k = 0; k < Kpad; k += 32) {
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane + 16);
            uint32_t b0 = 0, b1 = 0;
            if (b_col < 7 * L) {
                const uint8_t* column = b + b_col * Kpad + k + k_lane;
                b0 = *reinterpret_cast<const uint32_t*>(column);
                b1 = *reinterpret_cast<const uint32_t*>(column + 16);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        const unsigned c_col = warp * 8 + 2 * t;
        shared_c[g][c_col] = c0;
        shared_c[g][c_col + 1] = c1;
        shared_c[g + 8][c_col] = c2;
        shared_c[g + 8][c_col + 1] = c3;
        __syncthreads();

        if (tid < MT) {
            for (unsigned local_limb = 0; local_limb < 8; ++local_limb) {
                const unsigned limb = group * 8 + local_limb;
                if (limb < L) {
                    const int32_t* sums = shared_c[tid] + 7 * local_limb;
                    const uint64_t lo = uint64_t{static_cast<uint32_t>(sums[0])} +
                        (uint64_t{static_cast<uint32_t>(sums[1])} << 8) +
                        (uint64_t{static_cast<uint32_t>(sums[2])} << 16) +
                        (uint64_t{static_cast<uint32_t>(sums[3])} << 24);
                    const uint64_t hi = uint64_t{static_cast<uint32_t>(sums[4])} +
                        (uint64_t{static_cast<uint32_t>(sums[5])} << 8) +
                        (uint64_t{static_cast<uint32_t>(sums[6])} << 16);
                    const uint64_t total = carry + lo;
                    digits[tid][limb] = static_cast<uint32_t>(total);
                    carry = (total >> 32) + hi;
                }
            }
        }
        // All readers finish before the next group overwrites shared C.
        __syncthreads();
    }

    if (tid < MT) {
        digits[tid][L] = static_cast<uint32_t>(carry);
        digits[tid][L + 1] = static_cast<uint32_t>(carry >> 32);
        detail::reduce_near_power<W, L>(digits[tid], q, gap);
    }
    __syncthreads();
    for (unsigned word = tid; word < MT * L; word += threads) {
        const unsigned row = word / L;
        const unsigned coefficient = Transposed ? id + (first_row + row) * transpose_m
                                                : first_row + row;
        out[size_t(coefficient) * L + word % L] = digits[row][word % L];
    }
}

inline void launch_g1_crt(unsigned W, unsigned M,
                          const uint8_t* a, const uint8_t* b,
                          const double* invp, const uint32_t* q,
                          uint32_t gap, uint32_t* out,
                          cudaStream_t stream = nullptr) {
    if (M == 0 || M % 16 != 0)
        throw std::runtime_error("ByteCRT G1 M must be a positive multiple of 16");
    if (W == 1747)
        g1_crt16<1747, 55, 114, 480><<<M / 16, 224, 0, stream>>>(a, b, invp, q, gap, out);
    else if (W == 868)
        g1_crt16<868, 28, 57, 256><<<M / 16, 224, 0, stream>>>(a, b, invp, q, gap, out);
    else
        throw std::runtime_error("ByteCRT G1 W must be 1747 or 868");
}

// Full pipeline A_t has shape [256][Kpad/4][N/256] ordinary u32 words.
// Its prime words are input-dependent; alpha and padding stay fixed zero.
// Grid order tile*256+id gives each CTA 16 consecutive i for one id, while
// output coefficient id+i*256 preserves the official natural coefficient ABI.
inline void launch_g1_crt_transposed(unsigned W, unsigned M,
                                     const uint8_t* a_t, const uint8_t* b,
                                     const double* invp, const uint32_t* q,
                                     uint32_t gap, uint32_t* out,
                                     cudaStream_t stream = nullptr) {
    if (W == 1747 && M == 65536)
        g1_crt16<1747, 55, 114, 480, true><<<M / 16, 224, 0, stream>>>(a_t, b, invp, q, gap, out);
    else if (W == 868 && M == 32768)
        g1_crt16<868, 28, 57, 256, true><<<M / 16, 224, 0, stream>>>(a_t, b, invp, q, gap, out);
    else
        throw std::runtime_error("ByteCRT transposed G1 requires full staged geometries");
}

}  // namespace bytecrt

// Fixed setup selfcheck; integrate the call into init, outside whole-call timing.
void bytecrt_g1_smoke();
