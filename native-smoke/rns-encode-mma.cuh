// ByteCRT exact unsigned-byte tensor RNS input encoding, Apache-2.0.
#pragma once

#include "rns-encode.cuh"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bytecrt {

// Fixed setup only. B has 7*P column-major byte columns, each K bytes, where
// K=ceil(4*L/32)*32 (224 or 128). For w_l=2^(32*l) mod p[j], column 7*j+d
// contains byte_(d-i)(w_l) at k=4*l+i, with zero for byte indices outside 0..3.
// Consequently its dot with one input byte row is product digit sum d.
inline std::vector<uint8_t> make_encode_b(
    unsigned L, unsigned P, const std::vector<uint32_t>& primes,
    const std::vector<uint32_t>& powers_mont) {
    if (!((L == 55 && P == 114) || (L == 28 && P == 57)))
        throw std::runtime_error("ByteCRT tensor encoding requires (L,P)=(55,114) or (28,57)");
    if (primes.size() != P || powers_mont.size() != std::size_t{P} * L)
        throw std::runtime_error("ByteCRT tensor encoding setup array shape");
    const unsigned K = (4 * L + 31) / 32 * 32;
    std::vector<uint8_t> b(std::size_t{7} * P * K, 0);
    for (unsigned j = 0; j < P; ++j) {
        const uint32_t p = primes[j];
        if (p <= 1 || !(p & 1) || p >= (uint32_t{1} << 31))
            throw std::runtime_error("ByteCRT tensor encoding requires odd 1<p<2^31");
        uint32_t inverse = 1;
        for (unsigned step = 0; step < 5; ++step) inverse *= 2u - p * inverse;
        const uint32_t neg_inverse = 0u - inverse;
        for (unsigned limb = 0; limb < L; ++limb) {
            const uint32_t power = powers_mont[std::size_t{j} * L + limb];
            if (power >= p)
                throw std::runtime_error("ByteCRT tensor encoding Montgomery power must be canonical");
            const uint32_t weight = mont_mul_word(power, 1, p, neg_inverse);
            for (unsigned d = 0; d < 7; ++d)
                for (unsigned i = 0; i < 4; ++i)
                    if (d >= i && d - i < 4)
                        b[(std::size_t{7} * j + d) * K + 4 * limb + i] =
                            static_cast<uint8_t>(weight >> (8 * (d - i)));
        }
    }
    return b;
}

namespace detail {

// x=sum_l input_l*(2^(32*l) mod p) < L*2^32*p < 2^69.
// Seven nonnegative byte sums reconstruct x as three base-2^32 words.
// mu=floor(2^64/p)=UINT64_MAX/p for odd p>1. The low-word Barrett estimate
// undershoots by at most one, giving remainder<2*p. The high word is <28;
// multiplying by powers_mont[j*L+2] returns high*2^64 mod p.
__device__ __forceinline__ uint32_t encode_mma_reduce(
    const int32_t* sums, uint32_t p, uint32_t neg_inverse,
    uint32_t r2_mont, uint64_t mu) {
    const uint64_t lo = uint64_t{static_cast<uint32_t>(sums[0])} +
                        (uint64_t{static_cast<uint32_t>(sums[1])} << 8) +
                        (uint64_t{static_cast<uint32_t>(sums[2])} << 16) +
                        (uint64_t{static_cast<uint32_t>(sums[3])} << 24);
    const uint64_t hi = (lo >> 32) + uint64_t{static_cast<uint32_t>(sums[4])} +
                        (uint64_t{static_cast<uint32_t>(sums[5])} << 8) +
                        (uint64_t{static_cast<uint32_t>(sums[6])} << 16);
    const uint32_t word0 = static_cast<uint32_t>(lo);
    const uint32_t word1 = static_cast<uint32_t>(hi);
    const uint32_t high = static_cast<uint32_t>(hi >> 32);
    const uint64_t lo64 = uint64_t{word0} | (uint64_t{word1} << 32);
    const uint64_t qhat = __umul64hi(lo64, mu);
    const uint64_t remainder = lo64 - qhat * p;
    const uint32_t low_residue = static_cast<uint32_t>(
        remainder >= p ? remainder - p : remainder);
    const uint32_t high_residue = mont_mul_word(high, r2_mont, p, neg_inverse);
    // Both residues are canonical and p<2^31; their sum fits u32.
    const uint32_t sum = low_residue + high_residue;
    return sum >= p ? sum - p : sum;
}

}  // namespace detail

// MT=16, NT=56, seven full warps: the register gather/scatter follows G1's
// m16n8k32 unsigned-byte layout. Each CTA loads its input tile once, then
// iterates over eight primes (seven byte-product digits each) per group.
template <unsigned L, unsigned P, unsigned Kpad>
__global__ void encode_mma16(unsigned N,
                            const uint32_t* __restrict__ inputs,
                            const uint8_t* __restrict__ b,
                            const uint32_t* __restrict__ primes,
                            const uint32_t* __restrict__ neg_inverses,
                            const uint32_t* __restrict__ powers_mont,
                            const uint64_t* __restrict__ mu,
                            uint32_t* __restrict__ ab) {
    static_assert((L == 55 && P == 114 && Kpad == 224) ||
                  (L == 28 && P == 57 && Kpad == 128), "tensor RNS geometry");
    static_assert(4 * L * 255 * 255 < (uint32_t{1} << 31), "exact s32 byte dots");
    constexpr unsigned MT = 16, NT = 56, threads = 224;
    constexpr unsigned groups = (P + 7) / 8;
    constexpr unsigned word_stride = Kpad / 4;
    __shared__ __align__(16) uint8_t shared_a[MT][Kpad];
    __shared__ int32_t shared_c[MT][NT + 1];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    const unsigned first_coefficient = blockIdx.x * MT;
    const unsigned operand = blockIdx.y;
    const std::size_t input_base = std::size_t{operand} * N * L;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    for (unsigned word = tid; word < MT * word_stride; word += threads) {
        const unsigned coefficient = first_coefficient + word / word_stride;
        const unsigned limb = word % word_stride;
        shared_words[word] = coefficient < N && limb < L
            ? inputs[input_base + std::size_t{coefficient} * L + limb]
            : 0;
    }
    __syncthreads();

    for (unsigned group = 0; group < groups; ++group) {
        const unsigned b_col = group * NT + warp * 8 + g;
        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        #pragma unroll
        for (unsigned k = 0; k < Kpad; k += 32) {
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane + 16);
            uint32_t b0 = 0, b1 = 0;
            if (b_col < 7 * P) {
                const uint8_t* column = b + std::size_t{b_col} * Kpad + k + k_lane;
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

        if (tid < 128) {
            const unsigned local_prime = tid / MT;
            const unsigned coefficient = first_coefficient + tid % MT;
            const unsigned prime = group * 8 + local_prime;
            if (coefficient < N && prime < P) {
                const int32_t* sums = shared_c[tid % MT] + 7 * local_prime;
                const uint32_t residue = detail::encode_mma_reduce(
                    sums, primes[prime], neg_inverses[prime],
                    powers_mont[std::size_t{prime} * L + 2], mu[prime]);
                ab[(std::size_t{operand} * P + prime) * N + coefficient] = residue;
            }
        }
        // All 224 threads reach both group barriers, including tail lanes.
        __syncthreads();
    }
}

// Inputs are A then B, coefficient-major [2][N][L] little-endian u32 limbs.
// b is make_encode_b's fixed byte matrix; mu[j]=UINT64_MAX/p[j]; the other
// setup contracts match launch_encode. Output is compact canonical [2][P][N].
// The asynchronous launcher uses the caller's stream and CUDA status checks.
inline void launch_encode_mma(unsigned N, unsigned L, unsigned P,
                               const uint32_t* d_inputs,
                               const uint8_t* d_b,
                               const uint32_t* d_p,
                               const uint32_t* d_neg_inverse,
                               const uint32_t* d_powers_mont,
                               const uint64_t* d_mu,
                               uint32_t* d_ab,
                               cudaStream_t stream = nullptr) {
    if (N == 0)
        throw std::runtime_error("ByteCRT tensor encoding N must be positive");
    const dim3 grid((N - 1) / 16 + 1, 2);
    if (L == 55 && P == 114)
        encode_mma16<55, 114, 224><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_neg_inverse, d_powers_mont, d_mu, d_ab);
    else if (L == 28 && P == 57)
        encode_mma16<28, 57, 128><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_neg_inverse, d_powers_mont, d_mu, d_ab);
    else
        throw std::runtime_error("ByteCRT tensor encoding requires (L,P)=(55,114) or (28,57)");
}

}  // namespace bytecrt
