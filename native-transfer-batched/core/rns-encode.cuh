// ByteCRT wide-limb to canonical RNS encoding, Apache-2.0.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace bytecrt {

// R = 2^32, p odd and p < 2^31, neg_inverse = -p^-1 (mod R).
// a may use all 32 bits; b is canonical (b < p). The sum below fits u64:
// x < R*p and t*p < R*p, so x+t*p < 2*R*p < 2^64.
// Its low word is zero and u < 2*p; one subtraction gives [0,p).
inline __host__ __device__ uint32_t mont_mul_word(uint32_t a, uint32_t b,
                                                 uint32_t p,
                                                 uint32_t neg_inverse) {
    const uint64_t x = uint64_t{a} * b;
    const uint32_t t = static_cast<uint32_t>(x) * neg_inverse;
    const uint32_t u = static_cast<uint32_t>((x + uint64_t{t} * p) >> 32);
    return u >= p ? u - p : u;
}

namespace detail {

// A CTA shares 16 input coefficients among eight primes. Each half warp owns
// one prime and writes 16 consecutive coefficients of its output polynomial.
// The odd shared stride gives distinct banks for the 16 coefficient lanes.
template <unsigned L, unsigned P>
__global__ void encode_rns16x8(unsigned N,
                              const uint32_t* __restrict__ inputs,
                              const uint32_t* __restrict__ primes,
                              const uint32_t* __restrict__ neg_inverses,
                              const uint32_t* __restrict__ powers_mont,
                              uint32_t* __restrict__ ab) {
    constexpr unsigned coefficients = 16, prime_tile = 8, threads = 128;
    constexpr unsigned shared_stride = L | 1;
    static_assert(threads == coefficients * prime_tile, "RNS tile shape");
    static_assert((L == 55 && P == 114) || (L == 28 && P == 57),
                  "ByteCRT full-width RNS geometries");
    __shared__ uint32_t shared_inputs[coefficients][shared_stride];

    const unsigned tid = threadIdx.x;
    const unsigned first_coefficient = blockIdx.x * coefficients;
    const unsigned operand = blockIdx.z;
    const std::size_t input_base = std::size_t{operand} * N * L;
    for (unsigned word = tid; word < coefficients * L; word += threads) {
        const unsigned local_coefficient = word / L;
        const unsigned limb = word % L;
        const unsigned coefficient = first_coefficient + local_coefficient;
        shared_inputs[local_coefficient][limb] = coefficient < N
            ? inputs[input_base + std::size_t{coefficient} * L + limb]
            : 0;
    }
    __syncthreads();

    const unsigned local_coefficient = tid % coefficients;
    const unsigned coefficient = first_coefficient + local_coefficient;
    const unsigned prime = blockIdx.y * prime_tile + tid / coefficients;
    // Padding threads have completed the shared load and its sole barrier.
    if (coefficient >= N || prime >= P) return;

    const uint32_t p = primes[prime];
    const uint32_t neg_inverse = neg_inverses[prime];
    const std::size_t power_base = std::size_t{prime} * L;
    uint32_t residue = 0;
    #pragma unroll
    for (unsigned limb = 0; limb < L; ++limb) {
        const uint32_t term = mont_mul_word(shared_inputs[local_coefficient][limb],
                                            powers_mont[power_base + limb],
                                            p, neg_inverse);
        // Both addends are below p < 2^31, so this canonical sum fits u32.
        const uint32_t sum = residue + term;
        residue = sum >= p ? sum - p : sum;
    }
    ab[(std::size_t{operand} * P + prime) * N + coefficient] = residue;
}

}  // namespace detail

// Input: [2][N][L] contiguous u32 limbs, A followed by B, little endian.
// Setup: p[P], neg_inverse[P], powers_mont[P][L], with
// powers_mont[j][limb] = 2^(32*limb)*R mod p[j], all canonical.
// Output: tightly packed canonical ab[2][P][N]. Coefficient and prime tails
// stay within these extents; the caller provides disjoint input/output buffers.
// This asynchronous launcher uses the caller's stream and CUDA error checks.
inline void launch_encode(unsigned N, unsigned L, unsigned P,
                           const uint32_t* d_inputs,
                           const uint32_t* d_p,
                           const uint32_t* d_neg_inverse,
                           const uint32_t* d_powers_mont,
                           uint32_t* d_ab,
                           cudaStream_t stream = nullptr) {
    if (N == 0)
        throw std::runtime_error("ByteCRT RNS encoding N must be positive");
    const dim3 grid((N - 1) / 16 + 1, (P + 7) / 8, 2);
    if (L == 55 && P == 114)
        detail::encode_rns16x8<55, 114><<<grid, 128, 0, stream>>>(
            N, d_inputs, d_p, d_neg_inverse, d_powers_mont, d_ab);
    else if (L == 28 && P == 57)
        detail::encode_rns16x8<28, 57><<<grid, 128, 0, stream>>>(
            N, d_inputs, d_p, d_neg_inverse, d_powers_mont, d_ab);
    else
        throw std::runtime_error("ByteCRT RNS encoding requires (L,P)=(55,114) or (28,57)");
}

}  // namespace bytecrt
