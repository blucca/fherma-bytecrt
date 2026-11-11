// ByteCRT exact unsigned-byte tensor RNS input encoding, Apache-2.0.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bytecrt {

// Fixed setup only. B has 4*P column-major byte columns, each K bytes.
// At input byte k, weight=2^(8*k) mod p[j]; column 4*j+d stores byte_d(weight).
// This reduces powers before byte expansion: x=sum_k inputByte[k]*weight[k].
inline std::vector<uint8_t> make_encode_b(
    unsigned L, unsigned P, const std::vector<uint32_t>& primes) {
    if (!((L == 55 && P == 114) || (L == 28 && P == 57)))
        throw std::runtime_error("ByteCRT tensor encoding requires (L,P)=(55,114) or (28,57)");
    if (primes.size() != P)
        throw std::runtime_error("ByteCRT tensor encoding prime array shape");
    const unsigned K = (4 * L + 31) / 32 * 32;
    std::vector<uint8_t> b(std::size_t{4} * P * K, 0);
    for (unsigned j = 0; j < P; ++j) {
        const uint32_t p = primes[j];
        if (p <= 1 || !(p & 1) || p >= (uint32_t{1} << 31))
            throw std::runtime_error("ByteCRT tensor encoding requires odd 1<p<2^31");
        uint32_t power = 1;
        for (unsigned byte = 0; byte < 4 * L; ++byte) {
            for (unsigned d = 0; d < 4; ++d)
                b[(std::size_t{4} * j + d) * K + byte] =
                    static_cast<uint8_t>(power >> (8 * d));
            power = uint32_t(uint64_t{power} * 256 % p);
        }
    }
    return b;
}

namespace detail {

// x < 4*L*255*p < 2^47. Four exact byte-dot sums reconstruct this u64.
// mu=floor(2^64/p)=UINT64_MAX/p for odd p>1; qhat undershoots by at most one.
// One conditional subtraction produces the same canonical coefficient mod p.
__device__ __forceinline__ uint32_t encode_mma_reduce(
    const int32_t* sums, uint32_t p, uint64_t mu) {
    const uint64_t x = uint64_t{static_cast<uint32_t>(sums[0])} +
                       (uint64_t{static_cast<uint32_t>(sums[1])} << 8) +
                       (uint64_t{static_cast<uint32_t>(sums[2])} << 16) +
                       (uint64_t{static_cast<uint32_t>(sums[3])} << 24);
    const uint64_t qhat = __umul64hi(x, mu);
    const uint64_t remainder = x - qhat * p;
    return static_cast<uint32_t>(remainder >= p ? remainder - p : remainder);
}

}  // namespace detail

// MT=16, NT=56, seven full warps: the register gather/scatter follows G1's
// m16n8k32 unsigned-byte layout. Each CTA loads its input tile once, then
// iterates over fourteen primes (four byte-power digits each) per group.
template <unsigned L, unsigned P, unsigned Kpad, bool Transposed = false>
__global__ void encode_mma16(unsigned N,
                            const uint32_t* __restrict__ inputs,
                            const uint8_t* __restrict__ b,
                            const uint32_t* __restrict__ primes,
                            const uint64_t* __restrict__ mu,
                            uint32_t* __restrict__ ab) {
    static_assert((L == 55 && P == 114 && Kpad == 224) ||
                  (L == 28 && P == 57 && Kpad == 128), "tensor RNS geometry");
    static_assert(4 * L * 255 * 255 < (uint32_t{1} << 31), "exact s32 byte dots");
    constexpr unsigned MT = 16, NT = 56, threads = 224;
    constexpr unsigned groups = (P + 13) / 14;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned word_stride = shared_stride / 4;
    // +16 bytes makes each MMA half-row gather bank-distinct at both widths.
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];
    __shared__ int32_t shared_c[MT][NT + 1];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    // Transposed launch: blockIdx.x = tile*256 + id, row r selects
    // coefficient id + (tile*16+r)*256; each row retains contiguous limbs.
    constexpr unsigned transpose_m = 256;
    const unsigned id = Transposed ? blockIdx.x % transpose_m : 0;
    const unsigned first_row = (Transposed ? blockIdx.x / transpose_m : blockIdx.x) * MT;
    const unsigned operand = blockIdx.y;
    const std::size_t input_base = std::size_t{operand} * N * L;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    for (unsigned word = tid; word < MT * word_stride; word += threads) {
        const unsigned row = first_row + word / word_stride;
        const unsigned coefficient = Transposed ? id + row * transpose_m : row;
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
            if (b_col < 4 * P) {
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

        if (tid < threads) {
            const unsigned local_prime = tid / MT;
            const unsigned row = first_row + tid % MT;
            const unsigned coefficient = Transposed ? id + row * transpose_m : row;
            const unsigned prime = group * 14 + local_prime;
            if (coefficient < N && prime < P) {
                const int32_t* sums = shared_c[tid % MT] + 4 * local_prime;
                const uint32_t residue = detail::encode_mma_reduce(
                    sums, primes[prime], mu[prime]);
                const unsigned destination = Transposed ? id * (N / transpose_m) + row : coefficient;
                ab[(std::size_t{operand} * P + prime) * N + destination] = residue;
            }
        }
        // All 224 threads reach both group barriers, including tail lanes.
        __syncthreads();
    }
}

// Fixed MT32: two independent m16 accumulators reuse each B0/B1 load.
// The second A gather happens after the first MMA, keeping its A registers
// dead before the next gather. Each half-warp emits both row r and r+16.
template <unsigned L, unsigned P, unsigned Kpad, bool Transposed = false>
__global__ void encode_mma32(unsigned N,
                            const uint32_t* __restrict__ inputs,
                            const uint8_t* __restrict__ b,
                            const uint32_t* __restrict__ primes,
                            const uint64_t* __restrict__ mu,
                            uint32_t* __restrict__ ab) {
    static_assert((L == 55 && P == 114 && Kpad == 224) ||
                  (L == 28 && P == 57 && Kpad == 128), "tensor RNS geometry");
    static_assert(4 * L * 255 * 255 < (uint32_t{1} << 31), "exact s32 byte dots");
    constexpr unsigned MT = 32, NT = 56, threads = 224;
    constexpr unsigned groups = (P + 13) / 14;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned word_stride = shared_stride / 4;
    // +16 bytes makes each MMA half-row gather bank-distinct at both widths.
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];
    __shared__ int32_t shared_c[MT][NT + 1];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    // Transposed launch: blockIdx.x = tile*256 + id, row r selects
    // coefficient id + (tile*32+r)*256; each row retains contiguous limbs.
    constexpr unsigned transpose_m = 256;
    const unsigned id = Transposed ? blockIdx.x % transpose_m : 0;
    const unsigned first_row = (Transposed ? blockIdx.x / transpose_m : blockIdx.x) * MT;
    const unsigned operand = blockIdx.y;
    const std::size_t input_base = std::size_t{operand} * N * L;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    for (unsigned word = tid; word < MT * word_stride; word += threads) {
        const unsigned row = first_row + word / word_stride;
        const unsigned coefficient = Transposed ? id + row * transpose_m : row;
        const unsigned limb = word % word_stride;
        shared_words[word] = coefficient < N && limb < L
            ? inputs[input_base + std::size_t{coefficient} * L + limb]
            : 0;
    }
    __syncthreads();

    for (unsigned group = 0; group < groups; ++group) {
        const unsigned b_col = group * NT + warp * 8 + g;
        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        int32_t c4 = 0, c5 = 0, c6 = 0, c7 = 0;
        #pragma unroll
        for (unsigned k = 0; k < Kpad; k += 32) {
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane + 16);
            uint32_t b0 = 0, b1 = 0;
            if (b_col < 4 * P) {
                const uint8_t* column = b + std::size_t{b_col} * Kpad + k + k_lane;
                b0 = *reinterpret_cast<const uint32_t*>(column);
                b1 = *reinterpret_cast<const uint32_t*>(column + 16);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            const uint32_t a4 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + k + k_lane);
            const uint32_t a5 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + k + k_lane);
            const uint32_t a6 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + k + k_lane + 16);
            const uint32_t a7 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + k + k_lane + 16);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(c4), "+r"(c5), "+r"(c6), "+r"(c7)
                : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b0), "r"(b1));
        }
        const unsigned c_col = warp * 8 + 2 * t;
        shared_c[g][c_col] = c0;
        shared_c[g][c_col + 1] = c1;
        shared_c[g + 8][c_col] = c2;
        shared_c[g + 8][c_col + 1] = c3;
        shared_c[g + 16][c_col] = c4;
        shared_c[g + 16][c_col + 1] = c5;
        shared_c[g + 24][c_col] = c6;
        shared_c[g + 24][c_col + 1] = c7;
        __syncthreads();

        const unsigned local_prime = tid / 16;
        const unsigned prime = group * 14 + local_prime;
        if (prime < P) {
            #pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                const unsigned shared_row = tid % 16 + half * 16;
                const unsigned row = first_row + shared_row;
                const unsigned coefficient = Transposed ? id + row * transpose_m : row;
                if (coefficient < N) {
                    const int32_t* sums = shared_c[shared_row] + 4 * local_prime;
                    const uint32_t residue = detail::encode_mma_reduce(
                        sums, primes[prime], mu[prime]);
                    const unsigned destination = Transposed ? id * (N / transpose_m) + row : coefficient;
                    ab[(std::size_t{operand} * P + prime) * N + destination] = residue;
                }
            }
        }
        // All 224 threads reach both group barriers, including tail lanes.
        __syncthreads();
    }
}

// Inputs are A then B, coefficient-major [2][N][L] little-endian u32 limbs.
// b is make_encode_b's fixed byte-power matrix; mu[j]=UINT64_MAX/p[j].
// Output is compact canonical [2][P][N].
// The asynchronous launcher uses the caller's stream and CUDA status checks.
inline void launch_encode_mma(unsigned N, unsigned L, unsigned P,
                               const uint32_t* d_inputs,
                               const uint8_t* d_b,
                               const uint32_t* d_p,
                               const uint64_t* d_mu,
                               uint32_t* d_ab,
                               cudaStream_t stream = nullptr) {
    if (N == 0)
        throw std::runtime_error("ByteCRT tensor encoding N must be positive");
    const dim3 grid((N - 1) / 16 + 1, 2);
    if (L == 55 && P == 114)
        encode_mma16<55, 114, 224><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_mu, d_ab);
    else if (L == 28 && P == 57)
        encode_mma16<28, 57, 128><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_mu, d_ab);
    else
        throw std::runtime_error("ByteCRT tensor encoding requires (L,P)=(55,114) or (28,57)");
}

// Full-size staged-NTT input layout: ab_t[(operand*P+prime)*N+id*K+i],
// K=N/256, i=tile*16+r, natural coefficient=id+i*256. Each half warp
// stores 16 contiguous canonical words. Natural launch_encode_mma is retained.
inline void launch_encode_mma_transposed(unsigned N, unsigned L, unsigned P,
                                          const uint32_t* d_inputs,
                                          const uint8_t* d_b,
                                          const uint32_t* d_p,
                                          const uint64_t* d_mu,
                                          uint32_t* d_ab_t,
                                          cudaStream_t stream = nullptr) {
    const dim3 grid(N / 16, 2);
    if (N == 65536 && L == 55 && P == 114)
        encode_mma16<55, 114, 224, true><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_mu, d_ab_t);
    else if (N == 32768 && L == 28 && P == 57)
        encode_mma16<28, 57, 128, true><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_mu, d_ab_t);
    else
        throw std::runtime_error("ByteCRT transposed tensor encoding requires full staged geometries");
}

// Full-size MT32 production launcher; the MT16 launchers above stay intact.
inline void launch_encode_mma32_transposed(unsigned N, unsigned L, unsigned P,
                                            const uint32_t* d_inputs,
                                            const uint8_t* d_b,
                                            const uint32_t* d_p,
                                            const uint64_t* d_mu,
                                            uint32_t* d_ab_t,
                                            cudaStream_t stream = nullptr) {
    const dim3 grid(N / 32, 2);
    if (N == 65536 && L == 55 && P == 114)
        encode_mma32<55, 114, 224, true><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_mu, d_ab_t);
    else if (N == 32768 && L == 28 && P == 57)
        encode_mma32<28, 57, 128, true><<<grid, 224, 0, stream>>>(
            N, d_inputs, d_b, d_p, d_mu, d_ab_t);
    else
        throw std::runtime_error("ByteCRT MT32 encoding requires full staged geometries");
}

}  // namespace bytecrt
