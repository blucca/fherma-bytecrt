// SPDX-License-Identifier: Apache-2.0
// Runtime-prime cuPQC 0.6 staged negacyclic product on canonical RNS planes.
#pragma once

#include <cuda_runtime.h>
#include <cupqc/ntt.hpp>
#include "rns-encode.cuh"

#include <cstddef>
#include <cstdint>

namespace bytecrt::ntt {

constexpr uint32_t kSubSize = 256;
constexpr uint32_t kThreads = 128;

template<uint32_t N, cupqc::nttDirection D>
using Staged = decltype(cupqc::Algorithm<cupqc::algorithm::NTT>()
                       + cupqc::Direction<D>() + cupqc::Precision<uint32_t>()
                       + cupqc::Size<N>() + cupqc::SubSize<kSubSize>()
                       + cupqc::Block() + cupqc::BlockDim<kThreads>());

// mul() uses descriptor Size: the fused tile holds 256, not N, elements.
using Pointwise = decltype(cupqc::Algorithm<cupqc::algorithm::NTT>()
                          + cupqc::Direction<cupqc::nttDirection::FORWARD>()
                          + cupqc::Precision<uint32_t>() + cupqc::Size<kSubSize>()
                          + cupqc::Block() + cupqc::BlockDim<kThreads>());

template<uint32_t N, cupqc::nttDirection D>
__global__ void make_table(uint32_t* out, uint32_t p, uint32_t root) {
    Staged<N, D>().make_twiddles(out, p, root);  // <<<1, 1>>>
}

template<uint32_t N, cupqc::nttDirection D>
__global__ void mont_table(uint32_t* out, uint32_t p) {
    Staged<N, D>().transform_twiddles_to_mont(out, p);  // <<<1, 128>>>
}

// ab[(operand * prime_count + prime) * N + coefficient], operands A then B.
// Tables are [prime][N]: tw/inv_tw/psi in Montgomery form, psi_inv ordinary.
// Prime contract: 2^30 < p < 2^31, p == 1 mod 2N, psi^N == p-1,
// forward root = psi^2, inverse root = (psi^2)^-1, N_inv ordinary N^-1 mod p.
template<uint32_t N>
__global__ void forward_first(uint32_t* ab, const uint32_t* tw,
                              const uint32_t* psi,
                              const cupqc::nttConst<uint32_t>* constants,
                              uint32_t prime_count) {
    extern __shared__ uint32_t s[];
    constexpr uint32_t K = N / kSubSize;
    const uint32_t prime = blockIdx.y % prime_count;
    const auto mc = constants[prime];
    auto* poly = ab + size_t(blockIdx.y) * N;
    Staged<N, cupqc::nttDirection::FORWARD> f;
    f.stage_1_load_to_mont(s, poly, blockIdx.x, mc);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < K; i += blockDim.x) {
        // SDK forward stage 1: s[i] <- poly[id + i*M], strided.
        const uint32_t coefficient = blockIdx.x + i * kSubSize;
        // Both factors are Montgomery: the product retains that domain.
        s[i] = bytecrt::mont_mul_word(s[i], psi[size_t(prime) * N + coefficient],
                                       mc.p, mc.n_dash);
    }
    __syncthreads();
    f.stage_1_execute(s, tw + size_t(prime) * N, mc.p);
    __syncthreads();
    f.stage_1_store(s, poly, blockIdx.x);
}

// Coalesced input alternative; retains the C3 natural forward_first above.
// ab_t and psi_t use id*K+i; psi_t holds ordinary psi^coefficient*R^2 mod p.
// One Montgomery product converts canonical input and twists it in the same step.
// ab receives the SDK natural id+i*256 scatter.
// ab_t and ab are disjoint: a natural scatter can overwrite another CTA's
// unread transposed input when the two pointers alias.
template<uint32_t N>
__global__ void forward_first_coalesced(const uint32_t* ab_t, uint32_t* ab,
                                        const uint32_t* tw, const uint32_t* psi_t,
                                        const cupqc::nttConst<uint32_t>* constants,
                                        uint32_t prime_count) {
    static_assert(N == 65536 || N == 32768, "ByteCRT coalesced NTT geometries");
    extern __shared__ uint32_t s[];
    constexpr uint32_t K = N / kSubSize;
    const uint32_t prime = blockIdx.y % prime_count;
    const auto mc = constants[prime];
    const auto* input = ab_t + size_t(blockIdx.y) * N + size_t(blockIdx.x) * K;
    const auto* psi = psi_t + size_t(prime) * N + size_t(blockIdx.x) * K;
    auto* poly = ab + size_t(blockIdx.y) * N;
    Staged<N, cupqc::nttDirection::FORWARD> f;
    for (uint32_t i = threadIdx.x; i < K; i += blockDim.x)
        s[i] = bytecrt::mont_mul_word(input[i], psi[i], mc.p, mc.n_dash);
    __syncthreads();
    // grid.x=256 and blockIdx.x=id preserve the runtime SDK's stage-local id.
    f.stage_1_execute(s, tw + size_t(prime) * N, mc.p);
    __syncthreads();
    f.stage_1_store(s, poly, blockIdx.x);
}

template<uint32_t N>
__global__ void fused_middle(const uint32_t* ab, uint32_t* c,
                             const uint32_t* tw, const uint32_t* inv_tw,
                             const cupqc::nttConst<uint32_t>* constants,
                             uint32_t prime_count) {
    extern __shared__ uint32_t sa[];
    uint32_t* sb = sa + kSubSize;
    const uint32_t prime = blockIdx.y;
    const uint32_t p = constants[prime].p;
    Staged<N, cupqc::nttDirection::FORWARD> f;
    Staged<N, cupqc::nttDirection::INVERSE> inv;
    // Forward stage 2 and inverse stage 1 share the contiguous id*M+i tile.
    f.stage_2_load(sa, ab + size_t(prime) * N, blockIdx.x);
    f.stage_2_load(sb, ab + size_t(prime_count + prime) * N, blockIdx.x);
    __syncthreads();
    f.stage_2_execute(sa, tw + size_t(prime) * N, p);
    __syncthreads();
    f.stage_2_execute(sb, tw + size_t(prime) * N, p);
    __syncthreads();
    Pointwise().mul(sa, sb, p);
    __syncthreads();
    inv.stage_1_execute(sa, inv_tw + size_t(prime) * N, p);
    __syncthreads();
    inv.stage_1_store(sa, c + size_t(prime) * N, blockIdx.x);
}

template<uint32_t N>
__global__ void inverse_last(uint32_t* c, const uint32_t* inv_tw,
                             const uint32_t* psi_inv,
                             const cupqc::nttConst<uint32_t>* constants,
                             const uint32_t* n_inverse) {
    extern __shared__ uint32_t s[];
    constexpr uint32_t K = N / kSubSize;
    const uint32_t prime = blockIdx.y;
    const auto mc = constants[prime];
    auto* poly = c + size_t(prime) * N;
    Staged<N, cupqc::nttDirection::INVERSE> inv;
    inv.stage_2_load(s, poly, blockIdx.x);
    __syncthreads();
    // SDK applies the ordinary N^-1 once, here in the final inverse stage.
    inv.stage_2_execute(s, inv_tw + size_t(prime) * N, mc.p, n_inverse[prime]);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < K; i += blockDim.x) {
        // SDK inverse stage 2 scatter: poly[id + i*M] <- s[i].
        const uint32_t coefficient = blockIdx.x + i * kSubSize;
        const uint32_t ordinary = inv.from_mont(s[i], mc);
        poly[coefficient] = uint32_t(uint64_t(ordinary)
            * psi_inv[size_t(prime) * N + coefficient] % mc.p);
    }
}

// Mutates ab into forward-stage-1 intermediates; c receives canonical product.
// All three passes use the caller's stream. Return checks immediate launch
// errors; the caller synchronizes/checks asynchronous execution before handoff.
template<uint32_t N>
inline cudaError_t launch_product(uint32_t* ab, uint32_t* c,
                                  const uint32_t* tw, const uint32_t* inv_tw,
                                  const uint32_t* psi, const uint32_t* psi_inv,
                                  const cupqc::nttConst<uint32_t>* constants,
                                  const uint32_t* n_inverse,
                                  uint32_t prime_count, cudaStream_t stream = nullptr) {
    static_assert(N == 65536 || N == 32768, "ByteCRT staged NTT supported geometries");
    constexpr uint32_t K = N / kSubSize;
    constexpr size_t first = cupqc::fwd_stage_1_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    constexpr size_t middle = 2 * cupqc::fwd_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    constexpr size_t last = cupqc::inv_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    forward_first<N><<<dim3(kSubSize, 2 * prime_count), kThreads, first, stream>>>(
        ab, tw, psi, constants, prime_count);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    fused_middle<N><<<dim3(K, prime_count), kThreads, middle, stream>>>(
        ab, c, tw, inv_tw, constants, prime_count);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    inverse_last<N><<<dim3(kSubSize, prime_count), kThreads, last, stream>>>(
        c, inv_tw, psi_inv, constants, n_inverse);
    return cudaGetLastError();
}

}  // namespace bytecrt::ntt
