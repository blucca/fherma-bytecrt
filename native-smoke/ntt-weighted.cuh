// SPDX-License-Identifier: Apache-2.0
// Final staged inverse + untwist + CRT inverse + weighted boundary packing.
#pragma once

#include "ntt-pipeline.cuh"
#include "rns-encode.cuh"

namespace bytecrt {

template<uint32_t N, uint32_t P, uint32_t Kpad, bool Transposed = false>
__global__ void inverse_weighted(const uint32_t* c, uint32_t* weighted,
                                const uint32_t* inv_tw,
                                const uint32_t* untwist_crt,
                                const cupqc::nttConst<uint32_t>* constants,
                                const uint32_t* n_inverse) {
    static_assert((N == 65536 && P == 114 && Kpad == 480) ||
                  (N == 32768 && P == 57 && Kpad == 256), "weighted CRT geometry");
    extern __shared__ uint32_t s[];
    constexpr uint32_t K = N / ntt::kSubSize;
    const uint32_t prime = blockIdx.y;
    const auto mc = constants[prime];
    ntt::Staged<N, cupqc::nttDirection::INVERSE> inv;
    inv.stage_2_load(s, c + size_t(prime) * N, blockIdx.x);
    __syncthreads();
    inv.stage_2_execute(s, inv_tw + size_t(prime) * N, mc.p, n_inverse[prime]);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < K; i += blockDim.x) {
        const uint32_t coefficient = blockIdx.x + i * ntt::kSubSize;
        // s[i] is (negacyclic coefficient * psi^i) in Montgomery form.
        // The ordinary factor psi^-i * inv(P/p) cancels both transforms with
        // one Montgomery product, returning canonical weighted t_j for G1.
        // Transposed factors use [prime][id][i], just like coalesced psi.
        // Transposed A uses [id][prime_word][i]; each warp writes 32 words.
        const size_t factor_index = size_t(prime) * N +
            (Transposed ? size_t(blockIdx.x) * K + i : coefficient);
        const size_t destination = Transposed ?
            (size_t(blockIdx.x) * (Kpad / 4) + prime) * K + i :
            size_t(coefficient) * (Kpad / 4) + prime;
        const uint32_t t = mont_mul_word(s[i], untwist_crt[factor_index], mc.p, mc.n_dash);
        weighted[destination] = t;
    }
    // The alpha word and padding are fixed-zero global storage. G1 creates
    // the input-dependent alpha in its shared copy, without changing this A.
}

}  // namespace bytecrt
