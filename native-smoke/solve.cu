// SPDX-License-Identifier: Apache-2.0
// Exact wide negacyclic product: coefficient RNS -> cuPQC staged NTT -> ByteCRT.
// Fixed roots, moduli and byte weights are prepared solely from Point in init.
#include "fherma.h"
#include "rns-plan.hpp"
#include "rns-encode.cuh"
#include "rns-encode-mma.cuh"
#include "ntt-weighted.cuh"
#include "g1-crt.cuh"

#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

void bytecrt_pack_inputs(uint32_t*, const uint32_t*, const uint32_t*, std::size_t);

namespace {

using Clock = std::chrono::steady_clock;
using Constant = cupqc::nttConst<uint32_t>;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ByteCRT ") + what + ": " + cudaGetErrorString(e));
}

struct State {
    bytecrt::RnsPlan plan;
    uint32_t* inputs = nullptr;
    uint32_t* ab = nullptr;
    uint32_t* ab_t = nullptr;  // disjoint transposed input for coalesced forward stage 1
    uint32_t* c = nullptr;
    uint32_t* tw = nullptr;
    uint32_t* inv_tw = nullptr;
    uint32_t* psi = nullptr;
    uint32_t* untwist_crt = nullptr;
    uint32_t* weighted = nullptr;
    uint32_t* out = nullptr;
    uint32_t* hostout = nullptr;
    uint32_t* hostinput = nullptr;
    uint8_t* encode_b = nullptr;
    uint64_t* encode_mu = nullptr;
    uint32_t* q = nullptr;
    uint32_t* primes = nullptr;
    uint32_t* n_inverse = nullptr;
    uint8_t* b_matrix = nullptr;
    double* invp = nullptr;
    Constant* constants = nullptr;
    std::vector<void*> allocations;
    std::array<cudaEvent_t, 6> events{};
    std::array<cudaEvent_t, 2> transfer_events{};
    unsigned calls = 0;
    ~State() {
        for (auto e : events) if (e) cudaEventDestroy(e);
        for (auto e : transfer_events) if (e) cudaEventDestroy(e);
        for (void* p : allocations) cudaFree(p);
        if (hostout) cudaFreeHost(hostout);
        if (hostinput) cudaFreeHost(hostinput);
    }
};

template<class T>
void allocate(State& s, T*& p, size_t count) {
    void* raw = nullptr;
    check(cudaMalloc(&raw, count * sizeof(T)), "allocate");
    s.allocations.push_back(raw);
    p = static_cast<T*>(raw);
}

template<class T>
void fixed_copy(State& s, T*& p, const std::vector<T>& v) {
    allocate(s, p, v.size());
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "fixed copy");
}

template<uint32_t N>
void initialize_tables(State& s) {
    using namespace bytecrt::ntt;
    for (unsigned j = 0; j < s.plan.pcount; ++j) {
        const size_t offset = size_t(j) * N;
        const uint32_t p = s.plan.primes[j];
        make_table<N, cupqc::nttDirection::FORWARD><<<1, 1>>>(s.tw + offset, p, s.plan.omega_roots[j]);
        check(cudaGetLastError(), "forward twiddle generation");
        mont_table<N, cupqc::nttDirection::FORWARD><<<1, kThreads>>>(s.tw + offset, p);
        check(cudaGetLastError(), "forward twiddle Montgomery");
        make_table<N, cupqc::nttDirection::INVERSE><<<1, 1>>>(s.inv_tw + offset, p, s.plan.omega_inverse[j]);
        check(cudaGetLastError(), "inverse twiddle generation");
        mont_table<N, cupqc::nttDirection::INVERSE><<<1, kThreads>>>(s.inv_tw + offset, p);
        check(cudaGetLastError(), "inverse twiddle Montgomery");
    }
    check(cudaDeviceSynchronize(), "fixed tables");
}

void mark(State& s, bool profile, unsigned event) {
    if (profile) check(cudaEventRecord(s.events[event]), "profile event");
}

template<uint32_t N, uint32_t P, uint32_t Kpad>
void product(State& s, bool profile) {
    using namespace bytecrt::ntt;
    constexpr uint32_t K = N / kSubSize;
    constexpr size_t first = cupqc::fwd_stage_1_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    constexpr size_t middle = 2 * cupqc::fwd_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    constexpr size_t last = cupqc::inv_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    mark(s, profile, 0);
    bytecrt::launch_encode_mma_transposed(N, s.plan.l, P, s.inputs, s.encode_b, s.primes,
                                           s.encode_mu, s.ab_t);
    check(cudaGetLastError(), "coefficient RNS encoding");
    mark(s, profile, 1);
    forward_first_coalesced<N><<<dim3(kSubSize, 2 * P), kThreads, first>>>(
        s.ab_t, s.ab, s.tw, s.psi, s.constants, P);
    check(cudaGetLastError(), "forward first");
    mark(s, profile, 2);
    fused_middle<N><<<dim3(K, P), kThreads, middle>>>(s.ab, s.c, s.tw, s.inv_tw, s.constants, P);
    check(cudaGetLastError(), "fused spectral middle");
    mark(s, profile, 3);
    bytecrt::inverse_weighted<N, P, Kpad, true><<<dim3(kSubSize, P), kThreads, last>>>(
        s.c, s.weighted, s.inv_tw, s.untwist_crt, s.constants, s.n_inverse);
    check(cudaGetLastError(), "inverse weighted packing");
    mark(s, profile, 4);
    bytecrt::launch_g1_crt_transposed(s.plan.w, N, reinterpret_cast<const uint8_t*>(s.weighted),
                         s.b_matrix, s.invp, s.q, s.plan.gap, s.out);
    check(cudaGetLastError(), "fused byte CRT");
    mark(s, profile, 5);
}

}  // namespace

void* fherma_init(const fherma::Point& p) {
    auto s = std::make_unique<State>();
    s->plan = bytecrt::make_rns_plan(p.N, p.W, p.q.data);
    if (p.L != s->plan.l) throw std::runtime_error("ByteCRT point L mismatch");
    const size_t coeffs = size_t(p.N) * p.L;
    const size_t planes = size_t(p.N) * s->plan.pcount;
    allocate(*s, s->inputs, 2 * coeffs);
    allocate(*s, s->ab, 2 * planes);
    allocate(*s, s->ab_t, 2 * planes);
    allocate(*s, s->c, planes);
    allocate(*s, s->tw, planes);
    allocate(*s, s->inv_tw, planes);
    allocate(*s, s->weighted, size_t(p.N) * (s->plan.kpad / 4));
    check(cudaMemset(s->weighted, 0, size_t(p.N) * s->plan.kpad), "zero fixed CRT padding");
    allocate(*s, s->out, coeffs);
    check(cudaMallocHost(&s->hostout, coeffs * sizeof(uint32_t)), "allocate pinned output");
    check(cudaMallocHost(&s->hostinput, 2 * coeffs * sizeof(uint32_t)), "allocate pinned input");
    fixed_copy(*s, s->encode_b, bytecrt::make_encode_b(p.L, s->plan.pcount, s->plan.primes));
    std::vector<uint64_t> encode_mu(s->plan.pcount);
    for (unsigned j = 0; j < s->plan.pcount; ++j) encode_mu[j] = UINT64_MAX / s->plan.primes[j];
    fixed_copy(*s, s->encode_mu, encode_mu);
    fixed_copy(*s, s->q, p.q.data);
    fixed_copy(*s, s->primes, s->plan.primes);
    fixed_copy(*s, s->n_inverse, s->plan.n_inverse);
    fixed_copy(*s, s->b_matrix, s->plan.b_matrix);
    fixed_copy(*s, s->invp, s->plan.invp);
    std::vector<Constant> constants(s->plan.pcount);
    std::vector<uint32_t> psi(planes), untwist_crt(planes);
    for (unsigned j = 0; j < s->plan.pcount; ++j) {
        const uint32_t prime = s->plan.primes[j];
        constants[j] = Constant(prime);
        if (uint32_t(uint64_t(prime) * constants[j].n_dash) != UINT32_MAX)
            throw std::runtime_error("ByteCRT Montgomery inverse");
        const uint32_t r_mod_p = s->plan.powers_mont[size_t(j) * p.L];
        uint32_t forward_power = 1, inverse_power = s->plan.crt_inverse[j];
        const uint32_t inverse_step = bytecrt::rns_detail::pow_mod(s->plan.psi_roots[j], prime - 2, prime);
        for (uint32_t i = 0; i < p.N; ++i) {
            const size_t psi_index = size_t(j) * p.N +
                size_t(i % bytecrt::ntt::kSubSize) * (p.N / bytecrt::ntt::kSubSize) +
                i / bytecrt::ntt::kSubSize;
            psi[psi_index] = uint32_t(uint64_t(forward_power) * r_mod_p % prime); // transposed Montgomery
            untwist_crt[psi_index] = inverse_power; // transposed ordinary CRT untwist
            forward_power = uint32_t(uint64_t(forward_power) * s->plan.psi_roots[j] % prime);
            inverse_power = uint32_t(uint64_t(inverse_power) * inverse_step % prime);
        }
    }
    fixed_copy(*s, s->constants, constants);
    fixed_copy(*s, s->psi, psi);
    fixed_copy(*s, s->untwist_crt, untwist_crt);
    if (p.N == 65536) initialize_tables<65536>(*s);
    else initialize_tables<32768>(*s);
    for (auto& e : s->events) check(cudaEventCreate(&e), "profile event setup");
    for (auto& e : s->transfer_events) check(cudaEventCreate(&e), "transfer event setup");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "device properties");
    std::fprintf(stderr, "ByteCRT full setup N=%u W=%u L=%u P=%u gap=%u: %s, SMs=%d\n",
                 p.N, p.W, p.L, s->plan.pcount, s->plan.gap, props.name, props.multiProcessorCount);
    return s.release();
}

fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto& s = *static_cast<State*>(state);
    const size_t coeffs = size_t(s.plan.n) * s.plan.l;
    const size_t bytes = coeffs * sizeof(uint32_t);
    if (in.a.data.size() != coeffs || in.b.data.size() != coeffs)
        throw std::runtime_error("ByteCRT input coefficient count");
    // Both official warm-ups report phases; scored calls stay event-free.
    const unsigned call = s.calls++;
    const bool profile = call < 2;
    const auto t0 = Clock::now();
    bytecrt_pack_inputs(s.hostinput, in.a.data.data(), in.b.data.data(), coeffs);
    const auto packed = Clock::now();
    if (profile) check(cudaEventRecord(s.transfer_events[0]), "H2D start event");
    check(cudaMemcpyAsync(s.inputs, s.hostinput, 2 * bytes, cudaMemcpyHostToDevice), "copy packed inputs");
    if (profile) check(cudaEventRecord(s.transfer_events[1]), "H2D end event");
    if (s.plan.n == 65536) product<65536, 114, 480>(s, profile);
    else product<32768, 57, 256>(s, profile);
    check(cudaDeviceSynchronize(), "full product execute");
    const auto t2 = Clock::now();
    check(cudaMemcpy(s.hostout, s.out, bytes, cudaMemcpyDeviceToHost), "copy complete coefficients");
    const auto t3 = Clock::now();
    fherma::Outputs out;
    out.c.shape = {static_cast<int64_t>(s.plan.n), static_cast<int64_t>(s.plan.l)};
    out.c.data.assign(s.hostout, s.hostout + coeffs);
    const auto t4 = Clock::now();
    if (profile) {
        const char* names[] = {"encode", "forward1", "fused-middle", "inverse-weighted", "G1"};
        for (unsigned i = 0; i < 5; ++i) {
            float ms = 0;
            check(cudaEventElapsedTime(&ms, s.events[i], s.events[i + 1]), "profile elapsed");
            std::fprintf(stderr, "ByteCRT GPU %s: %.3f us (warm-up %u)\n", names[i], double(ms) * 1000, call + 1);
        }
        float transfer_ms = 0;
        check(cudaEventElapsedTime(&transfer_ms, s.transfer_events[0], s.transfer_events[1]), "H2D elapsed");
        std::fprintf(stderr, "ByteCRT GPU H2D: %.3f us (warm-up %u)\n", double(transfer_ms) * 1000, call + 1);
        const auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b-a).count(); };
        std::fprintf(stderr, "ByteCRT host input-pack=%.3f H2D+product=%.3f D2H=%.3f output-construct=%.3f us (warm-up %u)\n",
                     us(t0,packed), us(packed,t2), us(t2,t3), us(t3,t4), call + 1);
    }
    return out;
}

void fherma_free(void* state) { delete static_cast<State*>(state); }
