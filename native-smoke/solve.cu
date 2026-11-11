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
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

void bytecrt_pack_operand(uint32_t*, const uint32_t*, std::size_t);

namespace {

using Clock = std::chrono::steady_clock;
using Constant = cupqc::nttConst<uint32_t>;

// Fixed 512-CTA chunks keep two complete transposed tiles together.
// MT32 gives 16,384 natural coefficients per chunk: main=4, aux=2.
// Cap at four chunks: more splits add launch/DMA/OpenMP barriers for
// only ~37 us extra ideal main saving. Main target saving: 120-180 us.
constexpr unsigned kOutputBlocks = 512;
constexpr unsigned kOutputCoefficients = 32 * kOutputBlocks;
constexpr unsigned kMaxOutputChunks = 4;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ByteCRT ") + what + ": " + cudaGetErrorString(e));
}

// Fixed-setup locality readout; no affinity or allocation policy changes.
void print_setup_value(const char* label, const std::string& path,
                       const std::string& key = {}) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (key.empty() || line.compare(0, key.size(), key) == 0) {
            std::string value = key.empty() ? line : line.substr(key.size());
            const auto first = value.find_first_not_of(" \t");
            if (first != std::string::npos) value.erase(0, first);
            std::fprintf(stderr, "ByteCRT locality %s available=true value=%s\n",
                         label, value.c_str());
            return;
        }
    }
    std::fprintf(stderr, "ByteCRT locality %s available=false\n", label);
}

// Fixed-setup capability readout for the next input-staging decision.
void print_setup_attribute(const char* label, cudaDeviceAttr attribute) {
    int value = 0;
    const auto status = cudaDeviceGetAttribute(&value, attribute, 0);
    if (status != cudaSuccess) (void)cudaGetLastError();
    const int success = status == cudaSuccess;
    std::fprintf(stderr, "ByteCRT memory-attribute %s query-success=%d value=%d\n",
                 label, success, success && value != 0);
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
    float* invp = nullptr;
    Constant* constants = nullptr;
    std::vector<void*> allocations;
    std::array<cudaStream_t, 2> streams{};
    std::array<cudaEvent_t, kMaxOutputChunks> crt_done{};
    std::array<cudaEvent_t, kMaxOutputChunks> copied{};
    cudaEvent_t a_forwarded = nullptr;  // dependency only; timing disabled
    std::array<std::array<cudaEvent_t, 4>, 2> operand_events{};
    std::array<cudaEvent_t, 4> events{};  // joined middle / inverse / CRT
    unsigned calls = 0;
    ~State() {
        for (auto e : events) if (e) cudaEventDestroy(e);
        for (const auto& op : operand_events)
            for (auto e : op) if (e) cudaEventDestroy(e);
        for (auto e : crt_done) if (e) cudaEventDestroy(e);
        for (auto e : copied) if (e) cudaEventDestroy(e);
        if (a_forwarded) cudaEventDestroy(a_forwarded);
        for (auto stream : streams) if (stream) cudaStreamDestroy(stream);
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

void mark(State& s, bool profile, unsigned event, cudaStream_t stream) {
    if (profile) check(cudaEventRecord(s.events[event], stream), "profile event");
}

void mark_operand(State& s, bool profile, unsigned operand, unsigned event) {
    if (profile)
        check(cudaEventRecord(s.operand_events[operand][event], s.streams[operand]),
              "operand profile event");
}

template<uint32_t N, uint32_t P>
void enqueue_operand(State& s, bool profile, unsigned operand) {
    using namespace bytecrt::ntt;
    constexpr size_t first = cupqc::fwd_stage_1_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    const size_t coeffs = size_t(N) * s.plan.l;
    const size_t planes = size_t(N) * P;
    const auto stream = s.streams[operand];
    mark_operand(s, profile, operand, 0);
    check(cudaMemcpyAsync(s.inputs + operand * coeffs, s.hostinput + operand * coeffs,
                          coeffs * sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
          "copy packed operand");
    mark_operand(s, profile, operand, 1);
    bytecrt::launch_encode_mma32_transposed(N, s.plan.l, P, s.inputs + operand * coeffs,
        s.encode_b, s.primes, s.encode_mu, s.ab_t + operand * planes, stream, 1);
    check(cudaGetLastError(), "coefficient RNS encoding");
    mark_operand(s, profile, operand, 2);
    forward_first_coalesced<N><<<dim3(kSubSize, P), kThreads, first, stream>>>(
        s.ab_t + operand * planes, s.ab + operand * planes,
        s.tw, s.psi, s.constants, P);
    check(cudaGetLastError(), "forward first");
    mark_operand(s, profile, operand, 3);
    if (operand == 0)
        check(cudaEventRecord(s.a_forwarded, stream), "A-forwarded dependency");
}

template<uint32_t N, uint32_t P, uint32_t Kpad>
void finish_product(State& s, bool profile) {
    using namespace bytecrt::ntt;
    constexpr uint32_t K = N / kSubSize;
    constexpr size_t middle = 2 * cupqc::fwd_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    constexpr size_t last = cupqc::inv_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    const auto stream = s.streams[1];
    // B forward1 precedes this in stream 1; this event contributes A forward1.
    check(cudaStreamWaitEvent(stream, s.a_forwarded, 0), "join forwarded operands");
    mark(s, profile, 0, stream);
    fused_middle<N><<<dim3(K, P), kThreads, middle, stream>>>(
        s.ab, s.c, s.tw, s.inv_tw, s.constants, P);
    check(cudaGetLastError(), "fused spectral middle");
    mark(s, profile, 1, stream);
    bytecrt::inverse_weighted<N, P, Kpad, true><<<dim3(kSubSize, P), kThreads, last, stream>>>(
        s.c, s.weighted, s.inv_tw, s.untwist_crt, s.constants, s.n_inverse);
    check(cudaGetLastError(), "inverse weighted packing");
    mark(s, profile, 2, stream);
}

void enqueue_output(State& s, bool profile) {
    const auto stream = s.streams[1];
    // A-forward is an ancestor of every CRT done; reuse the A stream
    // for output after those events, keeping the two-stream DAG acyclic.
    const auto copy_stream = s.streams[0];
    const unsigned chunks = s.plan.n / kOutputCoefficients;
    const size_t words = size_t(kOutputCoefficients) * s.plan.l;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        bytecrt::launch_g1_crt32_transposed(s.plan.w, s.plan.n,
            reinterpret_cast<const uint8_t*>(s.weighted),
            s.b_matrix, s.invp, s.q, s.plan.gap, s.out, stream,
            chunk * kOutputBlocks, kOutputBlocks);
        check(cudaGetLastError(), "fused byte CRT chunk");
        if (chunk + 1 == chunks) mark(s, profile, 3, stream);
        check(cudaEventRecord(s.crt_done[chunk], stream), "CRT chunk ready");
        check(cudaStreamWaitEvent(copy_stream, s.crt_done[chunk], 0),
              "copy waits for CRT chunk");
        check(cudaMemcpyAsync(s.hostout + chunk * words, s.out + chunk * words,
                              words * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                              copy_stream), "copy coefficient chunk");
        check(cudaEventRecord(s.copied[chunk], copy_stream), "output chunk copied");
    }
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
    fixed_copy(*s, s->b_matrix, s->plan.b_byte_matrix);
    const std::vector<float> invp_float(s->plan.invp.begin(), s->plan.invp.end());
    fixed_copy(*s, s->invp, invp_float);
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
    for (auto& stream : s->streams)
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "operand stream setup");
    for (unsigned chunk = 0; chunk < p.N / kOutputCoefficients; ++chunk) {
        check(cudaEventCreateWithFlags(&s->crt_done[chunk], cudaEventDisableTiming),
              "CRT chunk event setup");
        check(cudaEventCreateWithFlags(&s->copied[chunk], cudaEventDisableTiming),
              "copied chunk event setup");
    }
    check(cudaEventCreateWithFlags(&s->a_forwarded, cudaEventDisableTiming),
          "A-forwarded event setup");
    for (auto& op : s->operand_events)
        for (auto& e : op) check(cudaEventCreate(&e), "operand profile event setup");
    for (auto& e : s->events) check(cudaEventCreate(&e), "profile event setup");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "device properties");
    std::fprintf(stderr, "ByteCRT full setup N=%u W=%u L=%u P=%u gap=%u: %s, SMs=%d\n",
                 p.N, p.W, p.L, s->plan.pcount, s->plan.gap, props.name, props.multiProcessorCount);
    char bdf[32];
    std::snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.0",
                  unsigned(props.pciDomainID), unsigned(props.pciBusID), unsigned(props.pciDeviceID));
    std::fprintf(stderr, "ByteCRT locality BDF=%s\n", bdf);
    const std::string pci = std::string("/sys/bus/pci/devices/") + bdf;
    print_setup_value("gpu-numa-node", pci + "/numa_node");
    print_setup_value("gpu-local-cpulist", pci + "/local_cpulist");
    print_setup_value("cpu-allowed-list", "/proc/self/status", "Cpus_allowed_list:");
    print_setup_attribute("PageableMemoryAccess", cudaDevAttrPageableMemoryAccess);
    print_setup_attribute("PageableMemoryAccessUsesHostPageTables",
                          cudaDevAttrPageableMemoryAccessUsesHostPageTables);
    print_setup_attribute("HostRegisterSupported", cudaDevAttrHostRegisterSupported);
    print_setup_attribute("HostRegisterReadOnlySupported", cudaDevAttrHostRegisterReadOnlySupported);
    print_setup_attribute("CanUseHostPointerForRegisteredMem",
                          cudaDevAttrCanUseHostPointerForRegisteredMem);
    return s.release();
}

fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto& s = *static_cast<State*>(state);
    const size_t coeffs = size_t(s.plan.n) * s.plan.l;
    const unsigned chunks = s.plan.n / kOutputCoefficients;
    const size_t chunk_words = size_t(kOutputCoefficients) * s.plan.l;
    if (in.a.data.size() != coeffs || in.b.data.size() != coeffs)
        throw std::runtime_error("ByteCRT input coefficient count");
    // Warm-ups report phases; scored calls record only required dependency events.
    const unsigned call = s.calls++;
    const bool profile = call < 2;
    const auto t0 = Clock::now();
    bytecrt_pack_operand(s.hostinput, in.a.data.data(), coeffs);
    const auto a_packed = Clock::now();
    if (s.plan.n == 65536) enqueue_operand<65536, 114>(s, profile, 0);
    else enqueue_operand<32768, 57>(s, profile, 0);
    const auto a_submitted = Clock::now();
    bytecrt_pack_operand(s.hostinput + coeffs, in.b.data.data(), coeffs);
    const auto b_packed = Clock::now();
    if (s.plan.n == 65536) {
        enqueue_operand<65536, 114>(s, profile, 1);
        finish_product<65536, 114, 480>(s, profile);
    } else {
        enqueue_operand<32768, 57>(s, profile, 1);
        finish_product<32768, 57, 256>(s, profile);
    }
    const auto output_begin = Clock::now();
    fherma::Outputs out;
    out.c.shape = {static_cast<int64_t>(s.plan.n), static_cast<int64_t>(s.plan.l)};
    // reserve owns the full allocation; insert constructs each copied range
    // through the standard vector API, without a full zero-fill pass.
    out.c.data.reserve(coeffs);
    enqueue_output(s, profile);
    const auto t2 = Clock::now();
    auto t3 = t2;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        check(cudaEventSynchronize(s.copied[chunk]), "wait for copied output chunk");
        if (chunk + 1 == chunks) t3 = Clock::now();
        const uint32_t* begin = s.hostout + chunk * chunk_words;
        out.c.data.insert(out.c.data.end(), begin, begin + chunk_words);
    }
    // Last copied implies all B CRT chunks and the joined A work completed;
    // the last standard insert barrier completes every returned owned word.
    const auto t4 = Clock::now();
    if (profile) {
        const auto elapsed = [](cudaEvent_t a, cudaEvent_t b) {
            float ms = 0;
            check(cudaEventElapsedTime(&ms, a, b), "profile elapsed");
            return double(ms) * 1000;
        };
        double encode = 0, forward = 0;
        for (unsigned operand = 0; operand < 2; ++operand) {
            const auto& e = s.operand_events[operand];
            const double op_encode = elapsed(e[1], e[2]);
            const double op_forward = elapsed(e[2], e[3]);
            encode += op_encode;
            forward += op_forward;
            std::fprintf(stderr,
                "ByteCRT GPU operand-%c H2D-span=%.3f encode=%.3f forward1=%.3f us (warm-up %u)\n",
                'A' + operand, elapsed(e[0], e[1]), op_encode, op_forward, call + 1);
        }
        // encode / forward1 are sums of per-stream elapsed spans. H2D is the
        // A-start to B-copy-end coverage; B's own H2D span can include queueing.
        std::fprintf(stderr, "ByteCRT GPU encode: %.3f us (warm-up %u)\n", encode, call + 1);
        std::fprintf(stderr, "ByteCRT GPU forward1: %.3f us (warm-up %u)\n", forward, call + 1);
        const char* names[] = {"fused-middle", "inverse-weighted", "G1"};
        for (unsigned i = 0; i < 3; ++i)
            std::fprintf(stderr, "ByteCRT GPU %s: %.3f us (warm-up %u)\n",
                         names[i], elapsed(s.events[i], s.events[i + 1]), call + 1);
        std::fprintf(stderr, "ByteCRT GPU H2D: %.3f us (warm-up %u)\n",
                     elapsed(s.operand_events[0][0], s.operand_events[1][1]), call + 1);
        std::fprintf(stderr, "ByteCRT GPU pipeline-wall=%.3f join-wait=%.3f us (warm-up %u)\n",
            elapsed(s.operand_events[0][0], s.events[3]),
            elapsed(s.operand_events[1][3], s.events[0]), call + 1);
        const auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b-a).count(); };
        // Four fields still add to whole host time. H2D+product ends at
        // output submission; D2H includes pending compute/copies and early
        // inserts; output-construct is the final insert. These overlap-aware
        // spans are supplemented by output enqueue/drain below.
        std::fprintf(stderr, "ByteCRT host input-pack=%.3f H2D+product=%.3f D2H=%.3f output-construct=%.3f us (warm-up %u)\n",
                     us(t0,a_packed), us(a_packed,t2), us(t2,t3), us(t3,t4), call + 1);
        std::fprintf(stderr, "ByteCRT host output-pipeline chunks=%u enqueue=%.3f drain=%.3f us (warm-up %u)\n",
            chunks, us(output_begin,t2), us(t2,t4), call + 1);
        std::fprintf(stderr, "ByteCRT host operand-pack A=%.3f B=%.3f total=%.3f us (warm-up %u)\n",
            us(t0,a_packed), us(a_submitted,b_packed),
            us(t0,a_packed) + us(a_submitted,b_packed), call + 1);
    }
    return out;
}

void fherma_free(void* state) { delete static_cast<State*>(state); }
