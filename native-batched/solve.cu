// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <malloc.h>
#include <climits>

// Reuse the exact C18 kernels and preserve its B=1 coefficient-to-coefficient path.
#define BYTECRT_BATCHED
#define fherma_init bytecrt_serial_init
#define fherma_run bytecrt_serial_run
#define fherma_free bytecrt_serial_free
#include "core/solve.cu"
#undef fherma_init
#undef fherma_run
#undef fherma_free

namespace {

constexpr unsigned kBatchSlots = 4;
enum BatchStream : unsigned { kH2D, kCompute, kD2H };

struct BatchState {
    unsigned batch = 0, slot_count = 0, calls = 0;
    std::array<std::unique_ptr<State>, kBatchSlots> slots{};
    std::array<cudaStream_t, 3> streams{};
    std::array<std::array<cudaEvent_t, 2>, 3> coverage{};

    ~BatchState() {
        // Streaming States have zero stream pointers. They own only their
        // arrays/events; these three global streams have this single owner.
        for (auto& slot : slots) slot.reset();
        for (const auto& pair : coverage)
            for (auto event : pair) if (event) cudaEventDestroy(event);
        for (auto stream : streams) if (stream) cudaStreamDestroy(stream);
    }
};

struct BatchHostProfile {
    double pack_us = 0, owned_us = 0;
};

void retire_product_streams(State& s) {
    for (auto& stream : s.streams) {
        check(cudaStreamDestroy(stream), "retire product stream");
        stream = nullptr;
    }
    check(cudaStreamDestroy(s.input_stream), "retire product input stream");
    s.input_stream = nullptr;
    s.profile_products = 0;
}

// One compute FIFO serializes every product through CRT. Only its workspace
// and Point-derived tables are shared; the H2D/D2H arrays stay slot-private.
// The existing inventory remains the sole owner list, including on teardown.
template<class T>
void borrow_device_allocation(State& s, T*& pointer, T* owner) {
    const auto owned = std::find(s.allocations.begin(), s.allocations.end(), pointer);
    if (owned == s.allocations.end())
        throw std::runtime_error("ByteCRT shared allocation missing from owner inventory");
    check(cudaFree(pointer), "release replicated compute allocation");
    s.allocations.erase(owned);
    pointer = owner;
}

void share_compute_storage(State& s, const State& owner) {
    borrow_device_allocation(s, s.tw, owner.tw);
    borrow_device_allocation(s, s.inv_tw, owner.inv_tw);
    borrow_device_allocation(s, s.psi, owner.psi);
    borrow_device_allocation(s, s.untwist_crt, owner.untwist_crt);
    borrow_device_allocation(s, s.encode_b, owner.encode_b);
    borrow_device_allocation(s, s.encode_mu, owner.encode_mu);
    borrow_device_allocation(s, s.q, owner.q);
    borrow_device_allocation(s, s.primes, owner.primes);
    borrow_device_allocation(s, s.n_inverse, owner.n_inverse);
    borrow_device_allocation(s, s.b_matrix, owner.b_matrix);
    borrow_device_allocation(s, s.invp, owner.invp);
    borrow_device_allocation(s, s.constants, owner.constants);
    borrow_device_allocation(s, s.ab, owner.ab);
    borrow_device_allocation(s, s.ab_t, owner.ab_t);
    borrow_device_allocation(s, s.c, owner.c);
    borrow_device_allocation(s, s.weighted, owner.weighted);
    // This borrowed State now owns only its independent device inputs/out.
    assert(s.allocations.size() == 2);
}

void batch_mark(BatchState& batch, bool profile, BatchStream which, unsigned end) {
    if (profile)
        check(cudaEventRecord(batch.coverage[which][end], batch.streams[which]),
              "batch coverage event");
}

template<uint32_t N, uint32_t P>
void batch_enqueue_operand(BatchState& batch, State& s, unsigned product,
                           unsigned operand, const uint32_t* source,
                           bool profile, BatchHostProfile& host) {
    using namespace bytecrt::ntt;
    constexpr size_t first = cupqc::fwd_stage_1_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    const size_t coeffs = size_t(s.plan.n) * s.plan.l;
    const size_t planes = size_t(s.plan.n) * s.plan.pcount;
    const unsigned chunks = s.plan.n / kInputCoefficients;
    const auto input = batch.streams[kH2D];
    const auto compute = batch.streams[kCompute];

    // This encode waits for the complete operand. Pack/copy it in one unit
    // to reduce CPU fork/join and DMA submissions; the other products overlap.
    const size_t offset = operand * coeffs;
    const auto begin = profile ? Clock::now() : Clock::time_point{};
    bytecrt_pack_operand(s.hostinput + offset, source, coeffs);
    if (profile)
        host.pack_us += std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
    if (product == 0 && operand == 0)
        batch_mark(batch, profile, kH2D, 0);
    check(cudaMemcpyAsync(s.inputs + offset, s.hostinput + offset,
                          coeffs * sizeof(uint32_t), cudaMemcpyHostToDevice, input),
          "batch input operand");
    if (product + 1 == batch.batch && operand == 1)
        batch_mark(batch, profile, kH2D, 1);
    // The complete pinned operand is now queued. Full MT32 retains the
    // original grid occupancy; cross-product overlap supplies throughput.
    const auto ready = s.input_ready[operand][chunks - 1];
    check(cudaEventRecord(ready, input), "batch operand ready");
    check(cudaStreamWaitEvent(compute, ready, 0), "batch encode waits input");
    if (product == 0 && operand == 0)
        batch_mark(batch, profile, kCompute, 0);
    bytecrt::launch_encode_mma32_transposed(N, s.plan.l, P,
        s.inputs + operand * coeffs, s.encode_b, s.primes, s.encode_mu,
        s.ab_t + operand * planes, compute, 1);
    check(cudaGetLastError(), "batch full coefficient RNS encoding");
    forward_first_coalesced<N><<<dim3(kSubSize, P), kThreads, first, compute>>>(
        s.ab_t + operand * planes, s.ab + operand * planes,
        s.tw, s.psi, s.constants, P);
    check(cudaGetLastError(), "batch forward first");
}

template<uint32_t N, uint32_t P, uint32_t Kpad>
void batch_finish_product(BatchState& batch, State& s) {
    using namespace bytecrt::ntt;
    constexpr uint32_t K = N / kSubSize;
    constexpr size_t middle = 2 * cupqc::fwd_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    constexpr size_t last = cupqc::inv_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    const auto compute = batch.streams[kCompute];
    // Both forwards precede this on the same compute stream.
    fused_middle<N><<<dim3(K, P), kThreads, middle, compute>>>(
        s.ab, s.c, s.tw, s.inv_tw, s.constants, P);
    check(cudaGetLastError(), "batch fused spectral middle");
    bytecrt::inverse_weighted<N, P, Kpad, true><<<dim3(kSubSize, P), kThreads, last, compute>>>(
        s.c, s.weighted, s.inv_tw, s.untwist_crt, s.constants, s.n_inverse);
    check(cudaGetLastError(), "batch inverse weighted packing");
}

void batch_enqueue_output(BatchState& batch, State& s, unsigned product, bool profile) {
    const auto compute = batch.streams[kCompute];
    const auto copy = batch.streams[kD2H];
    const unsigned chunks = s.plan.n / kOutputCoefficients;
    const size_t words = size_t(kOutputCoefficients) * s.plan.l;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        bytecrt::launch_g1_crt32_transposed(s.plan.w, s.plan.n,
            reinterpret_cast<const uint8_t*>(s.weighted),
            s.b_matrix, s.invp, s.q, s.plan.gap, s.out, compute,
            chunk * kOutputBlocks, kOutputBlocks);
        check(cudaGetLastError(), "batch byte CRT chunk");
        if (product + 1 == batch.batch && chunk + 1 == chunks)
            batch_mark(batch, profile, kCompute, 1);
        check(cudaEventRecord(s.crt_done[chunk], compute), "batch CRT ready");
        check(cudaStreamWaitEvent(copy, s.crt_done[chunk], 0), "batch copy waits CRT");
        if (product == 0 && chunk == 0)
            batch_mark(batch, profile, kD2H, 0);
        check(cudaMemcpyAsync(s.hostout + chunk * words, s.out + chunk * words,
                              words * sizeof(uint32_t), cudaMemcpyDeviceToHost, copy),
              "batch output chunk");
        if (product + 1 == batch.batch && chunk + 1 == chunks)
            batch_mark(batch, profile, kD2H, 1);
        check(cudaEventRecord(s.copied[chunk], copy), "batch output copied");
    }
}

void batch_drain_slot(State& s, std::vector<uint32_t>& output,
                      bool profile, BatchHostProfile& host) {
    const unsigned chunks = s.plan.n / kOutputCoefficients;
    const size_t words = size_t(s.plan.n) * s.plan.l;
    check(cudaEventSynchronize(s.copied[chunks - 1]), "batch wait whole copied product");
    const auto begin_time = profile ? Clock::now() : Clock::time_point{};
    const uint32_t* begin = s.hostout;
    output.insert(output.end(), begin, begin + words);
    if (profile)
        host.owned_us += std::chrono::duration<double, std::micro>(Clock::now() - begin_time).count();
    // The final D2H event completes this product's ancestors. The single owned
    // copy barrier completes every byte before this staging slot is reused.
}

}  // namespace

void* fherma_init(const fherma::Point& p) {
    if (p.B == 0 || p.B > 128)
        throw std::runtime_error("ByteCRT batch coverage B=1..128");
    if (p.B > 1 && (p.N != 65536 || p.W != 1747))
        throw std::runtime_error("ByteCRT streaming point N65536/W1747");
    if (p.B > 1) {
        // A 1.845 GB result otherwise gets a fresh mmap and ~450k first-touch
        // faults every call. This fixed-point process retains its ordinary
        // malloc heap pages; vector ownership/free semantics stay unchanged.
        // B<=128 at this point keeps result bytes below INT_MAX. Native peak
        // memory is checked against the spec's 32 GB limit.
        if (!mallopt(M_MMAP_THRESHOLD, INT_MAX) || !mallopt(M_TRIM_THRESHOLD, -1))
            throw std::runtime_error("ByteCRT batched heap policy rejected");
    }
    auto batch = std::make_unique<BatchState>();
    batch->batch = p.B;
    batch->slot_count = std::min(p.B, kBatchSlots);
    for (unsigned slot = 0; slot < batch->slot_count; ++slot) {
        batch->slots[slot].reset(static_cast<State*>(bytecrt_serial_init(p)));
        if (p.B > 1) {
            retire_product_streams(*batch->slots[slot]);
            if (slot) share_compute_storage(*batch->slots[slot], *batch->slots[0]);
        }
    }
    if (p.B > 1) {
        for (unsigned stream = 0; stream < batch->streams.size(); ++stream) {
            check(cudaStreamCreateWithFlags(&batch->streams[stream], cudaStreamNonBlocking),
                  "batch resource stream setup");
            for (auto& event : batch->coverage[stream])
                check(cudaEventCreate(&event), "batch coverage event setup");
        }
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, 0), "batch device properties");
        // One Point/compute owner and four private DMA slots use ~474 MB
        // explicit device storage +173.016 MB pinned staging. Serial run calls
        // and this single compute FIFO are the workspace-sharing contract.
        std::fprintf(stderr,
            "ByteCRT batch setup B=%u slots=%u resource-streams=3 async-engines=%d fixed-tables=shared compute-workspace=shared dma-io=private l2-cache-bytes=%d\n",
            p.B, batch->slot_count, props.asyncEngineCount, props.l2CacheSize);
    }
    return batch.release();
}

fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto& batch = *static_cast<BatchState*>(state);
    if (batch.batch == 1)
        return bytecrt_serial_run(batch.slots[0].get(), in);
    const auto& plan = batch.slots[0]->plan;
    const size_t words = size_t(plan.n) * plan.l;
    const size_t total = words * batch.batch;
    if (in.a.data.size() != total || in.b.data.size() != total)
        throw std::runtime_error("ByteCRT batch input coefficient count");
    const bool profile = batch.calls++ == 0;  // one unscored warmup
    const auto begin = profile ? Clock::now() : Clock::time_point{};
    BatchHostProfile host;
    fherma::Outputs out;
    out.c.shape = {int64_t(batch.batch), int64_t(plan.n), int64_t(plan.l)};
    out.c.data.reserve(total);
    for (unsigned product = 0; product < batch.batch; ++product) {
        auto& slot = *batch.slots[product % batch.slot_count];
        if (product >= batch.slot_count)
            batch_drain_slot(slot, out.c.data, profile, host);
        const size_t offset = size_t(product) * words;
        batch_enqueue_operand<65536, 114>(batch, slot, product, 0,
            in.a.data.data() + offset, profile, host);
        batch_enqueue_operand<65536, 114>(batch, slot, product, 1,
            in.b.data.data() + offset, profile, host);
        batch_finish_product<65536, 114, 480>(batch, slot);
        batch_enqueue_output(batch, slot, product, profile);
    }
    for (unsigned product = batch.batch - batch.slot_count; product < batch.batch; ++product)
        batch_drain_slot(*batch.slots[product % batch.slot_count], out.c.data, profile, host);
    if (profile) {
        const auto elapsed = [&](BatchStream stream) {
            float ms = 0;
            check(cudaEventElapsedTime(&ms, batch.coverage[stream][0], batch.coverage[stream][1]),
                  "batch coverage elapsed");
            return double(ms) * 1000;
        };
        const double whole = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        // Coverage spans include queue/host-supply gaps and overlap each other.
        std::fprintf(stderr,
            "ByteCRT batch warmup B=%u slots=%u whole=%.3f pack-sum=%.3f owned-sum=%.3f H2D-coverage=%.3f compute-coverage=%.3f D2H-coverage=%.3f us\n",
            batch.batch, batch.slot_count, whole, host.pack_us, host.owned_us,
            elapsed(kH2D), elapsed(kCompute), elapsed(kD2H));
    }
    return out;
}

void fherma_free(void* state) { delete static_cast<BatchState*>(state); }
