// SPDX-License-Identifier: Apache-2.0
#include <malloc.h>
#include <climits>
#include <algorithm>

// Frozen qualified ACA core, including the exact C18 mathematical kernels.
// The wrapper changes only where the original input/result bytes are copied.
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
    unsigned batch = 0, slot_count = 0;
    size_t words = 0, total = 0;
    std::array<std::unique_ptr<State>, kBatchSlots> slots{};
    std::array<cudaStream_t, 3> streams{};
    cudaEvent_t uploaded = nullptr, computed = nullptr;
    // Full original row-major u32[B,N,L] buffers, allocated only in init.
    // slots[0]->allocations owns these along with its legacy scratch/tables.
    uint32_t* raw_a = nullptr;
    uint32_t* raw_b = nullptr;
    uint32_t* raw_c = nullptr;

    ~BatchState() {
        for (auto& slot : slots) slot.reset();
        if (uploaded) cudaEventDestroy(uploaded);
        if (computed) cudaEventDestroy(computed);
        for (auto stream : streams) if (stream) cudaStreamDestroy(stream);
    }
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

bool input_shape(const std::vector<int64_t>& shape, const BatchState& batch) {
    const auto& plan = batch.slots[0]->plan;
    return shape.size() == 3 && shape[0] == int64_t(batch.batch) &&
           shape[1] == int64_t(plan.n) && shape[2] == int64_t(plan.l);
}

void validate_inputs(const BatchState& batch, const fherma::Inputs& in) {
    if (!input_shape(in.a.shape, batch) || !input_shape(in.b.shape, batch) ||
        in.a.data.size() != batch.total || in.b.data.size() != batch.total)
        throw std::runtime_error("ByteCRT transfer input shape/coefficient count");
}

// Device-only version of the original encoder/forward submission. B1 keeps
// its range encoder and two operand streams; the batched path keeps its full
// encoder on the one compute stream. All data conversion occurs here in run.
template<uint32_t N, uint32_t P, bool Ranged>
void enqueue_device_operand(State& s, unsigned operand, const uint32_t* source,
                            cudaStream_t stream) {
    using namespace bytecrt::ntt;
    constexpr size_t first = cupqc::fwd_stage_1_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    const size_t planes = size_t(s.plan.n) * s.plan.pcount;
    if constexpr (Ranged) {
        const unsigned chunks = s.plan.n / kInputCoefficients;
        for (unsigned chunk = 0; chunk < chunks; ++chunk) {
            bytecrt::launch_encode_mma32_transposed_range(N, s.plan.l, P,
                source, s.encode_b, s.primes, s.encode_mu,
                s.ab_t + operand * planes, stream,
                chunk * kInputBlocks, kInputBlocks);
            check(cudaGetLastError(), "device coefficient RNS chunk encoding");
        }
    } else {
        bytecrt::launch_encode_mma32_transposed(N, s.plan.l, P,
            source, s.encode_b, s.primes, s.encode_mu,
            s.ab_t + operand * planes, stream, 1);
        check(cudaGetLastError(), "device full coefficient RNS encoding");
    }
    forward_first_coalesced<N><<<dim3(kSubSize, P), kThreads, first, stream>>>(
        s.ab_t + operand * planes, s.ab + operand * planes,
        s.tw, s.psi, s.constants, P);
    check(cudaGetLastError(), "device forward first");
    if constexpr (Ranged) {
        if (operand == 0)
            check(cudaEventRecord(s.a_forwarded, stream), "A-forwarded dependency");
    }
}

template<uint32_t N, uint32_t P, uint32_t Kpad>
void finish_device_product(State& s, cudaStream_t stream) {
    using namespace bytecrt::ntt;
    constexpr uint32_t K = N / kSubSize;
    constexpr size_t middle = 2 * cupqc::fwd_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    constexpr size_t last = cupqc::inv_stage_2_ntt_shared_workspace_size<N, kSubSize, uint32_t>();
    fused_middle<N><<<dim3(K, P), kThreads, middle, stream>>>(
        s.ab, s.c, s.tw, s.inv_tw, s.constants, P);
    check(cudaGetLastError(), "device fused spectral middle");
    bytecrt::inverse_weighted<N, P, Kpad, true><<<dim3(kSubSize, P), kThreads, last, stream>>>(
        s.c, s.weighted, s.inv_tw, s.untwist_crt, s.constants, s.n_inverse);
    check(cudaGetLastError(), "device inverse weighted packing");
}

void enqueue_device_result(State& s, uint32_t* result, cudaStream_t stream) {
    const unsigned chunks = s.plan.n / kOutputCoefficients;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        // The frozen G1 kernel writes canonical natural [N,L] limbs directly
        // to this product's original-layout raw C range. No D2D copy is added.
        bytecrt::launch_g1_crt32_transposed(s.plan.w, s.plan.n,
            reinterpret_cast<const uint8_t*>(s.weighted),
            s.b_matrix, s.invp, s.q, s.plan.gap, result, stream,
            chunk * kOutputBlocks, kOutputBlocks);
        check(cudaGetLastError(), "device byte CRT chunk");
    }
}

void enqueue_result_copy(BatchState& batch, State& s, unsigned product,
                         cudaStream_t stream) {
    const unsigned chunks = s.plan.n / kOutputCoefficients;
    const size_t words = size_t(kOutputCoefficients) * s.plan.l;
    const uint32_t* result = batch.raw_c + size_t(product) * batch.words;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        check(cudaMemcpyAsync(s.hostout + chunk * words, result + chunk * words,
                              words * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
              "transfer output chunk");
        check(cudaEventRecord(s.copied[chunk], stream), "transfer output copied");
    }
}

void drain_result_copy(State& s, std::vector<uint32_t>& output) {
    const unsigned chunks = s.plan.n / kOutputCoefficients;
    const size_t words = size_t(kOutputCoefficients) * s.plan.l;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        check(cudaEventSynchronize(s.copied[chunk]), "transfer wait copied chunk");
        const uint32_t* begin = s.hostout + chunk * words;
        output.insert(output.end(), begin, begin + words);
    }
    // Every insert owns these bytes and completes its normal copy barrier
    // before this pinned slot is reused, including across consecutive calls.
}

}  // namespace

void* fherma_init(const fherma::Point& p) {
    // Intentional frozen point/size limit; no new mathematical dispatch.
    if (p.B == 0 || p.B > 128)
        throw std::runtime_error("ByteCRT batch coverage B=1..128");
    if (p.B > 1 && (p.N != 65536 || p.W != 1747))
        throw std::runtime_error("ByteCRT streaming point N65536/W1747");
    if (p.B > 1) {
        // Retain ACA's ordinary owned-vector heap policy. B128 result bytes
        // remain below INT_MAX; fresh native checks enforce the 32 GB limit.
        if (!mallopt(M_MMAP_THRESHOLD, INT_MAX) || !mallopt(M_TRIM_THRESHOLD, -1))
            throw std::runtime_error("ByteCRT batched heap policy rejected");
    }
    auto batch = std::make_unique<BatchState>();
    batch->batch = p.B;
    batch->slot_count = std::min(p.B, kBatchSlots);
    for (unsigned slot = 0; slot < batch->slot_count; ++slot) {
        batch->slots[slot].reset(static_cast<State*>(bytecrt_serial_init(p)));
        if (p.B > 1) retire_product_streams(*batch->slots[slot]);
    }
    if (p.B > 1) {
        for (auto& stream : batch->streams)
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "batch resource stream setup");
    }
    check(cudaEventCreateWithFlags(&batch->uploaded, cudaEventDisableTiming),
          "transfer uploaded event setup");
    check(cudaEventCreateWithFlags(&batch->computed, cudaEventDisableTiming),
          "transfer computed event setup");
    const auto& plan = batch->slots[0]->plan;
    batch->words = size_t(plan.n) * plan.l;
    batch->total = batch->words * batch->batch;
    allocate(*batch->slots[0], batch->raw_a, batch->total);
    allocate(*batch->slots[0], batch->raw_b, batch->total);
    allocate(*batch->slots[0], batch->raw_c, batch->total);
    // At B128 these three byte-layout arrays add 5,536,481,280 device bytes
    // to ACA's <=1.376 GB explicit legacy device arrays (four private States).
    std::fprintf(stderr,
        "ByteCRT transfer setup B=%u slots=%u raw-bytes=%zu math-core=frozen-ACA\n",
        p.B, batch->slot_count, 3 * batch->total * sizeof(uint32_t));
    return batch.release();
}

void fherma_h2d(void* state, const fherma::Inputs& in) {
    auto& batch = *static_cast<BatchState*>(state);
    validate_inputs(batch, in);
    const auto stream = batch.batch == 1 ? batch.slots[0]->input_stream : batch.streams[kH2D];
    const auto& plan = batch.slots[0]->plan;
    const unsigned chunks = plan.n / kInputCoefficients;
    const size_t words = size_t(kInputCoefficients) * plan.l;
    for (unsigned product = 0; product < batch.batch; ++product) {
        auto& slot = *batch.slots[product % batch.slot_count];
        if (product >= batch.slot_count)
            check(cudaEventSynchronize(slot.input_ready[1][chunks - 1]),
                  "transfer wait input staging reuse");
        const size_t product_offset = size_t(product) * batch.words;
        for (unsigned operand = 0; operand < 2; ++operand) {
            const uint32_t* source = (operand == 0 ? in.a.data.data() : in.b.data.data()) + product_offset;
            uint32_t* dest = (operand == 0 ? batch.raw_a : batch.raw_b) + product_offset;
            for (unsigned chunk = 0; chunk < chunks; ++chunk) {
                uint32_t* staging = slot.hostinput + operand * batch.words + chunk * words;
                // Frozen twelve-worker helper: disjoint memcpy only, with its
                // implicit barrier. Padding/RNS/transpose remain GPU run work.
                bytecrt_pack_operand(staging, source + chunk * words, words);
                check(cudaMemcpyAsync(dest + chunk * words, staging,
                                      words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
                      "transfer original input chunk");
                check(cudaEventRecord(slot.input_ready[operand][chunk], stream),
                      "transfer input chunk copied");
            }
        }
    }
    check(cudaEventRecord(batch.uploaded, stream), "transfer inputs uploaded");
    check(cudaStreamSynchronize(stream), "transfer complete input copies");
}

fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto& batch = *static_cast<BatchState*>(state);
    validate_inputs(batch, in);
    const auto& plan = batch.slots[0]->plan;
    fherma::Outputs out;
    out.c.shape = {int64_t(batch.batch), int64_t(plan.n), int64_t(plan.l)};
    // Reserve capacity without constructing or zero-filling the full result.
    // d2h appends the finished bytes into this independent owned vector.
    out.c.data.reserve(batch.total);
    cudaStream_t compute = nullptr;
    if (batch.batch == 1) {
        auto& s = *batch.slots[0];
        for (auto stream : s.streams)
            check(cudaStreamWaitEvent(stream, batch.uploaded, 0), "device operand waits input");
        compute = s.streams[1];
        if (plan.n == 65536) {
            enqueue_device_operand<65536, 114, true>(s, 0, batch.raw_a, s.streams[0]);
            enqueue_device_operand<65536, 114, true>(s, 1, batch.raw_b, compute);
            check(cudaStreamWaitEvent(compute, s.a_forwarded, 0), "join forwarded operands");
            finish_device_product<65536, 114, 480>(s, compute);
        } else {
            enqueue_device_operand<32768, 57, true>(s, 0, batch.raw_a, s.streams[0]);
            enqueue_device_operand<32768, 57, true>(s, 1, batch.raw_b, compute);
            check(cudaStreamWaitEvent(compute, s.a_forwarded, 0), "join forwarded operands");
            finish_device_product<32768, 57, 256>(s, compute);
        }
        enqueue_device_result(s, batch.raw_c, compute);
    } else {
        compute = batch.streams[kCompute];
        check(cudaStreamWaitEvent(compute, batch.uploaded, 0), "device compute waits input");
        for (unsigned product = 0; product < batch.batch; ++product) {
            auto& s = *batch.slots[product % batch.slot_count];
            const size_t offset = size_t(product) * batch.words;
            // One compute stream serializes every scratch reuse; all operands
            // already reside in their own complete raw device ranges.
            enqueue_device_operand<65536, 114, false>(s, 0, batch.raw_a + offset, compute);
            enqueue_device_operand<65536, 114, false>(s, 1, batch.raw_b + offset, compute);
            finish_device_product<65536, 114, 480>(s, compute);
            enqueue_device_result(s, batch.raw_c + offset, compute);
        }
    }
    check(cudaEventRecord(batch.computed, compute), "device results computed");
    check(cudaDeviceSynchronize(), "run complete all device operations");
    return out;
}

fherma::Outputs fherma_d2h(void* state, fherma::Outputs answer) {
    auto& batch = *static_cast<BatchState*>(state);
    if (!input_shape(answer.c.shape, batch) || !answer.c.data.empty())
        throw std::runtime_error("ByteCRT transfer output shape/empty owned vector");
    const auto stream = batch.batch == 1 ? batch.slots[0]->streams[0] : batch.streams[kD2H];
    check(cudaStreamWaitEvent(stream, batch.computed, 0), "transfer copy waits results");
    for (unsigned product = 0; product < batch.batch; ++product) {
        auto& slot = *batch.slots[product % batch.slot_count];
        if (product >= batch.slot_count) drain_result_copy(slot, answer.c.data);
        enqueue_result_copy(batch, slot, product, stream);
    }
    for (unsigned product = batch.batch - batch.slot_count; product < batch.batch; ++product)
        drain_result_copy(*batch.slots[product % batch.slot_count], answer.c.data);
    return answer;
}

void fherma_free(void* state) { delete static_cast<BatchState*>(state); }
