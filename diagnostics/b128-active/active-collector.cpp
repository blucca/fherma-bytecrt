// Fixed B128 diagnostic: CUPTI produces GPU records; timed host code writes fixed arrays.
#include "active-collector.hpp"
#include <cupti.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

static_assert(CUDA_VERSION == 12080, "active artifact requires actual CUDA 12.8 headers");
namespace bytecrt_active {
namespace {
constexpr size_t kBufferBytes = 4 * 1024 * 1024;
constexpr unsigned kBuffers = 8, kExternal = 131072;
constexpr uint64_t kExternalBase = uint64_t(128) << 32;
constexpr uint64_t kCopyBytes = 3604480;
constexpr CUpti_ActivityKind kKinds[] = {
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION};
void cupti_check(CUptiResult result, const char* what) {
    if (result == CUPTI_SUCCESS) return;
    const char* message = "CUPTI failure";
    cuptiGetResultString(result, &message);
    throw std::runtime_error(std::string(what) + ": " + message);
}
ClockPair clock_pair() {
    ClockPair pair;
    pair.host_before = host_ns();
    cupti_check(cuptiGetTimestamp(&pair.cupti), "CUPTI call timestamp");
    pair.host_after = host_ns();
    return pair;
}
struct Buffer {
    alignas(8) std::array<uint8_t, kBufferBytes> data{};
    std::atomic<size_t> valid{0};
    std::atomic<bool> completed{false};
};
struct External { uint32_t correlation = 0; uint64_t id = 0; };
struct Gpu {
    uint64_t start = 0, end = 0, bytes = 0;
    uint32_t correlation = 0, stream = 0, context = 0, device = 0;
    unsigned stage = 0;
};
struct CallGpu {
    std::array<Gpu, 1024> h2d{};
    std::array<Gpu, 1280> kernel{};
    std::array<Gpu, 512> d2h{};
    unsigned nh2d = 0, nkernel = 0, nd2h = 0, overflow = 0;
    unsigned by_correlation = 0, by_runtime_correlation = 0;
};
struct Summary {
    uint64_t active = 0, coverage = 0, idle = 0, head = 0, tail = 0, max_gap = 0;
    uint64_t bytes = 0;
    uint32_t stream = 0, context = 0, device = 0;
    bool valid = true;
};
unsigned stage_of(const char* name) {
    if (!name) return 0;
    if (std::strstr(name, "encode_mma32")) return 1;
    if (std::strstr(name, "forward_first_coalesced")) return 2;
    if (std::strstr(name, "fused_middle")) return 3;
    if (std::strstr(name, "inverse_weighted")) return 4;
    if (std::strstr(name, "g1_crt32")) return 5;
    return 0;
}
std::string json_quote(const char* value) {
    std::string out = "\"";
    if (value) for (const unsigned char c : std::string(value)) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) {
            char escaped[7]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        } else out += char(c);
    }
    return out + '"';
}
void varint(std::string& bytes, uint64_t value) {
    while (value >= 128) { bytes += char((value & 127) | 128); value >>= 7; }
    bytes += char(value);
}
std::string base64(const std::string& bytes) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((bytes.size() + 2) / 3) * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t a = uint8_t(bytes[i]);
        const uint32_t b = i + 1 < bytes.size() ? uint8_t(bytes[i + 1]) : 0;
        const uint32_t c = i + 2 < bytes.size() ? uint8_t(bytes[i + 2]) : 0;
        const uint32_t word = (a << 16) | (b << 8) | c;
        out += alphabet[(word >> 18) & 63]; out += alphabet[(word >> 12) & 63];
        out += i + 1 < bytes.size() ? alphabet[(word >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? alphabet[word & 63] : '=';
    }
    return out;
}
std::string gpu_trace(const Gpu* rows, unsigned count, uint64_t origin) {
    std::string bytes;
    bytes.reserve(size_t(count) * 10);
    uint64_t previous = origin;
    for (unsigned i = 0; i < count; ++i) {
        varint(bytes, rows[i].start - previous);
        varint(bytes, rows[i].end - rows[i].start);
        varint(bytes, rows[i].correlation); // Actual key matched to an external record.
        previous = rows[i].start;
    }
    return base64(bytes);
}
std::string host_trace(const Span* rows, unsigned count, uint64_t origin) {
    std::string bytes;
    bytes.reserve(size_t(count) * 7);
    uint64_t previous = origin;
    for (unsigned i = 0; i < count; ++i) {
        varint(bytes, rows[i].begin - previous);
        varint(bytes, rows[i].end - rows[i].begin);
        previous = rows[i].begin;
    }
    return base64(bytes);
}
Summary summarize(Gpu* rows, unsigned count, const Call& call) {
    Summary s;
    std::sort(rows, rows + count, [](const Gpu& a, const Gpu& b) { return a.start < b.start; });
    if (!count) { s.valid = false; return s; }
    s.stream = rows[0].stream; s.context = rows[0].context; s.device = rows[0].device;
    s.valid = call.closed && call.end.cupti >= call.begin.cupti;
    for (unsigned i = 0; i < count; ++i) {
        const auto& r = rows[i];
        s.valid &= r.start >= call.begin.cupti && r.end <= call.end.cupti && r.end > r.start;
        s.valid &= r.stream == s.stream && r.context == s.context && r.device == s.device;
        s.active += r.end - r.start; s.bytes += r.bytes;
        if (i) {
            s.valid &= r.start >= rows[i - 1].end;
            const uint64_t gap = r.start >= rows[i - 1].end ? r.start - rows[i - 1].end : 0;
            s.idle += gap; s.max_gap = std::max(s.max_gap, gap);
        }
    }
    s.coverage = rows[count - 1].end - rows[0].start;
    s.head = rows[0].start >= call.begin.cupti ? rows[0].start - call.begin.cupti : 0;
    s.tail = call.end.cupti >= rows[count - 1].end ? call.end.cupti - rows[count - 1].end : 0;
    return s;
}
uint64_t gpu_union(const CallGpu& gpu) {
    std::array<Span, 2816> rows{};
    unsigned count = 0;
    const auto append = [&](const auto& lane, unsigned n) {
        for (unsigned i = 0; i < n; ++i) rows[count++] = {lane[i].start, lane[i].end};
    };
    append(gpu.h2d, gpu.nh2d); append(gpu.kernel, gpu.nkernel); append(gpu.d2h, gpu.nd2h);
    std::sort(rows.begin(), rows.begin() + count, [](const Span& a, const Span& b) { return a.begin < b.begin; });
    uint64_t total = 0, begin = 0, end = 0;
    for (unsigned i = 0; i < count; ++i) {
        if (i == 0) { begin = rows[i].begin; end = rows[i].end; }
        else if (rows[i].begin <= end) end = std::max(end, rows[i].end);
        else { total += end - begin; begin = rows[i].begin; end = rows[i].end; }
    }
    return total + (count ? end - begin : 0);
}
void print_lane(const char* name, const Gpu* rows, unsigned count, const Summary& s, uint64_t origin) {
    const auto trace = s.valid ? gpu_trace(rows, count, origin) : std::string{};
    std::fprintf(stderr,
        "{\"name\":\"%s\",\"count\":%u,\"stream\":%u,\"context\":%u,\"device\":%u,"
        "\"bytes\":%llu,\"active_ns\":%llu,\"coverage_ns\":%llu,\"idle_ns\":%llu,"
        "\"head_ns\":%llu,\"tail_ns\":%llu,\"max_gap_ns\":%llu,\"intervals_b64\":\"%s\"}",
        name, count, s.stream, s.context, s.device,
        (unsigned long long)s.bytes, (unsigned long long)s.active, (unsigned long long)s.coverage,
        (unsigned long long)s.idle, (unsigned long long)s.head, (unsigned long long)s.tail,
        (unsigned long long)s.max_gap, trace.c_str());
}
void print_host_lane(const char* name, const Span* rows, unsigned count, uint64_t origin) {
    uint64_t sum = 0;
    for (unsigned i = 0; i < count; ++i) sum += rows[i].end - rows[i].begin;
    const auto trace = host_trace(rows, count, origin);
    std::fprintf(stderr, "\"%s\":{\"count\":%u,\"sum_ns\":%llu,\"intervals_b64\":\"%s\"}",
        name, count, (unsigned long long)sum, trace.c_str());
}
void print_pair(const ClockPair& p) {
    std::fprintf(stderr, "[%llu,%llu,%llu]", (unsigned long long)p.host_before,
        (unsigned long long)p.cupti, (unsigned long long)p.host_after);
}
}  // namespace

struct Collector::Impl {
    std::array<Buffer, kBuffers> buffers{};
    std::array<External, kExternal> external{};
    std::array<Call, kCalls> calls{};
    std::array<CallGpu, kCalls> gpu{};
    std::array<std::string, 6> names{};       // Assigned only while parsing in free.
    std::atomic<unsigned> requested{0}, completed{0}, exhausted{0}, callback_errors{0};
    std::atomic<uint64_t> dropped{0};
    unsigned call_count = 0, external_count = 0, external_overflow = 0;
    unsigned kernel_records = 0, memcpy_records = 0, external_records = 0, other_records = 0;
    unsigned unmapped_gpu = 0, conflicting_maps = 0, unexpected_copy = 0, protocol_errors = 0;
    uint32_t runtime_version = 0;
    bool finished = false;
    static Impl* active;
    static void CUPTIAPI request(uint8_t** buffer, size_t* size, size_t* max_records) {
        auto& p = *active;
        const unsigned index = p.requested.fetch_add(1, std::memory_order_relaxed);
        *max_records = 0;
        if (index >= kBuffers) {
            p.exhausted.fetch_add(1, std::memory_order_relaxed);
            *buffer = nullptr; *size = 0; return;
        }
        *buffer = p.buffers[index].data.data(); *size = kBufferBytes;
    }
    static void CUPTIAPI complete(CUcontext context, uint32_t stream, uint8_t* buffer,
                                  size_t size, size_t valid_size) {
        auto& p = *active;
        bool found = false;
        for (auto& slot : p.buffers) if (slot.data.data() == buffer) {
            found = valid_size <= size && size == kBufferBytes;
            slot.valid.store(found ? valid_size : 0, std::memory_order_release);
            slot.completed.store(true, std::memory_order_release);
            break;
        }
        if (!found) p.callback_errors.fetch_add(1, std::memory_order_relaxed);
        size_t dropped = 0;
        if (cuptiActivityGetNumDroppedRecords(context, stream, &dropped) != CUPTI_SUCCESS)
            p.callback_errors.fetch_add(1, std::memory_order_relaxed);
        p.dropped.fetch_add(dropped, std::memory_order_relaxed);
        p.completed.fetch_add(1, std::memory_order_relaxed);
        // Retain each filled buffer. Parsing, sorting, encoding and printing happen in free.
    }
    template<class F> void records(F consume) {
        for (auto& slot : buffers) if (slot.completed.load(std::memory_order_acquire)) {
            CUpti_Activity* record = nullptr;
            CUptiResult result;
            const size_t valid = slot.valid.load(std::memory_order_acquire);
            while ((result = cuptiActivityGetNextRecord(slot.data.data(), valid, &record)) == CUPTI_SUCCESS)
                consume(record);
            cupti_check(result == CUPTI_ERROR_MAX_LIMIT_REACHED ? CUPTI_SUCCESS : result,
                        "CUPTI record parse");
        }
    }
    uint64_t external_id(uint32_t correlation) const {
        if (!correlation) return 0;
        const auto first = external.begin(), last = first + external_count;
        const auto at = std::lower_bound(first, last, correlation,
            [](const External& row, uint32_t value) { return row.correlation < value; });
        return at != last && at->correlation == correlation ? at->id : 0;
    }
    void add(Gpu record, unsigned lane, uint32_t runtime_correlation) {
        const uint64_t first = external_id(record.correlation);
        const uint64_t second = external_id(runtime_correlation);
        if (first && second && first != second) { ++conflicting_maps; return; }
        const uint64_t id = first ? first : second;
        const unsigned call_id = unsigned(id & 0xffffffffu);
        if ((id & 0xffffffff00000000ull) != kExternalBase || call_id < 1 || call_id > call_count) {
            ++unmapped_gpu; return;
        }
        auto& output = gpu[call_id - 1];
        if (first) ++output.by_correlation;
        else { ++output.by_runtime_correlation; record.correlation = runtime_correlation; }
        if (lane == 0 && output.nh2d < output.h2d.size()) output.h2d[output.nh2d++] = record;
        else if (lane == 1 && output.nkernel < output.kernel.size()) output.kernel[output.nkernel++] = record;
        else if (lane == 2 && output.nd2h < output.d2h.size()) output.d2h[output.nd2h++] = record;
        else ++output.overflow;
    }
    void parse() {
        records([&](CUpti_Activity* record) {
            if (record->kind != CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) return;
            ++external_records;
            const auto& row = *reinterpret_cast<CUpti_ActivityExternalCorrelation*>(record);
            if (row.externalKind != CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) return;
            if (external_count < external.size()) external[external_count++] = {row.correlationId, row.externalId};
            else ++external_overflow;
        });
        std::sort(external.begin(), external.begin() + external_count,
            [](const External& a, const External& b) { return a.correlation < b.correlation; });
        for (unsigned i = 1; i < external_count; ++i)
            if (external[i].correlation == external[i - 1].correlation && external[i].id != external[i - 1].id)
                ++conflicting_maps;
        records([&](CUpti_Activity* record) {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                ++kernel_records;
                const auto& row = *reinterpret_cast<CUpti_ActivityKernel9*>(record);
                const unsigned stage = stage_of(row.name);
                if (stage && names[stage].empty()) names[stage] = row.name;
                add({row.start, row.end, 0, row.correlationId, row.streamId,
                     row.contextId, row.deviceId, stage}, 1, 0);
            } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                ++memcpy_records;
                const auto& row = *reinterpret_cast<CUpti_ActivityMemcpy5*>(record);
                if (row.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD || row.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH)
                    add({row.start, row.end, row.bytes, row.correlationId, row.streamId,
                         row.contextId, row.deviceId, 0},
                         row.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD ? 0 : 2, row.runtimeCorrelationId);
                else ++unexpected_copy;
            } else if (record->kind != CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) ++other_records;
        });
    }
    void print() {
        const bool session_valid = call_count == kCalls && dropped.load() == 0 && exhausted.load() == 0 &&
            callback_errors.load() == 0 && !external_overflow && !unmapped_gpu && !conflicting_maps &&
            !unexpected_copy && !protocol_errors && !other_records;
        std::fprintf(stderr,
            "{\"active_session\":1,\"batch\":128,\"calls\":%u,\"cuda_header_version\":%u,"
            "\"cupti_header_api_version\":%u,\"cupti_runtime_api_version\":%u,"
            "\"kernel_record\":\"CUpti_ActivityKernel9\",\"memcpy_record\":\"CUpti_ActivityMemcpy5\","
            "\"external_record\":\"CUpti_ActivityExternalCorrelation\","
            "\"runtime_tracing\":false,\"new_cuda_timing_events\":0,"
            "\"record_kinds\":[%u,%u,%u],\"kernel_records\":%u,\"memcpy_records\":%u,"
            "\"external_records\":%u,\"other_records\":%u,\"external_maps\":%u,"
            "\"buffer_requested\":%u,\"buffer_completed\":%u,\"buffer_exhausted\":%u,"
            "\"dropped\":%llu,\"callback_errors\":%u,\"external_overflow\":%u,\"unmapped_gpu\":%u,"
            "\"conflicting_maps\":%u,\"unexpected_copy\":%u,\"protocol_errors\":%u,\"valid\":%s,"
            "\"gpu_codec\":\"base64-uvarint-startdelta-duration-correlation\","
            "\"host_codec\":\"base64-uvarint-startdelta-duration\",\"kernel_names\":[",
            call_count, CUDA_VERSION, CUPTI_API_VERSION, runtime_version,
            unsigned(kKinds[0]), unsigned(kKinds[1]), unsigned(kKinds[2]),
            kernel_records, memcpy_records, external_records, other_records, external_count,
            requested.load(), completed.load(), exhausted.load(), (unsigned long long)dropped.load(),
            callback_errors.load(), external_overflow, unmapped_gpu, conflicting_maps, unexpected_copy,
            protocol_errors, session_valid ? "true" : "false");
        for (unsigned stage = 1; stage <= 5; ++stage)
            std::fprintf(stderr, "%s%s", stage == 1 ? "" : ",", json_quote(names[stage].c_str()).c_str());
        std::fprintf(stderr, "]}\n");
        for (unsigned c = 0; c < call_count; ++c) {
            auto& g = gpu[c]; const auto& call = calls[c];
            const auto h = summarize(g.h2d.data(), g.nh2d, call);
            const auto k = summarize(g.kernel.data(), g.nkernel, call);
            const auto d = summarize(g.d2h.data(), g.nd2h, call);
            bool valid = session_valid && h.valid && k.valid && d.valid && !g.overflow &&
                g.nh2d == 1024 && g.nkernel == 1280 && g.nd2h == 512 &&
                h.bytes == 3690987520ull && d.bytes == 1845493760ull &&
                call.packs == kPacks && call.drains == kDrains &&
                h.stream != k.stream && h.stream != d.stream && k.stream != d.stream &&
                h.context == k.context && h.context == d.context && h.device == k.device && h.device == d.device;
            std::array<uint64_t, 5> stages{};
            constexpr unsigned fifo_stage[] = {1, 2, 1, 2, 3, 4, 5, 5, 5, 5};
            uint64_t readiness_overlap = 0, post_readiness = 0, non_encode_idle = 0;
            for (unsigned i = 0; i < g.nkernel; ++i) {
                const auto& r = g.kernel[i];
                valid &= r.stage == fifo_stage[i % 10];
                if (r.stage >= 1 && r.stage <= 5) stages[r.stage - 1] += r.end - r.start;
                if (!i) continue;
                const uint64_t previous = g.kernel[i - 1].end;
                if (r.start < previous) continue;
                const uint64_t gap = r.start - previous;
                if (i % 10 == 0 || i % 10 == 2) {
                    const unsigned copy = (i / 10) * 8 + (i % 10 == 2 ? 7 : 3);
                    if (copy < g.nh2d) {
                        const uint64_t ready = g.h2d[copy].end;
                        valid &= ready <= r.start;
                        const uint64_t overlap = ready > previous ? std::min(ready, r.start) - previous : 0;
                        readiness_overlap += overlap; post_readiness += gap - overlap;
                    }
                } else non_encode_idle += gap;
            }
            for (unsigned i = 0; i < g.nh2d; ++i) valid &= g.h2d[i].bytes == kCopyBytes;
            for (unsigned i = 0; i < g.nd2h; ++i) {
                valid &= g.d2h[i].bytes == kCopyBytes;
                const unsigned crt = (i / 4) * 10 + 6 + i % 4;
                if (crt < g.nkernel) valid &= g.kernel[crt].end <= g.d2h[i].start;
            }
            const uint64_t origin = call.begin.host_before;
            std::array<Span, kDrains> waits{}, owned{};
            for (unsigned i = 0; i < call.packs; ++i)
                valid &= call.pack[i].begin >= origin && call.pack[i].end >= call.pack[i].begin &&
                    call.pack[i].end <= call.end.host_after;
            for (unsigned i = 0; i < call.drains; ++i) {
                const auto& row = call.drain[i];
                waits[i] = {row.begin, row.waited}; owned[i] = {row.waited, row.owned};
                valid &= row.begin >= origin && row.waited >= row.begin && row.owned >= row.waited &&
                    row.owned <= call.end.host_after;
            }
            std::fprintf(stderr,
                "{\"active_trace\":1,\"call\":%u,\"external_id\":%llu,\"scored\":%s,\"valid\":%s,"
                "\"cup_begin_ns\":%llu,\"cup_end_ns\":%llu,\"calibration\":{\"begin\":",
                c + 1, (unsigned long long)(kExternalBase | (c + 1)), c == 0 ? "false" : "true",
                valid ? "true" : "false", (unsigned long long)call.begin.cupti, (unsigned long long)call.end.cupti);
            print_pair(call.begin); std::fprintf(stderr, ",\"end\":"); print_pair(call.end);
            std::fprintf(stderr,
                "},\"by_correlation\":%u,\"by_runtime_correlation\":%u,\"gpu_overflow\":%u,"
                "\"gpu_union_ns\":%llu,\"stage_active_ns\":[%llu,%llu,%llu,%llu,%llu],"
                "\"compute_gaps\":{\"h2d_readiness_overlap_ns\":%llu,\"post_readiness_ns\":%llu,"
                "\"non_encode_idle_ns\":%llu},\"lanes\":[",
                g.by_correlation, g.by_runtime_correlation, g.overflow, (unsigned long long)gpu_union(g),
                (unsigned long long)stages[0], (unsigned long long)stages[1], (unsigned long long)stages[2],
                (unsigned long long)stages[3], (unsigned long long)stages[4],
                (unsigned long long)readiness_overlap, (unsigned long long)post_readiness,
                (unsigned long long)non_encode_idle);
            print_lane("h2d", g.h2d.data(), g.nh2d, h, call.begin.cupti); std::fprintf(stderr, ",");
            print_lane("kernel", g.kernel.data(), g.nkernel, k, call.begin.cupti); std::fprintf(stderr, ",");
            print_lane("d2h", g.d2h.data(), g.nd2h, d, call.begin.cupti);
            std::fprintf(stderr, "],\"host\":{\"origin_ns\":%llu,\"reserve\":[%llu,%llu],",
                (unsigned long long)origin, (unsigned long long)(call.reserve.begin - origin),
                (unsigned long long)(call.reserve.end - origin));
            print_host_lane("pack", call.pack.data(), call.packs, origin); std::fprintf(stderr, ",");
            print_host_lane("wait", waits.data(), call.drains, origin); std::fprintf(stderr, ",");
            print_host_lane("owned", owned.data(), call.drains, origin); std::fprintf(stderr, "}}\n");
        }
    }
};
Collector::Impl* Collector::Impl::active = nullptr;
Collector::Collector() : impl_(std::make_unique<Impl>()) {
    if (Impl::active) throw std::runtime_error("B128 active collector already initialized");
    Impl::active = impl_.get();
    cupti_check(cuptiGetVersion(&impl_->runtime_version), "CUPTI runtime version");
    if (impl_->runtime_version != CUPTI_API_VERSION)
        throw std::runtime_error("CUDA 12.8 CUPTI header/library API version mismatch; stop");
    cupti_check(cuptiActivityRegisterCallbacks(Impl::request, Impl::complete), "CUPTI buffer callbacks");
    for (const auto kind : kKinds) cupti_check(cuptiActivityEnable(kind), "CUPTI activity enable");
}
Collector::~Collector() = default;
Call* Collector::begin() {
    auto& p = *impl_;
    if (p.call_count >= kCalls) throw std::runtime_error("B128 diagnostic requires 1 warm + 3 fresh calls");
    auto* call = &p.calls[p.call_count++];
    call->begin = clock_pair();
    cupti_check(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,
                kExternalBase | p.call_count), "CUPTI push call ID");
    return call;
}
void Collector::end(Call* call) noexcept {
    uint64_t popped = 0;
    auto& p = *impl_;
    if (cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &popped) != CUPTI_SUCCESS ||
        popped != (kExternalBase | p.call_count)) ++p.protocol_errors;
    call->end.host_before = host_ns();
    if (cuptiGetTimestamp(&call->end.cupti) != CUPTI_SUCCESS) ++p.protocol_errors;
    call->end.host_after = host_ns();
    call->closed = true;
}
void Collector::finish() {
    auto& p = *impl_;
    if (p.finished) return;
    cupti_check(cuptiActivityFlushAll(0), "CUPTI completed-activity flush");
    for (const auto kind : kKinds) cupti_check(cuptiActivityDisable(kind), "CUPTI activity disable");
    cupti_check(cuptiActivityFlushAll(0), "CUPTI final buffer flush");
    p.parse(); p.print();
    p.finished = true; Impl::active = nullptr;
}
}  // namespace bytecrt_active
