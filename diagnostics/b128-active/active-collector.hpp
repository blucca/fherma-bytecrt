// B128-only, four-call CUPTI diagnostic. Every host trace array is allocated in init.
#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>

namespace bytecrt_active {
constexpr unsigned kCalls = 4, kPacks = 1024, kDrains = 512;
struct Span { uint64_t begin = 0, end = 0; };
struct Drain { uint64_t begin = 0, waited = 0, owned = 0; };
struct ClockPair { uint64_t host_before = 0, cupti = 0, host_after = 0; };
struct Call {
    ClockPair begin{}, end{};
    Span reserve{};
    std::array<Span, kPacks> pack{};
    std::array<Drain, kDrains> drain{};
    unsigned packs = 0, drains = 0;
    bool closed = false;
};
inline uint64_t host_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
class Collector {
public:
    Collector();
    ~Collector();
    Collector(const Collector&) = delete;
    Collector& operator=(const Collector&) = delete;
    Call* begin();
    void end(Call* call) noexcept;
    void finish();                         // Only fherma_free calls this; outside timing.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class CallScope {
public:
    explicit CallScope(Collector* collector)
        : collector_(collector), call_(collector ? collector->begin() : nullptr) {}
    ~CallScope() { if (collector_) collector_->end(call_); }
    Call* record() const { return call_; }
private:
    Collector* collector_;
    Call* call_;
};
}  // namespace bytecrt_active
