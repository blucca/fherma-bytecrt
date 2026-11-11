#include "fherma.h"
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
struct State { fherma::Point p; bool tuned; size_t calls = 0; uintptr_t prior_a = 0, prior_b = 0, prior_c = 0; };

static std::string mapping(const void* ptr) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(ptr);
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long begin = 0, end = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2 && x >= begin && x < end) {
            return line.find("[heap]") != std::string::npos ? "heap" : "anonymous_or_other";
        }
    }
    return "unmapped";
}
static long virtual_kib() {
    std::ifstream s("/proc/self/statm"); long pages = 0; s >> pages;
    return pages * (sysconf(_SC_PAGESIZE) / 1024);
}
static void json_ptr(const char* name, const void* ptr, uintptr_t prior) {
    const auto x = reinterpret_cast<uintptr_t>(ptr);
    std::cout << ",\"" << name << "_ptr\":\"0x" << std::hex << x << std::dec
              << "\",\"" << name << "_mapping\":\"" << mapping(ptr)
              << "\",\"" << name << "_same_previous\":" << (prior && x == prior ? "true" : "false");
}
void* fherma_init(const fherma::Point& p) {
    const char* mode = std::getenv("BATCH_HEAP_MODE");
    bool tuned = mode && std::strcmp(mode, "mallopt") == 0;
    if (tuned) { mallopt(M_MMAP_THRESHOLD, INT_MAX); mallopt(M_TRIM_THRESHOLD, -1); }
    auto* s = new State{p, tuned};
    std::cout << "{\"event\":\"init\",\"mode\":\"" << (tuned ? "mallopt" : "default")
              << "\",\"tensor_bytes\":" << uint64_t(p.B) * p.N * p.L * sizeof(uint32_t)
              << ",\"virtual_kib\":" << virtual_kib() << "}" << std::endl;
    return s;
}
fherma::Outputs fherma_run(void* opaque, const fherma::Inputs& in) {
    auto& s = *static_cast<State*>(opaque);
    const size_t count = in.a.data.size(), product = size_t(s.p.N) * s.p.L;
    if (in.b.data.size() != count) throw std::runtime_error("input size mismatch");
    rusage before{}, after{}, copy_before{}, copy_after{};
    getrusage(RUSAGE_SELF, &before);
    const auto started = Clock::now();
    fherma::Outputs out;
    out.c.shape = in.a.shape;
    out.c.data.reserve(count); // Full owned result; the vector's pages are first touched by inserts.
    const auto reserved = Clock::now();
    long copy_faults = 0; double copy_seconds = 0;
    {
        std::vector<uint32_t> temporary(product);
        for (size_t b = 0; b < s.p.B; ++b) {
            const size_t base = b * product;
            for (size_t j = 0; j < product; ++j) temporary[j] = in.a.data[base+j] ^ in.b.data[base+j];
            getrusage(RUSAGE_SELF, &copy_before);
            const auto c0 = Clock::now();
            out.c.data.insert(out.c.data.end(), temporary.begin(), temporary.end());
            const auto c1 = Clock::now();
            getrusage(RUSAGE_SELF, &copy_after);
            copy_seconds += std::chrono::duration<double>(c1-c0).count();
            copy_faults += copy_after.ru_minflt - copy_before.ru_minflt;
        }
    }
    const auto done = Clock::now();
    getrusage(RUSAGE_SELF, &after);
    // Core run and copy metrics stop before maps parsing / diagnostic I/O.
    std::cout << std::setprecision(12) << "{\"event\":\"run\",\"case\":" << s.calls
              << ",\"mode\":\"" << (s.tuned ? "mallopt" : "default") << "\",\"words\":" << count
              << ",\"reserve_s\":" << std::chrono::duration<double>(reserved-started).count()
              << ",\"run_core_s\":" << std::chrono::duration<double>(done-started).count()
              << ",\"copy_s\":" << copy_seconds << ",\"copy_minflt\":" << copy_faults
              << ",\"run_minflt\":" << after.ru_minflt-before.ru_minflt
              << ",\"run_majflt\":" << after.ru_majflt-before.ru_majflt
              << ",\"total_minflt\":" << after.ru_minflt << ",\"maxrss_kib\":" << after.ru_maxrss
              << ",\"virtual_kib\":" << virtual_kib();
    json_ptr("a", in.a.data.data(), s.prior_a);
    json_ptr("b", in.b.data.data(), s.prior_b);
    json_ptr("c", out.c.data.data(), s.prior_c);
    std::cout << "}" << std::endl;
    s.prior_a = reinterpret_cast<uintptr_t>(in.a.data.data());
    s.prior_b = reinterpret_cast<uintptr_t>(in.b.data.data());
    s.prior_c = reinterpret_cast<uintptr_t>(out.c.data.data());
    ++s.calls;
    return out;
}
void fherma_free(void* opaque) { delete static_cast<State*>(opaque); }
