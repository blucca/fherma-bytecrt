// CPU-only paired libc/AVX2 non-temporal experiment; native whole-call is measured separately.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <malloc.h>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>
#include <gnu/libc-version.h>

extern "C" void* __real_memcpy(void*, const void*, size_t);
extern "C" void* __real_memmove(void*, const void*, size_t);
using Clock = std::chrono::steady_clock;
constexpr size_t kChunkWords = 901120;
constexpr size_t kChunkBytes = 4 * kChunkWords;
constexpr size_t kProductWords = 4 * kChunkWords;
constexpr size_t kParallelBytes = 1536 * 1024;
static bool use_nt = false, measured = false;
static uint64_t wrapped_calls = 0, wrapped_bytes = 0;

// One worker fences after all its assigned tasks. Source/destination are disjoint.
static inline void nt_copy_unfenced(unsigned char* dst, const unsigned char* src,
                                    size_t bytes) {
    const size_t head = std::min(bytes, size_t(-reinterpret_cast<uintptr_t>(dst)) & 63);
    if (head) __real_memcpy(dst, src, head);
    dst += head; src += head; bytes -= head;
    for (; bytes >= 128; dst += 128, src += 128, bytes -= 128) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 64));
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 96));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst), a);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 32), b);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 64), c);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 96), d);
    }
    for (; bytes >= 32; dst += 32, src += 32, bytes -= 32)
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src)));
    if (bytes) __real_memcpy(dst, src, bytes);
}

static void copy_one(void* dest, const void* source, size_t bytes, bool nt) {
    if (nt) {
        nt_copy_unfenced(static_cast<unsigned char*>(dest),
                         static_cast<const unsigned char*>(source), bytes);
        _mm_sfence();
    } else __real_memcpy(dest, source, bytes);
}

static void parallel_copy(void* dest, const void* source, size_t bytes, bool nt,
                          bool word_tasks = false) {
    const int workers = std::min(12, omp_get_num_procs());
    const size_t units = word_tasks ? bytes / 4 : bytes;
    const size_t scale = word_tasks ? 4 : 1;
    const size_t per = units / 12, extra = units % 12;
    auto* dst = static_cast<unsigned char*>(dest);
    const auto* src = static_cast<const unsigned char*>(source);
    if (nt) {
#pragma omp parallel num_threads(workers)
        {
#pragma omp for schedule(static) nowait
            for (int task = 0; task < 12; ++task) {
                const size_t begin = scale * (per * task + std::min(extra, size_t(task)));
                const size_t count = scale * (per + (size_t(task) < extra));
                nt_copy_unfenced(dst + begin, src + begin, count);
            }
            _mm_sfence();
        }
    } else {
#pragma omp parallel for num_threads(workers) schedule(static)
        for (int task = 0; task < 12; ++task) {
            const size_t begin = scale * (per * task + std::min(extra, size_t(task)));
            const size_t count = scale * (per + (size_t(task) < extra));
            __real_memcpy(dst + begin, src + begin, count);
        }
    }
}

extern "C" void* __wrap_memcpy(void* dest, const void* source, size_t bytes) {
    if (bytes >= kParallelBytes) {
        if (measured) { ++wrapped_calls; wrapped_bytes += bytes; }
        parallel_copy(dest, source, bytes, use_nt);
        return dest;
    }
    return __real_memcpy(dest, source, bytes);
}
extern "C" void* __wrap_memmove(void* dest, const void* source, size_t bytes) {
    const uintptr_t dst = reinterpret_cast<uintptr_t>(dest), src = reinterpret_cast<uintptr_t>(source);
    const bool disjoint = dst >= src ? dst - src >= bytes : src - dst >= bytes;
    if (disjoint && bytes >= kParallelBytes) {
        if (measured) { ++wrapped_calls; wrapped_bytes += bytes; }
        parallel_copy(dest, source, bytes, use_nt);
        return dest;
    }
    return __real_memmove(dest, source, bytes);
}

static rusage usage() { rusage u{}; if (getrusage(RUSAGE_SELF, &u)) std::abort(); return u; }
static double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double cpu_ms(timeval a, timeval b) {
    return 1e3 * double(b.tv_sec - a.tv_sec) + 1e-3 * double(b.tv_usec - a.tv_usec);
}
static void pattern(unsigned char* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = uint8_t((i * 1315423911ull) ^ (i >> 11) ^ 0x543);
}
static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

static void semantics() {
    std::vector<unsigned char> src(kChunkBytes + 256), dst(kChunkBytes + 256);
    pattern(src.data(), src.size());
    const std::vector<size_t> sizes = {0,1,2,7,15,31,32,33,63,64,65,95,96,97,
        127,128,129,255,256,257,1023,1024,1025,300372,300373,300374,300376,kChunkBytes};
    const size_t src_offsets[] = {0,1,17,31,32,63};
    size_t tests = 0, verified_bytes = 0;
    for (size_t n : sizes) for (size_t d = 0; d < 64; ++d) for (size_t s : src_offsets) {
        std::fill(dst.begin(), dst.begin() + n + 192, uint8_t(0xa5));
        copy_one(dst.data() + 64 + d, src.data() + s, n, true);
        check(std::memcmp(dst.data() + 64 + d, src.data() + s, n) == 0, "primitive bytes differ");
        for (size_t i = 0; i < 64 + d; ++i) check(dst[i] == 0xa5, "head guard changed");
        for (size_t i = 64 + d + n; i < n + 192; ++i) check(dst[i] == 0xa5, "tail guard changed");
        ++tests; verified_bytes += n;
    }
    for (bool nt : {false,true}) for (bool words : {false,true})
        for (size_t d : {size_t(0),size_t(1),size_t(31),size_t(32),size_t(63)}) {
            std::fill(dst.begin(), dst.end(), uint8_t(0xa5));
            parallel_copy(dst.data() + 64 + d, src.data() + 17, kChunkBytes, nt, words);
            check(std::memcmp(dst.data() + 64 + d, src.data() + 17, kChunkBytes) == 0, "12-task bytes differ");
            for (size_t i = 0; i < 64 + d; ++i) check(dst[i] == 0xa5, "task head guard changed");
            for (size_t i = 64 + d + kChunkBytes; i < dst.size(); ++i) check(dst[i] == 0xa5, "task tail guard changed");
            ++tests; verified_bytes += kChunkBytes;
        }
    use_nt = true;
    std::vector<unsigned char> out;
    out.reserve(kChunkBytes);
    out.insert(out.end(), src.begin(), src.begin() + kChunkBytes);
    check(out.size() == kChunkBytes && std::memcmp(out.data(), src.data(), out.size()) == 0, "vector insert bytes differ");
    ++tests; verified_bytes += out.size();
    std::vector<unsigned char> expected = src;
    __real_memmove(expected.data() + 1, expected.data(), kChunkBytes);
    __wrap_memmove(src.data() + 1, src.data(), kChunkBytes);
    check(src == expected, "overlapping memmove fallback differs");
    ++tests; verified_bytes += kChunkBytes;
    std::printf("{\"phase\":\"semantics\",\"tests\":%zu,\"verified_bytes\":%zu,\"all_bytes_equal\":true,\"guards_equal\":true,\"workers\":%d,\"tasks\":12}\n",
        tests, verified_bytes, std::min(12, omp_get_num_procs()));
}

struct Bench {
    const char* name;
    size_t bytes, dest_slots, source_slots;
    bool parallel, words;
};
static void micro_bench() {
    // >32 MiB rings cover the remote EPYC LLC. Owned source models four output chunks.
    const Bench cases[] = {
        {"worker_hot",300374,1,1,false,false},
        {"worker_stream",300374,256,256,false,false},
        {"chunk_hot_owned",kChunkBytes,1,1,true,false},
        {"chunk_hot_pack",kChunkBytes,1,1,true,true},
        {"pack_stream",kChunkBytes,32,32,true,true},
        {"owned_stream",kChunkBytes,32,4,true,false}
    };
    for (const auto& b : cases) {
        std::vector<unsigned char> source(b.source_slots * b.bytes + 128);
        std::vector<unsigned char> dest(b.dest_slots * b.bytes + 128);
        pattern(source.data(), source.size());
        auto operation = [&](size_t i, bool nt) {
            auto* dst = dest.data() + 33 + (i % b.dest_slots) * b.bytes;
            const auto* src = source.data() + 17 + (i % b.source_slots) * b.bytes;
            if (b.parallel) parallel_copy(dst, src, b.bytes, nt, b.words);
            else copy_one(dst, src, b.bytes, nt);
        };
        const size_t cycles = std::max(size_t(1), (256 * 1024 * 1024 + b.dest_slots * b.bytes - 1) / (b.dest_slots * b.bytes));
        const size_t calls = cycles * b.dest_slots;
        for (size_t i = 0; i < b.dest_slots; ++i) operation(i, false);
        for (size_t pair = 0; pair < 5; ++pair) for (size_t order = 0; order < 2; ++order) {
            const bool nt = (pair + order) % 2;
            const auto before = usage(); const auto start = Clock::now();
            for (size_t i = 0; i < calls; ++i) operation(i, nt);
            const auto end = Clock::now(); const auto after = usage();
            for (size_t i = 0; i < b.dest_slots; ++i)
                check(std::memcmp(dest.data() + 33 + i * b.bytes,
                    source.data() + 17 + (i % b.source_slots) * b.bytes, b.bytes) == 0,
                    "throughput output bytes differ");
            const double ms = millis(start, end);
            std::printf("{\"phase\":\"micro\",\"case\":\"%s\",\"mode\":\"%s\",\"pair\":%zu,\"order\":%zu,\"copy_bytes\":%zu,\"calls\":%zu,\"transferred_bytes\":%zu,\"dest_slots\":%zu,\"source_slots\":%zu,\"workers\":%d,\"tasks\":%d,\"elapsed_ms\":%.6f,\"gb_s\":%.6f,\"minflt\":%ld,\"majflt\":%ld,\"user_ms\":%.3f,\"system_ms\":%.3f,\"all_bytes_equal\":true}\n",
                b.name, nt ? "nt" : "libc", pair, order, b.bytes, calls, calls * b.bytes,
                b.dest_slots, b.source_slots, b.parallel ? std::min(12, omp_get_num_procs()) : 1,
                b.parallel ? 12 : 1, ms, double(calls * b.bytes) / (ms * 1e6),
                after.ru_minflt - before.ru_minflt, after.ru_majflt - before.ru_majflt,
                cpu_ms(before.ru_utime,after.ru_utime),cpu_ms(before.ru_stime,after.ru_stime));
            std::fflush(stdout);
        }
    }
}

static bool heap_mapping(const void* pointer) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    FILE* maps = std::fopen("/proc/self/maps", "r"); check(maps != nullptr, "maps open failed");
    char line[768]; bool heap = false;
    while (std::fgets(line, sizeof(line), maps)) {
        unsigned long begin = 0, end = 0;
        if (std::sscanf(line, "%lx-%lx", &begin, &end) == 2 && begin <= address && address < end) {
            heap = std::strstr(line, "[heap]") != nullptr; break;
        }
    }
    std::fclose(maps); return heap;
}

static void owned_bench() {
    check(mallopt(M_MMAP_THRESHOLD, INT_MAX) && mallopt(M_TRIM_THRESHOLD, -1), "heap policy rejected");
    std::vector<uint32_t> source(kProductWords);
    for (size_t i = 0; i < source.size(); ++i) source[i] = uint32_t(i * 2654435761u) ^ 0xb128543u;
    constexpr size_t products = 128, total_words = products * kProductWords;
    std::vector<uint32_t> out;
    out.reserve(total_words);
    check(heap_mapping(out.data()), "owned vector allocation outside heap");
    for (size_t iteration = 0; iteration < 7; ++iteration) {
        // First call absorbs actual first-touch. Later pairs use AB/BA/AB.
        const bool nt = iteration > 0 && ((iteration - 1) / 2 + (iteration - 1) % 2) % 2;
        use_nt = nt; out.clear(); wrapped_calls = wrapped_bytes = 0;
        const auto before = usage(); measured = true; const auto start = Clock::now();
        for (size_t product = 0; product < products; ++product)
            for (size_t chunk = 0; chunk < 4; ++chunk) {
                const auto* begin = source.data() + chunk * kChunkWords;
                out.insert(out.end(), begin, begin + kChunkWords);
            }
        const auto end = Clock::now(); measured = false; const auto after = usage();
        check(out.size() == total_words && out.capacity() == total_words, "owned vector shape differs");
        check(wrapped_calls == products * 4 && wrapped_bytes == total_words * 4, "owned wrapper observation differs");
        for (size_t product = 0; product < products; ++product)
            check(std::memcmp(out.data() + product * kProductWords, source.data(), kProductWords * 4) == 0,
                "owned full-byte verification differs");
        const double ms = millis(start,end);
        std::printf("{\"phase\":\"owned\",\"mode\":\"%s\",\"warmup\":%s,\"pair\":%zu,\"order\":%zu,\"iteration\":%zu,\"products\":128,\"output_bytes\":%zu,\"wrapped_calls\":%llu,\"wrapped_bytes\":%llu,\"elapsed_ms\":%.6f,\"gb_s\":%.6f,\"minflt\":%ld,\"majflt\":%ld,\"maxrss_kib\":%ld,\"user_ms\":%.3f,\"system_ms\":%.3f,\"allocation_kind\":\"heap\",\"all_bytes_equal\":true,\"workers\":%d,\"tasks\":12}\n",
            nt ? "nt" : "libc", iteration == 0 ? "true" : "false", iteration ? (iteration - 1) / 2 : 0,
            iteration ? (iteration - 1) % 2 : 0, iteration, total_words * 4,
            (unsigned long long)wrapped_calls,(unsigned long long)wrapped_bytes,ms,double(total_words * 4)/(ms*1e6),
            after.ru_minflt-before.ru_minflt,after.ru_majflt-before.ru_majflt,after.ru_maxrss,
            cpu_ms(before.ru_utime,after.ru_utime),cpu_ms(before.ru_stime,after.ru_stime),
            std::min(12,omp_get_num_procs()));
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    std::fprintf(stderr,"glibc=%s gcc=%s omp_num_procs=%d workers=%d page_size=%ld chunk_bytes=%zu task_bytes_owned=300373/300374 task_bytes_pack=300372/300376\n",
        gnu_get_libc_version(),__VERSION__,omp_get_num_procs(),std::min(12,omp_get_num_procs()),sysconf(_SC_PAGESIZE),kChunkBytes);
    check(argc == 2, "usage: prototype --semantics|--micro|--owned");
    const std::string mode = argv[1];
    if (mode == "--semantics") semantics();
    else if (mode == "--micro") micro_bench();
    else if (mode == "--owned") owned_bench();
    else throw std::runtime_error("unknown mode");
}
