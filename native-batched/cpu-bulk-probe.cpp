// Round 543 private CPU experiment. Production files are unchanged.
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


static void print_bulk_row(const char* phase,bool bulk,size_t iteration,size_t bytes,size_t calls,
                           Clock::time_point start,Clock::time_point end,rusage before,rusage after) {
    std::printf("{\"phase\":\"%s\",\"mode\":\"%s\",\"warmup\":%s,\"pair\":%zu,\"order\":%zu,\"bytes\":%zu,\"calls\":%zu,\"elapsed_ms\":%.6f,\"minflt\":%ld,\"majflt\":%ld,\"maxrss_kib\":%ld,\"workers\":%d,\"tasks\":12,\"all_bytes_equal\":true}\n",
        phase,bulk?"bulk":"chunk",iteration==0?"true":"false",iteration?(iteration-1)/2:0,iteration?(iteration-1)%2:0,
        bytes,calls,millis(start,end),after.ru_minflt-before.ru_minflt,after.ru_majflt-before.ru_majflt,after.ru_maxrss,std::min(12,omp_get_num_procs()));
    std::fflush(stdout);
}
static bool bulk_mode(size_t iteration) { return iteration>0&&((iteration-1)/2+(iteration-1)%2)%2; }
static void pack_bulk() {
    constexpr size_t products=128,total_words=products*kProductWords,product_bytes=kProductWords*4;
    std::vector<uint32_t> a(total_words),b(total_words);
    for(size_t i=0;i<total_words;++i){a[i]=uint32_t(i*2654435761u)^0xa544;b[i]=uint32_t(i*2246822519u)^0xb544;}
    std::vector<unsigned char> buffers(8*product_bytes+128);
    auto* dest=reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(buffers.data())+63)&~uintptr_t(63));
    for(size_t iteration=0;iteration<7;++iteration) {
        const bool bulk=bulk_mode(iteration);const auto before=usage();const auto start=Clock::now();
        for(size_t product=0;product<products;++product)for(size_t operand=0;operand<2;++operand){
            const auto* source=reinterpret_cast<const unsigned char*>((operand?b:a).data()+product*kProductWords);
            auto* target=dest+(2*(product%4)+operand)*product_bytes;
            if(bulk)parallel_copy(target,source,product_bytes,true,true);
            else for(size_t chunk=0;chunk<4;++chunk)parallel_copy(target+chunk*kChunkBytes,source+chunk*kChunkBytes,kChunkBytes,true,true);
        }
        const auto end=Clock::now();const auto after=usage();
        for(size_t slot=0;slot<4;++slot)for(size_t operand=0;operand<2;++operand){
            const auto* source=(operand?b:a).data()+(products-4+slot)*kProductWords;
            check(std::memcmp(dest+(2*slot+operand)*product_bytes,source,product_bytes)==0,"full-operand staging differs");
        }
        print_bulk_row("pack",bulk,iteration,2*total_words*4,products*2*(bulk?1:4),start,end,before,after);
    }
}
static void owned_bulk() {
    constexpr size_t products=128,total_words=products*kProductWords;
    std::vector<uint32_t> source(kProductWords);
    for(size_t i=0;i<source.size();++i)source[i]=uint32_t(i*2654435761u)^0x544;
    std::vector<uint32_t> out;out.reserve(total_words);check(heap_mapping(out.data()),"owned allocation outside heap");
    for(size_t iteration=0;iteration<7;++iteration){
        const bool bulk=bulk_mode(iteration);const size_t unit_words=bulk?kProductWords:kChunkWords;
        out.clear();wrapped_calls=wrapped_bytes=0;const auto before=usage();measured=true;const auto start=Clock::now();
        for(size_t product=0;product<products;++product)for(size_t offset=0;offset<kProductWords;offset+=unit_words){
            const auto* begin=source.data()+offset;out.insert(out.end(),begin,begin+unit_words);
        }
        const auto end=Clock::now();measured=false;const auto after=usage();
        check(out.size()==total_words&&out.capacity()==total_words,"owned shape differs");
        check(wrapped_calls==products*(bulk?1:4)&&wrapped_bytes==total_words*4,"owned interception count differs");
        for(size_t product=0;product<products;++product)check(std::memcmp(out.data()+product*kProductWords,source.data(),kProductWords*4)==0,"owned all-product bytes differ");
        print_bulk_row("owned",bulk,iteration,total_words*4,wrapped_calls,start,end,before,after);
    }
}
int main(){
    check(mallopt(M_MMAP_THRESHOLD,INT_MAX)&&mallopt(M_TRIM_THRESHOLD,-1),"heap policy rejected");use_nt=true;
    std::fprintf(stderr,"glibc=%s gcc=%s workers=%d nt_common_backend=true chunks=3604480 product=14417920\n",gnu_get_libc_version(),__VERSION__,std::min(12,omp_get_num_procs()));
    pack_bulk();owned_bulk();
}
