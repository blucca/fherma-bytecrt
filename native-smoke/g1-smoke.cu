// ByteCRT G1: six fixed full-width CRT setup fixtures, Apache-2.0.
// Independent golden answers are signed x % q, serialized by the CPU generator.
#include "g1-crt.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Shape {
    uint32_t n, w, l, p, kpad, m, nc, gap;
};
static_assert(sizeof(Shape) == 8 * sizeof(uint32_t), "fixture shape ABI");

struct Fixture {
    Shape shape{};
    std::vector<uint32_t> q, golden;
    std::vector<double> invp;
    std::vector<uint8_t> b, a;
};

void read_exact(std::ifstream& file, void* dst, size_t bytes, const std::string& path) {
    if (!file.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("ByteCRT G1 truncated fixture " + path);
}

Fixture read_fixture(const char* name, uint32_t n, uint32_t w, uint32_t gap) {
    const std::string path = std::string("fixtures/") + name;
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("ByteCRT G1 open fixture " + path);
    const std::array<char, 8> magic = {'B', 'C', 'R', 'T', '0', '1', '\0', '\0'};
    std::array<char, 8> actual{};
    read_exact(file, actual.data(), actual.size(), path);
    if (actual != magic) throw std::runtime_error("ByteCRT G1 fixture magic " + path);
    Fixture f;
    read_exact(file, &f.shape, sizeof(f.shape), path);
    const Shape& s = f.shape;
    const bool large = w == 1747;
    if (s.n != n || s.w != w || s.l != (large ? 55u : 28u) ||
        s.p != (large ? 114u : 57u) || s.kpad != (large ? 480u : 256u) ||
        s.m != 16 || s.nc != 7 * s.l || s.gap != gap)
        throw std::runtime_error("ByteCRT G1 fixture shape " + path);
    f.q.resize(s.l);
    f.invp.resize(s.p);
    f.b.resize(static_cast<size_t>(s.kpad) * s.nc);
    f.a.resize(static_cast<size_t>(s.m) * s.kpad);
    f.golden.resize(static_cast<size_t>(s.m) * s.l);
    read_exact(file, f.q.data(), f.q.size() * sizeof(uint32_t), path);
    read_exact(file, f.invp.data(), f.invp.size() * sizeof(double), path);
    read_exact(file, f.b.data(), f.b.size(), path);
    read_exact(file, f.a.data(), f.a.size(), path);
    read_exact(file, f.golden.data(), f.golden.size() * sizeof(uint32_t), path);
    if (file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("ByteCRT G1 fixture trailing data " + path);

    const unsigned r = s.w - 32 * (s.l - 1);
    const uint32_t top_mask = (uint32_t{1} << r) - 1;
    for (unsigned limb = 0; limb < s.l; ++limb) {
        const uint32_t want = limb == 0 ? uint32_t{0} - gap :
                              (limb + 1 == s.l ? top_mask : UINT32_MAX);
        if (f.q[limb] != want)
            throw std::runtime_error("ByteCRT G1 fixture modulus " + path);
    }
    for (double inverse : f.invp)
        if (!std::isfinite(inverse) || inverse <= 0x1p-31 || inverse >= 0x1p-30)
            throw std::runtime_error("ByteCRT G1 fixture reciprocal " + path);
    for (unsigned row = 0; row < s.m; ++row) {
        for (unsigned byte = 4 * s.p; byte < s.kpad; ++byte)
            if (f.a[row * s.kpad + byte] != 0)
                throw std::runtime_error("ByteCRT G1 fixture A alpha/padding " + path);
        if (f.golden[row * s.l + s.l - 1] & ~top_mask)
            throw std::runtime_error("ByteCRT G1 fixture golden top limb " + path);
        if (row >= 6) {
            for (unsigned byte = 0; byte < 4 * s.p; ++byte)
                if (f.a[row * s.kpad + byte] != 0)
                    throw std::runtime_error("ByteCRT G1 fixture padded row " + path);
            for (unsigned limb = 0; limb < s.l; ++limb)
                if (f.golden[row * s.l + limb] != 0)
                    throw std::runtime_error("ByteCRT G1 fixture padded golden " + path);
        }
    }
    return f;
}

void require(cudaError_t e, const char* stage) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ByteCRT G1 ") + stage + ": " + cudaGetErrorString(e));
}

struct DeviceSlab {
    uint8_t* data = nullptr;
    ~DeviceSlab() { if (data) cudaFree(data); }
};

void run_fixture(const char* name, uint32_t n, uint32_t w, uint32_t gap) {
    const Fixture f = read_fixture(name, n, w, gap);
    const Shape& s = f.shape;
    // B and A lengths are multiples of 32; invp stays naturally eight-byte aligned.
    const size_t a_offset = f.b.size();
    const size_t invp_offset = a_offset + f.a.size();
    const std::vector<float> invp_float(f.invp.begin(), f.invp.end());
    const size_t q_offset = invp_offset + invp_float.size() * sizeof(float);
    const size_t out_offset = q_offset + f.q.size() * sizeof(uint32_t);
    const size_t out_bytes = f.golden.size() * sizeof(uint32_t);
    DeviceSlab device;
    require(cudaMalloc(&device.data, out_offset + out_bytes), "allocate");
    uint8_t* d_b = device.data;
    uint8_t* d_a = device.data + a_offset;
    float* d_invp = reinterpret_cast<float*>(device.data + invp_offset);
    uint32_t* d_q = reinterpret_cast<uint32_t*>(device.data + q_offset);
    uint32_t* d_out = reinterpret_cast<uint32_t*>(device.data + out_offset);
    require(cudaMemcpy(d_b, f.b.data(), f.b.size(), cudaMemcpyHostToDevice), "copy B");
    require(cudaMemcpy(d_a, f.a.data(), f.a.size(), cudaMemcpyHostToDevice), "copy A");
    require(cudaMemcpy(d_invp, invp_float.data(), invp_float.size() * sizeof(float), cudaMemcpyHostToDevice), "copy invp");
    require(cudaMemcpy(d_q, f.q.data(), f.q.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy q");
    bytecrt::launch_g1_crt(s.w, s.m, d_a, d_b, d_invp, d_q, s.gap, d_out);
    require(cudaGetLastError(), "launch");
    std::vector<uint32_t> got(f.golden.size());
    // The blocking D2H copy completes the fixed kernel before comparison/free.
    require(cudaMemcpy(got.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "copy output");
    require(cudaFree(device.data), "free");
    device.data = nullptr;
    for (size_t word = 0; word < got.size(); ++word)
        if (got[word] != f.golden[word])
            throw std::runtime_error("ByteCRT G1 " + std::string(name) +
                " row=" + std::to_string(word / s.l) + " limb=" + std::to_string(word % s.l) +
                " got=" + std::to_string(got[word]) + " want=" + std::to_string(f.golden[word]));
    std::fprintf(stderr, "ByteCRT G1 %s: %zu words PASS (6 full-width + 10 zero rows; fixed setup)\n",
                 name, got.size());
}
}  // namespace

void bytecrt_g1_smoke() {
    run_fixture("g1-n65536-w1747-g1.bin", 65536, 1747, 1);
    run_fixture("g1-n65536-w1747-g131071.bin", 65536, 1747, 131071);
    run_fixture("g1-n65536-w1747-g4294967295.bin", 65536, 1747, UINT32_MAX);
    run_fixture("g1-n32768-w868-g1.bin", 32768, 868, 1);
    run_fixture("g1-n32768-w868-g65535.bin", 32768, 868, 65535);
    run_fixture("g1-n32768-w868-g4294967295.bin", 32768, 868, UINT32_MAX);
    std::fprintf(stderr, "ByteCRT G1: 36 real full-width coefficients PASS (six fixed setup fixtures)\n");
}
