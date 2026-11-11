// SPDX-License-Identifier: Apache-2.0
// Fixed setup exact gate: both staged geometries, two real ByteCRT RNS primes.
#include "ntt-pipeline.cuh"

#include <array>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// These are the first two primes selected by the full-width G1 prime plan.
constexpr std::array<uint32_t, 2> kPrimes = {2147352577u, 2146959361u};
constexpr uint32_t kPrimeCount = uint32_t(kPrimes.size());

uint32_t modmul(uint32_t a, uint32_t b, uint32_t p) {
    return uint32_t(uint64_t(a) * b % p);
}

uint32_t modpow(uint32_t a, uint32_t exponent, uint32_t p) {
    uint32_t r = 1;
    while (exponent) {
        if (exponent & 1) r = modmul(r, a, p);
        a = modmul(a, a, p);
        exponent >>= 1;
    }
    return r;
}

bool is_prime(uint32_t p) {
    if (p < 2 || !(p & 1)) return p == 2;
    for (uint32_t d = 3; uint64_t(d) * d <= p; d += 2)
        if (p % d == 0) return false;
    return true;
}

struct Parameters {
    uint32_t p, psi, psi_inv, omega, omega_inv, n_inv;
};

Parameters checked_parameters(uint32_t n, uint32_t p) {
    const std::string label = "ByteCRT NTT N=" + std::to_string(n) + " p=" + std::to_string(p);
    if (n < 2 || (n & (n - 1)) || p <= (uint32_t{1} << 30) ||
        p >= (uint32_t{1} << 31) || (p - 1) % (2 * n) || !is_prime(p))
        throw std::runtime_error(label + " prime/size contract failed");
    uint32_t psi = 0;
    for (uint32_t seed = 2; seed < p; ++seed) {
        const uint32_t candidate = modpow(seed, (p - 1) / (2 * n), p);
        if (modpow(candidate, n, p) == p - 1) { psi = candidate; break; }
    }
    if (!psi) throw std::runtime_error(label + " primitive 2N-th root search failed");
    const Parameters r{p, psi, modpow(psi, p - 2, p), modmul(psi, psi, p),
                       modpow(modmul(psi, psi, p), p - 2, p), modpow(n, p - 2, p)};
    if (modpow(r.psi, n, p) != p - 1 || modpow(r.psi, 2 * n, p) != 1 ||
        modpow(r.omega, n, p) != 1 || modpow(r.omega, n / 2, p) != p - 1 ||
        modmul(r.psi, r.psi_inv, p) != 1 || modmul(r.omega, r.omega_inv, p) != 1 ||
        modmul(n, r.n_inv, p) != 1)
        throw std::runtime_error(label + " root/inverse/N_inv contract failed");
    return r;
}

uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

std::vector<uint32_t> dense_input(uint32_t n, uint32_t p, uint32_t operand) {
    std::vector<uint32_t> a(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t seed = (uint64_t(p) << 32) ^ uint64_t(n) ^
                             (UINT64_C(0x9e3779b97f4a7c15) * (uint64_t(i) + 1 + uint64_t(operand) * n));
        a[i] = uint32_t(mix64(seed) % p);
    }
    // Include canonical boundaries among otherwise dense deterministic data.
    a[0] = operand ? p - 1 : 0;
    a[1] = operand ? 1 : p - 1;
    a[n / 2] = p / 2;
    a[n - 1] = operand ? p - 1 : 1;
    return a;
}

// Independent conventional radix-2 DIT cyclic NTT, with explicit bit reversal.
// Roots, powers, butterflies and normalization use ordinary uint64 arithmetic.
void cyclic_ntt(std::vector<uint32_t>& a, uint32_t p, uint32_t root) {
    const uint32_t n = uint32_t(a.size());
    for (uint32_t i = 1, j = 0; i < n; ++i) {
        uint32_t bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (uint32_t length = 2; length <= n; length <<= 1) {
        const uint32_t step = modpow(root, n / length, p);
        for (uint32_t start = 0; start < n; start += length) {
            uint32_t w = 1;
            for (uint32_t j = 0; j < length / 2; ++j) {
                const uint32_t u = a[start + j], v = modmul(a[start + j + length / 2], w, p);
                const uint64_t sum = uint64_t(u) + v;
                a[start + j] = uint32_t(sum >= p ? sum - p : sum);
                a[start + j + length / 2] = u >= v ? u - v : uint32_t(uint64_t(u) + p - v);
                w = modmul(w, step, p);
            }
        }
    }
}

std::vector<uint32_t> cpu_product(std::vector<uint32_t> a, std::vector<uint32_t> b,
                                  const Parameters& r) {
    uint32_t power = 1;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = modmul(a[i], power, r.p);
        b[i] = modmul(b[i], power, r.p);
        power = modmul(power, r.psi, r.p);
    }
    cyclic_ntt(a, r.p, r.omega);
    cyclic_ntt(b, r.p, r.omega);
    for (size_t i = 0; i < a.size(); ++i) a[i] = modmul(a[i], b[i], r.p);
    cyclic_ntt(a, r.p, r.omega_inv);
    power = 1;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = modmul(modmul(a[i], r.n_inv, r.p), power, r.p);
        power = modmul(power, r.psi_inv, r.p);
    }
    return a;
}

void check_cpu_reference() {
    constexpr uint32_t n = 16;
    for (uint32_t p : kPrimes) {
        const Parameters r = checked_parameters(n, p);
        const auto a = dense_input(n, p, 0), b = dense_input(n, p, 1);
        std::vector<uint32_t> schoolbook(n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t k = (i + j) % n, v = modmul(a[i], b[j], p);
                schoolbook[k] = uint32_t((uint64_t(schoolbook[k]) +
                    (i + j < n ? uint64_t(v) : uint64_t(p) - v)) % p);
            }
        }
        if (cpu_product(a, b, r) != schoolbook)
            throw std::runtime_error("ByteCRT NTT CPU N=16 schoolbook self-check failed p=" + std::to_string(p));
    }
}

// CUDA setup and exact comparison.
void require(cudaError_t e, const std::string& stage) {
    if (e != cudaSuccess)
        throw std::runtime_error("ByteCRT NTT " + stage + ": " + cudaGetErrorString(e));
}

struct DeviceSlab {
    uint32_t* data = nullptr;
    ~DeviceSlab() { if (data) cudaFree(data); }
};

template<uint32_t N, cupqc::nttDirection D>
void initialize_table(uint32_t* table, const Parameters& r) {
    const bool forward = D == cupqc::nttDirection::FORWARD;
    const std::string label = "N=" + std::to_string(N) + " p=" + std::to_string(r.p) +
                              (forward ? " forward twiddles" : " inverse twiddles");
    bytecrt::ntt::make_table<N, D><<<1, 1>>>(table, r.p, forward ? r.omega : r.omega_inv);
    require(cudaGetLastError(), label + " generation launch");
    require(cudaDeviceSynchronize(), label + " generation execute");
    bytecrt::ntt::mont_table<N, D><<<1, bytecrt::ntt::kThreads>>>(table, r.p);
    require(cudaGetLastError(), label + " Montgomery launch");
    require(cudaDeviceSynchronize(), label + " Montgomery execute");
}

template<uint32_t N>
void run_geometry() {
    const std::string label = "N=" + std::to_string(N);
    constexpr size_t planes = size_t(kPrimeCount) * N;
    std::vector<uint32_t> ab(2 * planes), golden(planes), psi(planes), psi_inv(planes);
    std::array<Parameters, kPrimeCount> params{};
    std::array<cupqc::nttConst<uint32_t>, kPrimeCount> constants{};
    std::array<uint32_t, kPrimeCount> n_inverse{};
    for (uint32_t prime = 0; prime < kPrimeCount; ++prime) {
        const Parameters r = checked_parameters(N, kPrimes[prime]);
        params[prime] = r;
        constants[prime] = cupqc::nttConst<uint32_t>(r.p);
        const auto& mc = constants[prime];
        const uint32_t r_mod_p = uint32_t((uint64_t{1} << 32) % r.p);
        if (mc.p != r.p || uint32_t(uint64_t(r.p) * mc.n_dash) != UINT32_MAX ||
            mc.r2 != modmul(r_mod_p, r_mod_p, r.p))
            throw std::runtime_error("ByteCRT NTT " + label + " Montgomery constants failed p=" + std::to_string(r.p));
        n_inverse[prime] = r.n_inv;
        const auto a = dense_input(N, r.p, 0), b = dense_input(N, r.p, 1);
        const auto expected = cpu_product(a, b, r);
        uint32_t forward_power = 1, inverse_power = 1;
        for (uint32_t i = 0; i < N; ++i) {
            const size_t index = size_t(prime) * N + i;
            ab[index] = a[i];
            ab[planes + index] = b[i];
            golden[index] = expected[i];
            psi[index] = modmul(forward_power, r_mod_p, r.p);  // psi^i in Montgomery form
            psi_inv[index] = inverse_power;
            forward_power = modmul(forward_power, r.psi, r.p);
            inverse_power = modmul(inverse_power, r.psi_inv, r.p);
        }
    }

    DeviceSlab device;
    constexpr size_t plane_bytes = planes * sizeof(uint32_t);
    constexpr size_t bytes = 7 * plane_bytes + sizeof(n_inverse) + sizeof(constants);
    require(cudaMalloc(&device.data, bytes), label + " allocate");
    uint32_t* d_ab = device.data;
    uint32_t* d_c = d_ab + 2 * planes;
    uint32_t* d_tw = d_c + planes;
    uint32_t* d_inv_tw = d_tw + planes;
    uint32_t* d_psi = d_inv_tw + planes;
    uint32_t* d_psi_inv = d_psi + planes;
    uint32_t* d_n_inverse = d_psi_inv + planes;
    auto* d_constants = reinterpret_cast<cupqc::nttConst<uint32_t>*>(d_n_inverse + kPrimeCount);
    require(cudaMemcpy(d_ab, ab.data(), 2 * plane_bytes, cudaMemcpyHostToDevice), label + " copy inputs");
    require(cudaMemcpy(d_psi, psi.data(), plane_bytes, cudaMemcpyHostToDevice), label + " copy psi");
    require(cudaMemcpy(d_psi_inv, psi_inv.data(), plane_bytes, cudaMemcpyHostToDevice), label + " copy inverse psi");
    require(cudaMemcpy(d_n_inverse, n_inverse.data(), sizeof(n_inverse), cudaMemcpyHostToDevice), label + " copy N_inv");
    require(cudaMemcpy(d_constants, constants.data(), sizeof(constants), cudaMemcpyHostToDevice), label + " copy constants");
    for (uint32_t prime = 0; prime < kPrimeCount; ++prime) {
        initialize_table<N, cupqc::nttDirection::FORWARD>(d_tw + size_t(prime) * N, params[prime]);
        initialize_table<N, cupqc::nttDirection::INVERSE>(d_inv_tw + size_t(prime) * N, params[prime]);
    }
    require(bytecrt::ntt::launch_product<N>(d_ab, d_c, d_tw, d_inv_tw, d_psi,
        d_psi_inv, d_constants, d_n_inverse, kPrimeCount), label + " product launch");
    require(cudaDeviceSynchronize(), label + " product execute");
    std::vector<uint32_t> got(planes);
    require(cudaMemcpy(got.data(), d_c, plane_bytes, cudaMemcpyDeviceToHost), label + " copy output");
    require(cudaFree(device.data), label + " free");
    device.data = nullptr;

    size_t mismatches = 0, first = planes;
    for (size_t i = 0; i < planes; ++i) {
        if (got[i] != golden[i]) {
            if (!mismatches) first = i;
            ++mismatches;
        }
    }
    if (mismatches)
        throw std::runtime_error("ByteCRT NTT " + label + " p=" + std::to_string(kPrimes[first / N]) +
            " coefficient=" + std::to_string(first % N) + " got=" + std::to_string(got[first]) +
            " want=" + std::to_string(golden[first]) + " mismatches=" + std::to_string(mismatches) +
            "/" + std::to_string(planes));
    for (uint32_t prime = 0; prime < kPrimeCount; ++prime)
        std::fprintf(stderr, "ByteCRT NTT N=%u p=%u: %u coefficients PASS (fixed setup)\n", N, kPrimes[prime], N);
}

}  // namespace

// Invoke from fherma_init: fixtures, CPU golden and device execution stay in setup.
void bytecrt_ntt_smoke() {
    check_cpu_reference();
    run_geometry<65536>();
    run_geometry<32768>();
    std::fprintf(stderr, "ByteCRT NTT: 196608 canonical residues PASS (two sizes, two runtime primes; fixed setup)\n");
}

#ifdef BYTECRT_NTT_SMOKE_STANDALONE
int main() {
    try { bytecrt_ntt_smoke(); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    return 0;
}
#endif
