// ByteCRT fixed host RNS/CRT setup, Apache-2.0.
// C++17 standard library only. Build this plan during init, outside call timing.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bytecrt {

struct RnsPlan {
    unsigned n = 0, w = 0, l = 0, pcount = 0, kpad = 0;
    uint32_t gap = 0;
    std::vector<uint32_t> primes, psi_roots, omega_roots, omega_inverse;
    std::vector<uint32_t> n_inverse, crt_inverse, powers_mont;
    // Column-major: 7*l columns, kpad bytes per column (including zero padding).
    std::vector<uint8_t> b_matrix;
    std::vector<double> invp;
};

namespace rns_detail {

// Little-endian base-2^32 words; the empty vector represents zero.
using Words = std::vector<uint32_t>;

inline void trim(Words& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

inline unsigned bit_length(const Words& a) {
    if (a.empty()) return 0;
    unsigned top_bits = 0;
    for (uint32_t top = a.back(); top; top >>= 1) ++top_bits;
    return static_cast<unsigned>(32 * (a.size() - 1)) + top_bits;
}

inline int compare(const Words& a, const Words& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (std::size_t i = a.size(); i != 0; --i) {
        if (a[i - 1] != b[i - 1]) return a[i - 1] < b[i - 1] ? -1 : 1;
    }
    return 0;
}

inline void subtract(Words& a, const Words& b) {
    assert(compare(a, b) >= 0);
    uint64_t borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const uint64_t subtrahend = (i < b.size() ? uint64_t{b[i]} : 0) + borrow;
        const uint64_t word = a[i];
        a[i] = static_cast<uint32_t>(word - subtrahend);
        borrow = word < subtrahend;
    }
    assert(borrow == 0);
    trim(a);
}

inline void mul_small(Words& a, uint32_t b) {
    uint64_t carry = 0;
    for (uint32_t& word : a) {
        const uint64_t total = uint64_t{word} * b + carry;
        word = static_cast<uint32_t>(total);
        carry = total >> 32;
    }
    if (carry) a.push_back(static_cast<uint32_t>(carry));
    trim(a);
}

// Used once for the exact reference guard P > 4*N*(q-1)^2.
inline Words multiply(const Words& a, const Words& b) {
    Words out(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        uint64_t carry = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const uint64_t total = uint64_t{a[i]} * b[j] + out[i + j] + carry;
            out[i + j] = static_cast<uint32_t>(total);
            carry = total >> 32;
        }
        out[i + b.size()] = static_cast<uint32_t>(carry);
    }
    trim(out);
    return out;
}

inline Words div_small(const Words& a, uint32_t divisor, uint32_t& remainder) {
    assert(divisor != 0);
    Words quotient(a.size(), 0);
    uint64_t rem = 0;
    for (std::size_t i = a.size(); i != 0; --i) {
        const uint64_t numerator = (rem << 32) | a[i - 1];
        quotient[i - 1] = static_cast<uint32_t>(numerator / divisor);
        rem = numerator % divisor;
    }
    remainder = static_cast<uint32_t>(rem);
    trim(quotient);
    return quotient;
}

inline uint32_t mod_small(const Words& a, uint32_t divisor) {
    assert(divisor != 0);
    uint64_t remainder = 0;
    for (std::size_t i = a.size(); i != 0; --i)
        remainder = ((remainder << 32) | a[i - 1]) % divisor;
    return static_cast<uint32_t>(remainder);
}

// Straight bitwise division is sufficient for the fixed host-only setup.
inline Words mod_q(const Words& a, const Words& q) {
    assert(!q.empty() && q.back() != 0);
    Words remainder;
    remainder.reserve(q.size() + 1);
    for (unsigned bit = bit_length(a); bit != 0;) {
        --bit;
        uint64_t carry = (a[bit / 32] >> (bit % 32)) & 1u;
        for (uint32_t& word : remainder) {
            const uint64_t total = (uint64_t{word} << 1) | carry;
            word = static_cast<uint32_t>(total);
            carry = total >> 32;
        }
        if (carry) remainder.push_back(static_cast<uint32_t>(carry));
        if (compare(remainder, q) >= 0) subtract(remainder, q);
    }
    assert(compare(remainder, q) < 0);
    return remainder;
}

inline uint32_t pow_mod(uint32_t base, uint32_t exponent, uint32_t modulus) {
    uint64_t result = 1, power = base % modulus;
    for (; exponent; exponent >>= 1) {
        if (exponent & 1) result = result * power % modulus;
        power = power * power % modulus;
    }
    return static_cast<uint32_t>(result);
}

// Same deterministic unsigned-32 Miller-Rabin test as reference.py.
inline bool prime32(uint32_t p) {
    if (p < 2) return false;
    for (uint32_t a : {2u, 3u, 5u, 7u, 11u, 13u, 17u}) {
        if (p == a) return true;
        if (p % a == 0) return false;
    }
    uint32_t d = p - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (uint32_t a : {2u, 7u, 61u}) {
        if (a % p == 0) continue;
        uint32_t x = pow_mod(a, d, p);
        if (x == 1 || x == p - 1) continue;
        bool passed = false;
        for (uint32_t r = 1; r < s; ++r) {
            x = static_cast<uint32_t>(uint64_t{x} * x % p);
            if (x == p - 1) { passed = true; break; }
        }
        if (!passed) return false;
    }
    return true;
}

inline uint32_t primitive_2n_root(uint32_t p, unsigned n) {
    const uint32_t exponent = (p - 1) / (2 * n);
    for (uint32_t a = 2; a < p; ++a) {
        const uint32_t psi = pow_mod(a, exponent, p);
        if (pow_mod(psi, n, p) == p - 1) return psi;
    }
    throw std::runtime_error("ByteCRT RNS primitive root search failed");
}

}  // namespace rns_detail

inline RnsPlan make_rns_plan(unsigned N, unsigned W, const std::vector<uint32_t>& q) {
    using namespace rns_detail;
    RnsPlan plan;
    if (N == 65536 && W == 1747) {
        plan.l = 55; plan.pcount = 114; plan.kpad = 480;
    } else if (N == 32768 && W == 868) {
        plan.l = 28; plan.pcount = 57; plan.kpad = 256;
    } else {
        throw std::invalid_argument("ByteCRT RNS supports (65536,1747) and (32768,868)");
    }
    plan.n = N;
    plan.w = W;
    if (q.size() != plan.l)
        throw std::invalid_argument("ByteCRT RNS modulus limb count");
    plan.gap = uint32_t{0} - q.front();
    if (plan.gap == 0)
        throw std::invalid_argument("ByteCRT RNS gap must be in [1,UINT32_MAX]");
    const uint32_t top_mask = (uint32_t{1} << (W - 32 * (plan.l - 1))) - 1;
    for (unsigned limb = 0; limb < plan.l; ++limb) {
        const uint32_t expected = limb == 0 ? uint32_t{0} - plan.gap :
                                  (limb + 1 == plan.l ? top_mask : UINT32_MAX);
        if (q[limb] != expected)
            throw std::invalid_argument("ByteCRT RNS requires q=2^W-gap, gap<2^32");
    }
    assert(bit_length(q) == W);

    // Preserve the Python bit_length loop, including its final overshoot.
    unsigned n_bits = 0;
    for (unsigned value = N; value; value >>= 1) ++n_bits;
    const unsigned guard_bits = 2 * W + n_bits + 2;
    const uint32_t step = 2 * N;
    uint32_t p = ((uint32_t{1} << 31) - 2) / step * step + 1;
    Words product{1};
    [[maybe_unused]] unsigned previous_product_bits = 0;
    while (bit_length(product) < guard_bits) {
        while (p > (uint32_t{1} << 30) && !prime32(p)) p -= step;
        if (p <= (uint32_t{1} << 30))
            throw std::runtime_error("ByteCRT RNS prime search exhausted");
        assert(p < (uint32_t{1} << 31) && prime32(p));
        assert((p - 1) % step == 0);
        assert(plan.primes.empty() || plan.primes.back() > p);
        plan.primes.push_back(p);
        previous_product_bits = bit_length(product);
        mul_small(product, p);
        p -= step;
    }
    assert(previous_product_bits < guard_bits && bit_length(product) >= guard_bits);
    assert(plan.primes.size() == plan.pcount);
    Words q_minus_one = q;
    subtract(q_minus_one, Words{1});
    Words centered_bound = multiply(q_minus_one, q_minus_one);
    mul_small(centered_bound, 4 * N);
    assert(compare(product, centered_bound) > 0);

    const unsigned terms = plan.pcount + 1;
    assert(plan.kpad == (4 * terms + 31) / 32 * 32);
    assert(uint64_t{4} * terms * 255 * 255 < (uint64_t{1} << 31));
    // t_j < 2^31, alpha <= pcount, weights < q: wide sum < 2^(W+38).
    assert(uint64_t{plan.pcount} * (uint64_t{1} << 31) < (uint64_t{1} << 38));
    plan.psi_roots.resize(plan.pcount);
    plan.omega_roots.resize(plan.pcount);
    plan.omega_inverse.resize(plan.pcount);
    plan.n_inverse.resize(plan.pcount);
    plan.crt_inverse.resize(plan.pcount);
    plan.invp.resize(plan.pcount);
    plan.powers_mont.resize(static_cast<std::size_t>(plan.pcount) * plan.l);
    std::vector<Words> weights;
    weights.reserve(terms);
    for (unsigned j = 0; j < plan.pcount; ++j) {
        p = plan.primes[j];
        const uint32_t psi = primitive_2n_root(p, N);
        const uint32_t omega = static_cast<uint32_t>(uint64_t{psi} * psi % p);
        plan.psi_roots[j] = psi;
        plan.omega_roots[j] = omega;
        plan.omega_inverse[j] = pow_mod(omega, p - 2, p);
        plan.n_inverse[j] = pow_mod(N, p - 2, p);
        assert(pow_mod(psi, N, p) == p - 1 && pow_mod(psi, 2 * N, p) == 1);
        assert(pow_mod(omega, N, p) == 1 && pow_mod(omega, N / 2, p) == p - 1);
        assert(uint64_t{omega} * plan.omega_inverse[j] % p == 1);
        assert(uint64_t{N} * plan.n_inverse[j] % p == 1);

        uint32_t remainder = 0;
        const Words base = div_small(product, p, remainder);
        assert(remainder == 0);
        const uint32_t base_mod_p = mod_small(base, p);
        assert(base_mod_p != 0);
        plan.crt_inverse[j] = pow_mod(base_mod_p, p - 2, p);
        assert(uint64_t{base_mod_p} * plan.crt_inverse[j] % p == 1);
        plan.invp[j] = 1.0 / static_cast<double>(p);
        assert(plan.invp[j] > 0x1p-31 && plan.invp[j] < 0x1p-30);
        weights.push_back(mod_q(base, q));
        weights.back().resize(plan.l, 0);

        const uint64_t radix = (uint64_t{1} << 32) % p;
        uint64_t power = radix;
        for (unsigned limb = 0; limb < plan.l; ++limb) {
            // Montgomery form of 2^(32*limb): one extra radix factor.
            plan.powers_mont[j * plan.l + limb] = static_cast<uint32_t>(power);
            assert(power < p);
            power = power * radix % p;
        }
    }
    Words product_mod_q = mod_q(product, q);
    Words negative_product;
    if (!product_mod_q.empty()) {
        negative_product = q;
        subtract(negative_product, product_mod_q);
    }
    negative_product.resize(plan.l, 0);
    weights.push_back(negative_product);  // Alpha weight: (-P) mod q.

    plan.b_matrix.assign(static_cast<std::size_t>(7 * plan.l) * plan.kpad, 0);
    for (unsigned limb = 0; limb < plan.l; ++limb) {
        for (unsigned out_digit = 0; out_digit < 7; ++out_digit) {
            const std::size_t column = static_cast<std::size_t>(7 * limb + out_digit) * plan.kpad;
            for (unsigned j = 0; j < terms; ++j) {
                const uint32_t word = weights[j][limb];
                for (unsigned in_digit = 0; in_digit < 4; ++in_digit) {
                    const int digit = static_cast<int>(out_digit) - static_cast<int>(in_digit);
                    plan.b_matrix[column + 4 * j + in_digit] =
                        digit >= 0 && digit < 4 ? static_cast<uint8_t>(word >> (8 * digit)) : 0;
                }
            }
            for (unsigned byte = 4 * terms; byte < plan.kpad; ++byte)
                assert(plan.b_matrix[column + byte] == 0);
        }
    }
    return plan;
}

}  // namespace bytecrt
