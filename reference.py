#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 blucca
"""Independent CPU model of unsigned-byte MMA + a multi-limb CRT epilogue.
Each nonnegative dot product fits s32. No GPU, dependencies or official run.
"""
import json
import random


def prime32(p):
    if p < 2:
        return False
    for a in (2, 3, 5, 7, 11, 13, 17):
        if p == a:
            return True
        if p % a == 0:
            return False
    d, s = p - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic Miller-Rabin bases for unsigned 32-bit integers.
    for a in (2, 7, 61):
        if a % p == 0:
            continue
        x = pow(a, d, p)
        if x in (1, p - 1):
            continue
        for _ in range(s - 1):
            x = x * x % p
            if x == p - 1:
                break
        else:
            return False
    return True


def plan(n, bits, q):
    assert n >= 2 and n & (n - 1) == 0 and q.bit_length() == bits
    p, product, primes = ((1 << 31) - 2) // (2 * n) * (2 * n) + 1, 1, []
    while product.bit_length() < 2 * bits + n.bit_length() + 2:
        while p > 1 << 30 and not prime32(p):
            p -= 2 * n
        assert 1 << 30 < p < 1 << 31
        primes.append(p)
        product *= p
        p -= 2 * n
    assert product > 4 * n * (q - 1) ** 2
    bases = [product // p for p in primes]
    inverses = [pow(base % p, -1, p) for base, p in zip(bases, primes)]
    # Last row implements alpha * (-P mod q), in nonnegative representation.
    weights = [base % q for base in bases] + [q - product % q]
    return product, primes, bases, inverses, weights


def byte_matrices(terms, weights, limbs):
    inner = (4 * len(terms) + 31) // 32 * 32
    a = [v >> (8 * digit) & 255 for v in terms for digit in range(4)]
    a += [0] * (inner - len(a))
    columns = []
    for limb in range(limbs):
        for out_digit in range(7):
            col = []
            for weight in weights:
                word = weight >> (32 * limb) & 0xffffffff
                for in_digit in range(4):
                    digit = out_digit - in_digit
                    col.append(word >> (8 * digit) & 255 if 0 <= digit < 4 else 0)
            columns.append(col + [0] * (inner - len(col)))
    return a, columns


def epilogue(c, limbs):
    value, carry = 0, 0
    for limb in range(limbs):
        row = c[7 * limb:7 * limb + 7]
        lo = sum(row[d] << (8 * d) for d in range(4))
        hi = sum(row[d + 4] << (8 * d) for d in range(3))
        total = carry + lo
        assert total < 1 << 64 and hi < 1 << 64
        value |= (total & 0xffffffff) << (32 * limb)
        carry = (total >> 32) + hi
    assert carry < 1 << 64  # Two extra limbs cover this bound for both points.
    return value | carry << (32 * limbs)


def near_power_reduce(value, bits, gap):
    assert bits >= 64 and 0 < gap < 1 << 32 and value >= 0
    q, mask, folds = (1 << bits) - gap, (1 << bits) - 1, 0
    while value >> bits:
        value = (value & mask) + (value >> bits) * gap
        folds += 1
        assert folds <= 5
    return value - q if value >= q else value, folds


def check_point(n, bits):
    limbs, checked, largest_dot, largest_folds = (bits + 31) // 32, 0, 0, 0
    prime_count, inner = 0, 0
    for gap in (1, 2 * n - 1, 2 * n * 124 - 1):
        q = (1 << bits) - gap  # Synthetic full-width moduli, primality unnecessary here.
        product, primes, bases, inverses, weights = plan(n, bits, q)
        bound = n * (q - 1) ** 2
        rng = random.Random(n + bits + gap)
        fixtures = (0, 1, -1, bound, -bound, rng.randrange(-bound, bound + 1))
        for x in fixtures:
            terms = [x % p * inv % p for p, inv in zip(primes, inverses)]
            approximate = 0.0
            for t, p in zip(terms, primes):
                approximate += float(t) * (1.0 / float(p))
            alpha = int(approximate + 0.5)
            assert sum(t * b for t, b in zip(terms, bases)) - alpha * product == x
            a, columns = byte_matrices(terms + [alpha], weights, limbs)
            c = [sum(u * v for u, v in zip(a, col)) for col in columns]
            dot_bound = 4 * len(weights) * 255 * 255
            assert dot_bound < 1 << 31 and all(0 <= v <= dot_bound for v in c)
            wide = epilogue(c, limbs)
            expected = sum(t * w for t, w in zip(terms + [alpha], weights))
            assert wide == expected
            reduced, folds = near_power_reduce(wide, bits, gap)
            assert reduced == expected % q == x % q
            checked += 1
            largest_dot = max(largest_dot, max(c))
            largest_folds = max(largest_folds, folds)
            prime_count, inner = len(primes), len(a)
    return dict(n=n, bits=bits, limbs=limbs, prime_count=prime_count,
                padded_inner=inner, byte_output_columns=7 * limbs,
                coefficient_fixtures_checked=checked, largest_dot=largest_dot,
                largest_folds=largest_folds)


if __name__ == '__main__':
    assert prime32(2147483647) and not prime32(3215031751)
    # A signed-byte instruction changes 255*255 into (-1)*(-1); require u8.u8.
    assert 255 * 255 == 65025 and (-1) * (-1) == 1
    assert epilogue([1, 2, 3, 4, 5, 6, 7], 1) == sum(v << (8 * i) for i, v in enumerate([1, 2, 3, 4, 5, 6, 7]))
    rows = [check_point(65536, 1747), check_point(32768, 868)]
    assert [(r['prime_count'],r['padded_inner']) for r in rows] == [(114,480),(57,256)]
    print(json.dumps(dict(status='PASS', points=rows,
                         evidence='CPU byte-matrix, centered-CRT and carry/reduction model',
                         cuda_compiled=False, target_gpu_runs=0,
                         full_official_convolutions=0), indent=2))
