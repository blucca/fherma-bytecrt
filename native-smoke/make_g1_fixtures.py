#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed full-width CRT fixtures for native G1; independent signed x % q oracle.

ABI: BCRT01\0\0; 8 LE u32 fields; q[u32]; invp[f64]; B[u8,
column-major]; A[u8,row-major,alpha initially zero]; golden[u32,row-major].
"""
import json
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import byte_matrices, plan


def generate():
    directory = Path(__file__).parent / "fixtures"
    directory.mkdir(exist_ok=True)
    rows = []
    for n, bits, expected_p, expected_k in ((65536, 1747, 114, 480), (32768, 868, 57, 256)):
        limbs, m = (bits + 31) // 32, 16
        for gap in (1, 2 * n - 1, (1 << 32) - 1):
            q = (1 << bits) - gap
            product, primes, bases, inverses, weights = plan(n, bits, q)
            invp = [1.0 / p for p in primes]
            _, columns = byte_matrices([0] * len(weights), weights, limbs)
            inner = len(columns[0])
            assert (len(primes), inner) == (expected_p, expected_k)
            bound = n * (q - 1) ** 2
            rng = random.Random(n + bits + gap)
            values = [0, 1, -1, bound, -bound, rng.randrange(-bound, bound + 1)]
            values += [0] * (m - len(values))
            a, golden, largest_high_product = bytearray(), [], 0
            for value in values:
                terms = [value % p * inv % p for p, inv in zip(primes, inverses)]
                alpha = int(sum(t * ip for t, ip in zip(terms, invp)) + 0.5)
                assert sum(t * base for t, base in zip(terms, bases)) - alpha * product == value
                # Native G1 recomputes alpha from these terms in shared memory.
                raw = b"".join(t.to_bytes(4, "little") for t in terms + [0])
                a.extend(raw + bytes(inner - len(raw)))
                expected = value % q
                golden.extend((expected >> (32 * i)) & 0xffffffff for i in range(limbs))
                wide = sum(t * w for t, w in zip(terms + [alpha], weights))
                assert wide % q == expected and wide < (1 << (bits + 38))
                largest_high_product = max(largest_high_product, (wide >> bits) * gap)
            if gap == (1 << 32) - 1:
                assert largest_high_product >= 1 << 64  # Exercises the split-product path.
            fields = (n, bits, limbs, len(primes), inner, m, 7 * limbs, gap)
            q_words = [(q >> (32 * i)) & 0xffffffff for i in range(limbs)]
            b = bytes(v for column in columns for v in column)
            blob = (b"BCRT01\0\0" + struct.pack("<8I", *fields)
                    + struct.pack(f"<{limbs}I", *q_words)
                    + struct.pack(f"<{len(primes)}d", *invp)
                    + b + a + struct.pack(f"<{len(golden)}I", *golden))
            file = directory / f"g1-n{n}-w{bits}-g{gap}.bin"
            file.write_bytes(blob)
            saved = file.read_bytes()
            assert saved == blob and struct.unpack("<8I", saved[8:40]) == fields
            assert len(blob) == 40 + 4 * limbs + 8 * len(primes) + len(b) + len(a) + 4 * len(golden)
            rows.append(dict(file=file.name, n=n, bits=bits, gap=gap, primes=len(primes),
                             inner=inner, coefficients=6, padded_rows=m, bytes=len(blob),
                             high_product_crosses_u64=largest_high_product >= 1 << 64))
    return dict(status="PASS", fixture_files=rows, signed_coefficient_fixtures=36,
                padded_rows_checked=96, GPU_executions=0)


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
