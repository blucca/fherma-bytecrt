# SPDX-License-Identifier: Apache-2.0
"""An exact integer width example for a 31-bit radix-16 NTT port."""
import json

p = 2013265921
root = pow(31, (p - 1) // 16, p)
assert 2**30 < p < 2**31 and p % (2 * 65536) == 1
assert pow(root, 8, p) == p - 1 and pow(root, 16, p) == 1

# NTT row 8 has eight weights 1 and eight weights p-1; every input is p-1.
exact = 8 * p * (p - 1)
wrapped = exact & ((1 << 64) - 1)
assert exact.bit_length() == 65 and exact % p == 0
assert wrapped % p == 841097758
print(json.dumps({"p": p, "dot_bits": exact.bit_length(),
                  "exact_residue": exact % p, "u64_wrapped_residue": wrapped % p}))
