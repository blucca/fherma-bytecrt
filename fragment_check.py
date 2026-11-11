#!/usr/bin/env python3
"""CPU contract for unsigned mma.m16n8k32 (CUDA 12.8.1 / PTX ISA 8.7).

NVIDIA PTX §9.7.14.5.10, Matrix Fragments for mma.m16n8k32:
https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832
This checks register layout and integer arithmetic; native instruction execution
belongs to the target-GPU test. Python integers provide the independent oracle.
"""

import json


def a_cell(lane, reg, byte):
    g, t = divmod(lane, 4)
    return g + 8 * (reg % 2), 4 * t + 16 * (reg // 2) + byte


def b_cell(lane, reg, byte):
    g, t = divmod(lane, 4)
    return 4 * t + 16 * reg + byte, g


def c_cell(lane, reg):
    g, t = divmod(lane, 4)
    return g + 8 * (reg // 2), 2 * t + reg % 2


def pack(values):
    assert len(values) == 4 and all(0 <= v < 256 for v in values)
    return int.from_bytes(bytes(values), "little")


def unpack(word, byte):
    return (word >> (8 * byte)) & 255


def check():
    # Direct matrix fixtures include 0, 255 and unsigned values above 127.
    a = [[(r * 71 + k * 29) % 256 for k in range(32)] for r in range(16)]
    b = [[(k * 97 + c * 31 + 255) % 256 for c in range(8)] for k in range(32)]
    c = [[r * 1009 - col * 37 for col in range(8)] for r in range(16)]
    af = [[pack([a[r][k] for r, k in [a_cell(l, q, j) for j in range(4)]])
           for q in range(4)] for l in range(32)]
    bf = [[pack([b[k][col] for k, col in [b_cell(l, q, j) for j in range(4)]])
           for q in range(2)] for l in range(32)]
    cf = [[c[r][col] for r, col in [c_cell(l, q) for q in range(4)]]
          for l in range(32)]
    ac = [a_cell(l, q, j) for l in range(32) for q in range(4) for j in range(4)]
    bc = [b_cell(l, q, j) for l in range(32) for q in range(2) for j in range(4)]
    cc = [c_cell(l, q) for l in range(32) for q in range(4)]
    assert len(ac) == len(set(ac)) == 16 * 32
    assert len(bc) == len(set(bc)) == 32 * 8
    assert len(cc) == len(set(cc)) == 16 * 8
    # Independent inverse coordinate formulas reconstruct matrices from registers.
    aa = [[unpack(af[(r % 8) * 4 + (k % 16) // 4][2 * (k // 16) + r // 8], k % 4)
           for k in range(32)] for r in range(16)]
    bb = [[unpack(bf[col * 4 + (k % 16) // 4][k // 16], k % 4)
           for col in range(8)] for k in range(32)]
    c_back = [[cf[(r % 8) * 4 + col // 2][2 * (r // 8) + col % 2]
               for col in range(8)] for r in range(16)]
    assert aa == a and bb == b and c_back == c
    df = [[0] * 4 for _ in range(32)]
    for lane in range(32):
        for reg in range(4):
            row, col = c_cell(lane, reg)
            value = cf[lane][reg]
            for k in range(32):
                value += aa[row][k] * bb[k][col]
            assert -(1 << 31) <= value < 1 << 31
            df[lane][reg] = value
    d = [[df[(r % 8) * 4 + col // 2][2 * (r // 8) + col % 2]
          for col in range(8)] for r in range(16)]
    oracle = [[sum(x * y for x, y in zip(a[r], (b[k][col] for k in range(32)))) + c[r][col]
               for col in range(8)] for r in range(16)]
    assert d == oracle
    assert pack([0, 127, 128, 255]) == 0xFF807F00
    return {"status": "PASS", "ptx": "8.7", "cuda_docs": "12.8.1",
            "A_cells": len(ac), "B_cells": len(bc), "C_cells": len(cc),
            "inverse_and_byte_reconstruction": True, "integer_mma_cells": 128,
            "native_GPU_executions": 0}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
