#!/usr/bin/env python3
"""One source-closure check; direct bytes against the neighboring ACA baseline."""
import json
import re
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
baseline = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent.parent / 'native-batched'
assert baseline.is_dir(), 'Use a repository checkout or pass the ACA baseline directory'
core = sorted('core/' + p.name for p in (root/'core').iterdir() if p.is_file())
assert len(core) == 9
preserved = ['main.cpp', 'fherma.h'] + core
for name in preserved:
    assert (root/name).read_bytes() == (baseline/name).read_bytes(), name
files = preserved + ['solve.cu', 'CMakeLists.txt', 'active-collector.hpp', 'active-collector.cpp']
for name in files:
    for include in re.findall(r'^\s*#include\s*"([^\"]+)"', (root/name).read_text(), re.M):
        assert any(p.is_file() for p in ((root/name).parent/include, root/include, root/'core'/include)), (name, include)
solve = (root/'solve.cu').read_text()
cuda_lines = lambda text: [line for line in text.splitlines() if 'cuda' in line or '<<<' in line]
assert cuda_lines(solve) == cuda_lines((baseline/'solve.cu').read_text())
assert 'if (batch.batch == 1)\n        return bytecrt_serial_run(batch.slots[0].get(), in);' in solve
assert solve.count('bytecrt_pack_operand(') == solve.count('output.insert(output.end(), begin, begin + words);') == 1
assert solve.count('out.c.data.reserve(total);') == 1
collector = (root/'active-collector.cpp').read_text()
assert 'CUPTI_ACTIVITY_KIND_RUNTIME' not in collector and 'cudaEvent' not in collector
assert 'p.parse(); p.print();' in collector
print(json.dumps({'status':'pass','preserved_files':preserved,'cuda_api_kernel_line_order':'same bytes','local_include_closure':'complete','baseline':str(baseline)}))
