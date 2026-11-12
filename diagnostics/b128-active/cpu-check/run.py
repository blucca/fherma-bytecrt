#!/usr/bin/env python3
"""One synthetic collector/codec assertion. Scope: CPU protocol, not CUDA ABI/performance/exact."""
import importlib.util
import json
import subprocess
import sys
sys.dont_write_bytecode = True
from pathlib import Path

root = Path(__file__).resolve().parents[1]
work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
binary = work / 'synthetic-host-check'
log = work / 'synthetic-stderr.jsonl'
subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-I' + str(root / 'cpu-check'), str(root / 'active-collector.cpp'),
                str(root / 'cpu-check' / 'cupti-stub.cpp'), str(root / 'cpu-check' / 'synthetic.cpp'),
                '-o', str(binary)], check=True)
try:
    with log.open('w') as stderr:
        subprocess.run([str(binary)], stdout=subprocess.DEVNULL, stderr=stderr, check=True)
finally:
    binary.unlink(missing_ok=True)
spec = importlib.util.spec_from_file_location('trace_decoder', root / 'decode-trace.py')
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)
session, calls = decoder.decode_log(log)
assert session['cupti_header_api_version'] == session['cupti_runtime_api_version'] == 999
assert session['external_maps'] == 11264 and session['buffer_requested'] == session['buffer_completed'] == 1
for call in calls:
    assert call['by_runtime_correlation'] == 512 and call['by_correlation'] == 2304
    assert call['stage_active_ns'] == [1024, 1024, 512, 512, 2048]
    assert call['gpu_union_ns'] == 128 * 97
    assert [lane['active_ns'] for lane in call['lanes']] == [5120, 5120, 2560]
    assert [lane['idle_ns'] for lane in call['lanes']] == [121955, 121978, 124475]
    assert len(call['_critical_gaps']) == 1279 and len(call['_h2d_supply_gaps']) == 1024
    assert abs(decoder.host_to_gpu(call, ((call['calibration']['begin'][2] - call['calibration']['begin'][0]) / 2))) < .01
assert log.stat().st_size < 250 * 1024
result = {'status': 'pass', 'scope': 'synthetic CPU CUPTI callback/external mapping/codec/interval arithmetic',
          'native_cuda_abi': 'pending actual CUDA 12.8 image build', 'gpu_exact': 'pending root native',
          'fake_api_version': 999, 'calls': 4, 'kernels_per_call': 1280, 'h2d_per_call': 1024,
          'd2h_per_call': 512, 'dropped': 0, 'stderr_bytes': log.stat().st_size}
(work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, separators=(',', ':')))
