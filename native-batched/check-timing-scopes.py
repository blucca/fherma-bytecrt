#!/usr/bin/env python3
"""Read the published B128 receipts with the Python standard library."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load(name):
    return json.loads((HERE / name).read_text())

nt, bulk = (load(name) for name in ('native-nt-result.json', 'native-bulk-result.json'))
shared = load('native-workspace-result.json')
for result in (nt, bulk, shared):
    assert result['cases'] == result['passed'] == 2  # B1 and B128, one fresh seed each.

rows = [(label, result['unscored_first_warmup'], result['B128_ms'])
        for label, result in (('NT', nt), ('bulk', bulk))]
u = shared['B128_first_warmup_us']
rows.append(('shared', {key: u[source] / 1000 for key, source in (
    ('whole_ms', 'whole'), ('pack_ms', 'pack'), ('owned_ms', 'owned'),
    ('H2D_coverage_ms', 'H2D_span'), ('compute_coverage_ms', 'compute_span'),
    ('D2H_coverage_ms', 'D2H_span'))}, shared['B128_ms']))
print('method | warm whole | warm host copy sum | H2D span | compute span | D2H span | fresh whole (ms)')
for label, w, scored in rows:
    print(f"{label} | {w['whole_ms']:.3f} | {w['pack_ms'] + w['owned_ms']:.3f} | "
          f"{w['H2D_coverage_ms']:.3f} | {w['compute_coverage_ms']:.3f} | "
          f"{w['D2H_coverage_ms']:.3f} | {scored:.6f}")

warm_host_saving = sum(nt['unscored_first_warmup'][k] - bulk['unscored_first_warmup'][k]
                       for k in ('pack_ms', 'owned_ms'))
scored_saving = nt['B128_ms'] - bulk['B128_ms']
assert abs(warm_host_saving - 24.262005) < 1e-6
assert abs(scored_saving - 0.035068) < 1e-6
print(f'NT -> bulk: warm host sum -{warm_host_saving:.6f} ms; fresh whole -{scored_saving:.6f} ms')

output_bytes = 128 * 65536 * 55 * 4
link = nt['target_link']
assert link['speed_GT_s'] == 32 and link['width'] == 16
coding_bytes_per_s = link['speed_GT_s'] * 1e9 * link['width'] * (128 / 130) / 8
input_ms = 2 * output_bytes / coding_bytes_per_s * 1000
assert output_bytes == 1845493760 and abs(input_ms - 58.5728) < 1e-6
print(f'Payload: H2D {2 * output_bytes} B; D2H {output_bytes} B')
print(f'32 GT/s x16 coding-only floor: H2D {input_ms:.4f} ms; D2H {input_ms / 2:.4f} ms')
