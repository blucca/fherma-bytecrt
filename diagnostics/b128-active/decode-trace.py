#!/usr/bin/env python3
"""Decode the fixed B128 stderr contract. All coordinates returned are relative ns."""
import base64
import json
import statistics
import sys
from pathlib import Path


def intervals(payload, fields):
    raw = base64.b64decode(payload, validate=True)
    values, value, shift = [], 0, 0
    for byte in raw:
        assert shift < 64, 'uvarint overflow'
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
        else:
            assert value < 2**64
            values.append(value)
            value = shift = 0
    assert shift == 0 and len(values) % fields == 0, 'truncated trace'
    rows, previous = [], 0
    for i in range(0, len(values), fields):
        start = previous + values[i]
        rows.append((start, start + values[i + 1], *values[i + 2:i + fields]))
        previous = start
    return rows


def union_ns(rows):
    total = 0
    begin = end = None
    for start, stop, *_ in sorted(rows):
        if begin is None:
            begin, end = start, stop
        elif start <= end:
            end = max(end, stop)
        else:
            total += end - begin
            begin, end = start, stop
    return total + (end - begin if begin is not None else 0)


def host_to_gpu(call, relative):
    """Affine fit through midpoints; brackets expose the host/CUPTI sampling uncertainty."""
    first, last = call['calibration']['begin'], call['calibration']['end']
    h0, h1 = (first[0] + first[2]) / 2, (last[0] + last[2]) / 2
    host = call['host']['origin_ns'] + relative
    return (host - h0) * (last[1] - first[1]) / (h1 - h0)


def decode_log(path):
    objects = []
    for line in Path(path).read_text().splitlines():
        marker = next((line.find(token) for token in ('{"active_session":', '{"active_trace":') if token in line), -1)
        if marker >= 0:
            objects.append(json.loads(line[marker:]))
    sessions = [row for row in objects if row.get('active_session') == 1]
    calls = [row for row in objects if row.get('active_trace') == 1]
    assert len(sessions) == 1 and len(calls) == 4, 'one session and four calls required'
    session = sessions[0]
    assert session['valid'] and session['batch'] == 128 and session['calls'] == 4
    assert session['kernel_records'] == 5120 and session['memcpy_records'] == 6144
    assert session['runtime_tracing'] is False and session['new_cuda_timing_events'] == 0
    for key in ('dropped', 'unmapped_gpu', 'conflicting_maps', 'unexpected_copy',
                'other_records', 'callback_errors', 'external_overflow', 'buffer_exhausted', 'protocol_errors'):
        assert session[key] == 0, key
    for ordinal, call in enumerate(calls, 1):
        assert call['call'] == ordinal and call['external_id'] == (128 << 32) | ordinal
        assert call['scored'] is (ordinal != 1) and call['valid']
        begin, end = call['cup_begin_ns'], call['cup_end_ns']
        window = end - begin
        assert window > 0 and call['gpu_overflow'] == 0
        assert call['by_correlation'] + call['by_runtime_correlation'] == 2816
        for label, stamp in (('begin', begin), ('end', end)):
            pair = call['calibration'][label]
            assert len(pair) == 3 and pair[0] <= pair[2] and pair[1] == stamp
        assert call['calibration']['end'][0] > call['calibration']['begin'][2]
        lanes = {lane['name']: lane for lane in call['lanes']}
        assert set(lanes) == {'h2d', 'kernel', 'd2h'}
        assert len({lane['stream'] for lane in lanes.values()}) == 3
        assert len({lane['context'] for lane in lanes.values()}) == 1
        assert len({lane['device'] for lane in lanes.values()}) == 1
        decoded = {}
        for name, expected, payload in (('h2d', 1024, 3690987520), ('kernel', 1280, 0), ('d2h', 512, 1845493760)):
            lane = lanes[name]
            rows = intervals(lane['intervals_b64'], 3)
            assert lane['count'] == len(rows) == expected and lane['bytes'] == payload
            assert all(0 <= start < stop <= window and correlation > 0 for start, stop, correlation in rows)
            gaps = [rows[i][0] - rows[i - 1][1] for i in range(1, len(rows))]
            assert all(gap >= 0 for gap in gaps), 'resource FIFO overlap'
            assert lane['active_ns'] == sum(stop - start for start, stop, _ in rows)
            assert lane['coverage_ns'] == rows[-1][1] - rows[0][0]
            assert lane['idle_ns'] == sum(gaps) and lane['max_gap_ns'] == max(gaps, default=0)
            assert lane['head_ns'] == rows[0][0] and lane['tail_ns'] == window - rows[-1][1]
            decoded[name] = rows
        assert call['gpu_union_ns'] == union_ns(sum(decoded.values(), []))
        stages = [0] * 5
        stage_fifo = [0, 1, 0, 1, 2, 3, 4, 4, 4, 4]
        for i, (start, stop, _) in enumerate(decoded['kernel']):
            stages[stage_fifo[i % 10]] += stop - start
        assert stages == call['stage_active_ns']
        host = call['host']
        origin = host['origin_ns']
        assert origin == call['calibration']['begin'][0]
        reserve = host['reserve']
        assert 0 <= reserve[0] <= reserve[1] <= call['calibration']['end'][2] - origin
        for name, count in (('pack', 1024), ('wait', 512), ('owned', 512)):
            rows = intervals(host[name]['intervals_b64'], 2)
            assert len(rows) == host[name]['count'] == count
            assert all(0 <= start <= stop <= call['calibration']['end'][2] - origin for start, stop in rows)
            assert sum(stop - start for start, stop in rows) == host[name]['sum_ns']
            decoded[name] = rows
        assert all(wait[1] == owned[0] for wait, owned in zip(decoded['wait'], decoded['owned']))
        critical = []
        readiness = post = non_encode = 0
        for i, (start, stop, _) in enumerate(decoded['kernel']):
            if i == 0:
                assert decoded['h2d'][3][1] <= start
                continue
            previous = decoded['kernel'][i - 1][1]
            gap = start - previous
            dependency = None
            if i % 10 in (0, 2):
                dependency = decoded['h2d'][(i // 10) * 8 + (7 if i % 10 == 2 else 3)][1]
                assert dependency <= start
                part = max(0, min(start, dependency) - previous)
                readiness += part
                post += gap - part
            else:
                part = 0
                non_encode += gap
            critical.append({'product': i // 10, 'stage': i % 10, 'previous_end_ns': previous,
                             'start_ns': start, 'gap_ns': gap, 'h2d_end_ns': dependency,
                             'readiness_overlap_ns': part, 'post_readiness_ns': gap - part})
        assert call['compute_gaps'] == {'h2d_readiness_overlap_ns': readiness, 'post_readiness_ns': post,
                                        'non_encode_idle_ns': non_encode}
        for i, (start, _, _) in enumerate(decoded['d2h']):
            assert decoded['kernel'][(i // 4) * 10 + 6 + i % 4][1] <= start
        supply = []
        for i, (start, _, _) in enumerate(decoded['h2d']):
            previous = decoded['h2d'][i - 1][1] if i else 0
            packed = host_to_gpu(call, decoded['pack'][i][1])
            supply.append({'product': i // 8, 'operand': i // 4 % 2, 'chunk': i % 4,
                           'previous_end_ns': previous, 'start_ns': start, 'gap_ns': start - previous,
                           'pack_end_gpu_fit_ns': round(packed),
                           'pack_pending_overlap_ns': round(max(0, min(start, packed) - previous))})
        call['_decoded'] = decoded
        call['_critical_gaps'] = critical
        call['_h2d_supply_gaps'] = supply
    return session, calls


def summary(session, calls):
    out = {'valid': True, 'cupti_header_api_version': session['cupti_header_api_version'], 'calls': []}
    for call in calls:
        lanes = {lane['name']: lane for lane in call['lanes']}
        first, last = call['calibration']['begin'], call['calibration']['end']
        host_window = (last[0] + last[2] - first[0] - first[2]) / 2
        out['calls'].append({'call': call['call'], 'scored': call['scored'],
            'cupti_window_ms': (call['cup_end_ns'] - call['cup_begin_ns']) / 1e6,
            'host_cupti_bracket_ns': [first[2] - first[0], last[2] - last[0]],
            'host_cupti_fit_scale': (last[1] - first[1]) / host_window,
            'active_ms': {name: lane['active_ns'] / 1e6 for name, lane in lanes.items()},
            'coverage_ms': {name: lane['coverage_ns'] / 1e6 for name, lane in lanes.items()},
            'compute_idle_ms': lanes['kernel']['idle_ns'] / 1e6,
            'gpu_union_ms': call['gpu_union_ns'] / 1e6,
            'host_sum_ms': {name: call['host'][name]['sum_ns'] / 1e6 for name in ('pack', 'wait', 'owned')},
            'compute_gaps': call['compute_gaps'],
            'largest_compute_gaps': sorted(call['_critical_gaps'], key=lambda row: row['gap_ns'], reverse=True)[:10],
            'largest_h2d_supply_gaps': sorted(call['_h2d_supply_gaps'], key=lambda row: row['gap_ns'], reverse=True)[:10]})
    out['median_scored_cupti_window_ms'] = statistics.median(row['cupti_window_ms'] for row in out['calls'] if row['scored'])
    return out

if __name__ == '__main__':
    session, calls = decode_log(sys.argv[1])
    print(json.dumps(summary(session, calls), separators=(',', ':')))
