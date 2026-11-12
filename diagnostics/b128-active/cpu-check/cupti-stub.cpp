#include "cupti.h"
#include "../active-collector.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
namespace {
StubRequest request = nullptr;
StubComplete complete = nullptr;
uint8_t* buffer = nullptr;
size_t capacity = 0, valid = 0;
uint64_t external_id = 0, gpu_end = 0;
unsigned enabled = 0;
void emit_bytes(const void* record, size_t size) {
    if (!buffer || valid + size > capacity) {
        if (buffer) complete(nullptr, 0, buffer, capacity, valid);
        size_t max_records = 0;
        request(&buffer, &capacity, &max_records);
        assert(buffer && max_records == 0 && size <= capacity);
        valid = 0;
    }
    std::memcpy(buffer + valid, record, size); valid += size;
}
void mapping(uint32_t correlation) {
    CUpti_ActivityExternalCorrelation row;
    row.correlationId = correlation; row.externalId = external_id;
    emit_bytes(&row, sizeof(row));
}
}
CUptiResult cuptiGetResultString(CUptiResult, const char** text) { *text = "synthetic stub"; return CUPTI_SUCCESS; }
CUptiResult cuptiGetTimestamp(uint64_t* timestamp) {
    *timestamp = std::max(bytecrt_active::host_ns(), gpu_end); return CUPTI_SUCCESS;
}
CUptiResult cuptiGetVersion(uint32_t* version) { *version = CUPTI_API_VERSION; return CUPTI_SUCCESS; }
CUptiResult cuptiActivityRegisterCallbacks(StubRequest r, StubComplete c) { request = r; complete = c; return CUPTI_SUCCESS; }
CUptiResult cuptiActivityEnable(CUpti_ActivityKind) { ++enabled; return CUPTI_SUCCESS; }
CUptiResult cuptiActivityDisable(CUpti_ActivityKind) { assert(enabled); --enabled; return CUPTI_SUCCESS; }
CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t, size_t* count) { *count = 0; return CUPTI_SUCCESS; }
CUptiResult cuptiActivityGetNextRecord(uint8_t* data, size_t size, CUpti_Activity** row) {
    uint8_t* next = *row ? reinterpret_cast<uint8_t*>(*row) + (*row)->stub_size : data;
    if (next >= data + size) return CUPTI_ERROR_MAX_LIMIT_REACHED;
    *row = reinterpret_cast<CUpti_Activity*>(next);
    assert(next + (*row)->stub_size <= data + size); return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind, uint64_t id) {
    assert(!external_id); external_id = id; return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind, uint64_t* id) {
    assert(external_id); *id = external_id; external_id = 0; return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityFlushAll(uint32_t flags) {
    assert(flags == 0);
    if (buffer) complete(nullptr, 0, buffer, capacity, valid);
    buffer = nullptr; capacity = valid = 0; return CUPTI_SUCCESS;
}
void stub_emit(const CUpti_ActivityKernel9& row) {
    assert(external_id && enabled == 3);
    gpu_end = std::max(gpu_end, row.end);
    emit_bytes(&row, sizeof(row)); mapping(row.correlationId); // External arrives after GPU.
}
void stub_emit(const CUpti_ActivityMemcpy5& row) {
    assert(external_id && enabled == 3);
    gpu_end = std::max(gpu_end, row.end);
    emit_bytes(&row, sizeof(row));
    mapping(row.runtimeCorrelationId ? row.runtimeCorrelationId : row.correlationId);
}
