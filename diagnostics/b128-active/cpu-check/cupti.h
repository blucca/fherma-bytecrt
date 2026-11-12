// Synthetic host-path stub. Deliberately different record layout and API version.
// Actual CUDA 12.8 headers/library remain the native ABI gate.
#pragma once
#include <cstddef>
#include <cstdint>
#define CUDA_VERSION 12080
#define CUPTI_API_VERSION 999
#define CUPTIAPI
using CUcontext = void*;
enum CUptiResult { CUPTI_SUCCESS, CUPTI_ERROR_MAX_LIMIT_REACHED, CUPTI_ERROR_INVALID_PARAMETER };
enum CUpti_ActivityKind { CUPTI_ACTIVITY_KIND_MEMCPY = 1, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,
                         CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39 };
enum CUpti_ExternalCorrelationKind { CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 2 };
enum { CUPTI_ACTIVITY_MEMCPY_KIND_HTOD = 1, CUPTI_ACTIVITY_MEMCPY_KIND_DTOH = 2 };
struct CUpti_Activity { CUpti_ActivityKind kind; uint32_t stub_size; };
struct CUpti_ActivityKernel9 {
    CUpti_ActivityKind kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    uint32_t stub_size = sizeof(CUpti_ActivityKernel9);
    uint64_t start = 0, end = 0;
    uint32_t correlationId = 0, streamId = 0, contextId = 1, deviceId = 0;
    const char* name = nullptr;
};
struct CUpti_ActivityMemcpy5 {
    CUpti_ActivityKind kind = CUPTI_ACTIVITY_KIND_MEMCPY;
    uint32_t stub_size = sizeof(CUpti_ActivityMemcpy5);
    uint64_t start = 0, end = 0, bytes = 0;
    uint32_t correlationId = 0, runtimeCorrelationId = 0, streamId = 0, contextId = 1, deviceId = 0;
    uint32_t copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
};
struct CUpti_ActivityExternalCorrelation {
    CUpti_ActivityKind kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
    uint32_t stub_size = sizeof(CUpti_ActivityExternalCorrelation);
    CUpti_ExternalCorrelationKind externalKind = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;
    uint32_t correlationId = 0;
    uint64_t externalId = 0;
};
using StubRequest = void (*)(uint8_t**, size_t*, size_t*);
using StubComplete = void (*)(CUcontext, uint32_t, uint8_t*, size_t, size_t);
CUptiResult cuptiGetResultString(CUptiResult, const char**);
CUptiResult cuptiGetTimestamp(uint64_t*);
CUptiResult cuptiGetVersion(uint32_t*);
CUptiResult cuptiActivityRegisterCallbacks(StubRequest, StubComplete);
CUptiResult cuptiActivityEnable(CUpti_ActivityKind);
CUptiResult cuptiActivityDisable(CUpti_ActivityKind);
CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t, size_t*);
CUptiResult cuptiActivityGetNextRecord(uint8_t*, size_t, CUpti_Activity**);
CUptiResult cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind, uint64_t);
CUptiResult cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind, uint64_t*);
CUptiResult cuptiActivityFlushAll(uint32_t);
void stub_emit(const CUpti_ActivityKernel9&);
void stub_emit(const CUpti_ActivityMemcpy5&);
