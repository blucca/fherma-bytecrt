#include "../active-collector.hpp"
#include "cupti.h"
#include <cassert>
int main() {
    bytecrt_active::Collector collector;
    uint32_t correlation = 1000;
    const char* names[] = {"encode_mma32<true>", "forward_first_coalesced<65536>",
        "encode_mma32<true>", "forward_first_coalesced<65536>", "fused_middle<65536>",
        "inverse_weighted<65536,114,480,true>", "g1_crt32<true>", "g1_crt32<true>",
        "g1_crt32<true>", "g1_crt32<true>"};
    constexpr unsigned kernel_start[] = {36,42,76,82,88,94,100,110,120,130};
    for (unsigned c = 0; c < 4; ++c) {
        auto* call = collector.begin();
        call->reserve = {bytecrt_active::host_ns(), bytecrt_active::host_ns()};
        const uint64_t origin = call->begin.cupti + 1000;
        for (unsigned product = 0; product < 128; ++product) {
            const uint64_t base = origin + product * 1000;
            for (unsigned chunk = 0; chunk < 8; ++chunk) {
                auto& pack = call->pack[call->packs++];
                pack.begin = bytecrt_active::host_ns();
                CUpti_ActivityMemcpy5 row;
                row.start = base + chunk * 10; row.end = row.start + 5;
                row.bytes = 3604480; row.streamId = 11;
                const uint32_t key = correlation++;
                row.correlationId = chunk % 2 ? key + 2000000 : key;
                row.runtimeCorrelationId = chunk % 2 ? key : 0;
                stub_emit(row); pack.end = bytecrt_active::host_ns();
            }
            for (unsigned stage = 0; stage < 10; ++stage) {
                CUpti_ActivityKernel9 row;
                row.start = base + kernel_start[stage]; row.end = row.start + 4;
                row.correlationId = correlation++; row.streamId = 22; row.name = names[stage];
                stub_emit(row);
            }
            for (unsigned chunk = 0; chunk < 4; ++chunk) {
                CUpti_ActivityMemcpy5 row;
                row.start = base + 105 + chunk * 10; row.end = row.start + 5;
                row.bytes = 3604480; row.streamId = 33; row.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH;
                row.correlationId = correlation++; row.runtimeCorrelationId = row.correlationId;
                stub_emit(row);
                auto& drain = call->drain[call->drains++];
                drain.begin = bytecrt_active::host_ns(); drain.waited = bytecrt_active::host_ns();
                drain.owned = bytecrt_active::host_ns();
            }
        }
        collector.end(call);
        assert(call->packs == 1024 && call->drains == 512);
    }
    collector.finish();
}
