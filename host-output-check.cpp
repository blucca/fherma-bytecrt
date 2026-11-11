// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 blucca
// CPU-only construction model: vector resize + opaque overwrite versus range copy.
// The volatile call retains an external-transfer boundary; the copy is CPU memcpy.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
using Words = std::vector<uint32_t>;
using Clock = std::chrono::steady_clock;
void copy_words(uint32_t* dst, const uint32_t* src, size_t bytes) { std::memcpy(dst, src, bytes); }
using Transfer = void(*)(uint32_t*, const uint32_t*, size_t);
Transfer volatile opaque_transfer = copy_words;
Words resized(const Words& src) { Words out; out.resize(src.size()); opaque_transfer(out.data(),src.data(),src.size()*4); return out; }
Words assigned(const Words& src) { return Words(src.begin(),src.end()); }
double median(std::vector<double> values) { std::sort(values.begin(),values.end());return (values[(values.size()-1)/2]+values[values.size()/2])/2; }
void check(size_t n,size_t limbs) {
    Words src(n*limbs); for(size_t i=0;i<src.size();++i) src[i]=uint32_t(i*2654435761u+17);
    std::vector<double> time[2];
    for(int repeat=0;repeat<23;++repeat) for(int order=0;order<2;++order) {
        const int mode=(repeat+order)%2;
        const auto start=Clock::now(); auto out=mode?assigned(src):resized(src);
        const double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        assert(out==src);  // All words checked outside the timed construction.
        if(repeat>=3) time[mode].push_back(us);
    }
    assert(time[0].size()==20&&time[1].size()==20);
    std::cout<<"{\"n\":"<<n<<",\"limbs\":"<<limbs<<",\"bytes\":"<<src.size()*4
      <<",\"resized_opaque_overwrite_median_us\":"<<median(time[0])
      <<",\"range_construct_median_us\":"<<median(time[1])
      <<",\"samples_each\":20,\"all_words_equal\":true}";
}
int main() {
    static_assert(sizeof(uint32_t)==4);
    std::cout<<"{\"compiler\":\""<<__VERSION__<<"\",\"evidence\":\"CPU vector construction; memcpy emulates overwrite\",\"gpu_transfers_measured\":false,\"points\":[";
    check(65536,55); std::cout<<",";check(32768,28);std::cout<<"]}\n";
}
