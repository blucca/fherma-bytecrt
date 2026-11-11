#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
extern void bytecrt_pack_operand(uint32_t*,const uint32_t*,size_t);
extern "C" void* __wrap_memmove(void*,const void*,size_t);
int main(){
 constexpr size_t words=3604480, bytes=words*4;
 std::vector<uint32_t> src(words+128), dst(words+128);
 for(size_t i=0;i<src.size();++i)src[i]=uint32_t(i*2654435761u)^0x544;
 unsigned checks=0;
 for(size_t n:{size_t(0),size_t(1),size_t(11),size_t(17),words})for(size_t offset:{size_t(0),size_t(1),size_t(7),size_t(8),size_t(15)}){
  std::fill(dst.begin(),dst.end(),0xa5a5a5a5u);
  bytecrt_pack_operand(dst.data()+32+offset,src.data()+17,n);
  assert(std::memcmp(dst.data()+32+offset,src.data()+17,n*4)==0);
  for(size_t i=0;i<32+offset;++i)assert(dst[i]==0xa5a5a5a5u);
  for(size_t i=32+offset+n;i<dst.size();++i)assert(dst[i]==0xa5a5a5a5u);
  ++checks;
 }
 std::vector<uint32_t> out;out.reserve(words);out.insert(out.end(),src.begin(),src.begin()+words);
 assert(out.size()==words&&std::memcmp(out.data(),src.data(),bytes)==0);++checks;
 std::vector<uint32_t> expected=src;
 for(size_t i=words;i>0;--i)expected[i]=expected[i-1];
 __wrap_memmove(src.data()+1,src.data(),bytes);assert(src==expected);++checks;
 std::printf("{\"production_checks\":%u,\"pack_guards_equal\":true,\"owned_bytes_equal\":true,\"overlap_fallback_equal\":true}\n",checks);
}
