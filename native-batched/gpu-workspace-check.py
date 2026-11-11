from pathlib import Path
import json,re,subprocess,sys

artifact=Path(__file__).resolve().parent
if len(sys.argv)!=2: raise SystemExit('usage: gpu-workspace-check.py OUTPUT_DIR')
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
source=artifact/'solve.cu'
raw=source.read_text()
core=artifact/'core/solve.cu'
core_raw=core.read_text()
state_source='struct State {'+core_raw.split('struct State {',1)[1].split('\n};',1)[0]+'\n};\n'
allocate_source='template<class T>\nvoid allocate('+core_raw.split('template<class T>\nvoid allocate(',1)[1].split('\n}',1)[0]+'\n}\n'
fixed_fields=['tw','inv_tw','psi','untwist_crt','encode_b','encode_mu','q','primes','n_inverse','b_matrix','invp','constants']
scratch_fields=['ab','ab_t','c','weighted']
shared_fields=fixed_fields+scratch_fields
sharing=raw.split('void share_compute_storage(',1)[1].split('\n}',1)[0]
rebound=re.findall(r'borrow_device_allocation\(s, s\.(\w+), owner\.(\w+)\);',sharing)
assert len(rebound)==16 and all(a==b for a,b in rebound)
assert {a for a,b in rebound}==set(shared_fields)
core_init=core_raw.split('void* fherma_init(',1)[1].split('return s.release();',1)[0]
core_owned=re.findall(r'(?:allocate|fixed_copy)\(\*s, s->(\w+),',core_init)
assert len(core_owned)==18 and set(core_owned)==set(shared_fields+['inputs','out'])

body=raw.split('#undef fherma_free\n',1)[1]
body,count=re.subn(r'([\w:]+<[^;\n]*?>)<<<(.*?)>>>\(',lambda m:m[1]+'(Launch{'+m[2]+'}, ',body,flags=re.S)
assert count==3
header=artifact/'fherma.h'
prefix=r'''
#include "FHERMA_HEADER"
#include <malloc.h>
#include <climits>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using Clock=std::chrono::steady_clock;
struct Node;
using Ref=std::shared_ptr<Node>;
struct Node { std::vector<Ref> deps; std::function<void()> work; bool done=false,visiting=false; };
struct MockStream { Ref tail; unsigned role=9; };
struct MockEvent { Ref generation; };
using cudaStream_t=MockStream*;
using cudaEvent_t=MockEvent*;
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaStreamNonBlocking=1,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct Launch { dim3 grid; unsigned threads; size_t shared; cudaStream_t stream; };
struct cudaDeviceProp { int asyncEngineCount=2,l2CacheSize=6000000; };
struct Counts { size_t stream_create=0,stream_destroy=0,event_create=0,event_destroy=0,
 records=0,waits=0,syncs=0,h2d=0,d2h=0,pack=0,encode=0,forward=0,middle=0,inverse=0,crt=0,serial=0,device_alloc=0,device_free=0,host_alloc=0,host_free=0; };
Counts counts;
std::set<cudaStream_t> active_streams;
std::set<cudaEvent_t> active_events;
bool legacy_initializing=false;
unsigned next_role=0,create_attempt=0,fail_stream_attempt=0;
Ref enqueue(cudaStream_t stream,std::function<void()> work={},Ref dep={}) {
 assert(active_streams.count(stream)); auto node=std::make_shared<Node>();
 if(stream->tail) node->deps.push_back(stream->tail);
 if(dep) node->deps.push_back(dep);
 node->work=std::move(work);stream->tail=node;return node;
}
void execute(const Ref& node) {
 if(!node || node->done) return;
 assert(!node->visiting);node->visiting=true;
 for(const auto& dep:node->deps) execute(dep);
 if(node->work) node->work();
 node->done=true;node->visiting=false;
}
void check(cudaError_t error,const char* what) { if(error) throw std::runtime_error(what); }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream,unsigned flags) {
 assert(flags==cudaStreamNonBlocking);++create_attempt;
 if(create_attempt==fail_stream_attempt) return 1;
 *stream=new MockStream;(*stream)->role=legacy_initializing?9:next_role++;
 assert(active_streams.insert(*stream).second);++counts.stream_create;return 0;
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {
 assert(active_streams.count(stream));assert(!stream->tail || stream->tail->done);
 active_streams.erase(stream);delete stream;++counts.stream_destroy;return 0;
}
cudaError_t cudaEventCreate(cudaEvent_t* event) {
 *event=new MockEvent;assert(active_events.insert(*event).second);++counts.event_create;return 0;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
 assert(active_events.count(event));assert(!event->generation || event->generation->done);
 active_events.erase(event);delete event;++counts.event_destroy;return 0;
}
cudaError_t cudaEventRecord(cudaEvent_t event,cudaStream_t stream) {
 assert(active_events.count(event));
 // The actual wrapper recycles an event after its prior generation completed.
 assert(!event->generation || event->generation->done);
 event->generation=enqueue(stream);++counts.records;return 0;
}
cudaError_t cudaStreamWaitEvent(cudaStream_t stream,cudaEvent_t event,unsigned flags) {
 assert(flags==0 && active_events.count(event) && event->generation);
 enqueue(stream,{},event->generation);++counts.waits;return 0;
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) {
 assert(active_events.count(event) && event->generation);execute(event->generation);++counts.syncs;return 0;
}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t a,cudaEvent_t b) {
 assert(a->generation && b->generation && a->generation->done && b->generation->done);*ms=1;return 0;
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* props,int device) { assert(device==0);props->asyncEngineCount=2;return 0; }
cudaError_t cudaGetLastError() { return 0; }
namespace cupqc {
 template<unsigned N,unsigned M,class T> constexpr size_t fwd_stage_1_ntt_shared_workspace_size(){return N/M*sizeof(T);}
 template<unsigned N,unsigned M,class T> constexpr size_t fwd_stage_2_ntt_shared_workspace_size(){return M*sizeof(T);}
 template<unsigned N,unsigned M,class T> constexpr size_t inv_stage_2_ntt_shared_workspace_size(){return M*sizeof(T);}
}
constexpr unsigned kInputCoefficients=2,kOutputCoefficients=2,kOutputBlocks=1;
constexpr unsigned kMaxInputChunks=4,kMaxOutputChunks=4;
struct Plan { unsigned n=8,w=1747,l=2,pcount=2,gap=1,kpad=8; };
namespace bytecrt { using RnsPlan=Plan; }
using Constant=uint32_t;
struct State;
enum BufferKind : unsigned { Inputs,Ab,AbT,C,Weighted,Out,HostIn,HostOut,Tw,InvTw,Psi,Untwist,EncodeB,EncodeMu,Q,Primes,NInverse,BMatrix,Invp,Constants };
struct Location { State* owner; unsigned kind; size_t offset; };
struct Allocation {
 std::unique_ptr<unsigned char[]> storage;size_t bytes=0;bool host=false,live=true;
 State* owner=nullptr;unsigned kind=99,frees=0;
};
std::map<uintptr_t,Allocation> buffers;
std::map<State*,unsigned> generations;
constexpr uint32_t poison=0xdeadbeef;
cudaError_t mock_malloc(void** output,size_t bytes,bool host) {
 assert(bytes);Allocation a;a.storage=std::make_unique<unsigned char[]>(bytes);a.bytes=bytes;a.host=host;
 std::memset(a.storage.get(),0xef,bytes);*output=a.storage.get();
 assert(buffers.emplace(reinterpret_cast<uintptr_t>(*output),std::move(a)).second);
 if(host)++counts.host_alloc;else ++counts.device_alloc;return 0;
}
cudaError_t cudaMalloc(void** output,size_t bytes) {return mock_malloc(output,bytes,false);}
template<class T>cudaError_t cudaMallocHost(T** output,size_t bytes) {
 void* raw=nullptr;auto status=mock_malloc(&raw,bytes,true);*output=static_cast<T*>(raw);return status;
}
cudaError_t mock_free(void* pointer,bool host) {
 auto it=buffers.find(reinterpret_cast<uintptr_t>(pointer));assert(it!=buffers.end());auto& a=it->second;
 assert(a.host==host && a.live && a.frees==0);a.live=false;++a.frees;std::memset(pointer,0xdd,a.bytes);
 if(host)++counts.host_free;else ++counts.device_free;return 0;
}
cudaError_t cudaFree(void* pointer) {return mock_free(pointer,false);}
cudaError_t cudaFreeHost(void* pointer) {return mock_free(pointer,true);}
ACTUAL_STATE_SOURCE
ACTUAL_ALLOCATE_SOURCE
template<class T>void attach(State& state,T* pointer,unsigned kind) {
 auto& a=buffers.at(reinterpret_cast<uintptr_t>(pointer));assert(a.live && !a.owner);a.owner=&state;a.kind=kind;
}
template<class T>void make_device(State& state,T*& pointer,size_t elements,unsigned kind) {
 allocate(state,pointer,elements);attach(state,pointer,kind);
 if(kind>=Tw)std::memset(pointer,kind,buffers.at(reinterpret_cast<uintptr_t>(pointer)).bytes);
}
Location locate(const void* pointer,size_t bytes=0) {
 const auto ptr=reinterpret_cast<uintptr_t>(pointer);
 auto it=buffers.upper_bound(ptr);assert(it!=buffers.begin());--it;const auto& a=it->second;
 assert(a.live && a.owner && ptr+bytes<=it->first+a.bytes);
 assert((ptr-it->first)%sizeof(uint32_t)==0);
 return {a.owner,a.kind,(ptr-it->first)/sizeof(uint32_t)};
}
State* compute_owner(State* logical) {return locate(logical->ab).owner;}
void fixed_arg(const void* pointer,State* owner,unsigned kind) {
 const auto a=locate(pointer);assert(a.owner==owner && a.kind==kind && a.offset==0);
}
cudaError_t cudaMemcpyAsync(void* dest,const void* src,size_t bytes,int kind,cudaStream_t stream) {
 const auto d=locate(dest,bytes),s=locate(src,bytes);assert(d.owner==s.owner);
 if(kind==cudaMemcpyHostToDevice) {
  assert(stream->role==0 && d.kind==0 && s.kind==6);++counts.h2d;
 } else { assert(kind==cudaMemcpyDeviceToHost && stream->role==2 && d.kind==7 && s.kind==5);++counts.d2h; }
 enqueue(stream,[=]{(void)locate(dest,bytes);(void)locate(src,bytes);std::memcpy(dest,src,bytes);});return 0;
}
void bytecrt_pack_operand(uint32_t* dest,const uint32_t* src,size_t words) {
 const auto d=locate(dest,words*sizeof(uint32_t));assert(d.kind==6);
 if(d.offset==0) {
  auto& state=*d.owner;
  if(generations[&state]) for(auto event:state.copied)assert(event->generation && event->generation->done);
  ++generations[&state];std::fill(state.hostout,state.hostout+size_t(state.plan.n)*state.plan.l,poison);
 }
 std::memcpy(dest,src,words*sizeof(uint32_t));++counts.pack;
}
void* bytecrt_serial_init(const fherma::Point& p) {
 auto s=std::make_unique<State>();s->batch=p.B;s->profile_products=p.B==1?1:2;generations[s.get()]=0;
 const size_t words=size_t(s->plan.n)*s->plan.l,planes=size_t(s->plan.n)*s->plan.pcount;
 make_device(*s,s->inputs,2*words,Inputs);make_device(*s,s->ab,2*planes,Ab);
 make_device(*s,s->ab_t,2*planes,AbT);make_device(*s,s->c,planes,C);
 make_device(*s,s->tw,planes,Tw);make_device(*s,s->inv_tw,planes,InvTw);
 make_device(*s,s->weighted,size_t(s->plan.n)*(s->plan.kpad/4),Weighted);
 std::memset(s->weighted,0,size_t(s->plan.n)*s->plan.kpad);
 make_device(*s,s->out,words,Out);
 check(cudaMallocHost(&s->hostout,words*sizeof(uint32_t)),"host out");attach(*s,s->hostout,HostOut);
 check(cudaMallocHost(&s->hostinput,2*words*sizeof(uint32_t)),"host input");attach(*s,s->hostinput,HostIn);
 make_device(*s,s->encode_b,s->plan.l*s->plan.pcount,EncodeB);
 make_device(*s,s->encode_mu,s->plan.pcount,EncodeMu);make_device(*s,s->q,s->plan.l,Q);
 make_device(*s,s->primes,s->plan.pcount,Primes);make_device(*s,s->n_inverse,s->plan.pcount,NInverse);
 make_device(*s,s->b_matrix,s->plan.pcount*s->plan.l,BMatrix);
 make_device(*s,s->invp,s->plan.pcount,Invp);make_device(*s,s->constants,s->plan.pcount,Constants);
 make_device(*s,s->psi,planes,Psi);make_device(*s,s->untwist_crt,planes,Untwist);
 assert(s->allocations.size()==18);legacy_initializing=true;
 for(auto& stream:s->streams)check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"legacy stream");
 check(cudaStreamCreateWithFlags(&s->input_stream,cudaStreamNonBlocking),"legacy input");
 legacy_initializing=false;
 for(auto& op:s->input_ready)for(auto& e:op)check(cudaEventCreate(&e),"legacy ready");
 for(auto& e:s->crt_done)check(cudaEventCreate(&e),"legacy CRT");
 for(auto& e:s->copied)check(cudaEventCreate(&e),"legacy copied");
 for(auto& op:s->input_chunk_events)for(auto& chunk:op)for(auto& e:chunk)check(cudaEventCreate(&e),"legacy input profile");
 check(cudaEventCreate(&s->a_forwarded),"legacy forwarded");
 for(auto& op:s->operand_events)for(auto& e:op)check(cudaEventCreate(&e),"legacy operand profile");
 for(auto& e:s->events)check(cudaEventCreate(&e),"legacy profile");
 return s.release();
}
fherma::Outputs bytecrt_serial_run(void* state,const fherma::Inputs& in) {
 auto& s=*static_cast<State*>(state);assert(s.batch==1);++counts.serial;
 const size_t words=size_t(s.plan.n)*s.plan.l;assert(in.a.data.size()==words && in.b.data.size()==words);
 fherma::Outputs out;out.c.shape={1,s.plan.n,s.plan.l};
 for(size_t i=0;i<words;++i)out.c.data.push_back(in.a.data[i]+in.b.data[i]);
 return out;
}
namespace bytecrt {
 void launch_encode_mma32_transposed(unsigned N,unsigned L,unsigned P,const uint32_t* input,
 const uint8_t* encode_b,const uint32_t* primes,const uint64_t* encode_mu,uint32_t* output,cudaStream_t stream,unsigned operands) {
  const auto a=locate(input),b=locate(output);assert(compute_owner(a.owner)==b.owner && a.kind==Inputs && b.kind==AbT);
  fixed_arg(encode_b,b.owner,EncodeB);fixed_arg(primes,b.owner,Primes);fixed_arg(encode_mu,b.owner,EncodeMu);
  assert(N==65536 && L==2 && P==114 && operands==1 && stream->role==1);++counts.encode;
  const size_t words=size_t(a.owner->plan.n)*L;
  enqueue(stream,[=]{std::copy(input,input+words,output);});
 }
 namespace ntt {
  constexpr unsigned kSubSize=2,kThreads=128;
  template<unsigned N>void forward_first_coalesced(Launch launch,const uint32_t* input,uint32_t* output,
   const uint32_t* tw,const uint32_t* psi,const Constant* constants,unsigned P) {
   assert(N==65536 && P==114 && launch.stream->role==1 && launch.threads==128);
   const auto a=locate(input),b=locate(output);assert(a.owner==b.owner && a.kind==AbT && b.kind==Ab);
   fixed_arg(tw,a.owner,Tw);fixed_arg(psi,a.owner,Psi);fixed_arg(constants,a.owner,Constants);
   const size_t words=size_t(a.owner->plan.n)*a.owner->plan.l;++counts.forward;
   enqueue(launch.stream,[=]{std::copy(input,input+words,output);});
  }
  template<unsigned N>void fused_middle(Launch launch,const uint32_t* ab,uint32_t* output,
   const uint32_t* tw,const uint32_t* inv_tw,const Constant* constants,unsigned P) {
   assert(N==65536 && P==114 && launch.stream->role==1);
   const auto a=locate(ab),b=locate(output);assert(a.owner==b.owner && a.kind==Ab && b.kind==C);
   fixed_arg(tw,a.owner,Tw);fixed_arg(inv_tw,a.owner,InvTw);fixed_arg(constants,a.owner,Constants);
   const size_t words=size_t(a.owner->plan.n)*a.owner->plan.l;++counts.middle;
   enqueue(launch.stream,[=]{for(size_t i=0;i<words;++i)output[i]=ab[i]+ab[words+i];});
  }
 }
 template<unsigned N,unsigned P,unsigned Kpad,bool Transposed>void inverse_weighted(Launch launch,
  const uint32_t* input,uint32_t* output,const uint32_t* inv_tw,const uint32_t* untwist,
  const Constant* constants,const uint32_t* n_inverse) {
  assert(N==65536 && P==114 && Kpad==480 && Transposed && launch.stream->role==1);
  const auto a=locate(input),b=locate(output);assert(a.owner==b.owner && a.kind==C && b.kind==Weighted);
  fixed_arg(inv_tw,a.owner,InvTw);fixed_arg(untwist,a.owner,Untwist);fixed_arg(constants,a.owner,Constants);fixed_arg(n_inverse,a.owner,NInverse);
  const size_t words=size_t(a.owner->plan.n)*a.owner->plan.l;++counts.inverse;
  enqueue(launch.stream,[=]{std::copy(input,input+words,output);});
 }
 void launch_g1_crt32_transposed(unsigned W,unsigned N,const uint8_t* weighted,const uint8_t* b_matrix,
  const float* invp,const uint32_t* q,uint32_t,uint32_t* out,cudaStream_t stream,unsigned first,unsigned blocks) {
  assert(W==1747 && N==8 && blocks==1 && first<4 && stream->role==1);
  const auto a=locate(weighted),b=locate(out);assert(a.owner==compute_owner(b.owner) && a.kind==Weighted && b.kind==Out);
  fixed_arg(b_matrix,a.owner,BMatrix);fixed_arg(invp,a.owner,Invp);fixed_arg(q,a.owner,Q);
  const size_t begin=first*kOutputCoefficients*a.owner->plan.l,words=kOutputCoefficients*a.owner->plan.l;
  const auto* input=reinterpret_cast<const uint32_t*>(weighted);++counts.crt;
  enqueue(stream,[=]{std::copy(input+begin,input+begin+words,out+begin);});
 }
}
'''.replace('FHERMA_HEADER',str(header)).replace('ACTUAL_STATE_SOURCE',state_source).replace('ACTUAL_ALLOCATE_SOURCE',allocate_source)
suffix=r'''
std::vector<void*> shared_pointers(const State& s) {
 return {s.tw,s.inv_tw,s.psi,s.untwist_crt,s.encode_b,s.encode_mu,s.q,s.primes,s.n_inverse,s.b_matrix,s.invp,s.constants,s.ab,s.ab_t,s.c,s.weighted};
}
std::vector<cudaEvent_t> private_events(const State& s) {
 std::vector<cudaEvent_t> result;
 for(const auto& op:s.input_ready)for(auto e:op)result.push_back(e);
 for(const auto& op:s.input_chunk_events)for(const auto& chunk:op)for(auto e:chunk)result.push_back(e);
 for(auto e:s.crt_done)result.push_back(e);
 for(auto e:s.copied)result.push_back(e);
 result.push_back(s.a_forwarded);
 for(const auto& op:s.operand_events)for(auto e:op)result.push_back(e);
 for(auto e:s.events)result.push_back(e);
 return result;
}
void assert_inventory(const BatchState& batch) {
 const auto& owner=*batch.slots[0];assert(owner.allocations.size()==18);
 std::set<void*> parent_inventory(owner.allocations.begin(),owner.allocations.end());assert(parent_inventory.size()==18);
 auto shared=shared_pointers(owner);assert(std::set<void*>(shared.begin(),shared.end()).size()==16);
 for(auto p:shared)assert(parent_inventory.count(p) && buffers.at(reinterpret_cast<uintptr_t>(p)).live);
 std::set<void*> private_io;std::set<cudaEvent_t> events;
 for(unsigned i=0;i<batch.slot_count;++i) {
  const auto& s=*batch.slots[i];assert(s.inputs && s.out && s.hostinput && s.hostout);
  for(auto p:std::vector<void*>{s.inputs,s.out,s.hostinput,s.hostout})assert(private_io.insert(p).second);
  for(auto e:private_events(s))assert(e && events.insert(e).second);
  if(i) {
   assert(s.allocations.size()==2 && std::set<void*>(s.allocations.begin(),s.allocations.end())==std::set<void*>({s.inputs,s.out}));
   assert(shared_pointers(s)==shared);
   for(auto p:shared)assert(std::find(s.allocations.begin(),s.allocations.end(),p)==s.allocations.end());
  }
 }
 assert(private_io.size()==4*batch.slot_count && events.size()==61*batch.slot_count);
}
void assert_all_freed() {
 for(const auto& pair:buffers)assert(!pair.second.live && pair.second.frees==1);
 assert(counts.device_alloc==counts.device_free && counts.host_alloc==counts.host_free);
}
void reset() {
 assert(active_streams.empty() && active_events.empty());assert_all_freed();counts={};buffers.clear();generations.clear();
 next_role=create_attempt=fail_stream_attempt=0;legacy_initializing=false;
}
int main() {
 reset();fherma::Point p;p.B=1;p.N=65536;p.W=1747;p.L=55;
 auto* single=fherma_init(p);auto& single_batch=*static_cast<BatchState*>(single);
 assert(single_batch.slots[0]->allocations.size()==18 && counts.device_free==0);
 fherma::Inputs one;one.a.shape=one.b.shape={1,8,2};
 for(unsigned i=0;i<16;++i){one.a.data.push_back(i+1);one.b.data.push_back(i+2);}
 const auto one_a=one.a.data,one_b=one.b.data;
 auto single_result=fherma_run(single,one);
 assert((single_result.c.shape==std::vector<int64_t>{1,8,2}) && single_result.c.data.size()==16);
 for(unsigned i=0;i<16;++i)assert(single_result.c.data[i]==2*i+3);
 assert(one.a.data==one_a && one.b.data==one_b && one.a.shape==one.b.shape);
 assert(counts.serial==1 && counts.records==0 && counts.h2d==0 && counts.d2h==0);
 fherma_free(single);assert(counts.stream_create==3 && counts.stream_destroy==3);assert_all_freed();reset();
 // Actual lifted helper + actual core State destructor, both destruction orders.
 for(unsigned owner_first=0;owner_first<2;++owner_first) {
  p.B=2;auto* handle=fherma_init(p);auto& batch=*static_cast<BatchState*>(handle);assert_inventory(batch);
  assert(counts.device_alloc==36 && counts.device_free==16 && counts.host_alloc==4);
  auto owner=std::move(batch.slots[0]),borrower=std::move(batch.slots[1]);
  if(owner_first) {owner.reset();assert(counts.device_free==34);borrower.reset();}
  else {borrower.reset();assert(counts.device_free==18);owner.reset();}
  assert_all_freed();fherma_free(handle);reset();
 }
 p.B=128;auto* state=fherma_init(p);auto& batch=*static_cast<BatchState*>(state);
 assert(batch.slot_count==4 && counts.stream_create==15 && counts.stream_destroy==12);
 assert(counts.device_alloc==72 && counts.device_free==48 && counts.host_alloc==8);
 assert_inventory(batch);
 for(auto& s:batch.slots) {assert(s && !s->input_stream);for(auto stream:s->streams)assert(!stream);}
 std::map<void*,std::vector<unsigned char>> fixed_before;
 const auto shared=shared_pointers(*batch.slots[0]);
 for(unsigned i=0;i<12;++i) {auto* ptr=static_cast<unsigned char*>(shared[i]);const auto bytes=buffers.at(reinterpret_cast<uintptr_t>(ptr)).bytes;fixed_before[ptr]={ptr,ptr+bytes};}
 fherma::Inputs in;in.a.shape=in.b.shape={128,8,2};in.a.data.resize(128*16);in.b.data.resize(128*16);
 fherma::Outputs first_owned;
 for(unsigned call=0;call<2;++call) {
  for(unsigned j=0;j<128;++j)for(unsigned i=0;i<16;++i) {
   in.a.data[j*16+i]=100000*call+1000*j+i+1;in.b.data[j*16+i]=50000*call+100*j+3*i+2;
  }
  const auto a=in.a.data,b=in.b.data;const auto ashape=in.a.shape,bshape=in.b.shape;const auto before=counts;
  auto out=fherma_run(state,in);
  assert((out.c.shape==std::vector<int64_t>{128,8,2}) && out.c.data.size()==128*16);
  for(unsigned j=0;j<128;++j)for(unsigned i=0;i<16;++i)assert(out.c.data[j*16+i]==150000*call+1100*j+4*i+3);
  assert(in.a.data==a && in.b.data==b && in.a.shape==ashape && in.b.shape==bshape);
  assert(counts.records-before.records==1280+(call==0?6:0));
  assert(counts.waits-before.waits==768 && counts.syncs-before.syncs==128);
  assert(counts.h2d-before.h2d==256 && counts.d2h-before.d2h==512 && counts.pack-before.pack==256);
  assert(counts.encode-before.encode==256 && counts.forward-before.forward==256);
  assert(counts.middle-before.middle==128 && counts.inverse-before.inverse==128 && counts.crt-before.crt==512);
  assert(counts.device_alloc==before.device_alloc && counts.device_free==before.device_free && counts.host_alloc==before.host_alloc);
  for(auto stream:batch.streams)assert(stream->tail && stream->tail->done);
  for(auto& s:batch.slots)assert(generations[s.get()]==32*(call+1));
  for(const auto& pair:fixed_before)assert(std::memcmp(pair.first,pair.second.data(),pair.second.size())==0);
  if(call==0)first_owned=std::move(out);
  else for(unsigned j=0;j<128;++j)for(unsigned i=0;i<16;++i)assert(first_owned.c.data[j*16+i]==1100*j+4*i+3);
 }
 // Input size failure occurs before queue submission or device allocation.
 in.a.data.pop_back();const auto old=counts.records;bool bad=false;
 try{fherma_run(state,in);}catch(const std::runtime_error&){bad=true;}assert(bad && counts.records==old);
 fherma_free(state);
 assert_all_freed();assert(counts.stream_create==counts.stream_destroy && counts.event_create==counts.event_destroy);reset();
 // Retain the prior single setup-failure check, now covering shared inventory cleanup.
 fail_stream_attempt=14;bool failed=false;
 try{fherma_init(p);}catch(const std::runtime_error&){failed=true;}
 assert(failed && counts.stream_create==13 && counts.stream_destroy==13);
 assert_all_freed();assert(counts.event_create==counts.event_destroy);reset();
 std::cout<<"PASS: actual lifted shared helper and core State/allocate; child2/owner18 single-free inventory in both destruction orders; B1 delegation; two B128 generations; 256 complete owned products; private deferred DMA/events and one compute FIFO\n";
}
'''
cpp=root/'source-check.cpp'
cpp.write_text(prefix+body+suffix)
compile=subprocess.run(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror',str(cpp),'-o',str(root/'source-check')],capture_output=True,text=True)
(root/'compile.log').write_text(compile.stdout+compile.stderr)
assert compile.returncode==0,compile.stderr
run=subprocess.run([str(root/'source-check')],capture_output=True,text=True)
(root/'check.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stdout+run.stderr
result={'status':'PASS','scope':'actual shared wrapper/helper plus actual core State/allocate/destructor; scaled 8x2 arithmetic, deferred DMA/events, real host control flow','B1_path':True,'B128_calls':2,'full_owned_products':256,'shared_fixed_fields':fixed_fields,'shared_scratch_fields':scratch_fields,'owner_device_allocations':18,'borrower_device_allocations':2,'single_free_both_destruction_orders':True,'private_DMA_IO_events':True,'fixed_tables_unchanged':True,'previous_owned_output_unchanged':True,'per_B128_counts':{'pack':256,'H2D':256,'D2H':512,'encode':256,'forward':256,'middle':128,'inverse':128,'CRT':512,'owned_sync_insert':128},'init_cleanup_checked':True,'source_extracted':True,'GPU_execution':0}
(root/'orchestration-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(run.stdout.strip());print(json.dumps(result))
