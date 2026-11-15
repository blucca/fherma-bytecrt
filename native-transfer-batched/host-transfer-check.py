from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess

parser = argparse.ArgumentParser(description='Lift the actual wrapper into a small deferred-DMA host proof')
parser.add_argument('--out', required=True, type=Path, help='temporary output directory')
args = parser.parse_args()
artifact = Path(__file__).resolve().parent
repo = artifact.parent
root = args.out.resolve()
root.mkdir(parents=True, exist_ok=True)
source = artifact / 'solve.cu'
raw = source.read_text()
body = raw.split('#undef fherma_free\n', 1)[1]
body, launch_count = re.subn(
    r'([\w:]+<[^;\n]*?>)<<<(.*?)>>>\(',
    lambda m: m[1] + '(Launch{' + m[2] + '}, ', body, flags=re.S)
assert launch_count == 3
# Reduced mock dimensions retain the production N dispatch via a separate tag.
assert body.count("if (plan.n == 65536)") == 1
body = body.replace("if (plan.n == 65536)", "if (s.dispatch_n == 65536)")

prefix = r'''
#include "FHERMA_HEADER"
#include <malloc.h>
#include <climits>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

enum class Phase { idle, init, h2d, run, d2h };
Phase phase=Phase::idle;
struct Node;
using Ref=std::shared_ptr<Node>;
struct Node { std::vector<Ref> deps; std::function<void()> work; bool done=false, visiting=false; };
struct MockStream { Ref tail; unsigned role=99; };
struct MockEvent { Ref generation; };
using cudaStream_t=MockStream*;
using cudaEvent_t=MockEvent*;
using cudaError_t=int;
constexpr int cudaSuccess=0, cudaStreamNonBlocking=1, cudaEventDisableTiming=2,
    cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct Launch { dim3 grid; unsigned threads; size_t shared; cudaStream_t stream; };
struct Counts {
    size_t streams=0, streams_freed=0, events=0, events_freed=0, allocations=0,
        records=0, waits=0, event_sync=0, stream_sync=0, device_sync=0,
        pack=0, h2d=0, d2h=0, full_encode=0, range_encode=0,
        forward=0, middle=0, inverse=0, crt=0;
};
Counts counts;
std::set<cudaStream_t> active_streams;
std::set<cudaEvent_t> active_events;
bool legacy_initializing=false;
unsigned legacy_role=0, global_role=0;

Ref enqueue(cudaStream_t stream,std::function<void()> work={},Ref dep={}) {
    assert(active_streams.count(stream));
    auto node=std::make_shared<Node>();
    if(stream->tail)node->deps.push_back(stream->tail);
    if(dep)node->deps.push_back(dep);
    node->work=std::move(work);stream->tail=node;return node;
}
void execute(const Ref& node) {
    if(!node || node->done)return;
    assert(!node->visiting);node->visiting=true;
    for(const auto& dep:node->deps)execute(dep);
    if(node->work)node->work();
    node->done=true;node->visiting=false;
}
void check(cudaError_t e,const char* what){if(e)throw std::runtime_error(what);}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream,unsigned flags) {
    assert(phase==Phase::init && flags==cudaStreamNonBlocking);
    *stream=new MockStream;
    (*stream)->role=legacy_initializing ? legacy_role++%3 : 3+global_role++;
    assert(active_streams.insert(*stream).second);++counts.streams;return 0;
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    assert(active_streams.count(stream) && (!stream->tail || stream->tail->done));
    active_streams.erase(stream);delete stream;++counts.streams_freed;return 0;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event,unsigned flags) {
    assert(phase==Phase::init && flags==cudaEventDisableTiming);
    *event=new MockEvent;assert(active_events.insert(*event).second);++counts.events;return 0;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
    assert(active_events.count(event) && (!event->generation || event->generation->done));
    active_events.erase(event);delete event;++counts.events_freed;return 0;
}
cudaError_t cudaEventRecord(cudaEvent_t event,cudaStream_t stream) {
    assert(active_events.count(event) && (!event->generation || event->generation->done));
    event->generation=enqueue(stream);++counts.records;return 0;
}
cudaError_t cudaStreamWaitEvent(cudaStream_t stream,cudaEvent_t event,unsigned flags) {
    assert(flags==0 && active_events.count(event) && event->generation);
    enqueue(stream,{},event->generation);++counts.waits;return 0;
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    assert(active_events.count(event) && event->generation);
    execute(event->generation);++counts.event_sync;return 0;
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    assert(phase==Phase::h2d);execute(stream->tail);++counts.stream_sync;return 0;
}
cudaError_t cudaDeviceSynchronize() {
    assert(phase==Phase::run);
    for(auto stream:active_streams)execute(stream->tail);
    ++counts.device_sync;return 0;
}
cudaError_t cudaGetLastError(){return 0;}

namespace cupqc {
template<unsigned N,unsigned M,class T>constexpr size_t fwd_stage_1_ntt_shared_workspace_size(){return N/M*sizeof(T);}
template<unsigned N,unsigned M,class T>constexpr size_t fwd_stage_2_ntt_shared_workspace_size(){return M*sizeof(T);}
template<unsigned N,unsigned M,class T>constexpr size_t inv_stage_2_ntt_shared_workspace_size(){return M*sizeof(T);}
}
constexpr unsigned kInputCoefficients=2, kInputBlocks=1, kOutputCoefficients=2, kOutputBlocks=1;
struct Plan { unsigned n=8,w=1747,l=2,pcount=2,gap=1; };
struct State;
struct Buffer { State* owner; unsigned kind; size_t count; };
struct Location { State* owner; unsigned kind; size_t offset,count; };
std::map<uintptr_t,Buffer> buffers;
constexpr uint32_t poison=0xdeadbeef;
struct State {
    Plan plan; unsigned batch=1, profile_products=1, dispatch_n=65536;
    uint32_t *inputs=nullptr,*ab=nullptr,*ab_t=nullptr,*c=nullptr,*weighted=nullptr,*out=nullptr,
        *hostinput=nullptr,*hostout=nullptr,*primes=nullptr,*tw=nullptr,*psi=nullptr,*inv_tw=nullptr,
        *untwist_crt=nullptr,*constants=nullptr,*n_inverse=nullptr,*q=nullptr;
    uint8_t *encode_b=nullptr,*b_matrix=nullptr;
    uint64_t* encode_mu=nullptr;float* invp=nullptr;
    std::array<cudaStream_t,2> streams{};cudaStream_t input_stream=nullptr;
    std::array<std::array<cudaEvent_t,4>,2> input_ready{};
    std::array<cudaEvent_t,4> copied{};cudaEvent_t a_forwarded=nullptr;
    std::array<std::vector<uint32_t>,8> store;
    std::vector<std::unique_ptr<uint32_t[]>> raw;
    explicit State(const fherma::Point& p):batch(p.B),dispatch_n(p.N) {
        if(p.N==32768){plan.n=4;plan.w=868;}
        const size_t words=size_t(plan.n)*plan.l;
        for(unsigned k=0;k<8;++k) {
            store[k].assign((k==0 || k==1 || k==2 || k==6)?2*words:words,poison);
            assert(buffers.emplace(reinterpret_cast<uintptr_t>(store[k].data()),Buffer{this,k,store[k].size()}).second);
        }
        inputs=store[0].data();ab=store[1].data();ab_t=store[2].data();c=store[3].data();
        weighted=store[4].data();out=store[5].data();hostinput=store[6].data();hostout=store[7].data();
    }
    ~State() {
        for(auto stream:streams)if(stream)cudaStreamDestroy(stream);
        if(input_stream)cudaStreamDestroy(input_stream);
        for(const auto& op:input_ready)for(auto event:op)if(event)cudaEventDestroy(event);
        for(auto event:copied)if(event)cudaEventDestroy(event);
        if(a_forwarded)cudaEventDestroy(a_forwarded);
        for(auto& data:store)assert(buffers.erase(reinterpret_cast<uintptr_t>(data.data()))==1);
        for(auto& data:raw)assert(buffers.erase(reinterpret_cast<uintptr_t>(data.get()))==1);
    }
};
Location locate(const void* pointer,size_t bytes=0) {
    const uintptr_t ptr=reinterpret_cast<uintptr_t>(pointer);
    auto it=buffers.upper_bound(ptr);assert(it!=buffers.begin());--it;
    assert(ptr+bytes<=it->first+it->second.count*sizeof(uint32_t));
    assert((ptr-it->first)%sizeof(uint32_t)==0);
    return {it->second.owner,it->second.kind,(ptr-it->first)/sizeof(uint32_t),it->second.count};
}
template<class T>void allocate(State& s,T*& pointer,size_t count) {
    static_assert(std::is_same_v<T,uint32_t>);assert(phase==Phase::init);
    auto data=std::make_unique<uint32_t[]>(count);std::fill(data.get(),data.get()+count,poison);
    pointer=data.get();unsigned kind=8+s.raw.size();s.raw.push_back(std::move(data));
    assert(buffers.emplace(reinterpret_cast<uintptr_t>(pointer),Buffer{&s,kind,count}).second);
    ++counts.allocations;
}
struct PendingRead { const void* source; size_t bytes; Ref node; };
std::vector<PendingRead> pending_h2d;
cudaError_t cudaMemcpyAsync(void* dest,const void* source,size_t bytes,int kind,cudaStream_t stream) {
    const auto d=locate(dest,bytes),s=locate(source,bytes);
    if(kind==cudaMemcpyHostToDevice) {
        assert(phase==Phase::h2d && (d.kind==8 || d.kind==9) && s.kind==6);
        assert(stream->role==2 || stream->role==3);++counts.h2d;
    } else {
        assert(phase==Phase::d2h && kind==cudaMemcpyDeviceToHost && d.kind==7 && s.kind==10);
        assert(stream->role==0 || stream->role==5);++counts.d2h;
    }
    const auto scheduled=phase;
    auto node=enqueue(stream,[=]{assert(phase==scheduled);std::memcpy(dest,source,bytes);});
    if(kind==cudaMemcpyHostToDevice)pending_h2d.push_back({source,bytes,node});
    return 0;
}
void bytecrt_pack_operand_impl(uint32_t*,const uint32_t*,size_t);
void bytecrt_pack_operand(uint32_t* dest,const uint32_t* source,size_t words) {
    assert(phase==Phase::h2d && locate(dest,words*sizeof(uint32_t)).kind==6);
    const auto begin=reinterpret_cast<uintptr_t>(dest),end=begin+words*sizeof(uint32_t);
    for(const auto& read:pending_h2d) {
        const auto first=reinterpret_cast<uintptr_t>(read.source),last=first+read.bytes;
        if(begin<last && first<end)assert(read.node->done);
    }
    // Execute the actual frozen io-copy.cpp helper, not a replacement pack.
    bytecrt_pack_operand_impl(dest,source,words);++counts.pack;
}
void* bytecrt_serial_init(const fherma::Point& p) {
    assert(phase==Phase::init);auto s=std::make_unique<State>(p);legacy_initializing=true;
    for(auto& stream:s->streams)check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"legacy operand");
    check(cudaStreamCreateWithFlags(&s->input_stream,cudaStreamNonBlocking),"legacy input");
    legacy_initializing=false;
    for(auto& op:s->input_ready)for(auto& event:op)check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"legacy input event");
    for(auto& event:s->copied)check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"legacy copied event");
    check(cudaEventCreateWithFlags(&s->a_forwarded,cudaEventDisableTiming),"legacy A event");
    return s.release();
}
void kernel_stream(cudaStream_t stream){assert(phase==Phase::run && (stream->role<=1 || stream->role==4));}
void encoder(unsigned N,unsigned L,unsigned P,const uint32_t* input,uint32_t* output,
             cudaStream_t stream,bool ranged,unsigned first,unsigned blocks) {
    kernel_stream(stream);const auto a=locate(input),b=locate(output);auto& s=*b.owner;
    assert((a.kind==8 || a.kind==9) && b.kind==2 && N==s.dispatch_n && L==s.plan.l);
    assert(P==(N==65536?114u:57u));
    const size_t pair=size_t(s.plan.n)*L;
    assert(a.offset%pair==0 && b.offset==(a.kind==8?0:pair));
    if(ranged){assert(stream->role==(a.kind==8?0u:1u));++counts.range_encode;}
    else {assert(stream->role==4);++counts.full_encode;}
    const size_t begin=ranged?size_t(first)*kInputCoefficients*L:0;
    const size_t words=ranged?size_t(blocks)*kInputCoefficients*L:pair;
    assert(begin+words<=pair);
    enqueue(stream,[=]{assert(phase==Phase::run);std::copy(input+begin,input+begin+words,output+begin);});
}
namespace bytecrt {
void launch_encode_mma32_transposed(unsigned N,unsigned L,unsigned P,const uint32_t* input,
    const uint8_t*,const uint32_t*,const uint64_t*,uint32_t* output,cudaStream_t stream,unsigned operands) {
    assert(operands==1);encoder(N,L,P,input,output,stream,false,0,0);
}
void launch_encode_mma32_transposed_range(unsigned N,unsigned L,unsigned P,const uint32_t* input,
    const uint8_t*,const uint32_t*,const uint64_t*,uint32_t* output,cudaStream_t stream,unsigned first,unsigned blocks) {
    encoder(N,L,P,input,output,stream,true,first,blocks);
}
namespace ntt {
constexpr unsigned kSubSize=2,kThreads=128;
template<unsigned N>void forward_first_coalesced(Launch launch,const uint32_t* input,uint32_t* output,
    const uint32_t*,const uint32_t*,const uint32_t*,unsigned P) {
    kernel_stream(launch.stream);const auto a=locate(input),b=locate(output);
    assert(a.owner==b.owner && a.kind==2 && b.kind==1 && a.offset==b.offset);
    assert(N==a.owner->dispatch_n && P==(N==65536?114u:57u) && launch.threads==128);
    const size_t words=size_t(a.owner->plan.n)*a.owner->plan.l;++counts.forward;
    enqueue(launch.stream,[=]{assert(phase==Phase::run);std::copy(input,input+words,output);});
}
template<unsigned N>void fused_middle(Launch launch,const uint32_t* ab,uint32_t* output,
    const uint32_t*,const uint32_t*,const uint32_t*,unsigned P) {
    kernel_stream(launch.stream);const auto a=locate(ab),b=locate(output);
    assert(a.owner==b.owner && a.kind==1 && b.kind==3 && N==a.owner->dispatch_n);
    assert(P==(N==65536?114u:57u) && (launch.stream->role==1 || launch.stream->role==4));
    const size_t words=size_t(a.owner->plan.n)*a.owner->plan.l;++counts.middle;
    enqueue(launch.stream,[=]{assert(phase==Phase::run);for(size_t i=0;i<words;++i)output[i]=ab[i]+ab[words+i];});
}
}
template<unsigned N,unsigned P,unsigned Kpad,bool Transposed>void inverse_weighted(Launch launch,
    const uint32_t* input,uint32_t* output,const uint32_t*,const uint32_t*,const uint32_t*,const uint32_t*) {
    kernel_stream(launch.stream);const auto a=locate(input),b=locate(output);
    assert(a.owner==b.owner && a.kind==3 && b.kind==4 && N==a.owner->dispatch_n);
    assert(P==(N==65536?114u:57u) && Kpad==(N==65536?480u:256u) && Transposed);
    const size_t words=size_t(a.owner->plan.n)*a.owner->plan.l;++counts.inverse;
    enqueue(launch.stream,[=]{assert(phase==Phase::run);std::copy(input,input+words,output);});
}
void launch_g1_crt32_transposed(unsigned W,unsigned N,const uint8_t* weighted,const uint8_t*,
    const float*,const uint32_t*,uint32_t,uint32_t* output,cudaStream_t stream,unsigned first,unsigned blocks) {
    kernel_stream(stream);const auto a=locate(weighted),b=locate(output);auto& s=*a.owner;
    assert(a.kind==4 && b.kind==10 && W==s.plan.w && N==s.plan.n && blocks==1);
    const size_t pair=size_t(N)*s.plan.l,begin=size_t(first)*kOutputCoefficients*s.plan.l;
    const size_t words=size_t(blocks)*kOutputCoefficients*s.plan.l;
    assert(b.offset%pair==0 && begin+words<=pair && b.offset+pair<=b.count);++counts.crt;
    const auto* input=reinterpret_cast<const uint32_t*>(weighted);
    enqueue(stream,[=]{assert(phase==Phase::run);std::copy(input+begin,input+begin+words,output+begin);});
}
}
'''.replace('FHERMA_HEADER', str(artifact / 'fherma.h'))

suffix = r'''
static_assert(std::is_same_v<decltype(&fherma_h2d),void(*)(void*,const fherma::Inputs&)>);
static_assert(std::is_same_v<decltype(&fherma_d2h),fherma::Outputs(*)(void*,fherma::Outputs)>);
size_t math_count(const Counts& c){return c.full_encode+c.range_encode+c.forward+c.middle+c.inverse+c.crt;}
void all_done(){for(auto stream:active_streams)assert(!stream->tail || stream->tail->done);}
void reset() {
    assert(active_streams.empty() && active_events.empty() && buffers.empty());
    pending_h2d.clear();counts={};legacy_role=global_role=0;phase=Phase::idle;
}
void check_case(unsigned B,unsigned N) {
    reset();fherma::Point p;p.B=B;p.N=N;p.W=N==65536?1747:868;p.L=N==65536?55:28;
    phase=Phase::init;auto* state=fherma_init(p);auto& batch=*static_cast<BatchState*>(state);
    assert(batch.slot_count==std::min(B,4u) && counts.allocations==3);
    assert(counts.streams==(B==1?3u:3*batch.slot_count+3));
    const size_t pair=batch.words,total=batch.total,chunks=batch.slots[0]->plan.n/kInputCoefficients;
    fherma::Inputs in;in.a.shape=in.b.shape={B,batch.slots[0]->plan.n,batch.slots[0]->plan.l};
    in.a.data.resize(total);in.b.data.resize(total);
    fherma::Outputs previous;std::vector<uint32_t> previous_bytes;
    for(unsigned call=0;call<2;++call) {
        std::vector<uint32_t> expected(total);
        for(unsigned product=0;product<B;++product)for(size_t i=0;i<pair;++i) {
            const size_t index=size_t(product)*pair+i;
            in.a.data[index]=100000*call+1000*product+i+1;
            in.b.data[index]=50000*call+100*product+3*i+2;
            expected[index]=in.a.data[index]+in.b.data[index];
        }
        const auto original_a=in.a.data,original_b=in.b.data;
        auto before=counts;phase=Phase::h2d;fherma_h2d(state,in);
        assert(math_count(counts)==math_count(before) && counts.allocations==before.allocations);
        assert(counts.h2d-before.h2d==2*B*chunks && counts.pack-before.pack==2*B*chunks);
        assert(counts.d2h==before.d2h && counts.stream_sync-before.stream_sync==1);
        assert(std::memcmp(batch.raw_a,original_a.data(),total*sizeof(uint32_t))==0);
        assert(std::memcmp(batch.raw_b,original_b.data(),total*sizeof(uint32_t))==0);all_done();
        before=counts;phase=Phase::run;auto out=fherma_run(state,in);
        assert(out.c.shape==in.a.shape && out.c.data.empty() && out.c.data.capacity()>=total);
        assert(counts.allocations==before.allocations && counts.h2d==before.h2d && counts.d2h==before.d2h && counts.pack==before.pack);
        assert(counts.full_encode-before.full_encode==(B==1?0u:2*B));
        assert(counts.range_encode-before.range_encode==(B==1?2*chunks:0u));
        assert(counts.forward-before.forward==2*B && counts.middle-before.middle==B);
        assert(counts.inverse-before.inverse==B && counts.crt-before.crt==B*chunks);
        assert(counts.device_sync-before.device_sync==1);
        assert(std::memcmp(batch.raw_c,expected.data(),total*sizeof(uint32_t))==0);all_done();
        assert(std::memcmp(batch.raw_a,original_a.data(),total*sizeof(uint32_t))==0);
        assert(std::memcmp(batch.raw_b,original_b.data(),total*sizeof(uint32_t))==0);
        before=counts;phase=Phase::d2h;out=fherma_d2h(state,std::move(out));
        assert(math_count(counts)==math_count(before) && counts.allocations==before.allocations);
        assert(counts.d2h-before.d2h==B*chunks && counts.h2d==before.h2d && counts.pack==before.pack);
        assert(out.c.shape==in.a.shape && out.c.data==expected);
        assert(in.a.data==original_a && in.b.data==original_b);all_done();
        for(const auto& entry:buffers)assert(reinterpret_cast<uintptr_t>(out.c.data.data())!=entry.first);
        if(call==0){previous_bytes=out.c.data;previous=std::move(out);}
        else {assert(previous.c.data==previous_bytes && previous.c.data.data()!=out.c.data.data());}
    }
    // A malformed shape fails before any DMA or kernels are submitted.
    const auto before=counts;in.a.shape[0]+=1;bool bad=false;phase=Phase::h2d;
    try{fherma_h2d(state,in);}catch(const std::runtime_error&){bad=true;}
    assert(bad && counts.h2d==before.h2d && math_count(counts)==math_count(before));
    phase=Phase::idle;fherma_free(state);
    assert(previous.c.data==previous_bytes && counts.streams==counts.streams_freed && counts.events==counts.events_freed);
    reset();std::cout<<"PASS B="<<B<<" dispatch-N="<<N<<": byte copies, direct product ranges, completed run and independent outputs across two calls\n";
}
int main(){check_case(1,65536);check_case(2,65536);check_case(128,65536);check_case(1,32768);}
'''

(root / 'source-check.cpp').write_text(prefix + body + suffix)
common = ['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-fopenmp']
compile_pack = subprocess.run(common + ['-Dbytecrt_pack_operand=bytecrt_pack_operand_impl',
    '-c', str(artifact / 'core/io-copy.cpp'), '-o', str(root / 'io-copy.o')],
    capture_output=True, text=True)
compile_check = subprocess.run(common + [str(root / 'source-check.cpp'),
    str(root / 'io-copy.o'), str(artifact / 'core/vector-copy.cpp'),
    '-Wl,--wrap=memcpy', '-Wl,--wrap=memmove', '-o', str(root / 'source-check')],
    capture_output=True, text=True) if compile_pack.returncode == 0 else compile_pack
(root / 'compile.log').write_text(compile_pack.stdout + compile_pack.stderr + compile_check.stdout + compile_check.stderr)
assert compile_pack.returncode == compile_check.returncode == 0, (root / 'compile.log').read_text()
run = subprocess.run([str(root / 'source-check')], capture_output=True, text=True,
                     env=dict(os.environ, OMP_WAIT_POLICY='PASSIVE'), timeout=120)
(root / 'check.log').write_text(run.stdout + run.stderr)
assert run.returncode == 0, (root / 'check.log').read_text()

def function(name):
    start = raw.index(name + '(')
    begin = raw.index('{', start)
    depth = 1
    at = begin + 1
    while depth:
        depth += (raw[at] == '{') - (raw[at] == '}')
        at += 1
    return raw[begin:at]

def uncomment(s):
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', s, flags=re.S)

callback_bodies = '\n'.join(function(name) for name in [
    'fherma_h2d', 'fherma_d2h', 'enqueue_result_copy', 'drain_result_copy', 'validate_inputs', 'input_shape'])
assert not re.search(r'<<<|launch_|cudaMemset|cudaMalloc|\.resize\(|\.assign\(', uncomment(callback_bodies))
run_body = uncomment(function('fherma_run'))
assert 'out.c.data.reserve(batch.total)' in run_body
assert not re.search(r'cudaMemcpy|cudaMemset|\.resize\(|\.insert\(|bytecrt_pack_operand|\.data\(\)', run_body)

frozen = 'aca09d65fe30e6c9497f7a0f038e9a9f4b7384df'
core_hashes = {}
for path in sorted((artifact / 'core').iterdir()):
    reference = subprocess.check_output(['git', 'show', f'{frozen}:native-batched/core/{path.name}'], cwd=repo)
    assert path.read_bytes() == reference, path
    core_hashes[path.name] = hashlib.sha256(reference).hexdigest()

ranges = []
for B, N, L in [(1, 65536, 55), (2, 65536, 55), (128, 65536, 55), (1, 32768, 28)]:
    pair = N * L
    products = [[j * pair, (j + 1) * pair] for j in range(B)]
    assert products[0][0] == 0 and products[-1][1] == B * pair
    assert all(left[1] == right[0] for left, right in zip(products, products[1:]))
    ranges.append({'shape': [B, N, L], 'pair_words': pair, 'operand_bytes': B * pair * 4,
                   'first_product_word_range': products[0], 'last_product_word_range': products[-1]})

result = {'status': 'pass', 'scope': 'small lifted actual-wrapper host method proof; mathematical kernels are mocked',
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'generated_header_sha256': hashlib.sha256((artifact / 'fherma.h').read_bytes()).hexdigest(),
          'rewritten_cuda_launch_sites': launch_count, 'reduced_dimension_dispatch_predicates': 1,
          'production_ranges': ranges, 'frozen_core_sha256': core_hashes,
          'checks': ['B1/B2/B128 row-major shape and pair ranges', 'B1 auxiliary dispatch N32768/P57/Kpad256',
                     'deferred DMA and pinned staging reuse fences', 'exact original A/B byte preservation',
                     'G1 direct raw-C product ordering', 'run returns with all mock device work complete',
                     'two consecutive calls with retained independent owned outputs after free',
                     'h2d/d2h zero mathematical launches and zero device allocations',
                     'run reserve-only empty result, no bulk host zero-fill or byte copies',
                     'exact C++ callback function types', 'all core files frozen ACA byte-identical'],
          'stdout': run.stdout, 'real_cuda_measurements': 0}
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(run.stdout, end='')
print('PASS: static callback/run audit and all frozen core bytes; result:', root / 'result.json')
