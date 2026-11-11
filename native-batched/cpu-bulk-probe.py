#!/usr/bin/env python3
"""CPU-only NT mechanism gate: existing primitive, 2 CPUs / 4 GiB / 180 s."""
import datetime,json,os,pathlib,resource,statistics,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)
START=time.monotonic(); TOTAL=180
AFFINITY=sorted(os.sched_getaffinity(0))[:2]
STATE={'state':'compiling','selected_affinity':AFFINITY,'address_space_limit_bytes':4*1024**3,'total_wall_limit_seconds':TOTAL,'native_jobs':0}
def save(name,data): (ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
def caps():
 os.sched_setaffinity(0,AFFINITY)
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
 resource.setrlimit(resource.RLIMIT_CPU,(180,180))
def checkpoint(**kw):
 STATE.update(kw,total_wall_seconds=time.monotonic()-START); save('checkpoint.json',STATE)
def run(cmd,name):
 with (ROOT/(name+'.jsonl')).open('w') as out,(ROOT/(name+'.log')).open('w') as err:
  r=subprocess.run(cmd,stdout=out,stderr=err,timeout=max(1,TOTAL-(time.monotonic()-START)),preexec_fn=caps,env={**os.environ,'OMP_DYNAMIC':'FALSE'})
 if r.returncode: raise RuntimeError(name+' returncode '+str(r.returncode))
 return [json.loads(x) for x in (ROOT/(name+'.jsonl')).read_text().splitlines() if x]
def checked(cmd):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=10)
 return {'returncode':r.returncode,'output':r.stdout}
try:
 save('environment.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'affinity':AFFINITY,'cpu':checked(['lscpu']),'compiler':checked(['g++','--version']),'libc':checked(['ldd','--version']),'meminfo':pathlib.Path('/proc/meminfo').read_text()})
 compile_cmd=['g++','-O3','-mavx2','-std=c++17','-fopenmp','-Wl,--wrap=memcpy','-Wl,--wrap=memmove','cpu-bulk-probe.cpp','-o','cpu-bulk-probe']
 checkpoint(compile_command=compile_cmd)
 with (ROOT/'compile.log').open('w') as log:
  r=subprocess.run(compile_cmd,stdout=log,stderr=subprocess.STDOUT,timeout=min(30,TOTAL-(time.monotonic()-START)),preexec_fn=caps)
 if r.returncode:raise RuntimeError('compile returncode '+str(r.returncode))
 checkpoint(state='running')
 rows=run([str(ROOT/'cpu-bulk-probe')],'observations')
 assert len(rows)==14 and all(x['all_bytes_equal'] for x in rows)
 summary={};pairs={}
 for phase in ['pack','owned']:
  paired=[];pairs[phase]=[]
  for pair in range(3):
   one={r['mode']:r for r in rows if r['phase']==phase and not r['warmup'] and r['pair']==pair}
   assert set(one)=={'chunk','bulk'} and one['chunk']['bytes']==one['bulk']['bytes']
   pairs[phase].append(one);paired.append(one['bulk']['elapsed_ms']/one['chunk']['elapsed_ms'])
  summary[phase]={'paired_ratios':paired,'median_paired_ratio':statistics.median(paired),'reduction_percent':100*(1-statistics.median(paired)),'chunk_median_ms':statistics.median(x['chunk']['elapsed_ms'] for x in pairs[phase]),'bulk_median_ms':statistics.median(x['bulk']['elapsed_ms'] for x in pairs[phase])}
 aggregate=[sum(pairs[p][i]['bulk']['elapsed_ms'] for p in pairs)/sum(pairs[p][i]['chunk']['elapsed_ms'] for p in pairs) for i in range(3)]
 gate={'scope':'2CPU ordinary-buffer B128 two-full-operand pack plus one full standard-owned vector; NT common backend; DMA/driver effect belongs to native','aggregate_paired_ratios':aggregate,'aggregate_reduction_percent':100*(1-statistics.median(aggregate)),'CPU_copy_threshold_percent':10,'CPU_copy_gate_passed':statistics.median(aggregate)<=.9}
 checkpoint(state='complete',rows=len(rows),gate=gate)
 save('summary.json',{'phases':summary,'gate':gate});save('result.json',{'checkpoint':STATE,'summary':summary,'gate':gate,'rows':rows})
 print(json.dumps({'checkpoint':STATE,'summary':summary},indent=2))
except Exception as e:
 checkpoint(state='failed',error=type(e).__name__+': '+str(e));print(json.dumps(STATE));raise
