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
 checkpoint()
 compile_cmd=['g++','-O3','-mavx2','-std=c++17','-fopenmp','-Wl,--wrap=memcpy','-Wl,--wrap=memmove','cpu-nt-probe.cpp','-o','cpu-nt-probe']
 with (ROOT/'compile.log').open('w') as log:
  r=subprocess.run(compile_cmd,stdout=log,stderr=subprocess.STDOUT,timeout=min(30,TOTAL-(time.monotonic()-START)),preexec_fn=caps)
 if r.returncode: raise RuntimeError('compile returncode '+str(r.returncode))
 checkpoint(state='micro_running',compile_command=compile_cmd)
 rows=run([str(ROOT/'cpu-nt-probe'),'--micro'],'micro')
 assert len(rows)==60 and all(r['all_bytes_equal'] for r in rows)
 summary={}
 for name in sorted({r['case'] for r in rows}):
  case=[r for r in rows if r['case']==name]; paired=[]
  for pair in range(5):
   one={r['mode']:r for r in case if r['pair']==pair}
   assert set(one)=={'libc','nt'}
   assert one['libc']['transferred_bytes']==one['nt']['transferred_bytes']
   paired.append(one['nt']['elapsed_ms']/one['libc']['elapsed_ms'])
  ratio=statistics.median(paired)
  summary[name]={'libc_median_ms':statistics.median(r['elapsed_ms'] for r in case if r['mode']=='libc'),'nt_median_ms':statistics.median(r['elapsed_ms'] for r in case if r['mode']=='nt'),'paired_ratios':paired,'median_paired_ratio':ratio,'reduction_percent':100*(1-ratio),'all_bytes_equal':True}
 weights={'pack_stream':58.294586,'owned_stream':28.639138}
 weighted=sum(weights[c]*summary[c]['median_paired_ratio'] for c in weights)/sum(weights.values())
 gate={'weights_ms':weights,'weighted_ratio':weighted,'host_service_reduction_percent':100*(1-weighted),'threshold_percent':10,'passed':weighted<=.9,'scope':'2CPU ordinary-buffer streaming mechanism; native whole-call pending'}
 save('micro-summary.json',{'cases':summary,'gate':gate})
 checkpoint(state='owned_running' if gate['passed'] else 'complete_cpu_gate_failed',gate=gate)
 owned=[]
 if gate['passed']:
  owned=run([str(ROOT/'cpu-nt-probe'),'--owned'],'owned')
  assert len(owned)==7 and all(r['all_bytes_equal'] for r in owned)
  ratios=[]
  for pair in range(3):
   one={r['mode']:r for r in owned if not r['warmup'] and r['pair']==pair}
   assert set(one)=={'libc','nt'}
   ratios.append(one['nt']['elapsed_ms']/one['libc']['elapsed_ms'])
  save('owned-summary.json',{'paired_ratios':ratios,'median_paired_ratio':statistics.median(ratios),'reduction_percent':100*(1-statistics.median(ratios)),'all_bytes_equal':True,'output_bytes':owned[0]['output_bytes']})
  checkpoint(state='complete',owned_rows=len(owned))
 save('result.json',{'checkpoint':STATE,'micro_summary':summary,'gate':gate,'micro_rows':rows,'owned_rows':owned})
 print(json.dumps({'state':STATE['state'],'gate':gate,'micro_reduction_percent':{k:v['reduction_percent'] for k,v in summary.items()},'owned_rows':len(owned),'wall_seconds':time.monotonic()-START}))
except Exception as e:
 checkpoint(state='failed',error=type(e).__name__+': '+str(e)); print(json.dumps(STATE)); raise
