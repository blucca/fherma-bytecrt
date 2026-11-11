#!/usr/bin/env python3
"""Bounded CPU check of byte-identical generated main and full owned tensors."""
import array, hashlib, json, os, pathlib, resource, shutil, subprocess, sys, time
SOURCE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path.cwd()/'temp'/'bytecrt-harness-heap'
ROOT.mkdir(parents=True,exist_ok=True)
shutil.copy2(SOURCE/'solve.cpp',ROOT/'solve.cpp')
B, N, L, CASES = 128, 2048, 33, 4
AS_BYTES = 256 * 1024 * 1024
TOTAL_TIMEOUT = 60
started = time.monotonic()
receipt = {'status': 'started', 'shape': {'B': B, 'N': N, 'L': L, 'W': 64}, 'cases': CASES,
           'tensor_bytes': B*N*L*4, 'as_limit_bytes': AS_BYTES, 'total_timeout_s': TOTAL_TIMEOUT,
           'scope': 'CPU scaled generated-main allocator lifecycle; fake XOR arithmetic, not full-native correctness/performance gate',
           'runs': []}
def save():
    receipt['elapsed_s'] = time.monotonic() - started
    (ROOT/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
def remain():
    return max(0.1, TOTAL_TIMEOUT - (time.monotonic() - started))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()
def limit():
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
def make_fixture():
    base=ROOT/'fixture'; base.mkdir(exist_ok=True)
    (base/'point').mkdir(exist_ok=True)
    (base/'manifest.json').write_text(json.dumps({'point': {'B':B,'N':N,'W':64,'L':L}, 'cases':CASES})+'\n')
    q=array.array('I', [2147483647]*L)
    with open(base/'point'/'q.bin','wb') as f:q.tofile(f)
    # Distinct deterministic per-word data; streaming blocks avoid Python-scale peak memory.
    block_words=65536; count=B*N*L
    for name, mul, mask in [('a',2654435761,0x13579bdf),('b',2246822519,0x2468ace0)]:
        with open(base/(name+'.bin'),'wb') as f:
            for start in range(0,count,block_words):
                words=array.array('I', (((i*mul)&0xffffffff)^mask for i in range(start,min(start+block_words,count))))
                words.tofile(f)
    for i in range(CASES):
        case=base/'cases'/f'{i:06d}';case.mkdir(parents=True, exist_ok=True)
        for name in ['a','b']:
            dst=case/(name+'.bin')
            if dst.exists():dst.unlink()
            os.link(base/(name+'.bin'),dst)
    return base

def verify(base):
    count=B*N*L; detail=[]
    for i in range(CASES):
        path=base/'out'/f'{i:06d}'/'c.bin'; checked=0; bad=None
        if not path.exists():
            detail.append({'case':i, 'status':'missing'});continue
        with open(base/'a.bin','rb') as a, open(base/'b.bin','rb') as b, open(path,'rb') as c:
            while True:
                aa=a.read(262144);bb=b.read(262144);cc=c.read(262144)
                if not aa:break
                va=array.array('I');vb=array.array('I');vc=array.array('I')
                va.frombytes(aa);vb.frombytes(bb);vc.frombytes(cc)
                if len(vc)!=len(va):bad={'word':checked, 'reason':'length'};break
                for j,(x,y,z) in enumerate(zip(va,vb,vc)):
                    if z != x^y:bad={'word':checked+j,'expected':x^y,'actual':z};break
                if bad:break
                checked+=len(va)
            extra=c.read(1)
        detail.append({'case':i,'status':'pass' if bad is None and checked==count and not extra else 'failed',
                       'checked_words':checked,'file_bytes':path.stat().st_size,'mismatch':bad})
    return detail

save()
try:
    origin=SOURCE.parent
    for name in ['main.cpp','fherma.h']:shutil.copy2(origin/name,ROOT/name)
    receipt['generated_sources']={name:{'source_sha256':sha(origin/name),'copy_sha256':sha(ROOT/name),
        'byte_identical':(origin/name).read_bytes()==(ROOT/name).read_bytes()} for name in ['main.cpp','fherma.h']}
    if not all(x['byte_identical'] for x in receipt['generated_sources'].values()):raise RuntimeError('generated source identity failed')
    command=['g++','-std=c++17','-O2','-Wall','-Wextra','main.cpp','solve.cpp','-o','solution']
    build=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=remain())
    (ROOT/'build.log').write_text(build.stdout+build.stderr)
    receipt['build']={'command':command,'returncode':build.returncode}
    save()
    if build.returncode:raise RuntimeError('build failed')
    base=make_fixture();receipt['fixture_input_sha256']={n:sha(base/(n+'.bin')) for n in ['a','b']};save()
    for mode in ['default','mallopt']:
        out=base/'out'
        if out.exists():shutil.rmtree(out)
        env=os.environ.copy();env['BATCH_HEAP_MODE']=mode
        runstart=time.monotonic()
        try:
            p=subprocess.run([str(ROOT/'solution'),str(base)],env=env,capture_output=True,text=True,
                timeout=remain(),preexec_fn=limit)
            log=p.stdout;err=p.stderr;code=p.returncode;timeout=False
        except subprocess.TimeoutExpired as e:
            log=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout or ''
            err=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr or ''
            code=None;timeout=True
        (ROOT/(mode+'.jsonl')).write_text(log);(ROOT/(mode+'.stderr')).write_text(err)
        entry={'mode':mode,'returncode':code,'timeout':timeout,'wall_s':time.monotonic()-runstart,
               'observations':[json.loads(x) for x in log.splitlines() if x.strip().startswith('{')]}
        if (out/'results.json').exists():
            entry['generated_results']=json.loads((out/'results.json').read_text())
            shutil.copy2(out/'results.json',ROOT/(mode+'-results.json'))
        entry['verification']=verify(base)
        receipt['runs'].append(entry);save()
        # Preserve every full output from both allocator modes for independent verification.
        if out.exists():
            previous=ROOT/(mode+'-out')
            if previous.exists():shutil.rmtree(previous)
            out.rename(previous)
    receipt['status']='pass' if all(r['returncode']==0 and not r['timeout'] and
        len(r.get('generated_results',{}).get('cases',[]))==CASES and
        all(c['status']=='ok' for c in r.get('generated_results',{}).get('cases',[])) and
        all(v['status']=='pass' for v in r['verification']) for r in receipt['runs']) else 'failed'
except Exception as e:
    receipt['status']='failed';receipt['error']=str(e)
finally:
    save()
print(json.dumps(receipt,indent=2))
sys.exit(0 if receipt['status']=='pass' else 1)
