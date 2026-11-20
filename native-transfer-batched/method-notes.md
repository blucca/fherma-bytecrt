# Exactness and throughput prerequisites for a batched NTT port

The retained byte-copy implementation evaluates 128 independent negacyclic
products at `N=65536, W=1747, L=55`. Its [native result](native-results.json)
records **86.236234 ms for `run`**, with H2D and D2H measured separately.
The following is source analysis and a porting example; its timing models use
the measurement scopes stated below.

## Batch the dispatch; account for the same arithmetic

Each product currently submits two encoders, two forward passes, a fused
spectral middle, an inverse/weighting pass, and four CRT chunks. These ten
launches contain **122,880 CTAs per product**, or **15,728,640 per B128 call**.
Each NTT pass already has 29,184 CTAs; each encoder has 2,048.

A product dimension in the grid can combine these dispatches. The cuPQC staged
collective still executes on one complete block, using that product's pointers
and that prime's constants. `N` remains 65,536; the batch dimension selects an
independent polynomial ring.

An explicit lifetime plan can share one FixedPoint and three residue planes per
product: reuse the encoder plane for A, B and the spectral result; reuse the
two forward-output planes for weighted CRT input after the middle consumes
them. This plan uses **17.132 GB of device storage**, including raw A/B/C.
Reused weighted storage requires reconstructed zero alpha/padding planes.
These are allocation and lifetime calculations, with runtime peak memory and
throughput belonging to a future native implementation.

The original stage boundaries still move **51.640 GB of logical mutable data
per B128 call**, before fixed-table reads, sectors and spills. An inverse-zeroing
version of the lifetime plan adds 0.201 GB of padding stores. Under a model
where that traffic reaches DRAM at an effective 2 TB/s, its service is
**25.820 ms**. Cache residency and arithmetic/memory overlap belong in the
complete measurement. Grid concatenation changes dispatch and locality while
retaining the per-CTA NTT and MMA work.

## Tensor NTT: match the comparison and the modulus

The encoder and G1 CRT in this implementation already use exact `u8 × u8 → s32`
tensor instructions. The current NTT uses cuPQC's runtime-prime SIMT stages.

[TensorFHE](https://arxiv.org/abs/2212.14191) and
[WarpDrive](https://zhengfangyu.github.io/assets/papers/HPCA2025.pdf)
describe integer digit-sliced tensor NTTs. WarpDrive's radix-16 warp/register
method is particularly relevant. Its `N=65536` experiment uses an A100 and
36 primes with a 974-bit total modulus: about 27.06 bits per prime.
The reported **9.7× NTT** comparison is against TensorFHE. Relative to
WarpDrive's own optimized CUDA-butterfly variant, tensor execution improves
4–10%, with mixed CUDA/tensor execution adding 4–7%. Those comparisons use
the paper's workloads and hardware; the paper supplies a mechanism to study
alongside our cuPQC pipeline.

Our RNS primes occupy the **31-bit** domain. A radix-16 matrix dot therefore
needs a reconstruction that handles sums wider than `uint64_t` before modular
reduction. Each individual byte-pair accumulator can fit `s32` while their
combined, shifted integer sum needs additional bits.

### A valid 16-point row with a 65-bit integer dot

Take the standard NTT prime `p = 2013265921 = 15·2^27 + 1`. It also supports
`2N=131072`. The eighth row of a 16-point NTT has canonical weights
`(1, p−1, 1, p−1, …)`. With all sixteen input residues equal to `p−1`,

```text
unreduced dot = 8·p·(p−1) = 32425917333173698560
exact residue modulo p  = 0
uint64-wrapped residue  = 841097758
```

Run the independent integer example:

```sh
python3 check-radix16-dot.py
```

It checks the root, modulus domain and both residues using Python integers.
This establishes a width requirement for a 31-bit port. WarpDrive's published
workload and this wider-domain example have their respective input scopes.

## Measure the entire port

A 26.733948 ms B128 engineering budget allows **208.859 µs per product**.
At the retained **B1 first-warm CRT service of 170.592 µs**, holding that rate
fixed leaves **38.267 µs** for both encoders and the new NTT. This is an explicit
B1-to-B128 planning assumption, separate from the measured B128 `run` result.

The next complete arithmetic port therefore needs stage budgets for encoder,
NTT and CRT together, exact reconstruction for the actual primes, and a
fresh full-call measurement. Its byte-copy callbacks preserve original input
and output bytes; RNS conversion, every transform, carry and canonical
reduction remain inside `run`, which completes device work before returning.

Reference: WarpDrive, HPCA 2025,
[DOI 10.1109/HPCA61900.2025.00091](https://doi.org/10.1109/HPCA61900.2025.00091).
