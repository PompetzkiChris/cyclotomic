# `cyclotomic` — exact arithmetic in ℚ(ζₙ), in Racket

Racket 9.3 [cs]. **313 tests, all passing.** Racket and CUDA C++, and nothing
else: Racket drives the GPU directly through the CUDA driver API.

```
raco pkg install --link C:\GPU\pkg-cyclotomic
raco setup --pkgs cyclotomic     # renders the Scribble docs
raco test  --package cyclotomic
```

## Why Racket

Racket's numeric tower is **exact by default**. `1/6` is the rational one
sixth, not a float that happens to print that way. Inexactness is the thing you
have to ask for, with `exact->inexact`. So the arithmetic needs no defending;
only the boundary does. The test suite makes the distinction explicit:

```racket
(check-equal? v 1/6)                       ; passes
(check-false (= v (exact->inexact 1/6)))   ; passes
```

The exact rational and the nearest double to it are different objects, and
Racket says so.

## Delay on, delay off

Requiring a module must not *do* anything. Locating a DLL touches the
filesystem, opening it loads code, and making its dependencies findable mutates
the process PATH — and `cuda/nvrtc.rkt` did all three in its module body, so a
program that required the package and never compiled a kernel still paid for it
and still had its environment changed underneath it. Measured, before:

```
require nvrtc.rkt       : 101 ms
PATH mutated by require : #t
```

Everything derived now sits behind a promise: computed at most once, on first
use, and not at all if never used.

```
require nvrtc.rkt       :   5 ms      PATH mutated by require : #f
first use               :   7 ms      PATH mutated after use  : #t
```

The same applies to the field. `Φₙ`'s power table and unit group are pure
functions of `n`, wanted often but not always, and building them eagerly made
`make-field` do `O(n · φ(n))` work for a caller that only wanted the degree:

```
n=1260 deg  288 : make 15 ms, power table on demand   0 ms
n=2520 deg  576 : make  3 ms, power table on demand   5 ms
n=5040 deg 1152 : make  9 ms, power table on demand 121 ms
```

Each table is one `delay`, so it is built at most once and every later access is
a field read — `(eq? (cyclofield-pow F) (cyclofield-pow F))` is `#t`, tested.
`tests/purity-tests.rkt` checks that requiring `nvrtc.rkt` leaves PATH alone,
that the effect appears only on first use, and that a promise forced three times
computes once.

## Layout

```
poly.rkt        exact integer polynomials; Phi_n by the divisor recursion
field.rkt       Q(zeta_n): elements, Galois, norm/trace, inverse, embeddings
matrix.rkt      exact matrices: product, adjoint, kron, unitarity
mub.rkt         mutually unbiased bases in d = 2, 3, 6
exact-io.rkt    exact clock and exact decimal formatting
cuda/
  driver.rkt    CUDA driver API via ffi/unsafe: buffers, modules, launch,
                streams, pinned memory, occupancy
  nvrtc.rkt     compile CUDA C++ from a Racket string, at run time
  kernels.cu    int64-only kernels; kernels.ptx is the compiled artifact
  gpu.rkt       exact Z[zeta_n] matrices on the device, host and device-resident
  accel.rkt     installs the device onto mat*; declines safely
tests/
  field-tests.rkt        61
  gpu-tests.rkt         121
  mub-tests.rkt          31
  accel-tests.rkt        23
  hardening-tests.rkt     9
  no-float-tests.rkt     17
  nvrtc-tests.rkt        37
  purity-tests.rkt       14
tools/          probe, bench, profile, audit, sustained, waitmode, kernelcmp
scribblings/    Scribble documentation; raco setup renders it
refcheck/       an independent CUDA C++ implementation to check against
```

## The field

`Φₙ` is computed, not tabulated, from `xⁿ − 1 = ∏_{d|n} Φ_d(x)` by exact
integer polynomial division. The suite checks `Φ₁₀₅` has a `−2` coefficient —
the first cyclotomic polynomial whose coefficients leave `{−1,0,1}`, and a
thing a lookup table would get wrong.

An element is `φ(n)` exact rationals in the power basis. Multiplication is
convolution followed by reduction through the precomputed power table.
Inversion solves `M x = e₀` by Gaussian elimination over exact rationals — no
pivoting for magnitude is needed, because there is no magnitude to lose.

```racket
(define F (make-field 24))
(field-sqrt F 6)                    ; sqrt6, exactly
(cyc-expt (field-inv-sqrt F 6) 2)   ; 1/6
(cyc-norm (field-sqrt F 2))         ; 16
```

## Dimension 6

```racket
(mutually-unbiased? (map cdr (mubs-d6)) 6)   ; #t
(unbiasedness (cdr (first (mubs-d6)))
              (cdr (second (mubs-d6))))      ; 1/6
```

ℚ(ζ₂₄) is the smallest cyclotomic field holding √2, √3, √6 and *i* at once.
d=2 bases are built in ℚ(ζ₈), d=3 in ℚ(ζ₁₂), and both are lifted into ℚ(ζ₂₄)
before tensoring. `cyc-lift` refuses when `n ∤ m`.

Three pairwise unbiased bases in dimension 6 is the standard lower bound from
tensoring. Whether a fourth exists is a long-standing open problem, and nothing
here bears on it.

## The GPU, from Racket

`cuda/driver.rkt` binds the CUDA **driver** API (`nvcuda.dll`) through
`ffi/unsafe` — deliberately the driver API rather than the runtime API: a stable
C ABI, no host compiler at run time, and it loads PTX directly.

```
device 0: NVIDIA GeForce RTX 5090
  compute capability : sm_120      multiprocessors : 170
  memory bus width   : 512 bit     VRAM 30.2 GB free of 31.8 GB
host->device->host round trip of 1024 int64s: IDENTICAL
```

A matrix over ℚ(ζₙ) is `φ(n)` int64 coefficient planes over a common
denominator. Every product is bounds-checked **before** launch, from operand
magnitudes measured on the device by a reduction kernel, and audited after.
int64 does not round, it wraps — and a wrapped result still looks valid, which
is why it raises instead.

### Put it on the math path

`mat*` consults `current-mat*-hook`; requiring `cuda/accel.rkt` installs one
that clears denominators, runs the integer product on the device, and puts the
denominator back. Nothing downstream is rewritten:

```racket
(gpu-accel-install!)
(mutually-unbiased? (map cdr (mubs-d6)) 6)   ; 6 device products, 390 launches
```

The hook **declines** rather than fails — wrong field, past the int64 bound, no
device — returning `#f` so `mat*` falls through to `mat*/cpu`. A GPU problem can
slow this down; it cannot change an answer, and `accel-tests.rkt` checks both
paths agree entrywise.

`gpu-stats` makes "did it run on the card" a number rather than a belief.

### Four kernels, measured against each other

`current-gpu-kernel` selects; `'auto` is the default and picks on operand
width. All four are exact, all four agree entrywise, and all four are checked
against the independent CUDA C++ reference — a fast path checked only against
itself is not checked.

Two were wrong bets, kept because they are correct and because they make the
winner's margin measurable rather than asserted:

**`'fused`** — one launch for the whole field product instead of `deg²`, every
plane staged in shared memory. Cuts global traffic by a factor of `deg` and is
**slower for it**: 32 KB of shared memory per block collapses occupancy, and
that costs more than the traffic it saves. Measured 0.3–0.6x.

**`'rb`** — 64×64 output tile, 4×4 outputs per thread, eight times fewer shared
reads per multiply-add. Bought **1.0–1.1x**, which is the useful result: the
kernel is not starved of memory.

**`'w32`** — it is starved of integer throughput. There is no native 64×64
integer multiply on this hardware; `mul.lo.s64` is synthesised from several
32-bit operations. But `mul.wide.s32` is a single instruction, and every operand
is already bounds-checked, so when `|A|,|B| < 2³¹` the planes are narrowed to
int32 and the product runs one IMAD per multiply-add while accumulating in full
64-bit width. The PTX carries 275 `mul.wide.s32`. Nothing is rounded: the
accumulator stays int64 and the bound that authorised the launch still holds.

```
field      n     split   fused      rb     w32   best
Q(z8)    512      19      10       6       8    rb   3.0x
Q(z8)   1024      79     119      84      81    split 1.0x
Q(z8)   2048     252     546     251     232    w32  1.1x
Q(z8)   3072     536    1524     474     420    w32  1.3x
Q(z24)   512      11      29      13      10    w32  1.1x
Q(z24)  1024      97     261     104      92    w32  1.1x
Q(z24)  2048     503    1786     474     406    w32  1.2x
Q(z24)  3072    1343    5493    1127     848    w32  1.6x
```

### Feeding it

Profiling a 1024×1024 product put the device at **6%** of wall time; the rest
was building and tearing down `cyc` structs. Two fixes:

- `zmat-build` / `zmat-coeff` read and write plane bytes directly, skipping
  field-element structs for bulk data. Readback at n=1024: 594 ms → 75 ms.
- `dmat` keeps a matrix **resident on the device** across a chain of products,
  so nothing returns to the host between them.

| | rate | device share of wall |
|---|---|---|
| round-trip every product | 1160 Giga-int-ops/s | 14% |
| device-resident chain | **17894 Giga-int-ops/s** | **95%** |

24 chained 3072×3072 exact products over ℤ[ζ₂₄], 576 MB per matrix on the card:
peak 100% utilisation, **580 W** against a 575 W cap, 2917 MHz, 68 °C.

### Hardening

Found by writing tests that try to break it rather than tests that confirm it
works.

**Device memory on the error path.** `cuMemAlloc` is not GC-managed, so any
early exit leaked VRAM until the process died. `call-with-device-buffer` /
`call-with-device-buffers` wrap allocation in `dynamic-wind`, checked on all
three exits: normal return, a raise inside, and an escape continuation.

**A blocking driver call froze the whole runtime.** `#:blocking? #t` lets the
GC proceed; it does *not* let other Racket threads run, because Racket CS
schedules every green thread on one OS thread. Measured: a ticker thread
advanced **0** times across a 48 ms product. Fixed by not blocking — record a
`CUevent` and poll it with `(sleep 0)`. The same test now shows tens of
thousands of ticks during the call.

The cost belongs to the caller, so it is a parameter:

```racket
(current-gpu-wait 'poll)   ; default: other threads keep running
(current-gpu-wait 'block)  ; cuCtxSynchronize, maximum throughput
```

Polling costs about 2x on a short kernel and nothing on a long one.

**Immobile transfer buffers.** With the calls marked blocking, the GC may move
a byte string while the driver is reading it. Transfers stage through
`'atomic-interior` memory. The trap: that memory is GC-managed and must **not**
be passed to `free` — doing so is heap corruption, not a leak. It cost a
`0xC0000374` crash to learn.

**A bad module says why.** `cuModuleLoadDataEx` with the JIT log wired up:

```
load-ptx: CUDA error 200 loading PTX: device kernel image is invalid
JIT error log:
error   : Can't load this binary kind, as it's not recognized
```

Also `cuDevicePrimaryCtxRetain` instead of `cuCtxCreate`, so the context is
shared with every other CUDA library in the process rather than competing with
it; a compute-capability check before module load; the Φₙ reduction table
cached per field instead of re-uploaded per product; and one reused sync event.

## Write a CUDA kernel in Racket and run it

`cuda/nvrtc.rkt` binds NVRTC, so a kernel can be written as a Racket string,
compiled for the device that is actually present, and launched. No nvcc, no
build step, no host compiler:

```racket
(define-values (maj min) (cuda-compute-capability 0))
(define ptx
  (compile-cuda #<<EOF
extern "C" __global__ void scale_add(long long* v, long long s, long long a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = v[i] * s + a;
}
EOF
                #:arch (device-arch-string maj min)))
(define k (module-function (load-ptx ptx) "scale_add"))
```

A compile failure raises with NVRTC's own log attached, so the error names the
line and the reason:

```
NVRTC said: kernel.cu(1): error: "this" may only be used inside a nonstatic member function
```

Also bound: CUDA streams, page-locked host memory (`pinned-alloc`), async
transfers on a stream, and `max-potential-block-size`, which asks the driver
what block size keeps a kernel occupied instead of hardcoding one:

```
occupancy suggests grid 340, block 768
```

## No floating point, and no float ever involved

The mathematics never touched a float, but the *timing* and the *reporting*
did: `current-inexact-milliseconds` returns a flonum, and every "x.y GB" and
"n Giga-ops/s" was computed in floating point. That made the claim true only of
the parts one happens to care about, which is not what the claim says.

`exact-io.rkt` fixes it: clocks read `current-milliseconds` (exact integer),
ratios stay exact rationals, decimals are rendered by integer arithmetic, and
`dec` raises on a flonum rather than rendering it. Every `sleep` interval is an
exact rational.

`tests/no-float-tests.rkt` enforces it, scanning every source file for flonum
literals, exponent notation, `current-inexact-milliseconds`, `exact->inexact`
and `real->double-flonum`, with strings and comments stripped first:

```
scanned 25 .rkt files
.f32/.f64 in PTX : 0
mul.wide.s32     : 275
```

The only lines carrying a `FLOAT-OK` marker exist to prove a float is rejected
or distinguishable:

```racket
(make-cyc F (list 0.5))          ; must raise
(= v (exact->inexact 1/6))       ; must be false
```

### Runtime compilation does not open a door

NVRTC compiles whatever CUDA C++ it is handed, floats included, so runtime
compilation would have made "no floating point" a property of what the caller
writes rather than of this package. There is no escape hatch.

A PTX check alone would only say *no float instruction executes*. The claim
here is stronger — **no float is ever involved** — because NVRTC constant-folds
a float expression whose operands are compile-time known and emits an integer,
so a float can be used without leaving a trace in the PTX.

Two things could hide one from a scan of the source. Both are closed.

**The preprocessor is refused outright.** A `#define` could rename `float` to
anything and an `#include` could bring in a header full of them. So the source
handed to `compile-cuda` *is* the translation unit that gets compiled. NVRTC
has no standard headers to include anyway.

**Every float-producing name in CUDA carries one of a small set of
substrings** — `float`, `double`, `half`, `bfloat`, `fp16`, `fp8`, `tf32`,
`_Float`, `__fp16` — including the conversion intrinsics that manufacture a
float out of an integer bit pattern with no literal in sight, like
`__int_as_float` and `__int2half_rn`. Those are rejected case-insensitively,
along with every way of writing a float literal. All of these are refused:

```
float          double           __half          __nv_bfloat16
__int_as_float __int2half_rn    1.5             .5
1e9            0x1.8p3          #define         #include
```

Comments and string literals are stripped first, so a comment mentioning
`float`, or prose containing `1.5`, is not a hit, and a clean integer kernel
passes untouched. Both are tested.

And `load-ptx` runs the PTX check on every image it is given, so PTX from a
file, another toolchain or a string cannot bring floating point in either:

```
load-ptx: refusing floating point: this PTX contains 1 float construct (.f32).
```

So a float cannot be named, written as a literal, manufactured from an integer,
smuggled in behind a macro, or loaded as someone else's PTX.

## An independent implementation to check against

The Racket GPU path was only ever checked against Racket — same repository,
same kernel, same author. `refcheck/refcheck.cu` is a second implementation
written to disagree if anything is wrong:

* `Phi_n` recomputed in C++ from `x^n − 1 = ∏_{d|n} Φ_d(x)`, not ported.
* Its own power table.
* A deliberately different kernel: one thread per output **element**, whole
  convolution inline, no shared-memory tiling and no scatter. A mistake in the
  Racket kernel's tiling or its `R` scatter would not be repeated here.
* A CPU oracle accumulating in **128 bits**, so a product that silently wrapped
  int64 is detected rather than matched. MSVC has no `__int128`; it is built
  from `_umul128` with manual carry, and writing that surfaced a real sign bug
  the selftest caught immediately — overflow reported on a 1×1 product.

```
nvcc -O3 -arch=sm_120 -std=c++17 refcheck.cu -o refcheck.exe
refcheck.exe selftest
racket refcheck/crosscheck.rkt
```

Racket writes both operands as `.zmat` files, the C++ binary computes the
product independently, and the coefficients are compared. Nothing is shared but
the file format.

```
field   size     bits  cpp_ok  match
Q(z6)   96x96    9     ok      ALL 4 KERNELS IDENTICAL
Q(z8)   96x96    9     ok      ALL 4 KERNELS IDENTICAL
Q(z12)  96x96    9     ok      ALL 4 KERNELS IDENTICAL
Q(z24)  96x96    9     ok      ALL 4 KERNELS IDENTICAL
   ... 20 cases across four fields, sizes 1 to 96 ...

refcheck cmp on our output: IDENTICAL: 8192 coefficients
```

## Status

Two languages, and a test that keeps it that way: `no-float-tests.rkt` walks the
tree and fails on any source file that is not `.rkt`, `.cu`, `.ptx` or `.md`,
naming any script in another language outright. A helper in some third language
is exactly what creeps in when nobody is checking, and then it is a dependency.

Done: the field, matrices, MUB verification, the CUDA driver binding, NVRTC,
streams and pinned memory, four measured kernels, the accelerator on the math
path, device-resident chaining, the no-float enforcement, and an independent
C++ cross-check. 313 tests.

Not done, stated rather than hidden: the planes are still *stored* as int64, so
global read traffic is unchanged and only the shared tiles and the multiply got
narrower — storing narrow planes natively would halve global traffic again.
And there is no RNS path here yet, so a ℤ[ζₙ] product past the int64 bound
raises instead of splitting across primes.
