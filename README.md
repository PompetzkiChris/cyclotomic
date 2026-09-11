# `cyclotomic` — exact arithmetic in ℚ(ζₙ), in Racket

Racket 9.3 [cs]. **3074 tests, all passing.** Racket and CUDA C++, and nothing
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

The same applies to the field. The power table and the unit group are pure
functions of `n`, wanted often but not always, and building them eagerly made
`make-field` do `O(n · φ(n))` work for a caller that only wanted the degree.

`Φₙ` itself turned out to be one of them, and the larger one. It is needed to
reduce and for nothing else — the degree is `φ(n)`, which Euler's totient gives
from the factorisation of `n` without dividing any polynomials. At `n = 2520`
computing `Φₙ` cost 50 ms and the power table built from it cost 46, so a caller
who asked only for the degree was paying the larger of the two. Behind a promise
it costs nothing.

Each is one `delay`, so it is built at most once and every later access is a
field read — `(eq? (cyclofield-pow F) (cyclofield-pow F))` is `#t`, tested.

The test is a fact rather than a stopwatch. `promise-forced?` answers directly,
on any machine however busy, what a millisecond threshold can only guess at:

```racket
(define F (make-field 2520))                   ; phi(2520) = 576
(field-degree F)                               ; 576
(promise-forced? (cyclofield-phi-p F))         ; #f -- Phi_n not computed
(promise-forced? (cyclofield-pow-p F))         ; #f -- no power table
(void (cyclofield-pow F))
(promise-forced? (cyclofield-phi-p F))         ; #t -- and now both
```

`tests/purity-tests.rkt` also checks that requiring `nvrtc.rkt` leaves PATH
alone and that the effect appears only on first use — in a **subprocess whose
PATH has had the toolkit stripped first**, because on a machine where the CUDA
installer already put the toolkit on PATH, checking it in this process tests the
environment rather than this package.

## Layout

```
poly.rkt        exact integer polynomials; Phi_n by the divisor recursion
field.rkt       Q(zeta_n): elements, Galois, norm/trace, inverse, embeddings
karatsuba.rkt   Karatsuba for the cyclotomic convolution; the exact matrix that
                collapses its recombination and the Phi_n reduction into one
matrix.rkt      exact matrices: product, adjoint, kron, unitarity
mub.rkt         mutually unbiased bases in d = 2, 3, 6
exact-io.rkt    exact clock and exact decimal formatting
cuda/
  driver.rkt    CUDA driver API via ffi/unsafe: buffers, modules, launch,
                streams, pinned memory and pinned transfers, occupancy
  nvrtc.rkt     compile CUDA C++ from a Racket string, at run time
  kernels.cu    integer-only kernels; kernels.ptx is the compiled artifact
  gpu.rkt       exact Z[zeta_n] matrices on the device, host and device-resident
  accel.rkt     installs the device onto mat*; declines safely
tests/
  karatsuba-tests.rkt  2758
  gpu-tests.rkt         121
  field-tests.rkt        61
  nvrtc-tests.rkt        37
  mub-tests.rkt          31
  accel-tests.rkt        23
  no-float-tests.rkt     17
  purity-tests.rkt       17
  hardening-tests.rkt     9
tools/          probe, bench, profile, audit, sustained, waitmode, kernelcmp,
                breakdown
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

### Six kernels, measured against each other

`current-gpu-kernel` selects; `'auto` is the default and picks on degree and
operand width. All six are exact, all six agree entrywise, and all six are
checked against the independent CUDA C++ reference — a fast path checked only
against itself is not checked.

Two were wrong bets, kept because they are correct and because they make the
winner's margin measurable rather than asserted:

**`'fused`** — one launch for the whole field product instead of `deg²`, every
plane staged in shared memory as int64. Cuts global traffic by a factor of `deg`
and is **slower for it**: 32 KB of shared memory per block collapses occupancy,
and that costs more than the traffic it saves. Measured 0.1–0.3x.

**`'rb`** — 64×64 output tile, 4×4 outputs per thread, eight times fewer shared
reads per multiply-add. Bought **1.0–1.1x**, which is the useful result: the
kernel is not starved of memory.

**`'w32`** — it is starved of integer throughput. There is no native 64×64
integer multiply on this hardware; `mul.lo.s64` is synthesised from several
32-bit operations. But `mul.wide.s32` is a single instruction, and every operand
is already bounds-checked, so when `|A|,|B| < 2³¹` the planes are narrowed to
int32 and the product runs one IMAD per multiply-add while accumulating in full
64-bit width. Nothing is rounded: the accumulator stays int64 and the bound that
authorised the launch still holds.

**`'ultra`** — `'w32` still left three things on the table, all three visible in
the device's own numbers rather than guessed at. This machine reports 170 SMs,
1536 threads and 64 K registers per SM, shared memory in carve-outs of
{0, 8, 16, 32, 64, 100} KB, and 96 MiB of L2:

1. `deg²` launches. ℚ(ζ₂₄) pays 64, and each re-reads a whole A plane and a
   whole B plane out of L2.
2. `C` is accumulated with `+=`, so every launch reads it back and writes it
   again. At n = 1024 that is more traffic than the operands.
3. A 64×64 output tile is 256 blocks at n = 1024. A device with 170 SMs and room
   for 24 blocks on each cannot be filled by 256 blocks.

One launch, a 16×16 tile — 4096 blocks at n = 1024 — and `C` written exactly
once fixes all three. The tile can be that small because **the convolution
supplies its own arithmetic intensity**: one output element per thread already
does `deg²` multiply-accumulates against only `2·deg` shared reads, so `deg/2`
multiply-adds per read with no output blocking at all. The price is registers:
the raw convolution needs `2·deg−1` 64-bit accumulators, 30 of them at degree 8,
which is why the tile is 16×16 and one element per thread rather than 64×64 and
sixteen — sixteen outputs would want 480 registers against a hardware cap of
255. `deg` is a template parameter and not an argument, so there is one compiled
kernel per degree and every loop unrolls; ptxas reports **zero spills** at every
degree from 1 to 16.

**`'kara`** — and then the measurement says what to do next. Issuing only *some*
of the 64 plane products (deliberately wrong arithmetic, purely to time it)
gives 0.55 ms per product on top of 10 ms of fixed cost at n = 2048, and nothing
about unroll factor or occupancy target moves it. The kernel is multiply-bound,
so the thing to cut is multiplications — which is a question about the algebra,
not about CUDA.

Karatsuba does a length-`2H` convolution with three length-`H` ones:

```
Z₀ = a₀b₀,  Z₂ = a₁b₁,  Z₁ = (a₀+a₁)(b₀+b₁) − Z₀ − Z₂
ab = Z₀ + X·Z₁ + X²·Z₂
```

Recursively, `deg = 2^L` costs `3^L` multiplications: **27 instead of 64** at
degree 8, 9 instead of 16 at degree 4, 81 instead of 256 at degree 16. Exactly —
every coefficient of the result is the integer it would have been, because every
intermediate is an integer sum of integers. There is nothing here to round.

Only the forming of the `3^L` products happens on the device. The recombination
is linear, so [`karatsuba.rkt`](karatsuba.rkt) collapses it — together with the
`Φₙ` reduction — into a single `deg × 3^L` integer matrix `K` with
`out_t = Σⱼ K[t][j]·Pⱼ`. The device applies `K` once per output element and
unwinds no recursion of its own, so no algebra lives in CUDA that can disagree
with the algebra in Racket. The matrix is derived by folding basis vectors and
is checked against `cyc*` — the ordinary field multiplication the rest of the
package uses — in `tests/karatsuba-tests.rkt`.

The price is range, and it is paid in a checked certificate rather than hoped
for. Each multiplicand is a sum of up to `2^L` coefficients, so the host must
see `2^L·max|A|` and `2^L·max|B|` inside int32; and the output bound becomes
`rowsum(K)·4^L·k·maxA·maxB`, where `rowsum` is the **exact** maximum row sum of
`|K|` — 18 for ℚ(ζ₂₄), computed rather than estimated. When that does not fit
int64 the planner chooses `'ultra` instead, which has the smaller bound and the
larger multiply count. Nothing is ever truncated to make it fit.

Operands already on the card, so this is arithmetic and device memory and
nothing else:

```
field      n   split  fused     rb    w32  ultra   kara     best
Q(z8)   1024      6     48      6      4      3      3   ultra 2.0x
Q(z8)   2048     40    345     35     18     15     15   ultra 2.7x
Q(z8)   3072    134   1161    102     49     50     48   kara  2.8x
Q(z24)   512      3     23      6      4      1      1   ultra 3.0x
Q(z24)  1024     21    183     23     12      8      6   kara  3.5x
Q(z24)  2048    161   1374    139     65     55     39   kara  4.1x
Q(z24)  3072    538   4611    414    185    180    126   kara  4.3x
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

Then the kernel got fast enough that the kernel stopped being the problem. Split
one ℚ(ζ₂₄) 1024×1024 product into stages and 82 of its 94 ms were host traffic,
with the device accounting for 12:

- **The operand magnitudes were measured on the device.** Two reduction passes
  over device memory and two synchronisations, after uploading at double width
  to have something to measure — to learn a number the matrix knew when it was
  built. A `zmat` now carries its own exact `absmax` as a promise, filled in by
  the pass that wrote the coefficients. Two launches and two round trips gone.
- **The operands crossed PCIe at double width and were narrowed on the card.** A
  `zmat` now also carries its int32 image, written in the same pass, and it is
  uploaded directly: half the bytes, and both narrowing launches gone. It is a
  promise, so a caller who only reads coefficients back never pays for it, and
  it is `#f` rather than truncated when a coefficient does not fit int32.
- **Every transfer was unpinned**, which makes the driver bounce it through a
  pinned buffer of its own. Measured on this machine, 128 MB:

  | | rate |
  |---|---|
  | unpinned H2D | 4.1 GB/s |
  | unpinned D2H | 5.7 GB/s |
  | memcpy into pinned | 25.8 GB/s |
  | pinned H2D | **61.0 GB/s** |
  | pinned D2H | **55.9 GB/s** |
  | memcpy + pinned H2D, which is what a caller gets | **17.7 GB/s** |

  The memcpy is not an added cost — `copy-to-device!` already had to stage
  through immobile memory, because Racket CS may move a byte string during the
  GC that a blocking driver call permits. It is the remaining ceiling, though:
  Racket CS cannot alias foreign memory as a byte string (`make-sized-byte-string`
  on a pinned pointer raises `unsupported`), so the coefficients cannot be
  written into pinned memory in the first place.
- **Writing the coefficients went through `integer->integer-bytes`**, a generic
  conversion that dispatches on size, signedness and endianness at every call.
  8.4 million of them cost 141 ms; the same bytes written directly cost 8. The
  slow path is still there for any coefficient that does not fit the fast one,
  and `integer->integer-bytes` still raises rather than wrapping when a value
  exceeds int64.

The whole product as a caller sees it, PCIe and host allocation included. Best of
five, because at these sizes a fresh 268 MB result per product means the mean
mostly measures when Racket's collector ran:

```
field      n   split  fused     rb    w32  ultra   kara     best
Q(z8)   1024     14     52     14      9      9      9   w32  1.6x
Q(z8)   2048     73    383     77     44     45     42   kara 1.7x
Q(z24)  1024     40    205     40     24     20     17   kara 2.4x
Q(z24)  2048    294   1494    265    172    160    150   kara 2.0x
```

ℚ(ζ₂₄) at n = 1024 was **94 ms** before any of this and is **17 ms** after it.

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
Q(z6)   96x96    9     ok      ALL 6 KERNELS IDENTICAL
Q(z8)   96x96    9     ok      ALL 6 KERNELS IDENTICAL
Q(z12)  96x96    9     ok      ALL 6 KERNELS IDENTICAL
Q(z24)  96x96    9     ok      ALL 6 KERNELS IDENTICAL
   ... 20 cases across four fields, sizes 1 to 96 ...

refcheck cmp on our output: IDENTICAL: 8192 coefficients
```

## Status

Two languages, and a test that keeps it that way: `no-float-tests.rkt` walks the
tree and fails on any source file that is not `.rkt`, `.cu`, `.ptx` or `.md`,
naming any script in another language outright. A helper in some third language
is exactly what creeps in when nobody is checking, and then it is a dependency.

Done: the field, matrices, MUB verification, the CUDA driver binding, NVRTC,
streams and pinned memory, six measured kernels including the Karatsuba one, the
accelerator on the math path, device-resident chaining, the no-float
enforcement, and an independent C++ cross-check that every one of the six
kernels is checked against. 3074 tests.

Not done, stated rather than hidden:

- **No RNS path.** A ℤ[ζₙ] product past the int64 bound raises rather than
  splitting across machine-word primes and reconstructing with CRT. That is a
  different backend, not another kernel: it would remove the int64 wall while
  keeping an explicit certificate for unique reconstruction. Until then the
  bound is checked before the launch and audited after it, and the failure is an
  exception rather than a wrapped answer that still looks valid.
- **Karatsuba only at power-of-two degrees.** `'kara` covers `deg ∈ {2, 4, 8, 16}`,
  which is ℚ(ζₙ) for n ∈ {3,4,6}, {5,8,12}, {15,16,20,24,30}, {32,40,48} among
  others. A degree like 6 or 12 falls back to `'ultra`. Karatsuba generalises to
  a 3-way Toom–Cook split for degrees divisible by 3, exactly, and that is not
  written.
- **A `dmat` is resident as int64**, because that is what a product writes, so a
  chain of device-resident products still pays a narrowing launch per operand per
  product. Only `gpu-matmul` gets the host-narrowed upload. A width-tagged
  resident matrix would close that.
- **The coefficients cannot be written straight into pinned memory**, so every
  transfer pays one host memcpy at 25.8 GB/s in front of a 61 GB/s link. That is
  a Racket CS limitation, not a design choice: see the measurement above.
- **DP4A is unused.** For coefficients inside int8 the hardware can do four exact
  integer multiply-accumulates per instruction, which is the right instruction
  for the small-coefficient matrices this package actually multiplies most often.
  It does not compose with Karatsuba — the partial sums leave int8 immediately —
  so it would be a seventh kernel with its own certificate, chosen against
  `'kara` by measurement.
