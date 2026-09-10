# `cyclotomic` — exact arithmetic in ℚ(ζₙ), in Racket

Racket 9.3 [cs]. **245 tests, all passing.** Racket drives the GPU directly;
there is no Python anywhere in this path.

```
raco test C:\GPU\racket\tests\
```

## Why Racket is the right language for this

Racket's numeric tower is **exact by default**. `1/6` is the rational one
sixth, not a float that happens to print that way. Inexactness is the thing you
have to ask for, with `exact->inexact`.

In Python I had to build a wall — a `TypeError` on every constructor — to keep
floats out of the field. Here the ground state is already correct, and the one
remaining job is to reject an inexact number a caller hands in, which the
contracts do:

```racket
(make-cyc F (list 0.5))   ; contract violation, not a silent conversion
```

The test suite makes the distinction explicit:

```racket
(check-equal? v 1/6)                              ; passes
(check-false (= v (exact->inexact 1/6)))          ; passes
```

The exact rational and the nearest double to it are simply different objects,
and Racket says so.

## Layout

```
cyclotomic/
  poly.rkt      exact integer polynomials; Phi_n by the divisor recursion
  field.rkt     Q(zeta_n): elements, Galois, norm/trace, inverse, embeddings
  matrix.rkt    exact matrices: product, adjoint, kron, unitarity
  mub.rkt       mutually unbiased bases in d = 2, 3, 6
cuda/
  driver.rkt    CUDA driver API via ffi/unsafe; buffers, modules, launch, wait
  kernels.cu    int64-only kernels; kernels.ptx is the compiled artifact
  gpu.rkt       exact Z[zeta_n] matrices on the device
  accel.rkt     installs the device onto mat*; declines safely
  probe.rkt     what the driver reports about this machine
  profile.rkt   where wall time actually goes
  sustained.rkt device-resident chaining, the load test
  bench.rkt     GPU against pure Racket
  audit.rkt     leak / scheduler / GC-churn audit
  waitmode.rkt  poll vs block timing
tests/
  field-tests.rkt      61 tests
  gpu-tests.rkt       121 tests
  mub-tests.rkt        31 tests
  hardening-tests.rkt    9 tests
  accel-tests.rkt       23 tests
info.rkt        package definition
```

## The field

`Φₙ` is computed, not tabulated, from `xⁿ − 1 = ∏_{d|n} Φ_d(x)` by exact
integer polynomial division. The test suite checks `Φ₁₀₅` has a `−2`
coefficient — the first cyclotomic polynomial whose coefficients leave
`{−1,0,1}`, and a thing a lookup table would get wrong.

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
`ffi/unsafe` — deliberately the driver API rather than the runtime API: it is a
stable C ABI, it needs no host compiler at run time, and it loads PTX directly.
Racket owns the whole pipeline. Allocate, upload, launch, read back.

```
device 0: NVIDIA GeForce RTX 5090
  compute capability : sm_120
  multiprocessors    : 170
  memory bus width   : 512 bit
context created. VRAM 30.2 GB free of 31.8 GB
host->device->host round trip of 1024 int64s: IDENTICAL
```

`cuda/kernels.cu` is 64-bit integer arithmetic only. The build checks the
generated PTX for `.f32`/`.f64` and finds none — the claim that no floating
point is involved is mechanically verifiable, not a promise.

The product launches `φ(n)²` kernels, one per plane pair. Each computes
`Aᵢ·Bⱼ` tile-wise and scatters `R[i+j][t] ·` that straight into every output
plane, so no intermediate convolution plane is ever materialised: the working
set is exactly A + B + C.

### Measured

```
correctness against pure Racket
  Q(zeta_8)  96x96 : racket 267 ms   gpu  6 ms   EXACT
  Q(zeta_24) 96x96 : racket 492 ms   gpu  9 ms   EXACT

scale, GPU only
  Q(zeta_8)  1024x1024 :  30 ms   1141 Giga-integer-ops/s   16 launches
  Q(zeta_24) 1024x1024 : 122 ms   1124 Giga-integer-ops/s   64 launches
```

Every product is bounds-checked **before** launch, from operand magnitudes
measured on the device by a reduction kernel, and audited after. int64 does not
round, it wraps — and a wrapped result still looks valid, which is why it
raises instead.

Honest about the gap: this hand-written 16×16 tiled kernel reaches ~1.1
Tera-integer-ops/s, where cuBLAS-backed integer GEMM reaches ~8.6. The kernel
is straightforward and there is real headroom in it — tile size, register
blocking, wider loads. Exactness is settled; throughput is not finished.

### Hardening

Four things were wrong or fragile in the first version, found by writing tests
that try to break it rather than tests that confirm it works.

**Device memory on the error path.** `cuMemAlloc` is not GC-managed, so any
early exit leaked VRAM until the process died. `call-with-device-buffer` /
`call-with-device-buffers` wrap allocation in `dynamic-wind`, and
`tests/hardening-tests.rkt` checks all three exits: normal return, a raise
inside, and an escape continuation. 256 MB per buffer, so a leak would be
unmissable.

**A blocking driver call froze the whole runtime.** `#:blocking? #t` lets the
GC proceed; it does *not* let other Racket threads run, because Racket CS
schedules every green thread on one OS thread. Measured: a ticker thread
advanced **0** times across a 48 ms product. Fixed by not blocking — record a
`CUevent` and poll it with `(sleep 0)` between queries. Same test now shows the
ticker advancing tens of thousands of times during the call.

The cost is real and belongs to the caller, so it is a parameter:

```racket
(current-gpu-wait 'poll)   ; default: other threads keep running
(current-gpu-wait 'block)  ; cuCtxSynchronize, maximum throughput
```

| field | n | poll | block |
|---|---|---|---|
| ℚ(ζ₈) | 512 | 26 ms | 12 ms |
| ℚ(ζ₈) | 1024 | 92 ms | 90 ms |
| ℚ(ζ₂₄) | 512 | 30 ms | 16 ms |
| ℚ(ζ₂₄) | 1024 | 141 ms | 149 ms |

Polling costs about 2x on a short kernel and nothing on a long one.

**Immobile transfer buffers.** With the calls marked blocking, the GC may move
a byte string while the driver is reading it. Transfers now stage through
`'atomic-interior` memory. Note the trap: that memory is GC-managed and must
**not** be passed to `free` — doing so is heap corruption, not a leak. It cost
a `0xC0000374` crash to learn.

**A bad module now says why.** `cuModuleLoadDataEx` with the JIT log buffers
wired up:

```
load-ptx: CUDA error 200 loading PTX: device kernel image is invalid
JIT error log:
error   : Can't load this binary kind, as it's not recognized
```

Also switched from `cuCtxCreate` to `cuDevicePrimaryCtxRetain`, which shares
the context every other CUDA library in the process uses instead of competing
with it, and registered a plumber flush so the context is released even if the
program exits without calling `gpu-shutdown!`.

## An independent implementation to check against

The Racket GPU path was only ever checked against Racket -- same repository,
same kernel, same author. `refcheck/refcheck.cu` is a second implementation
written to disagree if anything is wrong:

* `Phi_n` recomputed in C++ from `x^n - 1 = prod_{d|n} Phi_d(x)`, not ported.
* Its own power table.
* A deliberately different kernel: one thread per output **element**, whole
  convolution inline, no shared-memory tiling and no scatter. A mistake in the
  Racket kernel's tiling or its `R` scatter would not be repeated here.
* A CPU oracle accumulating in **128 bits**, so a product that silently wrapped
  int64 is detected rather than matched. (MSVC has no `__int128`; it is built
  from `_umul128` with manual carry, and writing that surfaced a real sign bug
  the selftest caught immediately -- overflow reported on a 1x1 product.)

Build and run:

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
Q(z6)   96x96    9     ok      IDENTICAL
Q(z8)   96x96    9     ok      IDENTICAL
Q(z12)  96x96    9     ok      IDENTICAL
Q(z24)  96x96    9     ok      IDENTICAL
   ... 20 cases across four fields, sizes 1 to 96 ...

refcheck cmp on our output: IDENTICAL: 8192 coefficients
```

## Installing

```
raco pkg install --link C:\GPU\pkg-cyclotomic
racket -e '(require cyclotomic/field) (field-sqrt (make-field 24) 6)'
raco test --package cyclotomic
```

## Status

Done: the field, matrices, MUB verification, the CUDA FFI, exact GPU products,
the hardening above, 222 tests.

Next: kernel tuning, and an RNS path so ℤ[ζₙ] products past the int64 bound
stop raising and start splitting across primes.
