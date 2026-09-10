#lang scribble/manual
@require[@for-label[racket/base racket/contract racket/promise]]

@title{cyclotomic: exact arithmetic in @racket[Q(ζₙ)], with CUDA}
@author{cpompetzki}

Exact arithmetic in the cyclotomic fields @tt{Q(zeta_n)}, exact matrices over
them, mutually-unbiased-basis verification, and exact @tt{Z[zeta_n]} linear
algebra on NVIDIA GPUs through the CUDA driver API.

Two languages, Racket and CUDA C++, and no floating point anywhere --- both
enforced by the test suite rather than intended.

@table-of-contents[]

@section{The field}

@defmodule[cyclotomic/field]

@tt{Phi_n} is computed from @tt{x^n - 1 = prod_{d|n} Phi_d(x)} by exact integer
polynomial division, not tabulated. An element is @tt{phi(n)} exact rationals in
the power basis @tt{1, zeta, ..., zeta^(phi(n)-1)}.

@defproc[(make-field [n exact-positive-integer?]) cyclofield?]{
  The field @tt{Q(zeta_n)}. Fields are cached by @racket[n], so repeated calls
  return the same object.

  The power table and the unit group are @emph{promises}: a caller who only
  wants @racket[field-degree] does not pay to build them. At @tt{n = 5040} the
  power table costs about 121 ms, and it is built at most once, on first use.
}

@defproc[(field-degree [f cyclofield?]) exact-positive-integer?]{
  @tt{phi(n)}, available without forcing either table.
}

@defproc[(make-cyc [f cyclofield?] [coeffs (listof exact-rational?)]) cyc?]{
  An element from its coefficients in the power basis. The contract rejects an
  inexact number: there is no silent conversion.

  @racketblock[
    (make-cyc F (list 0.5))   (code:comment "contract violation")
  ]
}

@deftogether[(
  @defproc[(cyc+ [e cyc?] ...) cyc?]
  @defproc[(cyc* [e cyc?] ...) cyc?]
  @defproc[(cyc- [a cyc?] [b cyc?]) cyc?]
  @defproc[(cyc/ [a cyc?] [b cyc?]) cyc?]
  @defproc[(cyc-expt [a cyc?] [k exact-integer?]) cyc?]
  @defproc[(cyc-inverse [a cyc?]) cyc?]
)]{
  Field arithmetic. Multiplication is convolution followed by reduction through
  the power table; inversion solves @tt{M x = e_0} by Gaussian elimination over
  exact rationals, where no pivoting for magnitude is needed because there is no
  magnitude to lose.
}

@deftogether[(
  @defproc[(cyc-sigma [a cyc?] [k exact-integer?]) cyc?]
  @defproc[(cyc-conjugate [a cyc?]) cyc?]
  @defproc[(cyc-norm [a cyc?]) exact-rational?]
  @defproc[(cyc-trace [a cyc?]) exact-rational?]
  @defproc[(cyc-abs2 [a cyc?]) cyc?]
)]{
  The Galois action @tt{zeta -> zeta^k} for @racket[k] coprime to @tt{n},
  complex conjugation as @tt{sigma_(n-1)}, and the norm and trace, which are
  rational and are asserted to be.
}

@deftogether[(
  @defproc[(field-sqrt [f cyclofield?] [m exact-positive-integer?]) cyc?]
  @defproc[(field-inv-sqrt [f cyclofield?] [m exact-positive-integer?]) cyc?]
  @defproc[(field-i [f cyclofield?]) cyc?]
)]{
  @tt{sqrt(2) = zeta_8 + zeta_8^-1}, @tt{sqrt(3) = zeta_12 + zeta_12^-1},
  @tt{sqrt(6)} their product, and @tt{i}. Each raises if the field does not
  contain the root of unity it needs, rather than returning something
  approximate.

  @racketblock[
    (define F (make-field 24))
    (cyc-expt (field-inv-sqrt F 6) 2)   (code:comment "1/6, exactly")
  ]
}

@defproc[(cyc-lift [a cyc?] [g cyclofield?]) cyc?]{
  Embed into @tt{Q(zeta_m)} via @tt{zeta_n = zeta_m^(m/n)}. Raises when
  @tt{n} does not divide @tt{m}.
}

@section{Matrices}

@defmodule[cyclotomic/matrix]

@deftogether[(
  @defproc[(mat [f cyclofield?] [rows (listof (listof cyc?))]) matrix?]
  @defproc[(mat* [a matrix?] [b matrix?]) matrix?]
  @defproc[(mat*/cpu [a matrix?] [b matrix?]) matrix?]
  @defproc[(mat-adjoint [m matrix?]) matrix?]
  @defproc[(mat-kron [a matrix?] [b matrix?]) matrix?]
  @defproc[(mat-unitary? [m matrix?]) boolean?]
)]{
  Exact matrices. @racket[mat*] consults @racket[current-mat*-hook] first and
  falls back to @racket[mat*/cpu]; both give the same answer.
}

@defparam[current-mat*-hook hook (or/c #f (-> matrix? matrix? (or/c matrix? #f)))]{
  Installed by @racketmodname[cyclotomic/cuda/accel] to put the GPU on the
  ordinary math path. A hook returns @racket[#f] to decline, and
  @racket[mat*] then uses the exact CPU path. A GPU problem can slow a program
  down; it cannot change an answer.
}

@section{Mutually unbiased bases}

@defmodule[cyclotomic/mub]

Two orthonormal bases are unbiased when @tt{|<b_i,c_j>|^2 = 1/d} for every
@tt{i, j}. Here that is an equation between exact rationals, so the answer is
yes or no rather than a threshold.

@deftogether[(
  @defproc[(unbiasedness [b matrix?] [c matrix?]) (or/c exact-rational? #f)]
  @defproc[(mutually-unbiased? [ms (listof matrix?)] [d exact-positive-integer?]) boolean?]
)]{
  @racket[unbiasedness] returns the common value of @tt{|<b_i,c_j>|^2} or
  @racket[#f] if it is not constant.
}

@deftogether[(
  @defproc[(mubs-d2 [f cyclofield?]) list?]
  @defproc[(mubs-d3 [f cyclofield?]) list?]
  @defproc[(mubs-d6) list?]
)]{
  Complete sets in dimensions 2 and 3, and three pairwise unbiased bases in
  dimension 6 built by tensoring them in @tt{Q(zeta_24)} --- the smallest
  cyclotomic field holding @tt{sqrt2}, @tt{sqrt3}, @tt{sqrt6} and @tt{i} at
  once.

  Three in dimension 6 is the standard lower bound. Whether a fourth exists is a
  long-standing open problem, and nothing here bears on it.
}

@section{The CUDA driver}

@defmodule[cyclotomic/cuda/driver]

Bindings to @tt{nvcuda.dll} through @racketmodname[ffi/unsafe]. The driver API
rather than the runtime API: a stable C ABI, no host compiler at run time, and
it loads PTX directly. Every call is checked, and a failure raises with the
driver's own message.

@deftogether[(
  @defproc[(cuda-init!) void?]
  @defproc[(make-context [ordinal exact-nonnegative-integer?]) cuda-context?]
  @defproc[(context-destroy! [c cuda-context?]) void?]
)]{
  @racket[make-context] retains the device's @emph{primary} context, so it is
  shared with every other CUDA library in the process rather than competing
  with them.
}

@deftogether[(
  @defproc[(call-with-device-buffer [nbytes exact-positive-integer?]
                                    [proc (-> exact-nonnegative-integer? any)]) any]
  @defproc[(call-with-device-buffers [sizes (listof exact-positive-integer?)]
                                     [proc procedure?]) any]
)]{
  Device memory is not garbage collected, so these free it on every exit path:
  a normal return, a raise, or an escape continuation. All three are tested.
}

@defproc[(load-ptx [src (or/c path-string? bytes?)]) cuda-module?]{
  Load a PTX image. Raises with the JIT log when the module is rejected, and
  @bold{refuses any image containing a floating-point instruction} --- see
  @secref["nofloat"].
}

@defparam[current-gpu-wait mode (or/c 'poll 'block)]{
  How to wait for the device. @racket['poll] records a @tt{CUevent} and polls
  it with @racket[(sleep 0)], so other Racket threads keep running; a blocking
  driver call would freeze every green thread, since Racket CS schedules them
  all on one OS thread. @racket['block] uses @tt{cuCtxSynchronize} for maximum
  throughput. Polling costs about 2x on a short kernel and nothing on a long
  one.
}

@section{Putting the device on the math path}

@defmodule[cyclotomic/cuda/accel]

Requiring this module and calling @racket[gpu-accel-install!] makes every
@racket[mat*] in the program try the device first. Existing code is not
rewritten and does not need to know: both paths are exact, so the answer is the
same either way.

@deftogether[(
  @defproc[(gpu-accel-install! [#:device ordinal exact-nonnegative-integer? 0]) void?]
  @defproc[(gpu-accel-uninstall!) void?]
  @defproc[(gpu-accel-declines) exact-nonnegative-integer?]
)]{
  Entries of @tt{Q(zeta_n)} are rationals and the device path is
  @tt{Z[zeta_n]}, so the hook clears the common denominator, runs the integer
  product on the card, and puts the denominator back.

  It @emph{declines} rather than fails --- wrong field, a product past the
  @tt{int64} bound, no device, anything unexpected --- returning @racket[#f] so
  @racket[mat*] falls through to @racket[mat*/cpu].
}

@defparam[current-gpu-min-work n exact-nonnegative-integer?]{
  Products smaller than this many multiply-accumulates stay on the CPU.
  @racket[0], the default, always tries the device.
}

@section{Runtime compilation}

@defmodule[cyclotomic/cuda/nvrtc]

@defproc[(compile-cuda [src string?]
                       [#:name name string? "kernel.cu"]
                       [#:arch arch string? "compute_120"]
                       [#:options options (listof string?) '()])
         bytes?]{
  Compile CUDA C++ to PTX for the device actually present. No @tt{nvcc}, no
  build step, no host compiler.

  @racketblock[
    (define-values (maj min) (cuda-compute-capability 0))
    (compile-cuda src #:arch (device-arch-string maj min))
  ]

  Requiring this module does nothing at all: locating the library, loading it
  and putting its dependencies on @tt{PATH} all sit behind promises and happen
  on first use.

  A compile failure raises @racket[exn:fail:cuda-compile?] with NVRTC's own log
  attached. Floating point raises @racket[exn:fail:cuda-float?].
}

@section[#:tag "nofloat"]{No floating point}

Not "no floating point in the parts that matter" --- none, and no float ever
involved at any stage.

@itemlist[
  @item{The Racket sources contain no flonum. Clocks read
        @racket[current-milliseconds], which is an exact integer; ratios stay
        exact rationals; decimals are rendered by integer arithmetic.}
  @item{@racket[compile-cuda] refuses preprocessor directives, because a
        @tt{#define} could rename @tt{float} and an @tt{#include} could bring in
        a header full of them.}
  @item{It rejects every float-producing name --- including the intrinsics that
        manufacture one from an integer bit pattern, like @tt{__int_as_float}
        --- and every way of writing a float literal. Comments and string
        literals are stripped first.}
  @item{It then scans the generated PTX and refuses to return it if any float
        type or register appears, so a float that survived everything else
        still cannot reach the device.}
  @item{@racket[load-ptx] runs the same PTX check, so an image from a file or
        another toolchain cannot bring floating point in either.}
]

There is deliberately no option to permit it.

@section{Verification}

The GPU path is checked against an independent CUDA C++ implementation in
@tt{refcheck/}, written to disagree if anything is wrong: its own @tt{Phi_n},
its own power table, a deliberately different kernel, and a CPU oracle
accumulating in 128 bits so a product that silently wrapped @tt{int64} is
detected rather than matched.

All four GPU kernels agree with it, coefficient for coefficient, across
@tt{Q(zeta_6)}, @tt{Q(zeta_8)}, @tt{Q(zeta_12)} and @tt{Q(zeta_24)}.
