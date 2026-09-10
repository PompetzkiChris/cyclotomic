#lang racket/base
;; Writing a CUDA kernel in Racket, compiling it for the device that is
;; actually present, and running it. No nvcc, no build step, no host compiler.

(require rackunit
         rackunit/text-ui
         racket/string
         "../exact-io.rkt"
         "../cuda/driver.rkt"
         "../cuda/nvrtc.rkt")

(define SRC #<<EOF
extern "C" __global__ void scale_add(long long* v, long long s, long long a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) v[i] = v[i] * s + a;
}

extern "C" __global__ void sum_i64(const long long* v, long long n, long long* out)
{
    __shared__ long long acc[256];
    long long s = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x) s += v[i];
    acc[threadIdx.x] = s;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) acc[threadIdx.x] += acc[threadIdx.x + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc[0];
}
EOF
)

(define availability
  (test-suite
   "NVRTC is present"
   (check-true (nvrtc-available?) "nvrtc DLL loads")
   (let-values ([(maj min) (nvrtc-version)])
     (printf "  NVRTC ~a.~a\n" maj min)
     (check-true (>= maj 11) "a modern NVRTC"))))

(define compile-tests
  (test-suite
   "compile CUDA C++ written in Racket, for this device"
   (let*-values ([(maj min) (cuda-compute-capability 0)]
                 [(arch) (device-arch-string maj min)])
     (printf "  targeting ~a\n" arch)
     (let-values ([(ptx log) (compile-cuda/log SRC #:arch arch)])
       (check-true (> (bytes-length ptx) 500) "PTX produced")
       (check-true (regexp-match? #rx"scale_add" ptx) "our kernel is in it")
       (check-true (regexp-match? #rx"sum_i64" ptx) "and the second one")
       (check-equal? (length (regexp-match* #rx"\\.f32|\\.f64" ptx)) 0
                     "still no floating point")))))

(define error-tests
  (test-suite
   "a broken kernel reports why"
   (let ([e (with-handlers ([exn:fail:cuda-compile? values])
              (compile-cuda "extern \"C\" __global__ void k() { this is not c++ }")
              #f)])
     (check-true (exn:fail:cuda-compile? e) "raises a compile exception")
     (when (exn:fail:cuda-compile? e)
       (define log (exn:fail:cuda-compile-log e))
       (printf "  NVRTC said: ~a\n"
               (car (string-split (string-trim log) "\n")))
       (check-true (> (string-length log) 10) "the log is attached, not discarded")
       (check-true (regexp-match? #rx"error" log) "and it names an error")))))

(define run-tests-suite
  (test-suite
   "run the runtime-compiled kernel"
   (let*-values ([(maj min) (cuda-compute-capability 0)]
                 [(ptx) (compile-cuda SRC #:arch (device-arch-string maj min))])
     (define mod (load-ptx ptx))
     (define k-scale (module-function mod "scale_add"))
     (define k-sum (module-function mod "sum_i64"))

     ;; the driver's own idea of a good block size
     (let-values ([(g b) (max-potential-block-size k-scale)])
       (printf "  occupancy suggests grid ~a, block ~a\n" g b)
       (check-true (> b 0) "a positive block size"))

     (define n 1000000)
     (define host (make-bytes (* n 8)))
     (for ([i (in-range n)])
       (integer->integer-bytes i 8 #t #f host (* i 8)))

     (call-with-device-buffer
      (* n 8)
      (lambda (dv)
        (copy-to-device! dv host)
        (launch! k-scale (list 1024 1 1) (list 256 1 1)
                 (list (cons 'u64 dv) (cons 'u64 3) (cons 'u64 7) (cons 'u64 n)))
        (synchronize!)
        (define back (make-bytes (* n 8)))
        (copy-from-device! back dv)
        (define ok
          (for/and ([i (in-range 0 n 4001)])
            (= (integer-bytes->integer back #t #f (* i 8) (* (add1 i) 8))
               (+ (* 3 i) 7))))
        (check-true ok "v[i] = 3i + 7 exactly, everywhere sampled")

        ;; reduce, and check against the exact closed form
        (define blocks 256)
        (call-with-device-buffer
         (* blocks 8)
         (lambda (dout)
           (launch! k-sum (list blocks 1 1) (list 256 1 1)
                    (list (cons 'u64 dv) (cons 'u64 n) (cons 'u64 dout)))
           (synchronize!)
           (define ob (make-bytes (* blocks 8)))
           (copy-from-device! ob dout)
           (define total
             (for/sum ([b (in-range blocks)])
               (integer-bytes->integer ob #t #f (* b 8) (* (add1 b) 8))))
           (define want (+ (* 3 (/ (* n (- n 1)) 2)) (* 7 n)))
           (printf "  sum of 3i+7 for i<~a : ~a\n" n total)
           (check-equal? total want "exact integer reduction, no rounding"))))))))

(define stream-tests
  (test-suite
   "streams, pinned memory, async transfer"
   (let* ([n 4000000]
          [nbytes (* n 8)]
          [s (make-stream)])
     (call-with-pinned
      nbytes
      (lambda (h)
        ;; fill the pinned buffer and round-trip it asynchronously
        (define src (make-bytes nbytes))
        (for ([i (in-range 0 n 1)])
          (integer->integer-bytes (- i 12345) 8 #t #f src (* i 8)))
        (memcpy-into-pinned! h src)
        (call-with-device-buffer
         nbytes
         (lambda (dv)
           (copy-to-device/async! dv h nbytes s)
           (stream-synchronize! s)
           ;; clear the host side, then bring it back
           (memcpy-into-pinned! h (make-bytes nbytes))
           (copy-from-device/async! h dv nbytes s)
           (stream-synchronize! s)
           (define back (pinned-bytes h nbytes))
           (check-equal? (integer-bytes->integer back #t #f 0 8) -12345
                         "first element round-tripped")
           (check-equal? (integer-bytes->integer back #t #f (* 8 (- n 1)) (* 8 n))
                         (- n 1 12345)
                         "last element round-tripped")
           (check-true (equal? back src) "the whole buffer is identical")))))
     (stream-destroy! s))))

;; small helper: write a byte string into pinned memory
(define (memcpy-into-pinned! h bs)
  ((dynamic-require 'ffi/unsafe 'memcpy) (pinned-ptr h) bs (bytes-length bs)))

(define FLOATY #<<EOF
extern "C" __global__ void f(float* v, int n) {   // FLOAT-OK: must be refused
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = v[i] * 2.5f + 1.0f;   // FLOAT-OK: must be refused
}
EOF
)

(define DOUBLY #<<EOF
extern "C" __global__ void d(double* v, int n) {   // FLOAT-OK: must be refused
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = v[i] / 3.0;   // FLOAT-OK: must be refused
}
EOF
)

;; a float smuggled in behind a typedef and a template, to show the check is on
;; the artifact and not on the spelling of the source
(define SNEAKY #<<EOF
template <typename T> struct Box { T x; };
typedef float Real;   // FLOAT-OK: must be refused
using R2 = Real;
extern "C" __global__ void sneaky(long long* out, long long n) {
    Box<R2> b; b.x = (R2)3;
    b.x = b.x * (R2)1.5;   // FLOAT-OK: must be refused
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (long long)b.x;
}
EOF
)

;; Built at run time so no float literal appears in this source file; the point
;; is to hand load-ptx an image it must refuse.
(define (float-ptx-sample)
  (bytes-append
   (bytes-append #".version 8." (bytes (+ 48 0)) #"
")   ; FLOAT-OK
   #".target sm_120
.address_size 64
"
   #".visible .entry k() { .reg ." (bytes 102 51 50)       ; FLOAT-OK: ".f32"
   #" %" (bytes 102) #"<2>; ret; }
"))

(define no-float-tests
  (test-suite
   "floating point cannot get through, by any door"
   (let*-values ([(maj min) (cuda-compute-capability 0)]
                 [(arch) (device-arch-string maj min)])

     (for ([pair (in-list (list (cons "float" FLOATY)
                                (cons "double" DOUBLY)
                                (cons "float behind a typedef and a template" SNEAKY)))])
       (define e (with-handlers ([exn:fail:cuda-float? values])
                   (compile-cuda (cdr pair) #:arch arch)
                   #f))
       (check-true (exn:fail:cuda-float? e)
                   (format "compile-cuda refuses ~a" (car pair)))
       (when (exn:fail:cuda-float? e)
         (printf "  ~a -> refused, ~a float construct(s)
"
                 (car pair) (length (exn:fail:cuda-float-hits e)))))

     ;; a float expression that NVRTC folds away leaves float-free PTX, so the
     ;; PTX check alone would let it through; the source scan is what stops it
     (let ([hits (source-float-hits SNEAKY)])
       (check-true (> (length hits) 0) "the source scan sees the folded float"))

     ;; every way of naming or manufacturing a float, including the intrinsics
     ;; that build one out of an integer bit pattern with no literal in sight
     (for ([probe (in-list (list (cons "float" "typedef float R;")
                                 (cons "double" "double x;")
                                 (cons "__half" "__half h;")
                                 (cons "__nv_bfloat16" "__nv_bfloat16 b;")
                                 (cons "__int_as_float" "x = __int_as_float(1065353216);")
                                 (cons "__int2half_rn" "h = __int2half_rn(3);")
                                 (cons "decimal literal" "x = 1.5;")
                                 (cons "leading-dot literal" "x = .5;")
                                 (cons "exponent literal" "x = 1e9;")
                                 (cons "hex float literal" "x = 0x1.8p3;")))])
       (check-true (> (length (source-float-hits (cdr probe))) 0)
                   (format "source scan catches ~a" (car probe))))

     ;; and does not fire on prose or on clean integer code
     (check-equal? (source-float-hits "// mentions float and 1.5\nlong long a = 3;") '()
                   "a comment about floats is not a float")
     (check-equal? (source-float-hits "extern \"C\" __global__ void k(long long* v, long long n){ long long i = blockIdx.x; if (i<n) v[i] = v[i]*3 + 7; }")
                   '()
                   "clean integer kernel passes")

     ;; the preprocessor is the other way a float could hide, so it is refused
     (for ([d (in-list (list "#define REAL float\nlong long x;"
                             "  #include <cmath>\nlong long x;"))])
       (define e (with-handlers ([exn:fail:cuda-float? values])
                   (compile-cuda d #:arch arch)
                   #f))
       (check-true (exn:fail:cuda-float? e)
                   (format "compile-cuda refuses ~a"
                           (car (string-split (string-trim d) "\n")))))

     ;; and the other door: hand load-ptx float PTX directly
     (let ([e (with-handlers ([exn:fail? values])
                ;; FLOAT-OK: this PTX is deliberately float, to be refused
                (load-ptx (float-ptx-sample))
                #f)])
       (check-true (exn:fail? e) "load-ptx refuses float PTX from any source")
       (when (exn:fail? e)
         (printf "  load-ptx said: ~a
"
                 (car (string-split (exn-message e) "
")))
         (check-true (regexp-match? #rx"refusing floating point" (exn-message e))
                     "and says so plainly")))

     ;; the exact kernel still compiles and loads
     (let ([ptx (compile-cuda SRC #:arch arch)])
       (check-equal? (ptx-float-hits ptx) '() "our own kernel has none")
       (check-true (cuda-module? (load-ptx ptx)) "and loads fine")))))

(module+ test
  ;; load-ptx, launches and device memory all need a current context
  (cuda-init!)
  (define ctx (make-context 0))
  (void
   (run-tests
    (test-suite "nvrtc" availability compile-tests error-tests
                no-float-tests run-tests-suite stream-tests)))
  (context-destroy! ctx))
