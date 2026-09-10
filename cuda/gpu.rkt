#lang racket/base
;; Exact Z[zeta_n] matrix multiplication on the GPU, driven from Racket.
;;
;; A matrix is deg = phi(n) integer planes of int64, laid out contiguously as
;; one byte string: plane t, row-major, at offset t*rows*cols*8.
;;
;; The product launches deg^2 kernels, one per (i, j) plane pair. Each computes
;; A_i B_j tile-wise and scatters R[i+j][t] * that into every output plane t, so
;; no intermediate convolution plane is ever materialised: the working set is
;; exactly A + B + C.
;;
;; int64 does not round, it wraps, which is worse: the answer still looks
;; valid. So the bound is checked before launching, from the actual operand
;; magnitudes measured on the device, and the result is audited afterwards.

(require racket/vector
         racket/runtime-path
         "driver.rkt"
         "../field.rkt"
         "../matrix.rkt")

;; Resolved against THIS file's directory, not the caller's, and it survives
;; `raco exe` packaging.
(define-runtime-path ptx-file "kernels.ptx")

(provide (struct-out zmat)
         zmat-of-integers zmat-build zmat-coeff
         zmat->matrix
         gpu-init!
         gpu-shutdown!
         gpu-matmul
         gpu-absmax
         int64-max
         current-gpu-ready?
         gpu-stats gpu-reset-stats! gpu-stats-report
         current-gpu-kernel
         ;; device-resident matrices: upload once, multiply many times
         (struct-out dmat)
         zmat->dmat dmat->zmat dmat-free! call-with-dmat dmat*)

;; Which kernel runs a product. 'fused is one launch for the whole field
;; product; 'split is the original deg^2 launches. Both are exact and are
;; checked against each other; the selector exists so that can be tested and so
;; a field with deg > 8 can still run.
(define current-gpu-kernel (make-parameter 'auto))

;; Usage counters. "Did this actually run on the card" should be a number, not
;; a belief, so every product records what it did.
(define stat-matmuls 0)
(define stat-launches 0)
(define stat-up-bytes 0)
(define stat-down-bytes 0)
(define stat-device-ms 0.0)
(define stat-elements 0)

(define (gpu-reset-stats!)
  (set! stat-matmuls 0) (set! stat-launches 0)
  (set! stat-up-bytes 0) (set! stat-down-bytes 0)
  (set! stat-device-ms 0.0) (set! stat-elements 0)
  (void))

(define (gpu-stats)
  (hasheq 'matmuls stat-matmuls
          'launches stat-launches
          'uploaded-bytes stat-up-bytes
          'downloaded-bytes stat-down-bytes
          'device-ms stat-device-ms
          'output-elements stat-elements))

(define (gpu-stats-report [port (current-output-port)])
  (define s (gpu-stats))
  (fprintf port "GPU usage: ~a products, ~a kernel launches, ~a ms on device\n"
           (hash-ref s 'matmuls) (hash-ref s 'launches)
           (round (hash-ref s 'device-ms)))
  (fprintf port "           ~a MB up, ~a MB down, ~a output field elements\n"
           (round (/ (hash-ref s 'uploaded-bytes) 1048576.0))
           (round (/ (hash-ref s 'downloaded-bytes) 1048576.0))
           (hash-ref s 'output-elements)))

(define int64-max (sub1 (expt 2 63)))
(define int64-min (- (expt 2 63)))

;; planes : bytes, deg*rows*cols int64s little-endian
(struct zmat (field rows cols planes) #:transparent)

;; ------------------------------------------------------------------ context

(define ctx #f)
(define mod #f)
(define k-plane-madd #f)
(define k-zero #f)
(define k-absmax #f)
(define k-fused #f)
(define k-rb #f)
(define k-w32 #f)
(define k-narrow #f)
(define flush-handle #f)

;; The Phi_n reduction table is the same for every product over a given field,
;; so uploading it per product is pure waste -- a 24-product chain paid for 24
;; identical allocations and copies. Cached per field, freed on shutdown.
(define rtable-cache (make-hasheqv))

(define (field-reduction-table f)
  (define d (field-degree f))
  (define nrows (sub1 (* 2 d)))
  (define pow (cyclofield-pow f))
  (define buf (make-bytes (* nrows d 4) 0))
  (define maxR
    (for*/fold ([m 0]) ([mm (in-range nrows)] [t (in-range d)])
      (define v (vector-ref (vector-ref pow mm) t))
      (integer->integer-bytes v 4 #t #f buf (* 4 (+ (* mm d) t)))
      (max m (abs v))))
  (values buf maxR nrows))

;; (values device-ptr maxR nrows)
(define (device-reduction-table f)
  (define key (cyclofield-n f))
  (define hit (hash-ref rtable-cache key #f))
  (cond
    [hit (values (vector-ref hit 0) (vector-ref hit 1) (vector-ref hit 2))]
    [else
     (define-values (buf maxR nrows) (field-reduction-table f))
     (define p (device-alloc (bytes-length buf)))
     (copy-to-device! p buf)
     (hash-set! rtable-cache key (vector p maxR nrows))
     (values p maxR nrows)]))

(define (free-reduction-tables!)
  (for ([(k v) (in-hash rtable-cache)])
    (with-handlers ([exn:fail? void]) (device-free! (vector-ref v 0))))
  (hash-clear! rtable-cache))

(define (current-gpu-ready?) (and ctx #t))

(define (gpu-init! [ordinal 0])
  (unless ctx
    (cuda-init!)
    ;; Check the device can actually run what was compiled, before loading it.
    (let-values ([(maj min) (cuda-compute-capability ordinal)])
      (when (< maj 5)
        (error 'gpu-init! "device ~a is sm_~a~a; kernels are built for sm_120"
               (cuda-device-name ordinal) maj min)))
    (set! ctx (make-context ordinal))
    (set! mod (load-ptx ptx-file))
    (set! k-plane-madd (module-function mod "plane_madd"))
    (set! k-zero (module-function mod "zero_i64"))
    (set! k-absmax (module-function mod "absmax_i64"))
    (set! k-fused (module-function mod "fused_madd"))
    (set! k-rb (module-function mod "plane_madd_rb"))
    (set! k-w32 (module-function mod "plane_madd_w32"))
    (set! k-narrow (module-function mod "narrow_i64_i32"))
    ;; Release the context even if the program exits without calling shutdown.
    ;; Registered once, not once per init/shutdown cycle.
    (unless flush-handle
      (set! flush-handle
            (plumber-add-flush! (current-plumber) (lambda (h) (gpu-shutdown!))))))
  (void))

(define (gpu-shutdown!)
  (when ctx
    (free-reduction-tables!)
    (release-sync-event!)
    (when mod (void (unload-module! mod)))
    (context-destroy! ctx)
    (set! ctx #f) (set! mod #f)
    (set! k-plane-madd #f) (set! k-zero #f) (set! k-absmax #f) (set! k-fused #f) (set! k-rb #f) (set! k-w32 #f) (set! k-narrow #f))
  (when flush-handle
    (plumber-flush-handle-remove! flush-handle)
    (set! flush-handle #f))
  (void))

;; --------------------------------------------------------------- conversions

;; Build a zmat from a list-of-lists of cyc elements whose coefficients are all
;; exact integers (i.e. an element of Z[zeta_n], not merely Q(zeta_n)).
(define (zmat-of-integers f rows)
  (define r (length rows))
  (define c (length (car rows)))
  (define d (field-degree f))
  (define buf (make-bytes (* d r c 8) 0))
  (for ([row (in-list rows)] [i (in-naturals)])
    (for ([e (in-list row)] [j (in-naturals)])
      (define cs (cyc-coeffs e))
      (for ([t (in-range d)])
        (define v (vector-ref cs t))
        (unless (and (exact-integer? v))
          (error 'zmat-of-integers
                 "entry (~a,~a) has non-integer coefficient ~a; the GPU path is Z[zeta_n]"
                 i j v))
        (integer->integer-bytes v 8 #t #f buf
                                (* 8 (+ (* t r c) (* i c) j))))))
  (zmat f r c buf))

;; Bulk path. Building a zmat through lists of `cyc` structs costs far more
;; than the product itself -- profiling a 1024x1024 product over Q(zeta_24) put
;; the device at 6% of wall time and struct churn at the rest. `zmat-build`
;; writes coefficients straight into the plane bytes, and `zmat-coeff` reads one
;; back without constructing a field element. Use these for bulk data; use
;; zmat-of-integers / zmat->matrix when you actually want field elements.
(define (zmat-build f rows cols fill)
  (define d (field-degree f))
  (define buf (make-bytes (* d rows cols 8) 0))
  (define plane (* rows cols))
  (for ([t (in-range d)])
    (define base (* t plane))
    (for ([i (in-range rows)])
      (define row (* i cols))
      (for ([j (in-range cols)])
        (define v (fill i j t))
        (unless (eqv? v 0)
          (integer->integer-bytes v 8 #t #f buf (* 8 (+ base row j)))))))
  (zmat f rows cols buf))

(define (zmat-coeff z i j t)
  (define r (zmat-rows z))
  (define c (zmat-cols z))
  (define off (* 8 (+ (* t r c) (* i c) j)))
  (integer-bytes->integer (zmat-planes z) #t #f off (+ off 8)))

(define (zmat->matrix z)
  (define f (zmat-field z))
  (define d (field-degree f))
  (define r (zmat-rows z))
  (define c (zmat-cols z))
  (define buf (zmat-planes z))
  (mat f
       (for/list ([i (in-range r)])
         (for/list ([j (in-range c)])
           (make-cyc f
                     (for/list ([t (in-range d)])
                       (integer-bytes->integer
                        buf #t #f
                        (* 8 (+ (* t r c) (* i c) j))
                        (* 8 (add1 (+ (* t r c) (* i c) j))))))))))

;; ------------------------------------------------------------------ helpers

(define (ceil-div a b) (quotient (+ a b -1) b))

;; Largest |value| in a device buffer of int64s. Uses the reduction kernel.
(define (gpu-absmax dptr count)
  (define blocks 256)
  (define threads 256)
  (call-with-device-buffer
   (* blocks 8)
   (lambda (out)
     (launch! k-absmax (list blocks 1 1) (list threads 1 1)
              (list (cons 'u64 dptr) (cons 'u64 count) (cons 'u64 out)))
     (synchronize!)
     (define bs (make-bytes (* blocks 8)))
     (copy-from-device! bs out)
     (for/fold ([m 0]) ([b (in-range blocks)])
       (max m (integer-bytes->integer bs #t #f (* b 8) (* (add1 b) 8)))))))

;; ----------------------------------------------------------------- the product

(define (gpu-matmul A B #:audit? [audit? #t])
  (unless ctx (error 'gpu-matmul "call gpu-init! first"))
  (define f (zmat-field A))
  (unless (= (cyclofield-n f) (cyclofield-n (zmat-field B)))
    (error 'gpu-matmul "matrices are over different fields"))
  (define d (field-degree f))
  (define n (zmat-rows A))
  (define k (zmat-cols A))
  (define c (zmat-cols B))
  (unless (= k (zmat-rows B)) (error 'gpu-matmul "shape mismatch"))

  (define-values (dR maxR nrows) (device-reduction-table f))

  (define stride (* n c))
  (define nC (* d stride))

  ;; Every device buffer is released on the way out, including when the bound
  ;; check raises, a launch fails, or the caller escapes with a continuation.
  (define t-start (current-inexact-milliseconds))
  (call-with-device-buffers
   (list (bytes-length (zmat-planes A))
         (bytes-length (zmat-planes B))
         (* nC 8))
   (lambda (dA dB dC)
     (copy-to-device! dA (zmat-planes A))
     (copy-to-device! dB (zmat-planes B))
     (set! stat-up-bytes (+ stat-up-bytes
                            (bytes-length (zmat-planes A))
                            (bytes-length (zmat-planes B))))

     ;; overflow bound, from magnitudes measured on the device
     (define maxA (gpu-absmax dA (* d n k)))
     (define maxB (gpu-absmax dB (* d k c)))
     (define bound (* nrows d k maxA maxB (max maxR 1)))
     (when (> bound int64-max)
       (error 'gpu-matmul
              "|C| could reach ~a > int64 max ~a; this product needs the RNS path"
              bound int64-max))

     (define grid (list (ceil-div c 16) (ceil-div n 16) 1))
     (define block (list 16 16 1))
     (define launches (run-product! dA dB dC dR d n k c stride grid block maxA maxB))
     (synchronize!)

     (when audit?
       (define peak (gpu-absmax dC nC))
       (when (> peak bound)
         (error 'gpu-matmul "post-hoc audit failed: |C| = ~a exceeds the bound ~a"
                peak bound)))

     (define out (make-bytes (* nC 8)))
     (copy-from-device! out dC)
     (set! stat-down-bytes (+ stat-down-bytes (bytes-length out)))
     (set! stat-matmuls (add1 stat-matmuls))
     (set! stat-launches (+ stat-launches launches))
     (set! stat-elements (+ stat-elements (* n c)))
     (set! stat-device-ms (+ stat-device-ms
                             (- (current-inexact-milliseconds) t-start)))
     (zmat f n c out))))

;; ---------------------------------------------------------------------------
;; Device-resident matrices.
;;
;; gpu-matmul uploads both operands and downloads the result every time, which
;; is right for a one-off product and wrong for a chain of them: profiling put
;; the device at 14% of wall time with the rest in host traffic and struct
;; churn. A dmat stays on the card between products, so a sequence -- powers,
;; a search loop, repeated Gram matrices -- runs without touching the host.
;; ---------------------------------------------------------------------------

(struct dmat (field rows cols ptr) #:transparent)

(define (zmat->dmat z)
  (unless ctx (error 'zmat->dmat "call gpu-init! first"))
  (define bs (zmat-planes z))
  (define p (device-alloc (max 1 (bytes-length bs))))
  (copy-to-device! p bs)
  (set! stat-up-bytes (+ stat-up-bytes (bytes-length bs)))
  (dmat (zmat-field z) (zmat-rows z) (zmat-cols z) p))

(define (dmat->zmat dm)
  (define f (dmat-field dm))
  (define d (field-degree f))
  (define n (* d (dmat-rows dm) (dmat-cols dm) 8))
  (define out (make-bytes n))
  (copy-from-device! out (dmat-ptr dm))
  (set! stat-down-bytes (+ stat-down-bytes n))
  (zmat f (dmat-rows dm) (dmat-cols dm) out))

(define (dmat-free! dm) (device-free! (dmat-ptr dm)))

(define (call-with-dmat z proc)
  (define dm (zmat->dmat z))
  (dynamic-wind void (lambda () (proc dm)) (lambda () (dmat-free! dm))))

;; Product of two device-resident matrices, result device-resident.
;; The caller owns the result and must free it (or use call-with-dmat).
(define (dmat* A B #:audit? [audit? #t])
  (unless ctx (error 'dmat* "call gpu-init! first"))
  (define f (dmat-field A))
  (unless (= (cyclofield-n f) (cyclofield-n (dmat-field B)))
    (error 'dmat* "matrices are over different fields"))
  (define d (field-degree f))
  (define n (dmat-rows A))
  (define k (dmat-cols A))
  (define c (dmat-cols B))
  (unless (= k (dmat-rows B)) (error 'dmat* "shape mismatch"))

  (define-values (dR maxR nrows) (device-reduction-table f))
  (define stride (* n c))
  (define nC (* d stride))
  (define t-start (current-inexact-milliseconds))

  (define dA (dmat-ptr A))
  (define dB (dmat-ptr B))
  (define maxA (gpu-absmax dA (* d n k)))
  (define maxB (gpu-absmax dB (* d k c)))
  (define bound (* nrows d k maxA maxB (max maxR 1)))
  (when (> bound int64-max)
    (error 'dmat* "|C| could reach ~a > int64 max ~a" bound int64-max))

  (define dC (device-alloc (* nC 8)))
  (with-handlers ([(lambda (e) #t) (lambda (e) (device-free! dC) (raise e))])
    (define grid (list (ceil-div c 16) (ceil-div n 16) 1))
    (define block (list 16 16 1))
    (define launches (run-product! dA dB dC dR d n k c stride grid block maxA maxB))
    (synchronize!)
    (when audit?
      (define peak (gpu-absmax dC nC))
      (when (> peak bound)
        (error 'dmat* "post-hoc audit failed: |C| = ~a exceeds ~a" peak bound)))
    (set! stat-matmuls (add1 stat-matmuls))
    (set! stat-launches (+ stat-launches launches))
    (set! stat-elements (+ stat-elements (* n c)))
    (set! stat-device-ms (+ stat-device-ms (- (current-inexact-milliseconds) t-start)))
    (dmat f n c dC)))

;; Issue the kernels for one product; returns the number of launches made.
(define int32-max (sub1 (expt 2 31)))

(define (run-product! dA dB dC dR d n k c stride grid block [maxA #f] [maxB #f])
  (define mode
    (let ([m (current-gpu-kernel)])
      (if (eq? m 'auto)
          ;; narrow operands are the fast path when they fit; otherwise the
          ;; register-blocked int64 kernel
          (if (and maxA maxB (< maxA int32-max) (< maxB int32-max)) 'w32 'rb)
          m)))
  (cond
    [(eq? mode 'w32)
     ;; narrow both operands to int32, then one IMAD per multiply-add
     (define nA (* d n k))
     (define nB (* d k c))
     (call-with-device-buffers
      (list (* nA 4) (* nB 4))
      (lambda (wA wB)
        (launch! k-narrow (list 1024 1 1) (list 256 1 1)
                 (list (cons 'u64 dA) (cons 'u64 wA) (cons 'u64 nA)))
        (launch! k-narrow (list 1024 1 1) (list 256 1 1)
                 (list (cons 'u64 dB) (cons 'u64 wB) (cons 'u64 nB)))
        (launch! k-zero (list 1024 1 1) (list 256 1 1)
                 (list (cons 'u64 dC) (cons 'u64 (* d stride))))
        (define g (list (ceil-div c 64) (ceil-div n 64) 1))
        (define b (list 16 16 1))
        (for* ([i (in-range d)] [j (in-range d)])
          (launch! k-w32 g b
                   (list (cons 'u64 (+ wA (* 4 i n k)))
                         (cons 'u64 (+ wB (* 4 j k c)))
                         (cons 'u64 dC)
                         (cons 'u64 (+ dR (* 4 (+ i j) d)))
                         (cons 'i32 n) (cons 'i32 k) (cons 'i32 c)
                         (cons 'i32 d) (cons 'u64 stride))))
        ;; the copy-back must finish before the narrowed buffers are released
        (synchronize!)))
     (+ 3 (* d d))]
    [(and (eq? mode 'fused) (<= d 8))
     ;; one launch: all planes staged in shared memory, raw convolution in
     ;; registers, R folded once
     (launch! k-fused grid block
              (list (cons 'u64 dA) (cons 'u64 dB) (cons 'u64 dC) (cons 'u64 dR)
                    (cons 'i32 n) (cons 'i32 k) (cons 'i32 c) (cons 'i32 d)))
     1]
    [(eq? mode 'rb)
     ;; register-blocked: 64x64 output tile per block, 4x4 per thread
     (launch! k-zero (list 1024 1 1) (list 256 1 1)
              (list (cons 'u64 dC) (cons 'u64 (* d stride))))
     (define g (list (ceil-div c 64) (ceil-div n 64) 1))
     (define b (list 16 16 1))
     (for* ([i (in-range d)] [j (in-range d)])
       (launch! k-rb g b
                (list (cons 'u64 (+ dA (* 8 i n k)))
                      (cons 'u64 (+ dB (* 8 j k c)))
                      (cons 'u64 dC)
                      (cons 'u64 (+ dR (* 4 (+ i j) d)))
                      (cons 'i32 n) (cons 'i32 k) (cons 'i32 c)
                      (cons 'i32 d) (cons 'u64 stride))))
     (add1 (* d d))]
    [else
     (launch! k-zero (list 1024 1 1) (list 256 1 1)
              (list (cons 'u64 dC) (cons 'u64 (* d stride))))
     (for* ([i (in-range d)] [j (in-range d)])
       (launch! k-plane-madd grid block
                (list (cons 'u64 (+ dA (* 8 i n k)))
                      (cons 'u64 (+ dB (* 8 j k c)))
                      (cons 'u64 dC)
                      (cons 'u64 (+ dR (* 4 (+ i j) d)))
                      (cons 'i32 n) (cons 'i32 k) (cons 'i32 c)
                      (cons 'i32 d) (cons 'u64 stride))))
     (add1 (* d d))]))
