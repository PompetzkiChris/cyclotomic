#lang racket/base
;; Exact Z[zeta_n] matrix multiplication on the GPU, driven from Racket.
;;
;; A matrix is deg = phi(n) integer planes of int64, laid out contiguously as
;; one byte string: plane t, row-major, at offset t*rows*cols*8.
;;
;; The default path is one launch for the whole field product: the convolution
;; is accumulated in registers, Karatsuba makes it 3^L plane products instead of
;; 4^L, and the output planes are written exactly once. The deg^2-launch kernels
;; are still here, still exact, and still checked against it -- see kernels.cu
;; for what each one cost and what it bought.
;;
;; int64 does not round, it wraps, which is worse: the answer still looks
;; valid. So the bound is checked before launching -- from operand magnitudes the
;; matrices carry with them, and covering every intermediate the chosen kernel
;; forms, not just the final coefficient -- and the result is audited afterwards.

(require racket/vector
         racket/promise
         racket/unsafe/ops
         racket/runtime-path
         "../exact-io.rkt"
         "driver.rkt"
         "../field.rkt"
         "../karatsuba.rkt"
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
         zmat-absmax zmat-narrow make-zmat
         gpu-stats gpu-reset-stats! gpu-stats-report
         current-gpu-kernel
         ;; device-resident matrices: upload once, multiply many times
         (struct-out dmat)
         zmat->dmat dmat->zmat dmat-free! call-with-dmat dmat*)

;; Which kernel runs a product:
;;
;;   'auto   fewest multiplications the operands allow, then fewest launches
;;   'kara   one launch, narrow operands, 3^L plane products  (deg 2,4,8,16)
;;   'ultra  one launch, narrow operands, 4^L plane products  (deg 1,2,4,6,8,10,12,16)
;;   'w32    deg^2 launches, narrow operands, C accumulated
;;   'rb     deg^2 launches, int64 operands, 64x64 tile
;;   'fused  one launch, int64 planes staged in shared memory
;;   'split  deg^2 launches, one output element per thread -- the original
;;
;; All six are exact and all six are checked against each other and against the
;; independent CUDA C++ reference. The selector exists so that can be tested, so
;; a degree with no specialised kernel can still run, and so the cost of each
;; decision stays measurable instead of asserted.
(define current-gpu-kernel (make-parameter 'auto))

;; Usage counters. "Did this actually run on the card" should be a number, not
;; a belief, so every product records what it did.
(define stat-matmuls 0)
(define stat-launches 0)
(define stat-up-bytes 0)
(define stat-down-bytes 0)
(define stat-device-ms 0)
(define stat-elements 0)

(define (gpu-reset-stats!)
  (set! stat-matmuls 0) (set! stat-launches 0)
  (set! stat-up-bytes 0) (set! stat-down-bytes 0)
  (set! stat-device-ms 0) (set! stat-elements 0)
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
           (bytes->mb (hash-ref s 'uploaded-bytes))
           (bytes->mb (hash-ref s 'downloaded-bytes))
           (hash-ref s 'output-elements)))

(define int64-max (sub1 (expt 2 63)))
(define int64-min (- (expt 2 63)))
(define int32-max (sub1 (expt 2 31)))
(define int32-max-fx (sub1 (expt 2 31)))   ; same value, known to be a fixnum

;; planes  : bytes, deg*rows*cols int64s little-endian -- the canonical image
;; absmax-p: promise of the exact largest |coefficient| in it
;; narrow-p: promise of the same coefficients as int32, or #f if any of them
;;           does not fit in int32
;;
;; Both are derived, both are exact, and both are promises. A caller who only
;; wants to read coefficients back never pays for either. A caller who
;; multiplies forces them, and then:
;;
;;   - the overflow bound is computed from a number the host already knows,
;;     instead of from two reduction passes over device memory and the two
;;     synchronisations they need;
;;   - the operands cross PCIe at half the width, and the two device-side
;;     narrowing launches disappear with them.
;;
;; zmat-build and zmat-of-integers fill both in while they are already walking
;; every coefficient, so in the normal case forcing them costs nothing at all.
;; Forcing them on a zmat built from raw bytes costs one pass, which is the
;; honest price of data that arrived without them.
(struct zmat (field rows cols planes absmax-p narrow-p) #:transparent)

(define (zmat-absmax z) (force (zmat-absmax-p z)))
(define (zmat-narrow z) (force (zmat-narrow-p z)))

;; Little-endian stores, by hand. integer->integer-bytes is a generic
;; conversion that dispatches on size, signedness and endianness at every call;
;; writing 8.4 million coefficients through it costs 141 ms, and writing the
;; same bytes directly costs 8. Nothing about exactness changes -- these are the
;; same bytes, in the same order -- and the slow path below still handles any
;; coefficient that does not fit the fast one.
;;
;; 2^32 is a fixnum on Racket CS, so the two's-complement fold needs no bignum.
(define-syntax-rule (put-i32! bs o v)
  (let* ([off o]
         [u (if (unsafe-fx< v 0) (unsafe-fx+ v 4294967296) v)])
    (unsafe-bytes-set! bs off (unsafe-fxand u 255))
    (unsafe-bytes-set! bs (unsafe-fx+ off 1) (unsafe-fxand (unsafe-fxrshift u 8) 255))
    (unsafe-bytes-set! bs (unsafe-fx+ off 2) (unsafe-fxand (unsafe-fxrshift u 16) 255))
    (unsafe-bytes-set! bs (unsafe-fx+ off 3) (unsafe-fxand (unsafe-fxrshift u 24) 255))))

;; The 8-byte store of a value that fits int32: low word as above, high word all
;; ones when negative and left alone when not, since the buffer starts zeroed.
(define-syntax-rule (put-i64-from-i32! bs o v)
  (let ([off o])
    (put-i32! bs off v)
    (when (unsafe-fx< v 0)
      (unsafe-bytes-set! bs (unsafe-fx+ off 4) 255)
      (unsafe-bytes-set! bs (unsafe-fx+ off 5) 255)
      (unsafe-bytes-set! bs (unsafe-fx+ off 6) 255)
      (unsafe-bytes-set! bs (unsafe-fx+ off 7) 255))))

;; Largest |coefficient| in an int64 plane image.
(define (planes-absmax bs)
  (define n (quotient (bytes-length bs) 8))
  (for/fold ([m 0]) ([i (in-range n)])
    (max m (abs (integer-bytes->integer bs #t #f (* 8 i) (* 8 (add1 i)))))))

;; The same coefficients as int32, or #f when one of them does not fit. Nothing
;; is truncated to make it fit: either every coefficient is representable or
;; there is no narrow image and the int64 one is used.
(define (planes-narrow bs absmax)
  (and (<= absmax int32-max)
       (let* ([n (quotient (bytes-length bs) 8)]
              [out (make-bytes (* n 4))])
         (for ([i (in-range n)])
           (integer->integer-bytes
            (integer-bytes->integer bs #t #f (* 8 i) (* 8 (add1 i)))
            4 #t #f out (* 4 i)))
         out)))

;; A zmat from plane bytes alone: everything derived is deferred.
(define (make-zmat f rows cols planes #:absmax [absmax #f] #:narrow [narrow 'unknown])
  (define ap (if absmax (delay absmax) (delay (planes-absmax planes))))
  (zmat f rows cols planes ap
        (if (eq? narrow 'unknown)
            (delay (planes-narrow planes (force ap)))
            (delay narrow))))

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

;; One ultra kernel per degree, because phi(n) is known on the host and a
;; compile-time degree is what makes every loop in it unroll and every
;; accumulator a register. A degree with no instantiation is not handled here;
;; the selector falls back to the w32 path, which takes deg as an argument.
(define ultra-degrees '(1 2 4 6 8 10 12 16))
(define k-ultra (make-hasheqv))

;; And one Karatsuba kernel per power-of-two degree, which does 3^L plane
;; products where the direct one does 4^L. Keyed by L, not by degree.
(define kara-levels-available '(1 2 3 4))
(define k-kara (make-hasheqv))

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

;; The Karatsuba output matrix K, with the Phi_n reduction already composed
;; into it, and the exact max row sum of |K| that the overflow certificate
;; needs. Same field, same matrix, so it is uploaded once.
(define ktable-cache (make-hasheqv))

;; (values device-ptr rowsum np level) or #f when the degree is not a power of
;; two, in which case there is no Karatsuba kernel and the caller uses another.
(define (device-kara-table f)
  (define key (cyclofield-n f))
  (define hit (hash-ref ktable-cache key 'miss))
  (cond
    [(not (eq? hit 'miss))
     (and hit (values (vector-ref hit 0) (vector-ref hit 1)
                      (vector-ref hit 2) (vector-ref hit 3)))]
    [else
     (define l (kara-levels (field-degree f)))
     (cond
       [(or (not l) (not (memv l kara-levels-available)))
        (hash-set! ktable-cache key #f)
        #f]
       [else
        (define-values (K rowsum) (kara-output-matrix f))
        (define deg (field-degree f))
        (define np (kara-products l))
        (define buf (make-bytes (* deg np 4) 0))
        (for* ([t (in-range deg)] [j (in-range np)])
          (integer->integer-bytes (vector-ref (vector-ref K t) j) 4 #t #f
                                  buf (* 4 (+ (* t np) j))))
        (define p (device-alloc (bytes-length buf)))
        (copy-to-device! p buf)
        (hash-set! ktable-cache key (vector p rowsum np l))
        (values p rowsum np l)])]))

(define (free-reduction-tables!)
  (for ([(k v) (in-hash rtable-cache)])
    (with-handlers ([exn:fail? void]) (device-free! (vector-ref v 0))))
  (hash-clear! rtable-cache)
  (for ([(k v) (in-hash ktable-cache)])
    (when v (with-handlers ([exn:fail? void]) (device-free! (vector-ref v 0)))))
  (hash-clear! ktable-cache))

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
    (for ([d (in-list ultra-degrees)])
      (hash-set! k-ultra d (module-function mod (format "fused_w32_d~a" d))))
    (for ([l (in-list kara-levels-available)])
      (hash-set! k-kara l (module-function mod (format "kara_w32_l~a" l))))
    ;; Release the context even if the program exits without calling shutdown.
    ;; Registered once, not once per init/shutdown cycle.
    (unless flush-handle
      (set! flush-handle
            (plumber-add-flush! (current-plumber) (lambda (h) (gpu-shutdown!))))))
  (void))

(define (gpu-shutdown!)
  (when ctx
    (free-reduction-tables!)
    (free-staging!)
    (release-sync-event!)
    (when mod (void (unload-module! mod)))
    (context-destroy! ctx)
    (set! ctx #f) (set! mod #f)
    (set! k-plane-madd #f) (set! k-zero #f) (set! k-absmax #f) (set! k-fused #f) (set! k-rb #f) (set! k-w32 #f) (set! k-narrow #f)
    (hash-clear! k-ultra) (hash-clear! k-kara))
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
  (define peak 0)
  (for ([row (in-list rows)] [i (in-naturals)])
    (for ([e (in-list row)] [j (in-naturals)])
      (define cs (cyc-coeffs e))
      (for ([t (in-range d)])
        (define v (vector-ref cs t))
        (unless (and (exact-integer? v))
          (error 'zmat-of-integers
                 "entry (~a,~a) has non-integer coefficient ~a; the GPU path is Z[zeta_n]"
                 i j v))
        (when (> (abs v) peak) (set! peak (abs v)))
        (integer->integer-bytes v 8 #t #f buf
                                (* 8 (+ (* t r c) (* i c) j))))))
  (make-zmat f r c buf #:absmax peak))

;; Bulk path. Building a zmat through lists of `cyc` structs costs far more
;; than the product itself -- profiling a 1024x1024 product over Q(zeta_24) put
;; the device at 6% of wall time and struct churn at the rest. `zmat-build`
;; writes coefficients straight into the plane bytes, and `zmat-coeff` reads one
;; back without constructing a field element. Use these for bulk data; use
;; zmat-of-integers / zmat->matrix when you actually want field elements.
;; Both images are written in this one pass, and the peak is tracked as it
;; goes. The narrow image is written speculatively and abandoned the moment a
;; coefficient is seen that does not fit int32 -- so the common case pays one
;; extra 4-byte store per nonzero coefficient and nothing else, and the
;; uncommon case pays nothing at all after the first overflow.
(define (zmat-build f rows cols fill)
  (define d (field-degree f))
  (define count (* d rows cols))
  (define buf (make-bytes (* count 8) 0))
  (define narrow (make-bytes (* count 4) 0))
  (define plane (* rows cols))
  (define peak 0)
  (for ([t (in-range d)])
    (define base (* t plane))
    (for ([i (in-range rows)])
      (define row (+ base (* i cols)))
      (for ([j (in-range cols)])
        (define v (fill i j t))
        (unless (eqv? v 0)
          (define off (unsafe-fx+ row j))
          (cond
            ;; the overwhelmingly common case: a fixnum inside int32, so one
            ;; pair of direct stores and no conversion machinery at all
            [(and (fixnum? v) (unsafe-fx<= (unsafe-fxabs v) int32-max-fx))
             (when (unsafe-fx> (unsafe-fxabs v) peak) (set! peak (unsafe-fxabs v)))
             (put-i64-from-i32! buf (unsafe-fx* 8 off) v)
             (when narrow (put-i32! narrow (unsafe-fx* 4 off) v))]
            [else
             (unless (exact-integer? v)
               (error 'zmat-build "coefficient ~a at (~a,~a,~a) is not an exact integer"
                      v i j t))
             (define a (abs v))
             (when (> a peak) (set! peak a))
             ;; raises rather than wrapping if it does not fit int64
             (integer->integer-bytes v 8 #t #f buf (* 8 off))
             (set! narrow #f)])))))
  (make-zmat f rows cols buf #:absmax peak #:narrow narrow))

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

;; ------------------------------------------------------- pinned host staging
;;
;; An unpinned copy makes the driver bounce the data through a pinned buffer of
;; its own, which on this machine runs at about 9 GB/s against about 24 GB/s
;; for a copy that starts from pinned memory. The memcpy into the staging
;; buffer is not an extra cost: copy-to-device! already has to stage through
;; immobile memory, because Racket CS may move a byte string during the GC that
;; a blocking driver call permits.
;;
;; One buffer, grown on demand and reused, rather than a pin per product: the
;; pin itself costs more than the copy.
(define stage #f)

(define (staging-buffer nbytes)
  (when (and stage (< (pinned-size stage) nbytes))
    (pinned-free! stage)
    (set! stage #f))
  (unless stage (set! stage (pinned-alloc (max nbytes 65536))))
  stage)

(define (free-staging!)
  (when stage (with-handlers ([exn:fail? void]) (pinned-free! stage)) (set! stage #f)))

(define (upload! dptr bs)
  (define n (bytes-length bs))
  (define h (staging-buffer n))
  (pinned-fill! h bs n)
  (copy-to-device/pinned! dptr h n)
  (set! stat-up-bytes (+ stat-up-bytes n))
  (void))

(define (download! bs dptr)
  (define n (bytes-length bs))
  (define h (staging-buffer n))
  (copy-from-device/pinned! h dptr n)
  (pinned-read! bs h n)
  (set! stat-down-bytes (+ stat-down-bytes n))
  (void))

;; --------------------------------------------------------- operands on device
;;
;; The narrow kernels want int32 planes. When the matrix already carries its
;; int32 image -- which zmat-build and zmat-of-integers produce in the pass that
;; writes the int64 one -- that image is uploaded directly, and both the
;; double-width transfer and the device-side narrowing launch disappear. When it
;; does not, the int64 image is uploaded and narrowed on the device, which is
;; the old path and still exact.
(define (call-with-operand z narrow? count proc)
  (define bs (zmat-planes z))
  (cond
    [(not narrow?)
     (call-with-device-buffer (max 1 (bytes-length bs))
       (lambda (p) (upload! p bs) (proc p)))]
    [(zmat-narrow z)
     => (lambda (w)
          (call-with-device-buffer (max 1 (bytes-length w))
            (lambda (p) (upload! p w) (proc p))))]
    [else
     (call-with-device-buffer
      (max 1 (* count 4))
      (lambda (p)
        (call-with-device-buffer
         (max 1 (* count 8))
         (lambda (wide)
           (upload! wide bs)
           (launch! k-narrow (list 1024 1 1) (list 256 1 1)
                    (list (cons 'u64 wide) (cons 'u64 p) (cons 'u64 count)))
           (synchronize!)))
        (proc p)))]))

;; Which kernels take int32 planes.
(define (narrow-mode? mode) (and (memq mode '(kara ultra w32)) #t))

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

  ;; The operand magnitudes come from the matrices themselves, which knew them
  ;; when they were built. The old path measured them with two reduction passes
  ;; over device memory and two synchronisations, after uploading at double
  ;; width to have something to measure.
  (define maxA (zmat-absmax A))
  (define maxB (zmat-absmax B))
  (define-values (mode bound) (plan-product f d k maxA maxB nrows maxR))
  (when (> bound int64-max)
    (error 'gpu-matmul
           "|C| could reach ~a > int64 max ~a; this product needs the RNS path"
           bound int64-max))
  (define narrow? (narrow-mode? mode))

  ;; Every device buffer is released on the way out, including when a launch
  ;; fails or the caller escapes with a continuation.
  (define t-start (now-ms))
  (call-with-operand
   A narrow? (* d n k)
   (lambda (pA)
     (call-with-operand
      B narrow? (* d k c)
      (lambda (pB)
        (call-with-device-buffer
         (* nC 8)
         (lambda (dC)
           (define grid (list (ceil-div c 16) (ceil-div n 16) 1))
           (define block (list 16 16 1))
           (define launches
             (run-product! f mode pA pB dC dR d n k c stride grid block))
           (synchronize!)

           (define peak (and audit? (gpu-absmax dC nC)))
           (when (and peak (> peak bound))
             (error 'gpu-matmul "post-hoc audit failed: |C| = ~a exceeds the bound ~a"
                    peak bound))

           (define out (make-bytes (* nC 8)))
           (download! out dC)
           (set! stat-matmuls (add1 stat-matmuls))
           (set! stat-launches (+ stat-launches launches))
           (set! stat-elements (+ stat-elements (* n c)))
           (set! stat-device-ms (+ stat-device-ms (- (now-ms) t-start)))
           ;; When the audit ran, the result's own absmax is already known
           ;; exactly; there is no reason to make the next product rediscover it.
           (make-zmat f n c out #:absmax peak))))))))

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
  (make-zmat f (dmat-rows dm) (dmat-cols dm) out))

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
  (define t-start (now-ms))

  (define dA (dmat-ptr A))
  (define dB (dmat-ptr B))
  (define maxA (gpu-absmax dA (* d n k)))
  (define maxB (gpu-absmax dB (* d k c)))
  (define-values (mode bound) (plan-product f d k maxA maxB nrows maxR))
  (when (> bound int64-max)
    (error 'dmat* "|C| could reach ~a > int64 max ~a" bound int64-max))

  (define dC (device-alloc (* nC 8)))
  (with-handlers ([(lambda (e) #t) (lambda (e) (device-free! dC) (raise e))])
    (define grid (list (ceil-div c 16) (ceil-div n 16) 1))
    (define block (list 16 16 1))
    ;; A dmat is resident as int64 -- that is what the product writes, so it is
    ;; what a chain of products carries. The narrow kernels therefore need a
    ;; device-side narrowing pass here, unlike gpu-matmul, where the host
    ;; already has the int32 image and uploads it directly.
    (define launches
      (call-with-device-narrowed
       (narrow-mode? mode) dA (* d n k)
       (lambda (pA extraA)
         (call-with-device-narrowed
          (narrow-mode? mode) dB (* d k c)
          (lambda (pB extraB)
            (define made (run-product! f mode pA pB dC dR d n k c stride grid block))
            ;; the product must finish before the narrowed operands are released
            (synchronize!)
            (+ extraA extraB made))))))
    (when audit?
      (define peak (gpu-absmax dC nC))
      (when (> peak bound)
        (error 'dmat* "post-hoc audit failed: |C| = ~a exceeds ~a" peak bound)))
    (set! stat-matmuls (add1 stat-matmuls))
    (set! stat-launches (+ stat-launches launches))
    (set! stat-elements (+ stat-elements (* n c)))
    (set! stat-device-ms (+ stat-device-ms (- (now-ms) t-start)))
    (dmat f n c dC)))

;; Narrow an int64 device buffer to int32 for the duration of a call, or pass it
;; through untouched when the mode does not want it narrow. `proc` is given the
;; pointer and the number of extra launches it cost.
(define (call-with-device-narrowed narrow? dptr count proc)
  (cond
    [(not narrow?) (proc dptr 0)]
    [else
     (call-with-device-buffer
      (max 1 (* count 4))
      (lambda (p)
        (launch! k-narrow (list 1024 1 1) (list 256 1 1)
                 (list (cons 'u64 dptr) (cons 'u64 p) (cons 'u64 count)))
        (proc p 1)))]))

;; --------------------------------------------------------------- the planner
;;
;; Which kernel runs, and what the result is then bounded by, are one decision:
;; the Karatsuba kernel is faster and has a larger bound, so the bound is what
;; decides whether it may be used. Both are computed here, exactly, before
;; anything is launched.

;; (values mode bound). `bound` is the largest |C| the chosen kernel can
;; produce given the measured operand magnitudes -- every intermediate
;; included, not just the final coefficient.
(define (plan-product f d k maxA maxB nrows maxR)
  (define direct-bound (* nrows d k maxA maxB (max maxR 1)))
  (define narrow-ok? (and (<= maxA int32-max) (<= maxB int32-max)))
  ;; Karatsuba feeds sums of up to 2^L coefficients to a 32x32 multiply, and
  ;; its accumulators carry the 4^L that comes with them. rowsum is the exact
  ;; max row sum of |K|, so this bound is measured, not guessed.
  (define kara
    (let ([l (kara-levels d)])
      (and l (memv l kara-levels-available)
           (let*-values ([(g) (kara-operand-growth l)]
                         [(p rowsum np lv) (device-kara-table f)])
             (define b (* rowsum (kara-growth l) k maxA maxB))
             (and (<= (* g maxA) int32-max)
                  (<= (* g maxB) int32-max)
                  (<= b int64-max)
                  (cons l b))))))
  (define requested (current-gpu-kernel))
  (cond
    [(eq? requested 'kara)
     (unless kara
       (error 'gpu-matmul
              "kara needs a power-of-two degree and room for 2^L-fold operands; degree ~a, |A| ~a, |B| ~a"
              d maxA maxB))
     (values 'kara (cdr kara))]
    [(not (eq? requested 'auto))
     (when (and (memq requested '(ultra w32)) (not narrow-ok?))
       (error 'gpu-matmul
              "~a needs |A|,|B| <= ~a; measured ~a and ~a" requested int32-max maxA maxB))
     (values requested direct-bound)]
    ;; fewest multiplies first, then fewest launches, then int64 operands
    [kara (values 'kara (cdr kara))]
    [(and narrow-ok? (memv d ultra-degrees)) (values 'ultra direct-bound)]
    [narrow-ok? (values 'w32 direct-bound)]
    [else (values 'rb direct-bound)]))

;; Issue the kernels for one product; returns the number of launches made.
;; pA and pB are already in the width the mode wants: int32 for the narrow
;; kernels, int64 for the others. Widening or narrowing is the caller's job,
;; because the caller is the one that knows whether the host already has the
;; narrow image.
(define (run-product! f mode pA pB dC dR d n k c stride grid block)
  (cond
    [(eq? mode 'kara)
     ;; one launch, 3^L plane products instead of 4^L, C written once
     (define-values (dK rowsum np l) (device-kara-table f))
     (launch! (hash-ref k-kara l)
              (list (ceil-div c 16) (ceil-div n 16) 1) (list 16 16 1)
              (list (cons 'u64 pA) (cons 'u64 pB) (cons 'u64 dC) (cons 'u64 dK)
                    (cons 'i32 n) (cons 'i32 k) (cons 'i32 c)))
     1]
    [(eq? mode 'ultra)
     ;; one launch, 4^L plane products, C written once
     (unless (memv d ultra-degrees)
       (error 'gpu-matmul "no ultra kernel for degree ~a; degrees are ~a"
              d ultra-degrees))
     (launch! (hash-ref k-ultra d)
              (list (ceil-div c 16) (ceil-div n 16) 1) (list 16 16 1)
              (list (cons 'u64 pA) (cons 'u64 pB) (cons 'u64 dC) (cons 'u64 dR)
                    (cons 'i32 n) (cons 'i32 k) (cons 'i32 c)))
     1]
    [(eq? mode 'w32)
     ;; deg^2 launches of the register-blocked narrow kernel, C accumulated
     (launch! k-zero (list 1024 1 1) (list 256 1 1)
              (list (cons 'u64 dC) (cons 'u64 (* d stride))))
     (define g (list (ceil-div c 64) (ceil-div n 64) 1))
     (define b (list 16 16 1))
     (for* ([i (in-range d)] [j (in-range d)])
       (launch! k-w32 g b
                (list (cons 'u64 (+ pA (* 4 i n k)))
                      (cons 'u64 (+ pB (* 4 j k c)))
                      (cons 'u64 dC)
                      (cons 'u64 (+ dR (* 4 (+ i j) d)))
                      (cons 'i32 n) (cons 'i32 k) (cons 'i32 c)
                      (cons 'i32 d) (cons 'u64 stride))))
     (add1 (* d d))]
    [(and (eq? mode 'fused) (<= d 8))
     ;; one launch: all planes staged in shared memory, raw convolution in
     ;; registers, R folded once
     (launch! k-fused grid block
              (list (cons 'u64 pA) (cons 'u64 pB) (cons 'u64 dC) (cons 'u64 dR)
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
                (list (cons 'u64 (+ pA (* 8 i n k)))
                      (cons 'u64 (+ pB (* 8 j k c)))
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
                (list (cons 'u64 (+ pA (* 8 i n k)))
                      (cons 'u64 (+ pB (* 8 j k c)))
                      (cons 'u64 dC)
                      (cons 'u64 (+ dR (* 4 (+ i j) d)))
                      (cons 'i32 n) (cons 'i32 k) (cons 'i32 c)
                      (cons 'i32 d) (cons 'u64 stride))))
     (add1 (* d d))]))
