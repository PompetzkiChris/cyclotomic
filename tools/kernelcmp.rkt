#lang racket/base
;; Six kernels for the same exact product. They must agree entrywise; then the
;; only question is which is fastest.
;;
;;   'split  one launch per (i,j) plane pair, one output element per thread
;;   'fused  one launch total, every plane staged in shared memory as int64
;;   'rb     per plane pair, 64x64 tile, 4x4 outputs per thread, int64 operands
;;   'w32    the same tiling with operands narrowed to int32 -> mul.wide.s32,
;;           accumulating in int64; valid once the host has shown they fit
;;   'ultra  one launch, narrow operands, 16x16 tile, C written once, and one
;;           kernel compiled per degree so every loop unrolls
;;   'kara   the same with Karatsuba: 3^L plane products instead of 4^L
;;
;; Two tables, because there are two questions. The first has the operands
;; already on the card, so it measures arithmetic and device memory and nothing
;; else. The second is the whole product as a caller sees it, PCIe and host
;; allocation included -- and at the larger sizes that is what dominates, so it
;; reports the best of several runs rather than the mean: a 2048x2048 product
;; over Q(zeta_24) allocates a fresh 268 MB result every time, and the mean
;; mostly measures when Racket's collector decided to run.

(require racket/list
         "../exact-io.rkt"
         "../field.rkt"
         "../matrix.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(gpu-init!)

(define MODES '(split fused rb w32 ultra kara))

(printf "=== all six kernels must agree, exactly ===\n")
(define fails 0)
(for* ([ncyc (in-list '(6 8 12 24))]
       [sz (in-list '(1 3 17 40 96 130))])
  (define F (make-field ncyc))
  (define lim 256)
  (define A (zmat-build F sz sz (lambda (i j t) (- (random (* 2 lim)) lim))))
  (define B (zmat-build F sz sz (lambda (i j t) (- (random (* 2 lim)) lim))))
  (define results
    (for/list ([m (in-list MODES)])
      (parameterize ([current-gpu-kernel m]) (zmat-planes (gpu-matmul A B)))))
  (define same (for/and ([r (in-list (cdr results))]) (equal? r (car results))))
  (unless same (set! fails (add1 fails)))
  (printf "  Q(z~a) ~ax~a : ~a\n" ncyc sz sz (if same "IDENTICAL" "DIFFER")))
(printf "  ~a\n\n" (if (zero? fails) "all six agree" (format "~a MISMATCHES" fails)))

(define (pad v n)
  (define s (format "~a" v))
  (string-append (make-string (max 0 (- n (string-length s))) #\space) s))

(define (row ncyc n ts)
  (define bi (argmin (lambda (i) (list-ref ts i)) (range (length MODES))))
  (printf "Q(z~a) ~a ~a ~a ~a ~a ~a ~a   ~a ~ax\n"
          ncyc (pad n 5)
          (pad (round (first ts)) 6) (pad (round (second ts)) 6)
          (pad (round (third ts)) 6) (pad (round (fourth ts)) 6)
          (pad (round (fifth ts)) 6) (pad (round (sixth ts)) 6)
          (list-ref MODES bi)
          (ratio (first ts) (max 1 (list-ref ts bi)))))

(define header "field      n   split  fused     rb    w32  ultra   kara     best\n")

;; Best of `reps`, which for a measurement with this much host noise in it is
;; the estimate that is actually about the thing being measured.
(define (best-of reps thunk)
  (thunk)
  (for/fold ([b #f]) ([_ (in-range reps)])
    (define t (now-ms))
    (thunk)
    (define e (- (now-ms) t))
    (if (or (not b) (< e b)) e b)))

(printf "=== kernel only: operands device-resident, ms per product ===\n")
(printf "~a" header)
(for* ([ncyc (in-list '(8 24))]
       [n (in-list '(512 1024 2048 3072))])
  (define F (make-field ncyc))
  (define lim (if (= ncyc 8) 128 32))
  (define A (zmat-build F n n (lambda (i j t) (- (random (* 2 lim)) lim))))
  (row ncyc n
       (call-with-dmat
        A
        (lambda (dA)
          (for/list ([m (in-list MODES)])
            (parameterize ([current-gpu-kernel m])
              (best-of 5 (lambda () (dmat-free! (dmat* dA dA #:audit? #f))))))))))

(printf "\n=== whole product as a caller sees it, PCIe included ===\n")
(printf "~a" header)
(for* ([ncyc (in-list '(8 24))]
       [n (in-list '(512 1024 2048))])
  (define F (make-field ncyc))
  (define lim (if (= ncyc 8) 128 32))
  (define A (zmat-build F n n (lambda (i j t) (- (random (* 2 lim)) lim))))
  (row ncyc n
       (for/list ([m (in-list MODES)])
         (collect-garbage)
         (parameterize ([current-gpu-kernel m])
           (best-of 5 (lambda () (void (gpu-matmul A A #:audit? #f))))))))

(gpu-shutdown!)
(when (> fails 0) (exit 1))
