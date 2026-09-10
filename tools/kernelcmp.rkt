#lang racket/base
;; Four kernels for the same exact product. They must agree entrywise; then the
;; only question is which is fastest.
;;
;;   'split  one launch per (i,j) plane pair, one output element per thread
;;   'fused  one launch total, every plane staged in shared memory
;;   'rb     per plane pair, 64x64 tile, 4x4 outputs per thread, int64 operands
;;   'w32    same tiling but operands narrowed to int32 -> mul.wide.s32,
;;           accumulating in int64. Only valid when the operands fit, which is
;;           already known from the bound check.

(require racket/list
         "../field.rkt"
         "../matrix.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(gpu-init!)

(define MODES '(split fused rb w32))

(printf "=== all four kernels must agree, exactly ===\n")
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
(printf "  ~a\n\n" (if (zero? fails) "all four agree" (format "~a MISMATCHES" fails)))

(define (pad v n)
  (define s (format "~a" v))
  (string-append (make-string (max 0 (- n (string-length s))) #\space) s))

(printf "=== throughput, ms per product ===\n")
(printf "field      n     split   fused      rb     w32   best\n")
(for* ([ncyc (in-list '(8 24))]
       [n (in-list '(512 1024 2048 3072))])
  (define F (make-field ncyc))
  (define lim (if (= ncyc 8) 128 32))
  (define A (zmat-build F n n (lambda (i j t) (- (random (* 2 lim)) lim))))
  (define (timeit mode)
    (parameterize ([current-gpu-kernel mode])
      (void (gpu-matmul A A #:audit? #f))
      (define t (current-inexact-milliseconds))
      (for ([_ (in-range 3)]) (void (gpu-matmul A A #:audit? #f)))
      (/ (- (current-inexact-milliseconds) t) 3)))
  (define ts (for/list ([m (in-list MODES)]) (timeit m)))
  (define bi (argmin (lambda (i) (list-ref ts i)) '(0 1 2 3)))
  (printf "Q(z~a) ~a  ~a  ~a  ~a  ~a   ~a ~ax\n"
          ncyc
          (pad n 5)
          (pad (round (first ts)) 6)
          (pad (round (second ts)) 6)
          (pad (round (third ts)) 6)
          (pad (round (fourth ts)) 6)
          (list-ref MODES bi)
          (/ (round (* 10 (/ (first ts) (list-ref ts bi)))) 10.0)))

(gpu-shutdown!)
(when (> fails 0) (exit 1))
