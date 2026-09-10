#lang racket/base
;; Sustained exact work on the device: upload once, multiply many times.
;;
;; Chains products over Z[zeta_24] without returning to the host between them,
;; which is what it takes to actually load the card. Correctness is checked
;; against the round-tripping path at a small size first.

(require racket/list
         "../field.rkt"
         "../matrix.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(gpu-init!)
(define F (make-field 24))
(define deg (field-degree F))

;; ------------------------------------------------------- correctness first
(printf "=== device-resident products agree with the round-trip path ===\n")
(for ([n (in-list '(1 7 32 64))])
  (define Z (zmat-build F n n (lambda (i j t) (- (modulo (+ (* 5 i) (* 3 j) t) 17) 8))))
  (define viaHost (zmat->matrix (gpu-matmul Z Z)))
  (define viaDev
    (call-with-dmat Z
      (lambda (D)
        (define P (dmat* D D))
        (define out (dmat->zmat P))
        (dmat-free! P)
        (zmat->matrix out))))
  (printf "  ~ax~a : ~a\n" n n (if (mat=? viaHost viaDev) "IDENTICAL" "DIVERGED"))
  (unless (mat=? viaHost viaDev) (error 'sustained "dmat* disagreed")))

;; ------------------------------------------------------------- sustained
(printf "\n=== chained products, nothing returns to the host ===\n")
(define n 3072)
(define reps 24)
(printf "n = ~a over Q(zeta_24), ~a chained products, ~a launches each\n"
        n reps (add1 (* deg deg)))
(printf "one matrix = ~a MB on device\n"
        (round (/ (* deg n n 8) 1048576.0)))

;; small entries so 24 chained products stay inside the int64 bound
(define Z (zmat-build F n n (lambda (i j t) (if (= t 0) (if (= i j) 1 0) 0))))
(define Zp (zmat-build F n n (lambda (i j t)
                               (if (and (= t 1) (= j (modulo (+ i 1) n))) 1 0))))

(gpu-reset-stats!)
(define t0 (current-inexact-milliseconds))
(call-with-dmat Zp
  (lambda (D)
    (let loop ([acc (zmat->dmat Z)] [k 0])
      (cond
        [(= k reps)
         (define out (dmat->zmat acc))
         (dmat-free! acc)
         (printf "  final matrix nonzero coefficient at (0,~a,plane ~a) = ~a\n"
                 (modulo reps n) (modulo reps deg)
                 (zmat-coeff out 0 (modulo reps n) (modulo reps deg)))]
        [else
         (define nxt (dmat* acc D #:audit? #f))
         (dmat-free! acc)
         (loop nxt (add1 k))]))))
(define wall (- (current-inexact-milliseconds) t0))

(printf "\n  wall ~a ms\n" (round wall))
(gpu-stats-report)
(define s (gpu-stats))
(printf "  device share of wall : ~a%\n"
        (round (* 100 (/ (hash-ref s 'device-ms) wall))))
(printf "  sustained rate       : ~a Giga-integer-ops/s\n"
        (round (/ (* reps deg deg 2.0 (expt n 3)) (/ wall 1000.0) 1e9)))
(let-values ([(f t) (cuda-mem-info)])
  (printf "  VRAM free after      : ~a GB of ~a GB\n"
          (/ (round (/ (* 10.0 f) (expt 2 30))) 10.0)
          (/ (round (/ (* 10.0 t) (expt 2 30))) 10.0)))
(gpu-shutdown!)
