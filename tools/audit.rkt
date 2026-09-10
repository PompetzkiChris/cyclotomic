#lang racket/base
;; Audit the CUDA layer for the failure modes that actually bite:
;;   1. device memory leaked when a product raises part-way through
;;   2. VRAM not returned after many successful products
;;   3. the Racket scheduler frozen across a long kernel
;;   4. GC moving a byte string while the driver is reading it

(require racket/list
         "../matrix.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt"
         "../field.rkt")

(define F (make-field 24))
(define d (field-degree F))

(define (mb b) (/ (round (/ (* 10.0 b) (expt 2 20))) 10.0))
(define (free-now) (let-values ([(f t) (cuda-mem-info)]) f))

(define (rand-int-mat n lim)
  (for/list ([_ (in-range n)])
    (for/list ([_ (in-range n)])
      (make-cyc F (for/list ([_ (in-range d)]) (- (random (* 2 lim)) lim))))))

(gpu-init!)
(printf "baseline free: ~a MB\n\n" (mb (free-now)))

;; ---------------------------------------------------------------- 1. leak on error
(printf "1. device memory when a product RAISES mid-way\n")
(define before-err (free-now))
(define big (expt 2 40))
(define A-big (zmat-of-integers F (for/list ([_ (in-range 64)])
                                    (for/list ([_ (in-range 64)])
                                      (make-cyc F (for/list ([_ (in-range d)]) big))))))
(for ([i (in-range 25)])
  (with-handlers ([exn:fail? void])
    (gpu-matmul A-big A-big)))
(define after-err (free-now))
(printf "   25 raising products: free ~a MB -> ~a MB   leaked ~a MB\n"
        (mb before-err) (mb after-err) (mb (- before-err after-err)))
(printf "   verdict: ~a\n\n"
        (if (< (- before-err after-err) (* 4 1024 1024)) "OK" "LEAK"))

;; ---------------------------------------------------------------- 2. steady state
(printf "2. VRAM after many successful products\n")
(define A (zmat-of-integers F (rand-int-mat 128 30)))
(define B (zmat-of-integers F (rand-int-mat 128 30)))
(void (gpu-matmul A B))
(define before-loop (free-now))
(for ([i (in-range 50)]) (void (gpu-matmul A B #:audit? #f)))
(define after-loop (free-now))
(printf "   50 products: free ~a MB -> ~a MB   drift ~a MB\n"
        (mb before-loop) (mb after-loop) (mb (- before-loop after-loop)))
(printf "   verdict: ~a\n\n"
        (if (< (abs (- before-loop after-loop)) (* 4 1024 1024)) "OK" "LEAK"))

;; ---------------------------------------------------------------- 3. scheduler
(printf "3. does a long product freeze other Racket threads?\n")
(define ticks (box 0))
(define stop (box #f))
(define ticker
  (thread (lambda ()
            (let loop ()
              (unless (unbox stop)
                (set-box! ticks (add1 (unbox ticks)))
                (sleep 0.002)
                (loop))))))
(define Abig (zmat-of-integers F (rand-int-mat 640 8)))
(define t0 (current-inexact-milliseconds))
(void (gpu-matmul Abig Abig #:audit? #f))
(define elapsed (- (current-inexact-milliseconds) t0))
(set-box! stop #t)
(sync ticker)
(define expected (max 1 (inexact->exact (round (/ elapsed 2.0)))))
(printf "   product took ~a ms; ticker ran ~a times (roughly ~a if never blocked)\n"
        (round elapsed) (unbox ticks) expected)
(printf "   verdict: ~a\n\n"
        (if (> (unbox ticks) (max 2 (quotient expected 10))) "responsive" "SCHEDULER BLOCKED"))

;; ---------------------------------------------------------------- 4. GC pressure
(printf "4. transfers correct while the GC is churning\n")
(define ok (box #t))
(define churn
  (thread (lambda ()
            (let loop ([n 0])
              (when (< n 400)
                (void (make-bytes 400000 (modulo n 256)))
                (collect-garbage 'minor)
                (loop (add1 n)))))))
(define C1 (zmat->matrix (gpu-matmul A B)))
(sync churn)
(collect-garbage)
(define C2 (zmat->matrix (gpu-matmul A B)))
(printf "   product under GC churn reproduces: ~a\n"
        (if (mat=? C1 C2) "IDENTICAL" "DIVERGED"))
(printf "\nbaseline free at end: ~a MB\n" (mb (free-now)))

;; ---------------------------------------------------------------- 5. shutdown
(printf "
5. does shutdown release everything, including the caches?
")
(define before-shutdown (free-now))
(printf "   before shutdown        : ~a MB free
" (mb before-shutdown))
(gpu-shutdown!)
(gpu-init!)
(define after-cycle (free-now))
(printf "   after shutdown + reinit: ~a MB free
" (mb after-cycle))
(printf "   verdict: ~a
"
        (if (>= after-cycle before-shutdown) "OK, caches released" "SOMETHING RETAINED"))
(gpu-shutdown!)
