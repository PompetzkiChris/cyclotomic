#lang racket/base
;; Exact Z[zeta_n] products from Racket, on the GPU, timed against pure Racket.

(require racket/list
         "../exact-io.rkt"
         "../field.rkt"
         "../matrix.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(define (rand-rows f r c lim)
  (for/list ([_ (in-range r)])
    (for/list ([_ (in-range c)])
      (make-cyc f (for/list ([_ (in-range (field-degree f))])
                    (- (random (* 2 lim)) lim))))))

(define (gb b) (bytes->gb b))

(gpu-init!)
(printf "device: ~a\n" (cuda-device-name 0))
(let-values ([(free total) (cuda-mem-info)])
  (printf "VRAM  : ~a GB free of ~a GB\n\n" (gb free) (gb total)))

;; ---------------------------------------------------------------- agreement
(printf "correctness against pure Racket\n")
(for ([n (in-list '(8 24))])
  (define f (make-field n))
  (define sz 96)
  (define A (rand-rows f sz sz 40))
  (define B (rand-rows f sz sz 40))
  (define t0 (now-ms))
  (define cpu (mat* (mat f A) (mat f B)))
  (define t1 (now-ms))
  (define g (zmat->matrix (gpu-matmul (zmat-of-integers f A) (zmat-of-integers f B))))
  (define t2 (now-ms))
  (printf "  Q(zeta_~a) ~ax~a : racket ~a ms   gpu ~a ms   ~a\n"
          n sz sz
          (round (- t1 t0)) (round (- t2 t1))
          (if (mat=? g cpu) "EXACT" "MISMATCH"))
  (unless (mat=? g cpu) (error 'bench "GPU disagreed with pure Racket")))

;; -------------------------------------------------------------------- scale
(printf "\nscale, GPU only (every product bounds-checked and audited)\n")
(for ([n (in-list '(8 24))])
  (define f (make-field n))
  (define d (field-degree f))
  (for ([sz (in-list '(256 512 1024))])
    (define lim (case n [(8) 512] [else 128]))
    (define A (zmat-of-integers f (rand-rows f sz sz lim)))
    (define B (zmat-of-integers f (rand-rows f sz sz lim)))
    ;; warm up so the first launch's overhead is not counted
    (void (gpu-matmul A B #:audit? #f))
    (define t0 (now-ms))
    (define C (gpu-matmul A B))
    (define t1 (now-ms))
    (define secs (/ (- t1 t0) 1000))
    (define ops (* d d 2 (expt sz 3)))
    (printf "  Q(zeta_~a) ~ax~a : ~a ms   ~a Giga-integer-ops/s   ~a launches\n"
            n sz sz (round (- t1 t0))
            (if (zero? secs) "inf" (dec (/ ops secs 1000000000) 0))
            (* d d))))

(gpu-shutdown!)
(printf "\nEvery product above was bounds-checked before launch and audited after.\n")
