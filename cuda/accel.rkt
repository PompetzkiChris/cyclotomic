#lang racket/base
;; Put the GPU on the actual math path.
;;
;; Requiring this module makes every `mat*` in the program try the device
;; first. The answer is identical either way -- both paths are exact -- so
;; nothing downstream has to know or care which one ran. mub.rkt, the test
;; suite, anything: they all start using the card without being rewritten.
;;
;; Entries of Q(zeta_n) are rationals. The device path is Z[zeta_n], so the
;; common denominator is cleared first, the integer product runs on the card,
;; and the denominator is put back. Exact in, exact out.
;;
;; The hook DECLINES rather than fails: wrong field, product too large for
;; int64, device unavailable, anything unexpected -- it returns #f and mat*
;; falls through to the exact CPU path. A GPU problem can slow this down. It
;; cannot change an answer.

(require racket/list
         "../field.rkt"
         "../matrix.rkt"
         "driver.rkt"
         "gpu.rkt")

(provide gpu-accel-install!
         gpu-accel-uninstall!
         current-gpu-min-work
         gpu-accel-declines)

;; Products smaller than this many multiply-accumulates stay on the CPU;
;; 0 means always try the device.
(define current-gpu-min-work (make-parameter 0))

(define declines 0)
(define (gpu-accel-declines) declines)

(define (clear-denominators m)
  ;; returns (values rows-of-integer-cyc scale) with entry = original * scale
  (define f (mat-field m))
  (define r (mat-rows m))
  (define c (mat-cols m))
  (define L
    (for*/fold ([l 1]) ([i (in-range r)] [j (in-range c)]
                        [x (in-vector (cyc-coeffs (mat-ref m i j)))])
      (lcm l (denominator x))))
  (values (for/list ([i (in-range r)])
            (for/list ([j (in-range c)])
              (cyc-scale (mat-ref m i j) L)))
          L))

(define (gpu-mat* a b)
  (with-handlers ([exn:fail? (lambda (e) (set! declines (add1 declines)) #f)])
    (define f (mat-field a))
    (cond
      [(not (current-gpu-ready?)) (set! declines (add1 declines)) #f]
      [(not (= (cyclofield-n f) (cyclofield-n (mat-field b))))
       (set! declines (add1 declines)) #f]
      [(not (= (mat-cols a) (mat-rows b))) (set! declines (add1 declines)) #f]
      [(< (* (mat-rows a) (mat-cols a) (mat-cols b)) (current-gpu-min-work))
       (set! declines (add1 declines)) #f]
      [else
       (define-values (ra la) (clear-denominators a))
       (define-values (rb lb) (clear-denominators b))
       (define prod (zmat->matrix (gpu-matmul (zmat-of-integers f ra)
                                              (zmat-of-integers f rb))))
       ;; put the denominator back
       (define s (/ 1 (* la lb)))
       (mat f (for/list ([i (in-range (mat-rows prod))])
                (for/list ([j (in-range (mat-cols prod))])
                  (cyc-scale (mat-ref prod i j) s))))])))

(define (gpu-accel-install! #:device [ordinal 0])
  (gpu-init! ordinal)
  (current-mat*-hook gpu-mat*)
  (void))

(define (gpu-accel-uninstall!)
  (current-mat*-hook #f)
  (void))
