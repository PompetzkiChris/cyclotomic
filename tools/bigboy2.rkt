#lang racket/base
;; Maximal load: the largest exact Z[zeta_24] products that fit, plus a
;; sustained back-to-back chain to hold the card at steady state. Still audited
;; exact -- the bound is checked on every product.
(require racket/list cyclotomic/exact-io cyclotomic/field
         cyclotomic/cuda/driver cyclotomic/cuda/gpu)
(gpu-init!)
(define F (make-field 24))
(define (pad v n) (let ([s (format "~a" v)])
  (string-append (make-string (max 0 (- n (string-length s))) #\space) s)))
(define (vram) (let-values ([(free total) (cuda-mem-info)])
                 (round (/ (- total free) 1048576))))

(printf "=== single huge products (Q(zeta_24), Karatsuba) ===\n")
(printf "   N   bytes/mat   VRAM MiB   device ms    GMAC/s\n")
(for ([n (in-list '(4096 6144 8192))])
  (define lim 32)
  (define A (zmat-build F n n (lambda (i j t) (- (random (* 2 lim)) lim))))
  (call-with-dmat A (lambda (dA)
    (dmat-free! (dmat* dA dA #:audit? #t))     ; warm + audit once
    (define t (now-ms))
    (dmat-free! (dmat* dA dA #:audit? #f))
    (define ms (- (now-ms) t))
    (printf "~a  ~a MB  ~a  ~a ms   ~a\n"
            (pad n 5) (pad (round (/ (bytes-length (zmat-planes A)) 1048576)) 6)
            (pad (vram) 8) (pad ms 8)
            (pad (if (zero? ms) "inf" (dec (/ (* 27 n n n) ms 1000000) 0)) 8)))))

(printf "\n=== sustained chain: 40 back-to-back 4096x4096 products ===\n")
(let* ([n 4096] [lim 32]
       [A (zmat-build F n n (lambda (i j t) (- (random (* 2 lim)) lim)))])
  (call-with-dmat A (lambda (dA)
    (dmat-free! (dmat* dA dA #:audit? #f))
    (define reps 40)
    (define t (now-ms))
    (for ([_ (in-range reps)]) (dmat-free! (dmat* dA dA #:audit? #f)))
    (define ms (/ (- (now-ms) t) reps))
    (printf "sustained: ~a ms/product, ~a GMAC/s, ~a products in ~a ms\n"
            (round ms) (round (/ (* 27 n n n) ms 1000000))
            reps (- (now-ms) t)))))
(gpu-stats-report)
(gpu-shutdown!)
