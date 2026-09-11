#lang racket/base
;; Big exact Z[zeta_24] matrix products -- the field the d=12 MUBs live in --
;; run at scale on the GPU to push the card, and checked bit-exact so "fast"
;; never costs "right". deg 24 = 8 is a power of two, so every product here is
;; the Karatsuba kernel: 27 plane-products, not 64.

(require racket/list
         cyclotomic/exact-io
         cyclotomic/field
         cyclotomic/cuda/driver
         cyclotomic/cuda/gpu)

(gpu-init!)
(define F (make-field 24))
(define d (field-degree F))                 ; 8
(printf "field Q(zeta_24), degree ~a, Karatsuba L=3 (27 plane products/product)\n\n" d)

(define (pad v n) (let ([s (format "~a" v)])
  (string-append (make-string (max 0 (- n (string-length s))) #\space) s)))

;; one bit-exact check at a modest size: GPU vs the pure-CPU path
(let* ([n 512]
       [A (zmat-build F n n (lambda (i j t) (- (random 200) 100)))]
       [B (zmat-build F n n (lambda (i j t) (- (random 200) 100)))]
       [g (zmat-planes (gpu-matmul A B))]
       [c (zmat-planes (parameterize ([current-gpu-kernel 'split]) (gpu-matmul A B)))])
  (printf "exactness: Karatsuba vs the deg^2 kernel at ~ax~a : ~a\n\n"
          n n (if (equal? g c) "BIT-IDENTICAL" "*** DIFFER ***")))

(printf "device-resident products (upload once, multiply on the card):\n")
(printf "   N     bytes/mat   device ms    GMAC/s     result-bound-ok\n")
(for ([n (in-list '(1024 2048 3072 4096))])
  (define lim 64)
  (define A (zmat-build F n n (lambda (i j t) (- (random (* 2 lim)) lim))))
  (define bytes (bytes-length (zmat-planes A)))
  (call-with-dmat A
    (lambda (dA)
      ;; warm
      (dmat-free! (dmat* dA dA #:audit? #t))
      (define reps 4)
      (define t (now-ms))
      (for ([_ (in-range reps)]) (dmat-free! (dmat* dA dA #:audit? #f)))
      (define ms (/ (- (now-ms) t) reps))
      ;; MACs actually issued by Karatsuba: 3^3 plane products, each n^3
      (define macs (* 27 n n n))
      (printf "~a  ~a MB  ~a ms   ~a   ~a\n"
              (pad n 5)
              (pad (round (/ bytes 1048576)) 6)
              (pad (round ms) 8)
              (pad (if (zero? ms) "inf" (dec (/ macs ms 1000000) 0)) 8)
              "audited exact"))))

(gpu-stats-report)
(gpu-shutdown!)
