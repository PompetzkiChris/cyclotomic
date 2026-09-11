#lang racket/base
;; Karatsuba has to be exactly the product, not nearly it.
;;
;; The danger with a faster multiplication is that it is faster and wrong, and
;; wrong by a little, on some inputs. So it is checked three ways:
;;
;;   1. the algebra alone, against direct convolution, in Racket
;;   2. the matrix the kernel is actually handed, against cyc* -- the ordinary
;;      field multiplication the rest of the package uses
;;   3. the kernel itself, against the kernels that do 4^L multiplications
;;
;; and the range certificate is checked to hold the line rather than be hoped
;; for: operands too large for Karatsuba must make 'auto choose something else
;; and make an explicit request raise.

(require rackunit
         rackunit/text-ui
         racket/list
         racket/vector
         "../field.rkt"
         "../karatsuba.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(define (conv a b)
  (define la (vector-length a))
  (define lb (vector-length b))
  (define o (make-vector (sub1 (+ la lb)) 0))
  (for* ([i (in-range la)] [j (in-range lb)])
    (vector-set! o (+ i j) (+ (vector-ref o (+ i j))
                              (* (vector-ref a i) (vector-ref b j)))))
  o)

(define (rand-vec len lim)
  (for/vector ([_ (in-range len)]) (- (random (* 2 lim)) lim)))

(define levels-tests
  (test-suite
   "which degrees halve"
   (check-equal? (kara-levels 1) 0)
   (check-equal? (kara-levels 2) 1)
   (check-equal? (kara-levels 4) 2)
   (check-equal? (kara-levels 8) 3)
   (check-equal? (kara-levels 16) 4)
   (check-false (kara-levels 6) "6 is not a power of two")
   (check-false (kara-levels 10))
   (check-false (kara-levels 12))
   ;; and the whole point: fewer multiplications
   (for ([l (in-range 1 5)])
     (check-true (< (kara-products l) (expt 2 (* 2 l)))
                 (format "3^~a < 4^~a" l l)))
   (check-equal? (kara-products 3) 27 "27 multiplications at degree 8, not 64")))

(define algebra-tests
  (test-suite
   "fold after mac is the convolution, exactly"
   (for ([l (in-range 0 5)])
     (define deg (expt 2 l))
     (for ([_ (in-range 200)])
       (define a (rand-vec deg 5000))
       (define b (rand-vec deg 5000))
       (check-equal? (kara-fold (kara-mac a b) l) (conv a b)
                     (format "degree ~a" deg)))
     ;; and the product count is what it claims
     (check-equal? (vector-length (kara-mac (rand-vec deg 3) (rand-vec deg 3)))
                   (kara-products l)))
   ;; zero in, zero out, and no spurious dependence on order
   (let* ([a (rand-vec 8 100)]
          [z (make-vector 8 0)])
     (check-equal? (kara-fold (kara-mac a z) 3) (make-vector 15 0))
     (check-equal? (kara-fold (kara-mac a a) 3) (conv a a)))))

(define matrix-tests
  (test-suite
   "the matrix handed to the kernel reproduces cyc*"
   (for ([n (in-list '(3 4 5 6 8 12 16 24))])
     (define F (make-field n))
     (define deg (field-degree F))
     (define l (kara-levels deg))
     (when l
       (define-values (K rowsum) (kara-output-matrix F))
       (check-equal? (vector-length K) deg "one row per output coefficient")
       (check-equal? (vector-length (vector-ref K 0)) (kara-products l)
                     "one column per product")
       ;; rowsum is the exact max row sum of |K|, which the bound depends on
       (check-equal? rowsum
                     (for/fold ([best 0]) ([row (in-vector K)])
                       (max best (for/sum ([x (in-vector row)]) (abs x)))))
       (for ([_ (in-range 200)])
         (define av (rand-vec deg 300))
         (define bv (rand-vec deg 300))
         (define P (kara-mac av bv))
         (define via-K
           (for/vector ([t (in-range deg)])
             (for/sum ([j (in-range (kara-products l))])
               (* (vector-ref (vector-ref K t) j) (vector-ref P j)))))
         (check-equal? via-K
                       (cyc-coeffs (cyc* (make-cyc F (vector->list av))
                                         (make-cyc F (vector->list bv))))
                       (format "Q(zeta_~a)" n)))))))

(define growth-tests
  (test-suite
   "the range cost is stated, not discovered"
   ;; each multiplicand is a sum of up to 2^L coefficients
   (check-equal? (kara-operand-growth 3) 8)
   (check-equal? (kara-growth 3) 64 "and each accumulator carries 4^L")
   ;; the growth is real: build the worst case and watch it happen
   (for ([l (in-range 1 4)])
     (define deg (expt 2 l))
     (define lim 1000)
     (define a (make-vector deg lim))
     (define b (make-vector deg lim))
     (define P (kara-mac a b))
     (check-true (<= (for/fold ([m 0]) ([x (in-vector P)]) (max m (abs x)))
                     (* (kara-growth l) lim lim))
                 (format "no product exceeds 4^~a * lim^2" l)))))

;; ------------------------------------------------------------------ the device

(define device-tests
  (test-suite
   "the kernel agrees with the kernels that do more multiplications"
   (gpu-init!)
   (for* ([n (in-list '(3 4 6 8 12 16 24))]
          [sz (in-list '(1 2 15 16 17 63 130))])
     (define F (make-field n))
     (define A (zmat-build F sz (+ sz 3) (lambda (i j t) (- (random 128) 64))))
     (define B (zmat-build F (+ sz 3) (+ sz 1) (lambda (i j t) (- (random 128) 64))))
     (define ref (parameterize ([current-gpu-kernel 'split])
                   (zmat-planes (gpu-matmul A B))))
     (for ([m (in-list '(ultra kara))])
       (check-equal? (parameterize ([current-gpu-kernel m])
                       (zmat-planes (gpu-matmul A B)))
                     ref
                     (format "~a, Q(zeta_~a), ~ax~a" m n sz sz))))
   ;; 'auto reaches for Karatsuba, and that is one launch
   (let* ([F (make-field 24)]
          [A (zmat-build F 48 48 (lambda (i j t) (- (random 32) 16)))])
     (gpu-reset-stats!)
     (void (gpu-matmul A A))
     (check-equal? (hash-ref (gpu-stats) 'launches) 1
                   "one launch for the whole field product"))
   ;; and it uploads the narrow image, not the wide one
   (let* ([F (make-field 24)]
          [A (zmat-build F 64 64 (lambda (i j t) (- (random 32) 16)))]
          [d (field-degree F)])
     (check-equal? (bytes-length (zmat-narrow A)) (* d 64 64 4)
                   "the int32 image exists and is half the width")
     (gpu-reset-stats!)
     (void (gpu-matmul A A #:audit? #f))
     (check-equal? (hash-ref (gpu-stats) 'uploaded-bytes) (* 2 d 64 64 4)
                   "exactly two int32 operands crossed PCIe"))
   (gpu-shutdown!)))

(define certificate-tests
  (test-suite
   "operands too large for Karatsuba do not get it anyway"
   (gpu-init!)
   ;; Karatsuba feeds sums of up to 2^L coefficients to a 32x32 multiply, so at
   ;; degree 8 a coefficient of 2^28 is the boundary and 2^29 is past it -- while
   ;; the narrow kernels that multiply coefficients directly are still fine with
   ;; it. Only one operand is large, so the direct bound still fits int64 and
   ;; there is something for the fallback to do.
   (let* ([F (make-field 24)]
          [big (expt 2 29)]
          [A (zmat-build F 8 8 (lambda (i j t) (if (and (= i j) (= t 0)) big 0)))]
          [B (zmat-build F 8 8 (lambda (i j t) (if (= t 0) 1 0)))])
     (check-equal? (zmat-absmax A) big "the matrix knows its own peak")
     (check-true (and (zmat-narrow A) #t) "2^29 still fits int32")
     ;; an explicit request must refuse rather than narrow something that does
     ;; not fit
     (check-exn exn:fail?
                (lambda () (parameterize ([current-gpu-kernel 'kara])
                             (gpu-matmul A B)))
                "kara refuses operands it cannot represent")
     ;; and 'auto must still produce the right answer by another route
     (define via-auto (zmat-planes (gpu-matmul A B)))
     (define via-split (parameterize ([current-gpu-kernel 'split])
                         (zmat-planes (gpu-matmul A B))))
     (check-equal? via-auto via-split "auto fell back and got the same answer"))
   ;; a matrix with a coefficient past int32 has no narrow image at all, so
   ;; every narrow kernel is out and the int64 path takes it
   (let* ([F (make-field 8)]
          [A (zmat-build F 4 4 (lambda (i j t) (if (and (= i 0) (= j 0) (= t 0))
                                                   (expt 2 40) 0)))]
          [B (zmat-build F 4 4 (lambda (i j t) (if (= t 0) 1 0)))])
     (check-false (zmat-narrow A) "no int32 image when a coefficient does not fit")
     (check-equal? (zmat-absmax A) (expt 2 40))
     (check-exn exn:fail?
                (lambda () (parameterize ([current-gpu-kernel 'ultra])
                             (gpu-matmul A B)))
                "and the narrow kernels say so instead of truncating")
     (define via-auto (zmat-planes (gpu-matmul A B)))
     (define via-split (parameterize ([current-gpu-kernel 'split])
                         (zmat-planes (gpu-matmul A B))))
     (check-equal? via-auto via-split "the wide path agrees"))
   (gpu-shutdown!)))

(module+ test
  (void (run-tests (test-suite "karatsuba"
                               levels-tests algebra-tests matrix-tests
                               growth-tests device-tests certificate-tests))))
