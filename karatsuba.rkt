#lang racket/base
;; Karatsuba for the cyclotomic convolution, exactly.
;;
;; A product in Z[zeta_n] is the convolution of two length-deg coefficient
;; vectors followed by reduction mod Phi_n. Done directly that is deg^2
;; coefficient multiplications, and on the GPU each one is a full matrix
;; product -- 64 of them for Q(zeta_24). Measured on this device, time scales
;; almost linearly with that count: 0.55 ms per plane product at 2048x2048,
;; on top of 10 ms of fixed cost. So the way to make the product faster is to
;; do fewer of them, not to tune the loop that does them.
;;
;; Karatsuba does a length-2L convolution with three length-L convolutions
;; instead of four:
;;
;;   a = a0 + X a1,  b = b0 + X b1,   X = x^L
;;   Z0 = a0 b0,  Z2 = a1 b1,  Z1 = (a0+a1)(b0+b1) - Z0 - Z2
;;   a b = Z0 + X Z1 + X^2 Z2
;;
;; Recursively, length 2^L costs 3^L multiplications: 27 instead of 64 at
;; deg 8, 9 instead of 16 at deg 4, 3 instead of 4 at deg 2.
;;
;; This is not an approximation of the product. Every coefficient of the result
;; is the same integer it would have been; only the intermediate sums differ,
;; and they are integers too. There is nothing to round.
;;
;; The split here is deliberate: the recursion that *forms* the 3^L products
;; lives in the kernel, where it is 3^L multiply-accumulates per thread. The
;; recursion that *recombines* them is linear, so it collapses into a single
;; integer matrix, computed here, composed with the Phi_n reduction, and handed
;; to the kernel as a deg x 3^L table. The kernel does no recursion on the way
;; out -- one small matrix apply per output element -- and no algebra lives in
;; CUDA that can disagree with the algebra in Racket.

(require racket/list
         racket/vector
         "field.rkt")

(provide kara-levels
         kara-products
         kara-mac
         kara-fold
         kara-fold-matrix
         kara-output-matrix
         kara-growth
         kara-operand-growth)

;; How many halvings a degree admits: deg = 2^L, or #f if deg is not a power
;; of two. Only powers of two get a kernel; everything else uses the direct
;; product, which is correct for any degree.
(define (kara-levels deg)
  (and (exact-positive-integer? deg)
       (let loop ([d deg] [l 0])
         (cond [(= d 1) l]
               [(odd? d) #f]
               [else (loop (quotient d 2) (add1 l))]))))

(define (kara-products l) (expt 3 l))

;; ---------------------------------------------------------------- the products
;;
;; P_j for j < 3^L, in exactly the order the kernel forms them: the low half's
;; products first, then the high half's, then the sum half's. `a` and `b` may
;; hold anything that +, * and 0 make sense for, which is what lets the same
;; code be run on integers to compute and on basis vectors to derive.

(define (kara-mac a b [zero 0] [add +] [mul *])
  (define l (kara-levels (vector-length a)))
  (unless l (error 'kara-mac "length ~a is not a power of two" (vector-length a)))
  (define out (make-vector (kara-products l) zero))
  (let rec ([a a] [b b] [off 0])
    (define len (vector-length a))
    (cond
      [(= len 1)
       (vector-set! out off (mul (vector-ref a 0) (vector-ref b 0)))]
      [else
       (define h (quotient len 2))
       (define alo (vector-copy a 0 h))
       (define ahi (vector-copy a h len))
       (define blo (vector-copy b 0 h))
       (define bhi (vector-copy b h len))
       (define asum (for/vector ([x (in-vector alo)] [y (in-vector ahi)]) (add x y)))
       (define bsum (for/vector ([x (in-vector blo)] [y (in-vector bhi)]) (add x y)))
       (define m (kara-products (kara-levels h)))
       (rec alo blo off)
       (rec ahi bhi (+ off m))
       (rec asum bsum (+ off (* 2 m)))]))
  out)

;; ------------------------------------------------------------- recombination
;;
;; P (length 3^L) -> the raw convolution (length 2*2^L - 1). Linear in P, which
;; is the whole point: it can be applied to basis vectors to get a matrix.

(define (kara-fold P l [zero 0] [add +] [sub -])
  (let rec ([l l] [off 0])
    (cond
      [(zero? l) (vector (vector-ref P off))]
      [else
       (define h (expt 2 (sub1 l)))
       (define m (kara-products (sub1 l)))
       (define z0 (rec (sub1 l) off))
       (define z2 (rec (sub1 l) (+ off m)))
       (define zs (rec (sub1 l) (+ off (* 2 m))))
       (define cc (vector-length z0))
       (define zm (for/vector ([s (in-vector zs)] [x (in-vector z0)] [y (in-vector z2)])
                    (sub s (add x y))))
       (define out (make-vector (sub1 (* 2 (expt 2 l))) zero))
       (define (bump! base v)
         (for ([x (in-vector v)] [i (in-naturals)])
           (vector-set! out (+ base i) (add (vector-ref out (+ base i)) x))))
       (bump! 0 z0)
       (bump! h zm)
       (bump! (* 2 h) z2)
       out])))

;; The (2*2^L - 1) x 3^L integer matrix with raw = M P, obtained by folding each
;; basis vector. Exact by construction; checked against the direct convolution
;; in tests/karatsuba-tests.rkt.
(define (kara-fold-matrix l)
  (define np (kara-products l))
  (define nraw (sub1 (* 2 (expt 2 l))))
  (define cols
    (for/list ([j (in-range np)])
      (define e (make-vector np 0))
      (vector-set! e j 1)
      (kara-fold e l)))
  (for/vector ([m (in-range nraw)])
    (for/vector ([col (in-list cols)]) (vector-ref col m))))

;; What the kernel is actually given: out_t = sum_j K[t][j] P_j, with the
;; Phi_n reduction already folded in, so the device applies one matrix instead
;; of unwinding a recursion it could get wrong.
;;
;;   out_t = sum_m R[m][t] raw_m = sum_m R[m][t] sum_j M[m][j] P_j
;; (values K maxrowsum)  -- maxrowsum is the exact max_t sum_j |K[t][j]|, which
;; is what the overflow certificate needs.
(define (kara-output-matrix f)
  (define deg (field-degree f))
  (define l (kara-levels deg))
  (unless l (error 'kara-output-matrix "degree ~a is not a power of two" deg))
  (define np (kara-products l))
  (define M (kara-fold-matrix l))
  (define R (cyclofield-pow f))            ; R[m][t], m < 2*deg-1
  (define nraw (vector-length M))
  (define K
    (for/vector ([t (in-range deg)])
      (for/vector ([j (in-range np)])
        (for/sum ([m (in-range nraw)])
          (* (vector-ref (vector-ref R m) t)
             (vector-ref (vector-ref M m) j))))))
  (define rowsum
    (for/fold ([best 0]) ([row (in-vector K)])
      (max best (for/sum ([x (in-vector row)]) (abs x)))))
  (values K rowsum))

;; Each operand fed to a multiply is a sum of at most 2^L original
;; coefficients, one per halving, so |operand| <= 2^L * max|coeff|. The kernel
;; keeps operands in int32, which is what this has to respect.
(define (kara-operand-growth l) (expt 2 l))

;; And each product accumulator is bounded by k * (2^L maxA) * (2^L maxB),
;; which this returns as the 4^L factor; the output bound then multiplies by
;; the K row sum from kara-output-matrix, which is exact rather than estimated.
(define (kara-growth l) (expt 4 l))
