#lang racket/base
;; Exact integer polynomial arithmetic, and the cyclotomic polynomials.
;;
;; Polynomials are vectors of exact integers, index = degree, low to high.
;; Every operation here is exact; no inexact number is ever constructed.

(require racket/vector
         racket/contract
         (only-in math/number-theory divisors))

(provide
 (contract-out
  [poly-degree      (-> (vectorof exact-integer?) exact-integer?)]
  [poly-trim        (-> (vectorof exact-integer?) (vectorof exact-integer?))]
  [poly-mul         (-> (vectorof exact-integer?) (vectorof exact-integer?)
                        (vectorof exact-integer?))]
  [poly-sub         (-> (vectorof exact-integer?) (vectorof exact-integer?)
                        (vectorof exact-integer?))]
  [poly-divide-exact (-> (vectorof exact-integer?) (vectorof exact-integer?)
                         (vectorof exact-integer?))]
  [cyclotomic-poly  (-> exact-positive-integer? (vectorof exact-integer?))]
  [totient-of       (-> exact-positive-integer? exact-nonnegative-integer?)]))

;; ---------------------------------------------------------------- basics

(define (poly-trim p)
  (let loop ([i (sub1 (vector-length p))])
    (cond
      [(< i 0) (vector 0)]
      [(zero? (vector-ref p i)) (loop (sub1 i))]
      [else (vector-copy p 0 (add1 i))])))

(define (poly-degree p)
  (let ([t (poly-trim p)])
    (if (and (= 1 (vector-length t)) (zero? (vector-ref t 0)))
        -1
        (sub1 (vector-length t)))))

(define (poly-sub a b)
  (define n (max (vector-length a) (vector-length b)))
  (define out (make-vector n 0))
  (for ([i (in-range n)])
    (vector-set! out i
                 (- (if (< i (vector-length a)) (vector-ref a i) 0)
                    (if (< i (vector-length b)) (vector-ref b i) 0))))
  (poly-trim out))

(define (poly-mul a b)
  (define la (vector-length a))
  (define lb (vector-length b))
  (define out (make-vector (max 1 (+ la lb -1)) 0))
  (for* ([i (in-range la)]
         [j (in-range lb)])
    (define ai (vector-ref a i))
    (unless (zero? ai)
      (define bj (vector-ref b j))
      (unless (zero? bj)
        (vector-set! out (+ i j) (+ (vector-ref out (+ i j)) (* ai bj))))))
  (poly-trim out))

;; Exact division. Raises if the division leaves a remainder, which for the
;; cyclotomic recursion would mean the mathematics is wrong, not the input.
(define (poly-divide-exact num den)
  (define n (poly-trim num))
  (define d (poly-trim den))
  (define dd (poly-degree d))
  (when (< dd 0) (error 'poly-divide-exact "division by the zero polynomial"))
  (define lead (vector-ref d dd))
  (define rem (vector-copy n))
  (define qdeg (- (poly-degree n) dd))
  (cond
    [(< qdeg 0) (vector 0)]
    [else
     (define q (make-vector (add1 qdeg) 0))
     (for ([k (in-range qdeg -1 -1)])
       (define rk (vector-ref rem (+ k dd)))
       (unless (zero? rk)
         (define-values (c r) (quotient/remainder rk lead))
         (unless (zero? r)
           (error 'poly-divide-exact "not an exact division"))
         (vector-set! q k c)
         (for ([i (in-range (add1 dd))])
           (vector-set! rem (+ k i)
                        (- (vector-ref rem (+ k i)) (* c (vector-ref d i)))))))
     (unless (= -1 (poly-degree rem))
       (error 'poly-divide-exact "nonzero remainder"))
     (poly-trim q)]))

;; ---------------------------------------------------- cyclotomic polynomials

;; x^n - 1 = product over d | n of Phi_d(x),
;; so Phi_n = (x^n - 1) / product of Phi_d for proper divisors d.
(define cyclo-cache (make-hasheqv))

(define (cyclotomic-poly n)
  (hash-ref!
   cyclo-cache n
   (lambda ()
     (define xn-1
       (let ([v (make-vector (add1 n) 0)])
         (vector-set! v 0 -1)
         (vector-set! v n 1)
         v))
     (define lower
       (for/fold ([acc (vector 1)])
                 ([d (in-list (divisors n))]
                  #:when (< d n))
         (poly-mul acc (cyclotomic-poly d))))
     (poly-divide-exact xn-1 lower))))

(define (totient-of n)
  (sub1 (vector-length (cyclotomic-poly n))))
