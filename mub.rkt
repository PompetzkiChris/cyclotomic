#lang racket/base
;; Exact mutually-unbiased-basis verification over cyclotomic fields.
;;
;; Bases B and C of C^d are unbiased when |<b_i, c_j>|^2 = 1/d for every i, j.
;; Here that is an equation between exact rationals, so `mubs?` answers yes or
;; no. There is no threshold, and no place to put one.
;;
;; Convention: basis vectors are ROWS, <u,v> = sum_k u_k conj(v_k).

(require racket/contract
         racket/list
         "field.rkt"
         "matrix.rkt")

(provide
 (contract-out
  [unbiasedness   (-> matrix? matrix? (or/c exact-rational? #f))]
  [unbiased?      (-> matrix? matrix? exact-positive-integer? boolean?)]
  [mutually-unbiased? (-> (listof matrix?) exact-positive-integer? boolean?)]
  [mubs-d2        (-> cyclofield? (listof (cons/c string? matrix?)))]
  [mubs-d3        (-> cyclofield? (listof (cons/c string? matrix?)))]
  [mubs-d4        (-> cyclofield? (listof (cons/c string? matrix?)))]
  [mubs-d5        (-> cyclofield? (listof (cons/c string? matrix?)))]
  [mubs-d6        (-> (listof (cons/c string? matrix?)))]
  [mubs-d12       (-> (listof (cons/c string? matrix?)))]))

;; The common value of |<b_i,c_j>|^2, or #f if it is not constant / not rational.
(define (unbiasedness b c)
  (mat-constant-rational (mat-entrywise-abs2 (mat* b (mat-adjoint c)))))

(define (unbiased? b c d)
  (equal? (unbiasedness b c) (/ 1 d)))

(define (mutually-unbiased? ms d)
  (and (for/and ([m (in-list ms)]) (mat-unitary? m))
       (for*/and ([i (in-range (length ms))]
                  [j (in-range (add1 i) (length ms))])
         (unbiased? (list-ref ms i) (list-ref ms j) d))))

;; ------------------------------------------------------------- dimension 2
;; Eigenbases of Z, X, Y. The complete set of 3.
(define (mubs-d2 f)
  (define one (cyc-one f))
  (define zero (cyc-zero f))
  (define s (field-inv-sqrt f 2))
  (define i (field-i f))
  (list
   (cons "Z" (mat f (list (list one zero) (list zero one))))
   (cons "X" (mat f (list (list s s) (list s (cyc-negate s)))))
   (cons "Y" (mat f (list (list s (cyc* s i))
                          (list s (cyc-negate (cyc* s i))))))))

;; ------------------------------------------------------------- dimension 3
;; B_k[j][m] = omega^(k m^2 + j m) / sqrt3. Complete set of 4 with the
;; computational basis.
(define (mubs-d3 f)
  (define one (cyc-one f))
  (define zero (cyc-zero f))
  (define s (field-inv-sqrt f 3))
  (define w (cyc-zeta f (quotient (cyclofield-n f) 3)))
  (cons
   (cons "Z" (mat f (for/list ([a (in-range 3)])
                      (for/list ([b (in-range 3)])
                        (if (= a b) one zero)))))
   (for/list ([k (in-range 3)])
     (cons (format "B~a" k)
           (mat f (for/list ([j (in-range 3)])
                    (for/list ([m (in-range 3)])
                      (cyc* s (cyc-expt w (modulo (+ (* k m m) (* j m)) 3))))))))))

;; ------------------------------------------------------------- dimension 6
;; Tensoring a d=2 set with a d=3 set gives min(3,4) = 3 unbiased bases in
;; dimension 6. Everything is lifted into Q(zeta_24) first, the smallest
;; cyclotomic field holding both 1/sqrt2 and 1/sqrt3.
(define (mubs-d6)
  (define F8 (make-field 8))
  (define F12 (make-field 12))
  (define F24 (make-field 24))
  (define two (for/list ([p (in-list (mubs-d2 F8))])
                (cons (car p) (mat-lift (cdr p) F24))))
  (define three (for/list ([p (in-list (mubs-d3 F12))])
                  (cons (car p) (mat-lift (cdr p) F24))))
  (define (pick lst name) (cdr (assoc name lst)))
  (for/list ([pair (in-list '(("Z" "Z") ("X" "B0") ("Y" "B1")))])
    (define a (car pair))
    (define b (cadr pair))
    (cons (format "~a(x)~a" a b)
          (mat-kron (pick two a) (pick three b)))))

;; ------------------------------------------------------------- dimension 4
;; d = 4 = 2^2 is a prime power, so a COMPLETE set of 5 MUBs exists. Over Z/4
;; the Heisenberg-Weyl trick (which is what mubs-d3 is) does not give it -- 4 is
;; not prime. The complete set comes from the Galois ring GR(4,2) = Z4[xi] with
;; xi^2 = 3 + 3 xi and xi^3 = 1. Its Teichmuller set T = {0, 1, xi, xi^2} has
;; |T| = 4 = d, and the trace to Z4 is the Z4-linear map tr(a + b xi) = 2a + 3b
;; (NOT z + z^2 -- squaring is not the Frobenius in characteristic 4, the cross
;; term 2xi does not vanish).
;;
;;   (v_b^a)[x] = (1/2) i^{ tr((a + 2b) x) },   a, b, x in T
;;
;; gives, for each a in T, an orthonormal basis unbiased to the computational
;; one and to every other; 5 bases in all. 1/sqrt4 = 1/2 is rational, so the
;; only irrationality is i, and the field needs only 4 | n.
(define (mubs-d4 f)
  (define n (cyclofield-n f))
  (unless (zero? (modulo n 4))
    (error 'mubs-d4 "need 4 | n to hold i; Q(zeta_~a) does not" n))
  (define half (cyc-rational f 1/2))
  (define i (cyc-zeta f (quotient n 4)))     ; a primitive 4th root of unity
  ;; GR(4,2) arithmetic on (a . b) = a + b*xi in Z4
  (define (r+ p q) (cons (modulo (+ (car p) (car q)) 4) (modulo (+ (cdr p) (cdr q)) 4)))
  (define (r* p q)
    (let ([a (car p)] [b (cdr p)] [c (car q)] [d (cdr q)])
      (cons (modulo (+ (* a c) (* 3 b d)) 4)
            (modulo (+ (* a d) (* b c) (* 3 b d)) 4))))
  (define (r2 p) (cons (modulo (* 2 (car p)) 4) (modulo (* 2 (cdr p)) 4)))
  (define (tr z) (modulo (+ (* 2 (car z)) (* 3 (cdr z))) 4))
  (define T (list (cons 0 0) (cons 1 0) (cons 0 1) (r* (cons 0 1) (cons 0 1))))
  (define (basis-a a)
    (mat f (for/list ([b (in-list T)])
             (for/list ([x (in-list T)])
               (cyc* half (cyc-expt i (tr (r* (r+ a (r2 b)) x))))))))
  (cons
   (cons "Z" (mat f (for/list ([r (in-range 4)])
                      (for/list ([c (in-range 4)])
                        (if (= r c) (cyc-one f) (cyc-zero f))))))
   (for/list ([a (in-list T)] [k (in-naturals)])
     (cons (format "A~a" k) (basis-a a)))))

(define (mubs-d5 f)
  (define one (cyc-one f))
  (define zero (cyc-zero f))
  (define s (field-inv-sqrt f 5))
  (define w (cyc-zeta f (quotient (cyclofield-n f) 5)))
  (cons
   (cons "Z" (mat f (for/list ([a (in-range 5)])
                      (for/list ([b (in-range 5)]) (if (= a b) one zero)))))
   (for/list ([k (in-range 5)])
     (cons (format "B~a" k)
           (mat f (for/list ([j (in-range 5)])
                    (for/list ([m (in-range 5)])
                      (cyc* s (cyc-expt w (modulo (+ (* k m m) (* j m)) 5))))))))))

;; ------------------------------------------------------------- dimension 12
;; d = 12 = 4 * 3, and 4 and 3 are coprime prime powers. Tensoring a k-set in
;; d1 with a k-set in d2 index-wise gives a k-set in d1*d2, so min(5, 4) = 4
;; mutually unbiased bases in dimension 12 -- the standard tensor lower bound,
;; and more than the elementary 3 that 2 (x) 2 (x) 3 would give.
;;
;; Everything is built in Q(zeta_24), the smallest cyclotomic field holding both
;; i (from d=4) and 1/sqrt3 with a cube root of unity (from d=3). deg 24 = 8 is
;; a power of two, so a product here runs on the Karatsuba GPU kernel.
;;
;; Whether N(12) exceeds 4 is open; nothing here bears on it.
(define (mubs-d12)
  (define F12 (make-field 12))
  (define F24 (make-field 24))
  (define four (mubs-d4 F24))                                  ; already over Q(zeta_24)
  (define three (for/list ([p (in-list (mubs-d3 F12))])
                  (cons (car p) (mat-lift (cdr p) F24))))
  (define (pick lst name) (cdr (assoc name lst)))
  (for/list ([pair (in-list '(("Z" "Z") ("A0" "B0") ("A1" "B1") ("A2" "B2")))])
    (cons (format "~a(x)~a" (car pair) (cadr pair))
          (mat-kron (pick four (car pair)) (pick three (cadr pair))))))
