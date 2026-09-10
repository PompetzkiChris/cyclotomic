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
  [mubs-d6        (-> (listof (cons/c string? matrix?)))]))

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
