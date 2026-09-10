#lang racket/base
;; Exactness tests for the Racket cyclotomic field. Run with:  raco test field-tests.rkt

(require rackunit
         rackunit/text-ui
         racket/vector
         "../poly.rkt"
         "../field.rkt")

(define (v->list v) (vector->list v))

(define cyclotomic-tests
  (test-suite
   "cyclotomic polynomials"

   (check-equal? (v->list (cyclotomic-poly 1))  '(-1 1)          "Phi_1 = x - 1")
   (check-equal? (v->list (cyclotomic-poly 2))  '(1 1)           "Phi_2 = x + 1")
   (check-equal? (v->list (cyclotomic-poly 3))  '(1 1 1)         "Phi_3")
   (check-equal? (v->list (cyclotomic-poly 4))  '(1 0 1)         "Phi_4 = x^2+1")
   (check-equal? (v->list (cyclotomic-poly 6))  '(1 -1 1)        "Phi_6")
   (check-equal? (v->list (cyclotomic-poly 8))  '(1 0 0 0 1)     "Phi_8 = x^4+1")
   (check-equal? (v->list (cyclotomic-poly 12)) '(1 0 -1 0 1)    "Phi_12")
   (check-equal? (v->list (cyclotomic-poly 24)) '(1 0 0 0 -1 0 0 0 1) "Phi_24")

   ;; the first cyclotomic polynomial with a coefficient outside {-1,0,1}
   (check-equal? (vector-ref (cyclotomic-poly 105) 7) -2
                 "Phi_105 has a -2 coefficient")

   (check-equal? (totient-of 24) 8 "phi(24) = 8")
   (check-equal? (totient-of 105) 48 "phi(105) = 48")))

(define (field-suite n)
  (define F (make-field n))
  (define z (cyc-zeta F))
  (test-suite
   (format "Q(zeta_~a)" n)
   (check-true (cyc=? (cyc-expt z n) (cyc-one F)) "zeta^n = 1")
   (check-false (for/or ([k (in-range 1 n)]) (cyc=? (cyc-expt z k) (cyc-one F)))
                "zeta has exact order n")
   (check-equal? (cyc-trace (cyc-one F)) (field-degree F) "Tr(1) = phi(n)")))

(define structure-tests
  (test-suite
   "field structure"
   (field-suite 3) (field-suite 4) (field-suite 8) (field-suite 12) (field-suite 24)

   (let* ([F8 (make-field 8)]
          [s2 (field-sqrt F8 2)]
          [i  (field-i F8)])
     (check-true (cyc=? (cyc-expt s2 2) (cyc-rational F8 2)) "sqrt2^2 = 2")
     (check-true (cyc=? (cyc-expt i 2) (cyc-negate (cyc-one F8))) "i^2 = -1")
     (check-true (cyc=? (cyc-expt (field-inv-sqrt F8 2) 2) (cyc-rational F8 1/2))
                 "(1/sqrt2)^2 = 1/2  -- an exact rational, not 0.4999...")
     (check-true (cyc=? (cyc-zeta F8)
                        (cyc* (cyc+ (cyc-one F8) i) (field-inv-sqrt F8 2)))
                 "zeta_8 = (1+i)/sqrt2"))

   (let* ([F12 (make-field 12)]
          [s3 (field-sqrt F12 3)]
          [w  (cyc-zeta F12 4)])
     (check-true (cyc=? (cyc-expt s3 2) (cyc-rational F12 3)) "sqrt3^2 = 3")
     (check-true (cyc=? (cyc-expt w 3) (cyc-one F12)) "omega^3 = 1")
     (check-true (cyc-zero? (cyc+ (cyc-one F12) w (cyc-expt w 2)))
                 "1 + omega + omega^2 = 0")
     (check-exn exn:fail? (lambda () (field-sqrt F12 2))
                "Q(zeta_12) genuinely has no sqrt2"))

   (let* ([F24 (make-field 24)]
          [s2 (field-sqrt F24 2)]
          [s3 (field-sqrt F24 3)]
          [s6 (field-sqrt F24 6)])
     (check-equal? (field-degree F24) 8 "degree 8")
     (check-true (cyc=? (cyc-expt s2 2) (cyc-rational F24 2)) "sqrt2^2 = 2")
     (check-true (cyc=? (cyc-expt s3 2) (cyc-rational F24 3)) "sqrt3^2 = 3")
     (check-true (cyc=? (cyc-expt s6 2) (cyc-rational F24 6)) "sqrt6^2 = 6")
     (check-true (cyc=? (cyc* s2 s3) s6) "sqrt2 * sqrt3 = sqrt6")
     (check-true (cyc=? (cyc-expt (field-inv-sqrt F24 6) 2) (cyc-rational F24 1/6))
                 "(1/sqrt6)^2 = 1/6 exactly")
     (check-true (and (cyc-real? s2) (cyc-real? s3) (cyc-real? s6))
                 "the square roots are real")
     (check-false (cyc-real? (field-i F24)) "i is not real"))))


(define galois-tests
  (test-suite
   "Galois action, norm and trace"
   (let* ([F (make-field 24)]
          [s2 (field-sqrt F 2)]
          [s3 (field-sqrt F 3)]
          [z  (cyc-zeta F)])
     (check-equal? (length (cyclofield-units F)) 8 "|Gal| = phi(24) = 8")
     (check-true (for/and ([k (in-list (cyclofield-units F))])
                   (cyc=? (cyc-sigma (cyc* s2 s3) k)
                          (cyc* (cyc-sigma s2 k) (cyc-sigma s3 k))))
                 "sigma is multiplicative")
     (check-true (for/or ([k (in-list (cyclofield-units F))])
                   (cyc=? (cyc-sigma s2 k) (cyc-negate s2)))
                 "some sigma sends sqrt2 to -sqrt2")
     (check-equal? (cyc-norm z) 1 "N(zeta_24) = 1")
     (check-equal? (cyc-norm s2) 16 "N(sqrt2) = 16")
     (check-equal? (cyc-trace z) 0 "Tr(zeta_24) = 0")
     (check-true (cyc=? (cyc-conjugate s2) s2) "conjugation fixes sqrt2"))))

(define inverse-tests
  (test-suite
   "exact inversion"
   (let* ([F (make-field 24)]
          [a (make-cyc F (list 3/5 -2/7 0 11/3 0 1/9 -4/13 2))])
     (check-true (cyc=? (cyc* a (cyc-inverse a)) (cyc-one F))
                 "a * a^-1 = 1, exactly, in degree 8")
     (check-true (cyc=? (cyc/ (cyc* a (field-sqrt F 6)) (field-sqrt F 6)) a)
                 "division round trip"))
   (let ([F (make-field 8)])
     (check-true (cyc=? (cyc* (cyc+ (cyc-one F) (cyc-zeta F))
                              (cyc-inverse (cyc+ (cyc-one F) (cyc-zeta F))))
                        (cyc-one F))
                 "(1+zeta)^-1 exact"))
   (check-exn exn:fail? (lambda () (cyc-inverse (cyc-zero (make-field 8))))
              "0 is not invertible")))

(define embedding-tests
  (test-suite
   "embeddings"
   (let ([F8 (make-field 8)] [F12 (make-field 12)] [F24 (make-field 24)])
     (check-true (cyc=? (cyc-lift (field-sqrt F8 2) F24) (field-sqrt F24 2))
                 "sqrt2 lifts consistently")
     (check-true (cyc=? (cyc-lift (field-sqrt F12 3) F24) (field-sqrt F24 3))
                 "sqrt3 lifts consistently")
     (check-true (cyc=? (cyc-lift (field-i F8) F24) (field-i F24))
                 "i lifts consistently")
     (check-exn exn:fail? (lambda () (cyc-lift (field-sqrt F8 2) F12))
                "8 does not divide 12, so the embedding is refused"))))

(define exactness-tests
  (test-suite
   "exactness is the ground state"
   (let ([F (make-field 24)])
     (check-exn exn:fail:contract?
                (lambda () (make-cyc F (list 0.5)))  ; FLOAT-OK: the point of the test
                "an inexact number is rejected by the contract")
     (check-true (exact? (cyc-trace (field-sqrt F 2))) "traces are exact")
     (check-true (exact? (cyc-norm (field-sqrt F 6))) "norms are exact")
     (check-true (for/and ([x (in-vector (cyc-coeffs (field-inv-sqrt F 6)))])
                   (exact? x))
                 "every coefficient of 1/sqrt6 is exact"))))

(module+ test
  (void
   (run-tests
    (test-suite "cyclotomic"
                cyclotomic-tests
                structure-tests
                galois-tests
                inverse-tests
                embedding-tests
                exactness-tests))))
