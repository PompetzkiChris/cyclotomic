#lang racket/base
;; Exact MUB verification in Racket. Run with:  raco test mub-tests.rkt

(require rackunit
         rackunit/text-ui
         racket/list
         "../field.rkt"
         "../matrix.rkt"
         "../mub.rkt")

(define (bases lst) (map cdr lst))

(define matrix-tests
  (test-suite
   "exact matrices"
   (let* ([F (make-field 8)]
          [I (mat-identity F 3)])
     (check-true (mat-unitary? I) "identity is unitary")
     (check-true (mat=? (mat* I I) I) "I * I = I")
     (check-equal? (mat-rows I) 3)
     (check-equal? (mat-cols I) 3))
   (let* ([F (make-field 24)]
          [A (mat-identity F 2)]
          [B (mat-identity F 3)]
          [K (mat-kron A B)])
     (check-equal? (mat-rows K) 6 "kron gives a 6x6")
     (check-true (mat-unitary? K) "kron of unitaries is unitary"))))

(define d2-tests
  (test-suite
   "dimension 2 over Q(zeta_8)"
   (let* ([F (make-field 8)]
          [ms (mubs-d2 F)])
     (check-equal? (length ms) 3 "three bases")
     (for ([p (in-list ms)])
       (check-true (mat-unitary? (cdr p)) (format "~a is unitary" (car p))))
     (check-equal? (unbiasedness (cdr (first ms)) (cdr (second ms))) 1/2
                   "|<z,x>|^2 = 1/2 exactly")
     (check-true (mutually-unbiased? (bases ms) 2)
                 "3 mutually unbiased bases in dimension 2"))))

(define d3-tests
  (test-suite
   "dimension 3 over Q(zeta_12)"
   (let* ([F (make-field 12)]
          [ms (mubs-d3 F)])
     (check-equal? (length ms) 4 "four bases")
     (for ([p (in-list ms)])
       (check-true (mat-unitary? (cdr p)) (format "~a is unitary" (car p))))
     (check-equal? (unbiasedness (cdr (second ms)) (cdr (third ms))) 1/3
                   "|<b0,b1>|^2 = 1/3 exactly")
     (check-true (mutually-unbiased? (bases ms) 3)
                 "4 mutually unbiased bases in dimension 3"))))

(define d6-tests
  (test-suite
   "dimension 6 over Q(zeta_24)"
   (let* ([ms (mubs-d6)])
     (check-equal? (length ms) 3 "three bases")
     (for ([p (in-list ms)])
       (check-equal? (mat-rows (cdr p)) 6 "6x6")
       (check-true (mat-unitary? (cdr p)) (format "~a is unitary" (car p))))
     (check-equal? (unbiasedness (cdr (first ms)) (cdr (second ms))) 1/6
                   "|<b,c>|^2 = 1/6 -- the rational, not 0.1666...")
     (check-true (mutually-unbiased? (bases ms) 6)
                 "3 mutually unbiased bases in dimension 6, exactly"))))

(define exactness-tests
  (test-suite
   "no floating point anywhere"
   (let* ([ms (mubs-d6)]
          [v (unbiasedness (cdr (first ms)) (cdr (third ms)))])
     (check-true (exact? v) "the overlap is an exact number")
     (check-equal? v 1/6 "and it equals 1/6")
     (check-false (= v (exact->inexact 1/6))  ; FLOAT-OK: the point of the test
                  "1/6 is NOT equal to the nearest double to 1/6"))))

(module+ test
  (void
   (run-tests
    (test-suite "mub"
                d4-tests d12-tests matrix-tests d2-tests d3-tests d6-tests exactness-tests))))

(define d4-tests
  (test-suite
   "dimension 4: complete set of 5 via Galois ring GR(4,2)"
   (let ([d4 (mubs-d4 (make-field 24))])
     (check-equal? (length d4) 5 "five bases (computational + four)")
     (for ([p (in-list d4)]) (check-true (mat-unitary? (cdr p)) (format "~a unitary" (car p))))
     (check-true (mutually-unbiased? (map cdr d4) 4) "all pairwise |<.,.>|^2 = 1/4")
     (check-true (mutually-unbiased? (map cdr (mubs-d4 (make-field 8))) 4) "same over Q(zeta_8)")
     (check-exn exn:fail? (lambda () (mubs-d4 (make-field 3))) "refuses a field without i"))))

(define d12-tests
  (test-suite
   "dimension 12: four MUBs via 4 (x) 3"
   (let ([d12 (mubs-d12)])
     (check-equal? (length d12) 4 "four bases")
     (for ([p (in-list d12)]) (check-true (mat-unitary? (cdr p)) (format "~a unitary" (car p))))
     (for* ([i (in-range 4)] [j (in-range (add1 i) 4)])
       (check-equal? (unbiasedness (cdr (list-ref d12 i)) (cdr (list-ref d12 j))) 1/12
                     (format "|<~a,~a>|^2 = 1/12" i j)))
     (check-true (mutually-unbiased? (map cdr d12) 12) "mutually unbiased in d=12"))))
