#lang racket/base
;; Exact matrices over Q(zeta_n).
;;
;; A matrix is a vector of rows, each row a vector of cyc elements. Everything
;; is exact; there is no tolerance anywhere and no place one could be inserted.

(require racket/vector
         racket/contract
         "field.rkt")

(provide
 (contract-out
  [matrix?          (-> any/c boolean?)]
  [mat              (-> cyclofield? (listof (listof cyc?)) matrix?)]
  [mat-field        (-> matrix? cyclofield?)]
  [mat-rows         (-> matrix? exact-positive-integer?)]
  [mat-cols         (-> matrix? exact-positive-integer?)]
  [mat-ref          (-> matrix? exact-nonnegative-integer? exact-nonnegative-integer? cyc?)]
  [mat-identity     (-> cyclofield? exact-positive-integer? matrix?)]
  [mat*             (-> matrix? matrix? matrix?)]
  [mat-conjugate    (-> matrix? matrix?)]
  [mat-transpose    (-> matrix? matrix?)]
  [mat-adjoint      (-> matrix? matrix?)]
  [mat=?            (-> matrix? matrix? boolean?)]
  [mat-unitary?     (-> matrix? boolean?)]
  [mat-entrywise-abs2 (-> matrix? matrix?)]
  [mat-constant-rational (-> matrix? (or/c exact-rational? #f))]
  [mat-kron         (-> matrix? matrix? matrix?)]
  [mat-lift         (-> matrix? cyclofield? matrix?)]
  [mat*/cpu         (-> matrix? matrix? matrix?)])
 current-mat*-hook
 matrix-data)

;; Installed by cuda/accel.rkt. When set, mat* offers the product to it first;
;; the hook returns #f to decline (wrong field, non-integer entries, no device)
;; and mat* falls back to the exact CPU path. Nothing downstream has to know
;; which one ran, and the answer is identical either way.
(define current-mat*-hook (make-parameter #f))

(struct matrix (field data)
  #:transparent
  #:methods gen:custom-write
  [(define (write-proc m port mode)
     (fprintf port "#<matrix ~ax~a over ~a>"
              (mat-rows m) (mat-cols m) (matrix-field m)))])

(define (mat f rows)
  (define r (length rows))
  (when (zero? r) (error 'mat "matrix needs at least one row"))
  (define c (length (car rows)))
  (for ([row (in-list rows)])
    (unless (= c (length row))
      (error 'mat "ragged rows"))
    (for ([e (in-list row)])
      (unless (= (cyclofield-n (cyc-field e)) (cyclofield-n f))
        (error 'mat "entry ~a is not in ~a" e f))))
  (matrix f (for/vector ([row (in-list rows)]) (list->vector row))))

(define (mat-field m) (matrix-field m))
(define (mat-rows m) (vector-length (matrix-data m)))
(define (mat-cols m) (vector-length (vector-ref (matrix-data m) 0)))
(define (mat-ref m i j) (vector-ref (vector-ref (matrix-data m) i) j))

(define (build f r c fn)
  (matrix f (for/vector ([i (in-range r)])
              (for/vector ([j (in-range c)]) (fn i j)))))

(define (mat-identity f n)
  (build f n n (lambda (i j) (if (= i j) (cyc-one f) (cyc-zero f)))))

(define (mat* a b)
  (define hook (current-mat*-hook))
  (or (and hook (hook a b))
      (mat*/cpu a b)))

(define (mat*/cpu a b)
  (define f (matrix-field a))
  (unless (= (cyclofield-n f) (cyclofield-n (matrix-field b)))
    (error 'mat* "matrices are over different fields"))
  (define n (mat-rows a))
  (define k (mat-cols a))
  (unless (= k (mat-rows b)) (error 'mat* "shape mismatch"))
  (define c (mat-cols b))
  (build f n c
         (lambda (i j)
           (for/fold ([acc (cyc-zero f)]) ([t (in-range k)])
             (cyc+ acc (cyc* (mat-ref a i t) (mat-ref b t j)))))))

(define (mat-conjugate m)
  (build (matrix-field m) (mat-rows m) (mat-cols m)
         (lambda (i j) (cyc-conjugate (mat-ref m i j)))))

(define (mat-transpose m)
  (build (matrix-field m) (mat-cols m) (mat-rows m)
         (lambda (i j) (mat-ref m j i))))

(define (mat-adjoint m) (mat-transpose (mat-conjugate m)))

(define (mat=? a b)
  (and (= (mat-rows a) (mat-rows b))
       (= (mat-cols a) (mat-cols b))
       (for*/and ([i (in-range (mat-rows a))] [j (in-range (mat-cols a))])
         (cyc=? (mat-ref a i j) (mat-ref b i j)))))

(define (mat-unitary? m)
  (and (= (mat-rows m) (mat-cols m))
       (mat=? (mat* m (mat-adjoint m))
              (mat-identity (matrix-field m) (mat-rows m)))))

(define (mat-entrywise-abs2 m)
  (build (matrix-field m) (mat-rows m) (mat-cols m)
         (lambda (i j) (cyc-abs2 (mat-ref m i j)))))

;; If every entry is the same rational, return it; otherwise #f.
(define (mat-constant-rational m)
  (define first-val
    (and (cyc-rational? (mat-ref m 0 0)) (cyc->rational (mat-ref m 0 0))))
  (and first-val
       (for*/and ([i (in-range (mat-rows m))] [j (in-range (mat-cols m))])
         (define e (mat-ref m i j))
         (and (cyc-rational? e) (= first-val (cyc->rational e))))
       first-val))

(define (mat-kron a b)
  (define f (matrix-field a))
  (define ra (mat-rows a)) (define ca (mat-cols a))
  (define rb (mat-rows b)) (define cb (mat-cols b))
  (build f (* ra rb) (* ca cb)
         (lambda (i j)
           (cyc* (mat-ref a (quotient i rb) (quotient j cb))
                 (mat-ref b (remainder i rb) (remainder j cb))))))

(define (mat-lift m g)
  (build g (mat-rows m) (mat-cols m)
         (lambda (i j) (cyc-lift (mat-ref m i j) g))))
