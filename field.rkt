#lang racket/base
;; Exact arithmetic in the cyclotomic field Q(zeta_n) = Q[x]/Phi_n(x).
;;
;; Racket's numeric tower is exact by default, so this module does not have to
;; defend against floating point the way a Python version does -- 1/6 here IS
;; the rational one sixth. The one thing worth guarding is a caller handing in
;; an inexact number, which `exact-rational?` in the contracts rejects outright.
;;
;; An element is a vector of phi(n) exact rationals in the power basis
;; 1, zeta, ..., zeta^(phi(n)-1).

(require racket/vector
         racket/string
         racket/contract
         (only-in math/number-theory coprime?)
         "poly.rkt")

;; racket/base has exact-integer? but not exact-rational?; this is the guard
;; that keeps an inexact number from ever entering the field.
(define (exact-rational? x) (and (rational? x) (exact? x)))

(provide
 exact-rational?
 (struct-out cyclofield)
 (contract-out
  [make-field       (-> exact-positive-integer? cyclofield?)]
  [field-degree     (-> cyclofield? exact-positive-integer?)]
  [make-cyc         (-> cyclofield? (listof exact-rational?) cyc?)]
  [cyc?             (-> any/c boolean?)]
  [cyc-field        (-> cyc? cyclofield?)]
  [cyc-coeffs       (-> cyc? (vectorof exact-rational?))]
  [cyc-zero         (-> cyclofield? cyc?)]
  [cyc-one          (-> cyclofield? cyc?)]
  [cyc-rational     (-> cyclofield? exact-rational? cyc?)]
  [cyc-zeta         (->* (cyclofield?) (exact-integer?) cyc?)]
  [cyc-basis        (-> cyclofield? exact-nonnegative-integer? cyc?)]
  [cyc=?            (-> cyc? cyc? boolean?)]
  [cyc-zero?        (-> cyc? boolean?)]
  [cyc+             (->* () #:rest (listof cyc?) cyc?)]
  [cyc-             (-> cyc? cyc? cyc?)]
  [cyc-negate       (-> cyc? cyc?)]
  [cyc*             (->* () #:rest (listof cyc?) cyc?)]
  [cyc-scale        (-> cyc? exact-rational? cyc?)]
  [cyc-expt         (-> cyc? exact-integer? cyc?)]
  [cyc-inverse      (-> cyc? cyc?)]
  [cyc/             (-> cyc? cyc? cyc?)]
  [cyc-sigma        (-> cyc? exact-integer? cyc?)]
  [cyc-conjugate    (-> cyc? cyc?)]
  [cyc-norm         (-> cyc? exact-rational?)]
  [cyc-trace        (-> cyc? exact-rational?)]
  [cyc-abs2         (-> cyc? cyc?)]
  [cyc-rational?    (-> cyc? boolean?)]
  [cyc->rational    (-> cyc? exact-rational?)]
  [cyc-real?        (-> cyc? boolean?)]
  [cyc-lift         (-> cyc? cyclofield? cyc?)]
  [field-sqrt       (-> cyclofield? exact-positive-integer? cyc?)]
  [field-inv-sqrt   (-> cyclofield? exact-positive-integer? cyc?)]
  [field-i          (-> cyclofield? cyc?)]))

;; ------------------------------------------------------------------ the field

;; n        : the order of the root of unity
;; deg      : phi(n)
;; phi      : coefficients of Phi_n, low to high
;; pow      : vector of vectors; pow[m] is zeta^m in the power basis
;; units    : the k in [1,n] with gcd(k,n)=1, i.e. the Galois group
(struct cyclofield (n deg phi pow units)
  #:transparent
  #:methods gen:custom-write
  [(define (write-proc f port mode)
     (fprintf port "#<Q(zeta_~a) degree ~a>" (cyclofield-n f) (cyclofield-deg f)))])

(define field-cache (make-hasheqv))

(define (make-field n)
  (hash-ref!
   field-cache n
   (lambda ()
     (define phi (cyclotomic-poly n))
     (define deg (sub1 (vector-length phi)))
     (unless (= 1 (vector-ref phi deg))
       (error 'make-field "Phi_~a is not monic" n))
     (define size (add1 (max n (* 2 deg))))
     (define pow (make-vector size #f))
     ;; zeta^j = e_j for j < deg
     (for ([j (in-range (min deg size))])
       (define v (make-vector deg 0))
       (vector-set! v j 1)
       (vector-set! pow j v))
     ;; zeta^m = shift(zeta^(m-1)), reduced by x^deg = -(phi_0 + ... )
     (for ([m (in-range deg size)])
       (define prev (vector-ref pow (sub1 m)))
       (define top (vector-ref prev (sub1 deg)))
       (define v (make-vector deg 0))
       (for ([j (in-range 1 deg)])
         (vector-set! v j (vector-ref prev (sub1 j))))
       (unless (zero? top)
         (for ([j (in-range deg)])
           (vector-set! v j (- (vector-ref v j) (* top (vector-ref phi j))))))
       (vector-set! pow m v))
     (define units (for/list ([k (in-range 1 (add1 n))] #:when (coprime? k n)) k))
     (cyclofield n deg phi pow units))))

(define (field-degree f) (cyclofield-deg f))

;; ---------------------------------------------------------------- elements

(struct cyc (field coeffs)
  #:constructor-name make-elem
  #:transparent
  #:methods gen:custom-write
  [(define (write-proc e port mode)
     (define cs (cyc-coeffs e))
     (define parts
       (for/list ([k (in-naturals)] [a (in-vector cs)] #:unless (zero? a))
         (cond [(= k 0) (format "~a" a)]
               [(= k 1) (format "(~a)z" a)]
               [else    (format "(~a)z^~a" a k)])))
     (fprintf port "~a" (if (null? parts) "0" (string-join parts " + "))))])

(define (make-cyc f coeffs)
  (define d (cyclofield-deg f))
  (when (> (length coeffs) d)
    (error 'make-cyc "~a needs at most ~a coefficients, got ~a" f d (length coeffs)))
  (define v (make-vector d 0))
  (for ([a (in-list coeffs)] [i (in-naturals)])
    (vector-set! v i a))
  (make-elem f v))

(define (cyc-zero f) (make-elem f (make-vector (cyclofield-deg f) 0)))

(define (cyc-one f)
  (define v (make-vector (cyclofield-deg f) 0))
  (vector-set! v 0 1)
  (make-elem f v))

(define (cyc-rational f q)
  (define v (make-vector (cyclofield-deg f) 0))
  (vector-set! v 0 q)
  (make-elem f v))

(define (cyc-basis f k)
  (define v (make-vector (cyclofield-deg f) 0))
  (vector-set! v k 1)
  (make-elem f v))

(define (cyc-zeta f [power 1])
  (define n (cyclofield-n f))
  (make-elem f (vector-copy (vector-ref (cyclofield-pow f) (modulo power n)))))

(define (same-field! a b who)
  (unless (= (cyclofield-n (cyc-field a)) (cyclofield-n (cyc-field b)))
    (error who "cannot mix ~a with ~a; use cyc-lift"
           (cyc-field a) (cyc-field b))))

(define (cyc=? a b)
  (same-field! a b 'cyc=?)
  (equal? (cyc-coeffs a) (cyc-coeffs b)))

(define (cyc-zero? a)
  (for/and ([x (in-vector (cyc-coeffs a))]) (zero? x)))

;; ---------------------------------------------------------------- arithmetic

(define (cyc-add2 a b)
  (same-field! a b 'cyc+)
  (define va (cyc-coeffs a))
  (define vb (cyc-coeffs b))
  (make-elem (cyc-field a)
             (build-vector (vector-length va)
                           (lambda (i) (+ (vector-ref va i) (vector-ref vb i))))))

(define (cyc+ . es)
  (cond [(null? es) (error 'cyc+ "needs at least one argument")]
        [else (for/fold ([acc (car es)]) ([e (in-list (cdr es))]) (cyc-add2 acc e))]))

(define (cyc-negate a)
  (make-elem (cyc-field a)
             (vector-map (lambda (x) (- x)) (cyc-coeffs a))))

(define (cyc- a b) (cyc-add2 a (cyc-negate b)))

(define (cyc-scale a q)
  (make-elem (cyc-field a) (vector-map (lambda (x) (* q x)) (cyc-coeffs a))))

(define (cyc-mul2 a b)
  (same-field! a b 'cyc*)
  (define f (cyc-field a))
  (define d (cyclofield-deg f))
  (define pow (cyclofield-pow f))
  (define va (cyc-coeffs a))
  (define vb (cyc-coeffs b))
  ;; raw convolution, degree < 2d-1
  (define raw (make-vector (max 1 (- (* 2 d) 1)) 0))
  (for ([i (in-range d)])
    (define ai (vector-ref va i))
    (unless (zero? ai)
      (for ([j (in-range d)])
        (define bj (vector-ref vb j))
        (unless (zero? bj)
          (vector-set! raw (+ i j) (+ (vector-ref raw (+ i j)) (* ai bj)))))))
  ;; fold each power back into the basis
  (define out (make-vector d 0))
  (for ([m (in-range (vector-length raw))])
    (define c (vector-ref raw m))
    (unless (zero? c)
      (define row (vector-ref pow m))
      (for ([t (in-range d)])
        (define r (vector-ref row t))
        (unless (zero? r)
          (vector-set! out t (+ (vector-ref out t) (* c r)))))))
  (make-elem f out))

(define (cyc* . es)
  (cond [(null? es) (error 'cyc* "needs at least one argument")]
        [else (for/fold ([acc (car es)]) ([e (in-list (cdr es))]) (cyc-mul2 acc e))]))

(define (cyc-expt a k)
  (cond
    [(negative? k) (cyc-expt (cyc-inverse a) (- k))]
    [else
     (let loop ([r (cyc-one (cyc-field a))] [b a] [k k])
       (cond [(zero? k) r]
             [(odd? k) (loop (cyc-mul2 r b) (cyc-mul2 b b) (quotient k 2))]
             [else     (loop r (cyc-mul2 b b) (quotient k 2))]))]))

;; ------------------------------------------------------------------ inverse

;; Solve M x = e_0 over exact rationals, where M is multiplication-by-a in the
;; power basis. Plain Gaussian elimination; every entry stays exact.
(define (cyc-inverse a)
  (when (cyc-zero? a) (raise-argument-error 'cyc-inverse "nonzero element" a))
  (define f (cyc-field a))
  (define d (cyclofield-deg f))
  (define M (build-vector d (lambda (i)
                              (build-vector (add1 d)
                                            (lambda (j) 0)))))
  ;; column j of M is a * zeta^j
  (for ([j (in-range d)])
    (define col (cyc-coeffs (cyc-mul2 a (cyc-basis f j))))
    (for ([i (in-range d)])
      (vector-set! (vector-ref M i) j (vector-ref col i))))
  (vector-set! (vector-ref M 0) d 1)              ; right-hand side e_0
  (for ([i (in-range 1 d)])
    (vector-set! (vector-ref M i) d 0))
  ;; forward elimination with partial pivoting on nonzero entries
  (for ([col (in-range d)])
    (define piv (for/first ([r (in-range col d)]
                            #:unless (zero? (vector-ref (vector-ref M r) col)))
                  r))
    (unless piv (error 'cyc-inverse "singular; element is not invertible"))
    (unless (= piv col)
      (define tmp (vector-ref M col))
      (vector-set! M col (vector-ref M piv))
      (vector-set! M piv tmp))
    (define prow (vector-ref M col))
    (define p (vector-ref prow col))
    (for ([j (in-range col (add1 d))])
      (vector-set! prow j (/ (vector-ref prow j) p)))
    (for ([r (in-range d)] #:unless (= r col))
      (define row (vector-ref M r))
      (define factor (vector-ref row col))
      (unless (zero? factor)
        (for ([j (in-range col (add1 d))])
          (vector-set! row j (- (vector-ref row j)
                                (* factor (vector-ref prow j))))))))
  (make-elem f (build-vector d (lambda (i) (vector-ref (vector-ref M i) d)))))

(define (cyc/ a b) (cyc-mul2 a (cyc-inverse b)))

;; ------------------------------------------------------------------- Galois

(define (cyc-sigma a k)
  (define f (cyc-field a))
  (define n (cyclofield-n f))
  (unless (coprime? k n)
    (error 'cyc-sigma "k=~a must be coprime to n=~a" k n))
  (define d (cyclofield-deg f))
  (define pow (cyclofield-pow f))
  (define va (cyc-coeffs a))
  (define out (make-vector d 0))
  (for ([j (in-range d)])
    (define aj (vector-ref va j))
    (unless (zero? aj)
      (define row (vector-ref pow (modulo (* j k) n)))
      (for ([t (in-range d)])
        (define r (vector-ref row t))
        (unless (zero? r)
          (vector-set! out t (+ (vector-ref out t) (* aj r)))))))
  (make-elem f out))

(define (cyc-conjugate a)
  (cyc-sigma a (sub1 (cyclofield-n (cyc-field a)))))

(define (cyc-norm a)
  (define f (cyc-field a))
  (define p (for/fold ([acc (cyc-one f)]) ([k (in-list (cyclofield-units f))])
              (cyc-mul2 acc (cyc-sigma a k))))
  (unless (cyc-rational? p) (error 'cyc-norm "norm was not rational: ~a" p))
  (cyc->rational p))

(define (cyc-trace a)
  (define f (cyc-field a))
  (define s (for/fold ([acc (cyc-zero f)]) ([k (in-list (cyclofield-units f))])
              (cyc-add2 acc (cyc-sigma a k))))
  (unless (cyc-rational? s) (error 'cyc-trace "trace was not rational: ~a" s))
  (cyc->rational s))

(define (cyc-abs2 a) (cyc-mul2 a (cyc-conjugate a)))

(define (cyc-rational? a)
  (for/and ([i (in-naturals 1)] [x (in-vector (cyc-coeffs a) 1)]) (zero? x)))

(define (cyc->rational a)
  (unless (cyc-rational? a) (error 'cyc->rational "~a is not rational" a))
  (vector-ref (cyc-coeffs a) 0))

(define (cyc-real? a) (cyc=? a (cyc-conjugate a)))

;; --------------------------------------------------------------- embeddings

;; Q(zeta_n) -> Q(zeta_m) when n divides m, via zeta_n = zeta_m^(m/n).
(define (cyc-lift a g)
  (define f (cyc-field a))
  (define n (cyclofield-n f))
  (define m (cyclofield-n g))
  (unless (zero? (modulo m n))
    (error 'cyc-lift "~a does not divide ~a; no such embedding" n m))
  (define step (quotient m n))
  (for/fold ([acc (cyc-zero g)])
            ([j (in-naturals)] [aj (in-vector (cyc-coeffs a))])
    (if (zero? aj)
        acc
        (cyc-add2 acc (cyc-scale (cyc-zeta g (* j step)) aj)))))

;; ---------------------------------------------------------- named constants

(define (primitive-root f k)
  (define n (cyclofield-n f))
  (unless (zero? (modulo n k))
    (error 'field-sqrt "~a does not contain zeta_~a" f k))
  (cyc-zeta f (quotient n k)))

;; sqrt2 = zeta_8 + zeta_8^-1 ; sqrt3 = zeta_12 + zeta_12^-1 ; sqrt6 = product
(define (field-sqrt f m)
  (case m
    [(2) (let ([z (primitive-root f 8)])  (cyc-add2 z (cyc-expt z 7)))]
    [(3) (let ([z (primitive-root f 12)]) (cyc-add2 z (cyc-expt z 11)))]
    [(6) (cyc-mul2 (field-sqrt f 2) (field-sqrt f 3))]
    [else (error 'field-sqrt "sqrt(~a) not implemented" m)]))

(define (field-inv-sqrt f m)
  (cyc-scale (field-sqrt f m) (/ 1 m)))

(define (field-i f)
  (define n (cyclofield-n f))
  (unless (zero? (modulo n 4)) (error 'field-i "~a does not contain i" f))
  (cyc-zeta f (quotient n 4)))
