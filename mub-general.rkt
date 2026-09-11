#lang racket/base
;; The optimal known MUB set for any dimension. A prime power q = p^k has a
;; COMPLETE set of q+1 -- that case is solved -- from the field GF(q) (odd p) or
;; the Galois ring GR(4,k) (p=2). A composite dimension gets the tensor lower
;; bound min_i (q_i + 1) over its prime-power factors, which is the best general
;; construction; whether it can be beaten is open.
;;
;; Everything is exact and every set is checked by the |<Bi,Bj>|^2 = 1/d test in
;; mub.rkt before anyone trusts it.

(require racket/list
         racket/string
         (only-in math/number-theory prime? factorize)
         "field.rkt" "matrix.rkt" "mub.rkt")

(provide mubs-optimal optimal-count prime-power? pp-decompose)

;; ---- prime-power bookkeeping ----------------------------------------------
(define (pp-decompose q)          ; q = p^k -> (values p k) or #f
  (define fs (factorize q))
  (and (= 1 (length fs)) (values (caar fs) (cadar fs))))
(define (prime-power? q)
  (define fs (factorize q)) (= 1 (length fs)))
(define (factor-prime-powers d)   ; d -> list of the q_i = p_i^k_i
  (for/list ([pk (in-list (factorize d))]) (expt (car pk) (cadr pk))))

;; the cyclotomic order a factor's construction needs to be exact
(define (required-n q)
  (define fs (factorize q)) (define p (caar fs)) (define k (cadar fs))
  (cond
    [(= p 2) (if (<= k 1) 8 (if (odd? k) (* 4 (expt 2 (quotient (add1 k) 2))) 4))] ; ~ needs i and maybe sqrt2
    [(= k 1) (if (= 1 (modulo p 4)) p (* 4 p))]   ; odd prime: zeta_p, and i when p=3 mod4
    ;; odd prime power p^k via GF(p^k): always needs zeta_p; when k is ODD the
    ;; normalizer 1/sqrt(p^k) carries a sqrt(p), which needs the same field the
    ;; prime case does (zeta_p, plus i when p=3 mod4). k even -> 1/sqrt is rational.
    [(odd? k) (if (= 1 (modulo p 4)) p (* 4 p))]
    [else p]))

;; ---- odd prime p: Heisenberg-Weyl, complete set p+1 -----------------------
(define (mubs-prime p f)
  (define one (cyc-one f)) (define zero (cyc-zero f))
  (define s (field-inv-sqrt f p))
  (define w (cyc-zeta f (quotient (cyclofield-n f) p)))
  (cons
   (cons "Z" (mat f (for/list ([a (in-range p)]) (for/list ([b (in-range p)]) (if (= a b) one zero)))))
   (for/list ([k (in-range p)])
     (cons (format "B~a" k)
           (mat f (for/list ([j (in-range p)])
                    (for/list ([m (in-range p)])
                      (cyc* s (cyc-expt w (modulo (+ (* k m m) (* j m)) p))))))))))

;; ---- odd prime power p^k via GF(p^k) Wootters-Fields ----------------------
;; GF(p^k) = GF(p)[t]/(irr).  element = coefficient vector length k over GF(p).
(define (gf-make p k)
  ;; find a monic irreducible of degree k over GF(p) by search
  (define (poly-eval-nonzero? cf) #t)
  (define (mulmod a b irr)           ; a,b: coeff vectors length k; irr: length k+1 monic
    (define prod (make-vector (sub1 (* 2 k)) 0))
    (for* ([i (in-range k)] [j (in-range k)])
      (vector-set! prod (+ i j) (modulo (+ (vector-ref prod (+ i j)) (* (vector-ref a i) (vector-ref b j))) p)))
    ;; reduce degree >= k using irr (monic): x^k = -(irr_0 + ... + irr_{k-1} x^{k-1})
    (for ([deg (in-range (- (* 2 k) 2) (sub1 k) -1)])
      (define c (vector-ref prod deg))
      (unless (zero? c)
        (vector-set! prod deg 0)
        (for ([j (in-range k)])
          (define idx (- deg (- k j)))
          (vector-set! prod idx (modulo (- (vector-ref prod idx) (* c (vector-ref irr j))) p)))))
    (for/vector ([i (in-range k)]) (vector-ref prod i)))
  (define (irreducible? irr)
    ;; t has order p^k-1 iff generates; cheap check: t^(p^k-1)=1 and t^((p^k-1)/prime)!=1 for prime|order
    (define t (let ([v (make-vector k 0)]) (vector-set! v 1 1) v))
    (define one (let ([v (make-vector k 0)]) (vector-set! v 0 1) v))
    (define (pow b e) (let loop ([b b] [e e] [acc one])
                        (cond [(zero? e) acc]
                              [(odd? e) (loop (mulmod b b irr) (quotient e 2) (mulmod acc b irr))]
                              [else (loop (mulmod b b irr) (quotient e 2) acc)])))
    (define ord (sub1 (expt p k)))
    (and (equal? (pow t ord) one)
         (for/and ([pk (in-list (factorize ord))])
           (not (equal? (pow t (quotient ord (car pk))) one)))))
  (define irr
    (let loop ([n 0])
      (define irr (for/vector ([i (in-range k)]) (modulo (quotient n (expt p i)) p)))  ; low k coeffs
      (cond [(>= n (expt p k)) (error 'gf-make "no primitive poly for GF(~a^~a)" p k)]
            [(irreducible? irr) irr]
            [else (loop (add1 n))])))
  (values irr (lambda (a b) (mulmod a b irr))))

(define (mubs-gfpk p k f)
  (define-values (irr mul) (gf-make p k))
  (define q (expt p k))
  (define elts (for/list ([n (in-range q)])
                 (for/vector ([i (in-range k)]) (modulo (quotient n (expt p i)) p))))
  (define (add a b) (for/vector ([x (in-vector a)] [y (in-vector b)]) (modulo (+ x y) p)))
  (define (fld-trace z)                 ; GF(p^k) -> GF(p): sum of z^(p^i)
    (define (frob v) v) ; placeholder, replaced below
    z)
  ;; trace via matrix of Frobenius is heavy; use tr(z)=sum_i z^{p^i}
  (define (powp z i)                     ; z^(p^i)
    (let loop ([i i] [z z]) (if (zero? i) z (loop (sub1 i) (let pw ([e p] [b z] [acc (let ([v (make-vector k 0)]) (vector-set! v 0 1) v)])
                                                              (cond [(zero? e) acc]
                                                                    [(odd? e) (pw (quotient e 2) (mul b b) (mul acc b))]
                                                                    [else (pw (quotient e 2) (mul b b) acc)]))))))
  (define (trace z) (modulo (for/sum ([i (in-range k)]) (vector-ref (powp z i) 0)) p))
  (define s (if (even? k)
                (cyc-rational f (/ 1 (expt p (quotient k 2))))
                (cyc-scale (field-sqrt f p) (/ 1 (expt p (quotient (add1 k) 2))))))
  (define w (cyc-zeta f (quotient (cyclofield-n f) p)))
  (define (idx z) (for/sum ([c (in-vector z)] [i (in-naturals)]) (* c (expt p i))))
  (define one (cyc-one f)) (define zero (cyc-zero f))
  (cons
   (cons "Z" (mat f (for/list ([a (in-range q)]) (for/list ([b (in-range q)]) (if (= a b) one zero)))))
   (for/list ([a (in-list elts)] [ai (in-naturals)])
     (cons (format "B~a" ai)
           (mat f (for/list ([b (in-list elts)])
                    (for/list ([x (in-list elts)])
                      (define e (modulo (+ (trace (mul a (mul x x))) (trace (mul b x))) p))
                      (cyc* s (cyc-expt w e)))))))))

;; ---- prime power dispatcher -----------------------------------------------
(define (mubs-pp q f)
  (define fs (factorize q)) (define p (caar fs)) (define k (cadar fs))
  (cond
    [(and (= p 2) (= k 1)) (mubs-d2 f)]
    [(and (= p 2) (= k 2)) (mubs-d4 f)]
    [(= k 1)               (mubs-prime p f)]
    [(odd? p)              (mubs-gfpk p k f)]
    [else                  (mubs-gr2k k f)]))

;; ---- optimal set for any dimension ----------------------------------------
(define (optimal-count d)
  (apply min (for/list ([q (in-list (factor-prime-powers d))]) (add1 q))))

(define (mubs-optimal d)
  (define qs (factor-prime-powers d))
  (cond
    [(= 1 (length qs))
     (mubs-pp d (make-field (required-n d)))]
    [else
     (define L (apply lcm (map required-n qs)))
     (define F (make-field L))
     (define sets (for/list ([q (in-list qs)]) (mubs-pp q F)))   ; each already over F
     (define k (apply min (map length sets)))
     (for/list ([idx (in-range k)])
       (cons (string-join (for/list ([s (in-list sets)]) (car (list-ref s idx))) "(x)")
             (for/fold ([acc (cdr (list-ref (car sets) idx))]) ([s (in-list (cdr sets))])
               (mat-kron acc (cdr (list-ref s idx))))))]))
;; ---- 2^k via the Galois ring GR(4,k) --------------------------------------
;; R = Z4[x]/(h), h a basic primitive polynomial of degree k (its reduction mod
;; 2 is primitive over GF(2), got by the Graeffe lift h(x^2) = (-1)^k g(x)g(-x)).
;; Teichmuller set T = {0} U {xi^i}, xi = x, |T| = 2^k. Trace tr(z)=sum z^(2^i).
;; MUB: (v_b^a)[x] = i^{tr((a+2b)x)} / sqrt(2^k), a,b,x in T. Complete set 2^k+1.

;; -- primitive polynomial over GF(2), degree k: coeffs c_0..c_{k-1}, x^k = sum c_i x^i
(define (gf2-primitive k)
  (define (mul a b)                 ; a,b length k over GF(2), reduce by red
    (define pr (make-vector (sub1 (* 2 k)) 0))
    (for* ([i (in-range k)] [j (in-range k)])
      (when (and (= 1 (vector-ref a i)) (= 1 (vector-ref b j)))
        (vector-set! pr (+ i j) (modulo (add1 (vector-ref pr (+ i j))) 2))))
    (for ([deg (in-range (- (* 2 k) 2) (sub1 k) -1)])
      (when (= 1 (vector-ref pr deg))
        (vector-set! pr deg 0)
        (for ([j (in-range k)])
          (define idx (- deg (- k j)))
          (vector-set! pr idx (modulo (+ (vector-ref pr idx) (vector-ref red j)) 2)))))
    (for/vector ([i (in-range k)]) (vector-ref pr i)))
  (define red #f)
  (define (order-2k-1? r)
    (set! red r)
    (define x (let ([v (make-vector k 0)]) (vector-set! v 1 1) v))
    (define one (let ([v (make-vector k 0)]) (vector-set! v 0 1) v))
    (define (pw b e) (let loop ([b b][e e][acc one])
                       (cond [(zero? e) acc]
                             [(odd? e) (loop (mul b b) (quotient e 2) (mul acc b))]
                             [else (loop (mul b b) (quotient e 2) acc)])))
    (define ord (sub1 (expt 2 k)))
    (and (equal? (pw x ord) one)
         (for/and ([pk (in-list (factorize ord))]) (not (equal? (pw x (quotient ord (car pk))) one)))))
  (let loop ([n 0])
    (define r (for/vector ([i (in-range k)]) (modulo (quotient n (expt 2 i)) 2)))
    (cond [(>= n (expt 2 k)) (error 'gf2-primitive "none for k=~a" k)]
          [(order-2k-1? r) r]
          [else (loop (add1 n))])))

;; integer polynomial multiply
(define (ipoly* a b)
  (define r (make-vector (sub1 (+ (vector-length a) (vector-length b))) 0))
  (for* ([i (in-range (vector-length a))] [j (in-range (vector-length b))])
    (vector-set! r (+ i j) (+ (vector-ref r (+ i j)) (* (vector-ref a i) (vector-ref b j)))))
  r)

;; basic primitive h over Z4 as low coeffs c_0..c_{k-1} with x^k = -(sum c_i x^i)
(define (gr4-h k)
  (define red (gf2-primitive k))               ; x^k = sum red_i x^i over GF(2)
  ;; g(x) = x^k - sum red_i x^i  (integer lift, monic degree k)
  (define g (make-vector (add1 k) 0))
  (vector-set! g k 1)
  (for ([i (in-range k)]) (vector-set! g i (- (vector-ref red i))))
  ;; g(-x): flip sign of odd-degree coeffs
  (define gneg (for/vector ([c (in-vector g)] [i (in-naturals)]) (if (odd? i) (- c) c)))
  (define prod (ipoly* g gneg))                ; = (-1)^k h(x^2); only even powers nonzero
  ;; h(y): coefficient of y^m is prod[2m] * (-1)^k
  (define sign (if (odd? k) -1 1))
  (define h (for/vector ([m (in-range (add1 k))]) (modulo (* sign (vector-ref prod (* 2 m))) 4)))
  ;; h monic degree k; return low coeffs c_i with x^k = -(sum c_i x^i): c_i = h_i
  (for/vector ([i (in-range k)]) (vector-ref h i)))

(define (mubs-gr2k k f)
  (define c (gr4-h k))                          ; x^k = -(sum c_i x^i) mod 4
  (define (rmul a b)                            ; length-k over Z4
    (define pr (make-vector (sub1 (* 2 k)) 0))
    (for* ([i (in-range k)] [j (in-range k)])
      (vector-set! pr (+ i j) (modulo (+ (vector-ref pr (+ i j)) (* (vector-ref a i) (vector-ref b j))) 4)))
    (for ([deg (in-range (- (* 2 k) 2) (sub1 k) -1)])
      (define t (vector-ref pr deg))
      (unless (zero? t)
        (vector-set! pr deg 0)
        (for ([j (in-range k)])
          (define idx (- deg (- k j)))
          (vector-set! pr idx (modulo (- (vector-ref pr idx) (* t (vector-ref c j))) 4)))))
    (for/vector ([i (in-range k)]) (vector-ref pr i)))
  (define one (let ([v (make-vector k 0)]) (vector-set! v 0 1) v))
  (define xi  (let ([v (make-vector k 0)]) (vector-set! v 1 1) v))
  (define T (cons (make-vector k 0)             ; 0, then xi^0..xi^(2^k-2)
                  (let loop ([i 0] [z one] [acc '()])
                    (if (= i (sub1 (expt 2 k))) (reverse acc)
                        (loop (add1 i) (rmul z xi) (cons z acc))))))
  (define (radd a b) (for/vector ([x (in-vector a)] [y (in-vector b)]) (modulo (+ x y) 4)))
  (define (r2 a) (for/vector ([x (in-vector a)]) (modulo (* 2 x) 4)))
  (define (sq z) (rmul z z))
  (define (tr z)                                ; sum_{i=0}^{k-1} z^(2^i), constant coeff
    (let loop ([i 0] [zz z] [acc 0])
      (if (= i k) (modulo acc 4)
          (loop (add1 i) (sq zz) (+ acc (vector-ref zz 0))))))
  (define q (expt 2 k))
  (define s (if (even? k)
                (cyc-rational f (/ 1 (expt 2 (quotient k 2))))
                (cyc-scale (field-sqrt f 2) (/ 1 (expt 2 (quotient (add1 k) 2))))))
  (define i-unit (field-i f))
  (define one-c (cyc-one f)) (define zero-c (cyc-zero f))
  (cons
   (cons "Z" (mat f (for/list ([a (in-range q)]) (for/list ([b (in-range q)]) (if (= a b) one-c zero-c)))))
   (for/list ([a (in-list T)] [ai (in-naturals)])
     (cons (format "A~a" ai)
           (mat f (for/list ([b (in-list T)])
                    (for/list ([x (in-list T)])
                      (cyc* s (cyc-expt i-unit (tr (rmul (radd a (r2 b)) x)))))))))))
