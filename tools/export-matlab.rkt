#lang racket/base
;; Save every MUB result to MATLAB, exactly. Entries of Q(zeta_n) are stored as
;; integer numerator/denominator arrays in the power basis of zeta_n -- no float
;; is ever written, so nothing is rounded on the way out. Both the positive
;; results (pairs that ARE unbiased, |<Bi,Bj>|^2 = 1/d) and their complement
;; (the diagonal self-overlaps = 1, and anything that would fail) are recorded:
;; the boolean matrix `unbiased` carries both.
(require racket/list racket/string
         cyclotomic/field cyclotomic/matrix cyclotomic/mub)

(define out-dir "C:/ClaudeOutput/math-results")
(define m-file (string-append out-dir "/build_mubs.m"))

(define (num v) (numerator v))
(define (den v) (denominator v))

;; MATLAB literal for an integer row vector
(define (ivec xs) (string-append "[" (string-join (map number->string xs) " ") "]"))

(define (emit-dim port label d bases)
  (define f (mat-field (cdr (first bases))))
  (define n (cyclofield-n f))
  (define deg (field-degree f))
  (define nb (length bases))
  (fprintf port "R.~a.dim = ~a;\n" label d)
  (fprintf port "R.~a.field_n = ~a;   % entries live in Q(zeta_~a)\n" label n n)
  (fprintf port "R.~a.deg = ~a;        % coefficients per entry, power basis 1,z,...,z^(deg-1)\n" label deg)
  (fprintf port "R.~a.names = {~a};\n" label
           (string-join (map (lambda (b) (format "'~a'" (car b))) bases) " "))
  ;; coefficient tensors, one (d*d) x deg page per basis, entry (r,c) at row r*d+c+1
  (for ([b (in-list bases)] [e (in-naturals 1)])
    (define M (cdr b))
    (define nums '()) (define dens '())
    (for ([r (in-range d)])
      (for ([c (in-range d)])
        (define cs (cyc-coeffs (mat-ref M r c)))
        (for ([k (in-range deg)])
          (set! nums (cons (num (vector-ref cs k)) nums))
          (set! dens (cons (den (vector-ref cs k)) dens)))))
    (fprintf port "R.~a.coef_num{~a} = int64(reshape(~a, [~a ~a])');\n"
             label e (ivec (reverse nums)) deg (* d d))
    (fprintf port "R.~a.coef_den{~a} = int64(reshape(~a, [~a ~a])');\n"
             label e (ivec (reverse dens)) deg (* d d)))
  ;; pairwise squared overlaps, exact, and the positive/negative flag
  (define on '()) (define od '()) (define ub '())
  (for ([i (in-range nb)])
    (for ([j (in-range nb)])
      (define v (if (= i j)
                    ;; self-overlap of an orthonormal basis is 1 on the diagonal
                    ;; of <Bi,Bi>; report the constant only when it exists
                    (or (unbiasedness (cdr (list-ref bases i)) (cdr (list-ref bases j))) 1)
                    (or (unbiasedness (cdr (list-ref bases i)) (cdr (list-ref bases j))) -1)))
      (set! on (cons (num v) on))
      (set! od (cons (den v) od))
      (set! ub (cons (if (equal? v (/ 1 d)) 1 0) ub))))
  (fprintf port "R.~a.overlap_num = int64(reshape(~a, [~a ~a])');\n" label (ivec (reverse on)) nb nb)
  (fprintf port "R.~a.overlap_den = int64(reshape(~a, [~a ~a])');\n" label (ivec (reverse od)) nb nb)
  (fprintf port "R.~a.unbiased = logical(reshape(~a, [~a ~a])');\n" label (ivec (reverse ub)) nb nb)
  (fprintf port "R.~a.is_complete_mub = ~a;\n" label
           (if (mutually-unbiased? (map cdr bases) d) "true" "false"))
  (fprintf port "R.~a.n_unbiased_pairs = ~a;   %% POSITIVE results (off-diagonal = 1/d)\n"
           label (for*/sum ([i (in-range nb)] [j (in-range (add1 i) nb)])
                   (if (equal? (unbiasedness (cdr (list-ref bases i)) (cdr (list-ref bases j))) (/ 1 d)) 1 0)))
  (fprintf port "\n"))

(call-with-output-file m-file #:exists 'replace
  (lambda (port)
    (fprintf port "% cyclotomic MUB results -- EXACT (integer num/den, no float)\n")
    (fprintf port "% generated ~a\n\n" (current-seconds))
    (fprintf port "R = struct();\n\n")
    (emit-dim port "d2"  2  (mubs-d2 (make-field 8)))
    (emit-dim port "d3"  3  (mubs-d3 (make-field 12)))
    (emit-dim port "d4"  4  (mubs-d4 (make-field 24)))
    (emit-dim port "d6"  6  (mubs-d6))
    (emit-dim port "d12" 12 (mubs-d12))
    (fprintf port "save('~a/mubs_exact.mat','R','-v7.3');\n" out-dir)
    (fprintf port "disp('saved mubs_exact.mat');
")))
(printf "wrote ~a\n" m-file)
