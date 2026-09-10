#lang racket/base
;; "No floating point" has to be enforced, not asserted.
;;
;;   1. the sources contain no flonum anywhere, except lines explicitly marked
;;      FLOAT-OK -- and every such line exists to prove a float is rejected or
;;      distinguishable
;;   2. the compiled PTX contains no .f32/.f64 instruction
;;   3. the values the library computes and reports are exact, clock included

(require rackunit
         rackunit/text-ui
         racket/list
         racket/string
         racket/file
         racket/path
         racket/runtime-path
         "../exact-io.rkt"
         "../field.rkt"
         "../mub.rkt")

(define-runtime-path root "..")

(define (rkt-files dir)
  (for/fold ([acc '()]) ([p (in-directory dir)])
    (define s (path->string p))
    (if (and (regexp-match? #rx"[.]rkt$" s)
             (not (regexp-match? #rx"compiled" s)))
        (cons p acc)
        acc)))

;; Crude on purpose: it errs toward flagging. Strings and comments are stripped
;; first so prose about "0.4999..." does not trip it.
(define (float-hits path)
  (for/list ([line (in-list (file->lines path))]
             [i (in-naturals 1)]
             #:unless (regexp-match? #rx"FLOAT-OK" line)
             #:when (let* ([no-string (regexp-replace* #rx"\"[^\"]*\"" line "")]
                           [code (car (string-split (string-append no-string "\n") ";"))])
                      (or (regexp-match? #rx"[^A-Za-z0-9_-][0-9]+[.][0-9]" code)
                          (regexp-match? #rx"[0-9]e[-+]?[0-9]" code)
                          (regexp-match? #rx"current-inexact-milliseconds" code)
                          (regexp-match? #rx"exact->inexact" code)
                          (regexp-match? #rx"real->double-flonum" code))))
    (cons i line)))

(define source-tests
  (test-suite
   "no flonum in the sources"
   (let* ([files (rkt-files root)]
          [bad (for*/list ([f (in-list files)]
                           [h (in-list (float-hits f))])
                 (format "~a:~a  ~a" (file-name-from-path f) (car h)
                         (string-trim (cdr h))))])
     (printf "  scanned ~a .rkt files\n" (length files))
     (for ([b (in-list bad)]) (printf "  FLOAT: ~a\n" b))
     (check-equal? bad '() "no unmarked floating point in any source file"))))

(define ptx-tests
  (test-suite
   "no float instruction in the compiled kernels"
   (let* ([ptx (build-path root "cuda" "kernels.ptx")]
          [present (file-exists? ptx)]
          [txt (if present (file->string ptx) "")]
          [hits (length (regexp-match* #rx"\\.f32|\\.f64" txt))]
          [wide (length (regexp-match* #rx"mul\\.wide\\.s32" txt))])
     (check-true present "kernels.ptx present")
     (printf "  .f32/.f64 in PTX : ~a\n" hits)
     (printf "  mul.wide.s32     : ~a\n" wide)
     (check-equal? hits 0 "the PTX has no floating point instruction")
     (check-true (> wide 0) "and it does use the single-instruction 32x32->64 multiply"))))

(define value-tests
  (test-suite
   "the values are exact, clock and formatter included"
   (let* ([F (make-field 24)]
          [ms (mubs-d6)]
          [v (unbiasedness (cdr (first ms)) (cdr (second ms)))])
     (check-true (exact? (cyc-norm (field-sqrt F 6))) "norms exact")
     (check-true (for/and ([x (in-vector (cyc-coeffs (field-inv-sqrt F 6)))]) (exact? x))
                 "1/sqrt6 coefficients exact")
     (check-true (exact? v) "the d=6 overlap is exact")
     (check-equal? v 1/6 "and it is 1/6")
     (check-true (exact-integer? (now-ms)) "now-ms is an exact integer")
     (check-equal? (dec 1/6 4) "0.1667" "decimals rendered by integer arithmetic")
     (check-equal? (dec 3/2 1) "1.5")
     (check-equal? (dec -7/4 2) "-1.75")
     (check-equal? (dec 1000 0) "1000")
     (check-equal? (bytes->gb (expt 2 30)) "1.0")
     (check-exn exn:fail:contract?
                (lambda () (dec 0.5))   ; FLOAT-OK: dec must refuse a flonum
                "dec refuses a flonum rather than rendering it"))))

;; Racket and CUDA C++, and nothing else. Enforced rather than intended: a
;; helper script in some other language is exactly the thing that creeps in
;; when nobody is checking, and then it is a dependency.
(define two-languages-tests
  (test-suite
   "Racket and CUDA C++ only"
   (let* ([files (for/list ([p (in-directory root)]
                            #:unless (regexp-match? #rx"compiled|[.]git" (path->string p))
                            #:when (file-exists? p))
                   p)]
          [exts (for/list ([f (in-list files)])
                  (let ([s (path->string f)])
                    (cond [(regexp-match #rx"[.]([A-Za-z0-9]+)$" s) => cadr]
                          [else ""])))]
          ;; source we ship
          [source-exts '("rkt" "cu" "ptx" "md" "gitignore" "")]
          ;; things a build leaves behind; not shipped, and .gitignore excludes them
          [artifact-exts '("exe" "lib" "exp" "obj" "pdb" "zmat" "mat")]
          [stray (for/list ([f (in-list files)] [e (in-list exts)]
                            #:unless (or (member e source-exts)
                                         (member e artifact-exts)))
                   (format "~a" (file-name-from-path f)))]
          [foreign (for/list ([f (in-list files)] [e (in-list exts)]
                              #:when (member e '("py" "pyc" "pyw" "ipynb" "js" "mjs"
                                                 "ts" "sh" "bash" "bat" "cmd" "ps1"
                                                 "pl" "rb" "lua")))
                     (format "~a" (file-name-from-path f)))])
     (printf "  source extensions: ~a\n"
             (sort (remove-duplicates
                    (filter (lambda (e) (member e source-exts)) exts))
                   string<?))
     (for ([s (in-list stray)]) (printf "  STRAY: ~a\n" s))
     (for ([s (in-list foreign)]) (printf "  FOREIGN LANGUAGE: ~a\n" s))
     (check-equal? foreign '()
                   "no script in any language other than Racket and CUDA C++")
     (check-equal? stray '()
                   "no file outside .rkt, .cu, .ptx, .md and build artifacts"))))

(module+ test
  (void (run-tests (test-suite "no-float"
                               source-tests ptx-tests value-tests
                               two-languages-tests))))
