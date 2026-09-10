#lang racket/base
;; Requiring a module must not do anything.
;;
;; Locating a DLL touches the filesystem, opening it loads code, and making its
;; dependencies findable mutates the process PATH. None of that belongs in a
;; module body: a program that requires this package and never compiles a kernel
;; should pay nothing and change nothing. delay on, delay off.

(require rackunit
         rackunit/text-ui
         racket/promise
         racket/runtime-path
         "../exact-io.rkt"
         "../field.rkt"
         "../cuda/nvrtc.rkt")

(define-runtime-path root "..")

(define require-tests
  (test-suite
   "requiring a module has no observable effect"
   ;; This module has already required nvrtc.rkt above. If instantiation had
   ;; loaded the library or touched PATH, it would show here.
   (let ([path-before (getenv "PATH")])
     (check-true (string? path-before) "PATH readable")
     ;; nothing has forced NVRTC yet in this process
     (check-false (regexp-match? #rx"NVIDIA GPU Computing Toolkit"
                                 (or (getenv "PATH") ""))
                  "the toolkit bin is not on PATH merely from requiring nvrtc.rkt")
     ;; now use it on purpose, and only now does the effect appear
     (check-true (nvrtc-available?) "NVRTC becomes available when asked for")
     (check-true (regexp-match? #rx"NVIDIA GPU Computing Toolkit"
                                (or (getenv "PATH") ""))
                 "the PATH change happens on first use, not on require")
     (printf "  PATH changed only after nvrtc-available? was called\n"))))

(define promise-tests
  (test-suite
   "a promise computes once"
   (let* ([n (box 0)]
          [p (delay (set-box! n (add1 (unbox n))) 'value)])
     (check-equal? (unbox n) 0 "not computed until forced")
     (check-equal? (force p) 'value)
     (check-equal? (force p) 'value)
     (check-equal? (force p) 'value)
     (check-equal? (unbox n) 1 "forced three times, computed once"))))

(define field-laziness-tests
  (test-suite
   "field tables are built on demand, once"
   ;; a field whose tables are large: phi(2520) = 576, and the power table is
   ;; max(n, 2*deg)+1 rows of that width
   (let* ([t0 (now-ms)]
          [F (make-field 2520)]
          [t-make (elapsed-ms t0)])
     (check-equal? (field-degree F) 576 "degree is available without the tables")
     (printf "  make-field 2520 (degree 576): ~a ms\n" t-make)
     (check-true (< t-make 50) "constructing the field is cheap")
     (let* ([t1 (now-ms)]
            [_ (cyclofield-units F)]
            [t-units (elapsed-ms t1)]
            [t2 (now-ms)]
            [__ (cyclofield-units F)]
            [t-again (elapsed-ms t2)])
       (printf "  units: ~a ms first, ~a ms second\n" t-units t-again)
       (check-true (<= t-again t-units) "the second access is not slower")))
   ;; and the values are the same object every time, not rebuilt
   (let ([F (make-field 24)])
     (check-eq? (cyclofield-pow F) (cyclofield-pow F)
                "the power table is the same object on every access")
     (check-eq? (cyclofield-units F) (cyclofield-units F)
                "so is the unit group"))))

(module+ test
  (void (run-tests (test-suite "purity"
                               require-tests promise-tests field-laziness-tests))))
