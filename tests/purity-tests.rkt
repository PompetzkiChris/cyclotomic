#lang racket/base
;; Requiring a module must not do anything.
;;
;; Locating a DLL touches the filesystem, opening it loads code, and making its
;; dependencies findable mutates the process PATH. None of that belongs in a
;; module body: a program that requires this package and never compiles a kernel
;; should pay nothing and change nothing. delay on, delay off.

(require rackunit
         racket/string
         racket/port
         racket/system
         rackunit/text-ui
         racket/promise
         racket/runtime-path
         "../exact-io.rkt"
         "../field.rkt"
         "../cuda/nvrtc.rkt")

(define-runtime-path root "..")

;; The claim is that requiring nvrtc.rkt does not put the toolkit on PATH, and
;; that calling into it does. Checking that in THIS process only works on a
;; machine whose PATH does not already contain the toolkit -- and on a machine
;; where the CUDA installer put it there, the check passes or fails on the
;; environment rather than on this package. So it is checked in a subprocess
;; whose PATH has been stripped of the toolkit first, which is the same test on
;; every machine.
(define toolkit-rx #rx"NVIDIA GPU Computing Toolkit")

(define (path-without-toolkit)
  (define sep (if (eq? (system-type) 'windows) ";" ":"))
  (string-join
   (filter (lambda (e) (and (not (string=? e ""))
                            (not (regexp-match? toolkit-rx e))))
           (string-split (or (getenv "PATH") "") sep))
   sep))

;; (values before-require after-use) as symbols, from a process that starts
;; clean.
(define (probe-in-clean-subprocess)
  (define nvrtc-path (path->string (build-path root "cuda" "nvrtc.rkt")))
  (define code
    (format
     (string-append
      "(require (file ~s))"
      "(displayln (if (regexp-match? #rx\"NVIDIA GPU Computing Toolkit\""
      "                             (or (getenv \"PATH\") \"\")) 'dirty 'clean))"
      "(void (nvrtc-available?))"
      "(displayln (if (regexp-match? #rx\"NVIDIA GPU Computing Toolkit\""
      "                             (or (getenv \"PATH\") \"\")) 'added 'absent))")
     nvrtc-path))
  (define env (environment-variables-copy (current-environment-variables)))
  (environment-variables-set! env #"PATH" (string->bytes/utf-8 (path-without-toolkit)))
  (parameterize ([current-environment-variables env])
    (define-values (sp out in err)
      (subprocess #f #f #f (find-system-path 'exec-file) "-l" "racket/base" "-e" code))
    (close-output-port in)
    (define lines (port->lines out))
    (define errs (port->string err))
    (subprocess-wait sp)
    (close-input-port out) (close-input-port err)
    (unless (zero? (subprocess-status sp))
      (error 'purity-tests "subprocess failed: ~a" errs))
    (values (string->symbol (car lines)) (string->symbol (cadr lines)))))

(define require-tests
  (test-suite
   "requiring a module has no observable effect"
   (let-values ([(before after) (probe-in-clean-subprocess)])
     (check-eq? before 'clean
                "requiring nvrtc.rkt does not put the toolkit on PATH")
     (check-eq? after 'added
                "the PATH change happens on first use, not on require")
     (printf "  in a clean process: PATH ~a after require, ~a after first use\n"
             before after))
   ;; And the library is genuinely reachable on this machine.
   (check-true (nvrtc-available?) "NVRTC becomes available when asked for")))

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
   ;; Whether a promise has been forced is a fact, not a duration. Asking it
   ;; directly tests the claim -- nothing was built -- on any machine, however
   ;; busy, which a millisecond threshold cannot.
   (let* ([t0 (now-ms)]
          [F (make-field 2520)]      ; phi(2520) = 576
          [t-make (elapsed-ms t0)])
     (check-equal? (field-degree F) 576 "degree is available without the tables")
     (check-false (promise-forced? (cyclofield-phi-p F))
                  "make-field does not compute Phi_n")
     (check-false (promise-forced? (cyclofield-pow-p F))
                  "make-field does not build the power table")
     (check-false (promise-forced? (cyclofield-units-p F))
                  "make-field does not enumerate the unit group")
     (printf "  make-field 2520 (degree 576): ~a ms, nothing forced\n" t-make)
     (let ([t1 (now-ms)])
       (void (cyclofield-pow F))
       (printf "  forcing the power table: ~a ms\n" (elapsed-ms t1)))
     (check-true (promise-forced? (cyclofield-pow-p F)) "and then it is built")
     (check-true (promise-forced? (cyclofield-phi-p F))
                 "which needed Phi_n, so that is built too")
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
