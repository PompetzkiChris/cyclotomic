#lang racket/base
;; Cross-check the Racket CUDA path against the independent CUDA C++ reference.
;;
;; Racket writes both operands as .zmat files, refcheck.exe computes the product
;; with its own Phi_n, its own power table, its own kernel and a 128-bit CPU
;; oracle, and the two results are compared coefficient by coefficient.
;;
;; Nothing is shared between the two implementations except the file format, so
;; agreement is evidence about the mathematics rather than about one codebase.

(require racket/system
         racket/runtime-path
         racket/list
         "../field.rkt"
         "../matrix.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(define-runtime-path here ".")
(define exe (build-path here "refcheck.exe"))

(define (zmat-write! z path)
  (call-with-output-file path #:exists 'truncate
    (lambda (out)
      (write-bytes #"ZMAT" out)
      (for ([v (in-list (list (cyclofield-n (zmat-field z))
                              (field-degree (zmat-field z))
                              (zmat-rows z)
                              (zmat-cols z)))])
        (write-bytes (integer->integer-bytes v 4 #t #f) out))
      (write-bytes (zmat-planes z) out))))

(define (zmat-read F path)
  (call-with-input-file path
    (lambda (in)
      (define magic (read-bytes 4 in))
      (unless (equal? magic #"ZMAT") (error 'zmat-read "bad magic: ~a" magic))
      (define hdr (for/list ([_ (in-range 4)])
                    (integer-bytes->integer (read-bytes 4 in) #t #f)))
      (define rows (third hdr))
      (define cols (fourth hdr))
      (define planes (read-bytes (* (second hdr) rows cols 8) in))
      (zmat F rows cols planes))))

(define (run . args)
  (define out (open-output-string))
  (define ok (parameterize ([current-output-port out])
               (apply system* exe args)))
  (values ok (get-output-string out)))

(printf "independent cross-check: Racket CUDA vs CUDA C++ reference\n")
(printf "reference binary: ~a\n\n" exe)
(unless (file-exists? exe) (error 'crosscheck "refcheck.exe not built"))

(gpu-init!)
(define tmp (find-system-path 'temp-dir))
(define fails 0)

(printf "~a\t~a\t~a\t~a\t~a\n" "field" "size" "bits" "cpp_ok" "match")
(for* ([ncyc (in-list '(6 8 12 24))]
       [sz (in-list '(1 3 16 40 96))])
  (define F (make-field ncyc))
  (define bits 9)
  (define lim (arithmetic-shift 1 bits))
  (define A (zmat-build F sz sz (lambda (i j t) (- (random (* 2 lim)) lim))))
  (define B (zmat-build F sz sz (lambda (i j t) (- (random (* 2 lim)) lim))))

  (define pa (build-path tmp (format "cc_a_~a_~a.zmat" ncyc sz)))
  (define pb (build-path tmp (format "cc_b_~a_~a.zmat" ncyc sz)))
  (define pc (build-path tmp (format "cc_c_~a_~a.zmat" ncyc sz)))
  (zmat-write! A pa)
  (zmat-write! B pb)

  (define-values (ok msg)
    (run "mul" (path->string pa) (path->string pb) (path->string pc)))

  (define theirs (and ok (zmat-read F pc)))
  ;; every kernel mode must match the independent reference. A fast path that
  ;; is only ever checked against itself is not checked.
  (define per-mode
    (for/list ([m (in-list '(split fused rb w32))])
      (define ours (parameterize ([current-gpu-kernel m]) (gpu-matmul A B)))
      (cons m (and theirs (equal? (zmat-planes ours) (zmat-planes theirs))))))
  (define match? (for/and ([q (in-list per-mode)]) (cdr q)))
  (unless match? (set! fails (add1 fails)))
  (printf "Q(z~a)	~ax~a	~a	~a	~a
" ncyc sz sz bits
          (if ok "ok" "FAILED")
          (cond [match? "ALL 4 KERNELS IDENTICAL"]
                [(not theirs) "-"]
                [else (format "DIFFER ~a"
                              (for/list ([q (in-list per-mode)] #:unless (cdr q))
                                (car q)))]))
  (for ([p (in-list (list pa pb pc))]) (when (file-exists? p) (delete-file p))))

;; the reference's own comparator, on a product it did not produce
(printf "\ncross-check the other direction: refcheck cmp on our output\n")
(let* ([F (make-field 24)]
       [sz 32]
       [lim 256]
       [A (zmat-build F sz sz (lambda (i j t) (- (random (* 2 lim)) lim)))]
       [B (zmat-build F sz sz (lambda (i j t) (- (random (* 2 lim)) lim)))]
       [pa (build-path tmp "cc2_a.zmat")]
       [pb (build-path tmp "cc2_b.zmat")]
       [pref (build-path tmp "cc2_ref.zmat")]
       [pours (build-path tmp "cc2_ours.zmat")])
  (zmat-write! A pa)
  (zmat-write! B pb)
  (define-values (ok1 m1) (run "mul" (path->string pa) (path->string pb) (path->string pref)))
  (zmat-write! (gpu-matmul A B) pours)
  (define-values (ok2 m2) (run "cmp" (path->string pref) (path->string pours)))
  (printf "  ~a" m2)
  (unless ok2 (set! fails (add1 fails)))
  (for ([p (in-list (list pa pb pref pours))]) (when (file-exists? p) (delete-file p))))

(printf "\n~a\n" (if (zero? fails)
                     "Racket and the independent CUDA C++ reference agree everywhere."
                     (format "~a MISMATCHES" fails)))
(gpu-shutdown!)
(when (> fails 0) (exit 1))
