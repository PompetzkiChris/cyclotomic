#lang racket/base
;; The hardening has to be tested, not asserted.

(require rackunit
         "../exact-io.rkt"
         rackunit/text-ui
         racket/list
         "../field.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(define (free-now) (let-values ([(f t) (cuda-mem-info)]) f))
(define MB (* 1024 1024))
(define BIG (* 256 MB))

(define buffer-tests
  (test-suite
   "device buffers are released on every exit path"

   (let ([before (free-now)])
     (call-with-device-buffer BIG (lambda (p) (void)))
     (check-true (< (abs (- before (free-now))) (* 4 MB))
                 "normal return frees"))

   (let ([before (free-now)])
     (with-handlers ([exn:fail? void])
       (call-with-device-buffer BIG (lambda (p) (error 'boom "raise inside"))))
     (check-true (< (abs (- before (free-now))) (* 4 MB))
                 "raise inside frees"))

   (let ([before (free-now)])
     (let/ec escape
       (call-with-device-buffer BIG (lambda (p) (escape 'gone))))
     (check-true (< (abs (- before (free-now))) (* 4 MB))
                 "escape continuation frees"))

   (let ([before (free-now)])
     (with-handlers ([exn:fail? void])
       (call-with-device-buffers (list BIG BIG BIG)
                                 (lambda (a b c) (error 'boom "raise in the middle"))))
     (check-true (< (abs (- before (free-now))) (* 4 MB))
                 "multi-buffer raise frees all of them"))

   (let ([before (free-now)])
     (let/ec escape
       (call-with-device-buffers (list BIG BIG BIG)
                                 (lambda (a b c) (escape 'gone))))
     (check-true (< (abs (- before (free-now))) (* 4 MB))
                 "multi-buffer escape frees all of them"))))

(define ptx-tests
  (test-suite
   "a bad module says why"
   (let ([msg (with-handlers ([exn:fail? (lambda (e) (exn-message e))])
                (load-ptx #"this is not ptx at all\n")
                "NO ERROR RAISED")])
     (printf "\n  bad-PTX message:\n  ~a\n\n"
             (regexp-replace* #rx"\n" msg "\n  "))
     (check-true (regexp-match? #rx"load-ptx" msg) "raises from load-ptx")
     (check-true (regexp-match? #rx"JIT error log" msg) "includes the JIT log section")
     (check-true (> (string-length msg) 60) "the message is not empty boilerplate"))))

(define scheduler-tests
  (test-suite
   "a long kernel does not freeze the Racket scheduler"
   (let* ([F (make-field 24)]
          [d (field-degree F)]
          [n 768]
          [rows (for/list ([_ (in-range n)])
                  (for/list ([_ (in-range n)])
                    (make-cyc F (for/list ([_ (in-range d)]) (- (random 16) 8)))))]
          [A (zmat-of-integers F rows)])
     (void (gpu-matmul A A #:audit? #f))
     (let* ([ticks (box 0)]
            [stop (box #f)]
            [ticker (thread (lambda ()
                              (let loop ()
                                (unless (unbox stop)
                                  (set-box! ticks (add1 (unbox ticks)))
                                  (sleep 0)
                                  (loop)))))])
       (sleep 1/20)
       (let* ([t-before (unbox ticks)]
              [t0 (now-ms)]
              [_ (void (gpu-matmul A A #:audit? #f))]
              [ms (- (now-ms) t0)]
              [during (- (unbox ticks) t-before)])
         (set-box! stop #t)
         (sync ticker)
         (printf "  product ~a ms, ticker advanced ~a times DURING the call
"
                 (round ms) during)
         (check-true (> during 0)
                     "another thread made progress while the kernel ran"))))))

(module+ test
  (gpu-init!)
  (void
   (run-tests
    (test-suite "hardening" buffer-tests ptx-tests scheduler-tests)))
  (gpu-shutdown!))
