#lang racket/base
;; Where does the time actually go? Before optimising anything, split one
;; product into its stages and measure each. Every number below is wall time
;; around a synchronised device, in exact milliseconds.

(require racket/list
         "../exact-io.rkt"
         "../field.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(gpu-init!)

(define (pad v n)
  (define s (format "~a" v))
  (string-append (make-string (max 0 (- n (string-length s))) #\space) s))

(define (ms thunk [reps 3])
  (thunk)                                  ; warm
  (define t (now-ms))
  (for ([_ (in-range reps)]) (thunk))
  (/ (- (now-ms) t) reps))

(printf "field  n     build  matmul+audit  matmul  dmat*(resident)  audit-cost  host-cost\n")
(for* ([ncyc (in-list '(8 24))]
       [n (in-list '(512 1024 2048))])
  (define F (make-field ncyc))
  (define lim (if (= ncyc 8) 128 32))
  (define tb (now-ms))
  (define A (zmat-build F n n (lambda (i j t) (- (random (* 2 lim)) lim))))
  (define build (- (now-ms) tb))
  (define full (ms (lambda () (void (gpu-matmul A A #:audit? #t)))))
  (define noaudit (ms (lambda () (void (gpu-matmul A A #:audit? #f)))))
  (define resident
    (call-with-dmat A (lambda (dA)
      (ms (lambda () (dmat-free! (dmat* dA dA #:audit? #f)))))))
  (printf "Q(z~a) ~a  ~a  ~a  ~a  ~a  ~a  ~a\n"
          ncyc (pad n 5) (pad build 6)
          (pad (round full) 10) (pad (round noaudit) 8)
          (pad (round resident) 12)
          (pad (round (- full noaudit)) 9)
          (pad (round (- noaudit resident)) 9)))

(printf "\n=== per-stage, Q(z24) 1024x1024, w32 path ===\n")
(let* ([F (make-field 24)]
       [d (field-degree F)]
       [n 1024]
       [A (zmat-build F n n (lambda (i j t) (- (random 64) 32)))]
       [bs (zmat-planes A)]
       [nel (* d n n)])
  (printf "planes: ~a MB int64\n" (bytes->mb (bytes-length bs)))
  (define dA (device-alloc (bytes-length bs)))
  (printf "  H2D  (unpinned) : ~a ms  -> ~a GB/s\n"
          (pad (round (ms (lambda () (copy-to-device! dA bs) (synchronize!)) 5)) 5)
          (let ([t (ms (lambda () (copy-to-device! dA bs) (synchronize!)) 5)])
            (if (zero? t) "inf" (dec (/ (bytes-length bs) t 1000000) 1))))
  (define out (make-bytes (bytes-length bs)))
  (printf "  D2H  (unpinned) : ~a ms\n"
          (pad (round (ms (lambda () (copy-from-device! out dA) (synchronize!)) 5)) 5))
  (printf "  absmax pass     : ~a ms  (called 2x before launch, 1x for audit)\n"
          (pad (round (ms (lambda () (void (gpu-absmax dA nel))) 5)) 5))
  (device-free! dA))

(gpu-shutdown!)
