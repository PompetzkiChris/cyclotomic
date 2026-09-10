#lang racket/base
;; The GPU must be on the real math path, and must give the same answers.

(require rackunit
         rackunit/text-ui
         racket/list
         "../field.rkt"
         "../matrix.rkt"
         "../mub.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt"
         "../cuda/accel.rkt")

(define (rand-rows F n lim)
  (for/list ([_ (in-range n)])
    (for/list ([_ (in-range n)])
      (make-cyc F (for/list ([_ (in-range (field-degree F))])
                    (/ (- (random (* 2 lim)) lim) (add1 (random 6))))))))

(define agreement-tests
  (test-suite
   "accelerated mat* equals the CPU mat*, exactly"
   (for ([n (in-list '(8 12 24))])
     (let* ([F (make-field n)])
       (for ([sz (in-list '(1 2 6 17 33))])
         (let* ([A (mat F (rand-rows F sz 20))]
                [B (mat F (rand-rows F sz 20))]
                [cpu (mat*/cpu A B)]
                [gpu (parameterize ([current-mat*-hook (current-mat*-hook)])
                       (mat* A B))])
           (check-true (mat=? cpu gpu)
                       (format "Q(zeta_~a) ~ax~a: GPU result equals CPU result" n sz sz))))))))

(define usage-tests
  (test-suite
   "the device is actually being used, counted not assumed"
   (let* ([F (make-field 24)]
          [A (mat F (rand-rows F 24 15))]
          [B (mat F (rand-rows F 24 15))])
     (gpu-reset-stats!)
     (void (mat* A B))
     (let ([s (gpu-stats)])
       (check-true (> (hash-ref s 'matmuls) 0) "mat* reached the device")
       (check-true (> (hash-ref s 'launches) 0) "kernels were launched")
       (check-true (> (hash-ref s 'uploaded-bytes) 0) "data went to the device")
       (check-true (> (hash-ref s 'downloaded-bytes) 0) "data came back")))))

(define mub-on-gpu-tests
  (test-suite
   "the dimension-6 MUB verification runs on the device"
   (gpu-reset-stats!)
   (let ([ms (map cdr (mubs-d6))])
     (check-true (mutually-unbiased? ms 6)
                 "3 pairwise unbiased bases in dimension 6")
     (let ([s (gpu-stats)])
       (printf "  d=6 verification used ~a device products, ~a launches\n"
               (hash-ref s 'matmuls) (hash-ref s 'launches))
       (check-true (> (hash-ref s 'matmuls) 0)
                   "the MUB check went through the GPU, not around it")))))

(define fallback-tests
  (test-suite
   "declining is safe"
   (let* ([F (make-field 24)]
          [A (mat F (rand-rows F 8 10))]
          [B (mat F (rand-rows F 8 10))]
          [want (mat*/cpu A B)])
     ;; force every product onto the CPU by raising the work threshold
     (parameterize ([current-gpu-min-work 100000000])
       (check-true (mat=? (mat* A B) want) "declined product still exact"))
     ;; and with the hook removed entirely
     (parameterize ([current-mat*-hook #f])
       (check-true (mat=? (mat* A B) want) "uninstalled hook still exact")))))

(module+ test
  (gpu-accel-install!)
  (void
   (run-tests
    (test-suite "accel" agreement-tests usage-tests mub-on-gpu-tests fallback-tests)))
  (printf "\n")
  (gpu-stats-report)
  (printf "hook declines: ~a\n" (gpu-accel-declines))
  (gpu-accel-uninstall!)
  (gpu-shutdown!))
