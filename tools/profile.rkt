#lang racket/base
;; Where does the wall time go, before and after the bulk path?
(require racket/list
         "../field.rkt"
         "../matrix.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(gpu-init!)
(define F (make-field 24))
(define deg (field-degree F))

(define-syntax-rule (ms expr)
  (let* ([t (current-inexact-milliseconds)] [v expr])
    (values v (- (current-inexact-milliseconds) t))))

(printf "\n--- via cyc structs (zmat-of-integers / zmat->matrix) ---\n")
(printf "~a\t~a\t~a\t~a\t~a\t~a\n" "n" "gen" "toplanes" "gpu" "back" "gpu_share")
(for ([n (in-list '(256 512 1024))])
  (define-values (rows tgen)
    (ms (for/list ([_ (in-range n)])
          (for/list ([_ (in-range n)])
            (make-cyc F (for/list ([_ (in-range deg)]) (- (random 64) 32)))))))
  (define-values (A tconv) (ms (zmat-of-integers F rows)))
  (define-values (C tgpu)  (ms (gpu-matmul A A #:audit? #f)))
  (define-values (M tback) (ms (zmat->matrix C)))
  (define total (+ tgen tconv tgpu tback))
  (printf "~a\t~a\t~a\t~a\t~a\t~a%\n" n (round tgen) (round tconv) (round tgpu)
          (round tback) (round (* 100 (/ tgpu total)))))

(printf "\n--- via the bulk path (zmat-build / zmat-coeff) ---\n")
(printf "~a\t~a\t~a\t~a\t~a\n" "n" "build" "gpu" "readback" "gpu_share")
(for ([n (in-list '(256 512 1024 2048))])
  (define-values (A tbuild)
    (ms (zmat-build F n n (lambda (i j t) (- (random 64) 32)))))
  (define-values (C tgpu) (ms (gpu-matmul A A #:audit? #f)))
  (define-values (acc tback)
    (ms (for*/fold ([s 0]) ([i (in-range n)] [j (in-range n)])
          (+ s (zmat-coeff C i j 0)))))
  (define total (+ tbuild tgpu tback))
  (printf "~a\t~a\t~a\t~a\t~a%\n" n (round tbuild) (round tgpu) (round tback)
          (round (* 100 (/ tgpu total)))))

(printf "\n--- the two paths agree ---\n")
(let* ([n 32]
       [rows (for/list ([i (in-range n)])
               (for/list ([j (in-range n)])
                 (make-cyc F (for/list ([t (in-range deg)]) (+ (* 7 i) (* 3 j) t)))))]
       [A1 (zmat-of-integers F rows)]
       [A2 (zmat-build F n n (lambda (i j t) (+ (* 7 i) (* 3 j) t)))]
       [C1 (zmat->matrix (gpu-matmul A1 A1))]
       [C2 (zmat->matrix (gpu-matmul A2 A2))])
  (printf "identical: ~a\n" (mat=? C1 C2)))
(gpu-shutdown!)
