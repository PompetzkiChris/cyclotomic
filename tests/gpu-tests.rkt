#lang racket/base
;; The GPU must agree with pure Racket, entry for entry, or it is worthless.

(require rackunit
         rackunit/text-ui
         racket/list
         "../field.rkt"
         "../matrix.rkt"
         "../mub.rkt"
         "../cuda/driver.rkt"
         "../cuda/gpu.rkt")

(define (random-int-elem f lo hi)
  (make-cyc f (for/list ([_ (in-range (field-degree f))])
                (+ lo (random (- hi lo))))))

(define (random-int-matrix f r c lo hi)
  (for/list ([_ (in-range r)])
    (for/list ([_ (in-range c)]) (random-int-elem f lo hi))))

;; pure-Racket reference product
(define (cpu-product f rows-a rows-b)
  (mat* (mat f rows-a) (mat f rows-b)))

(define (agree? f rows-a rows-b)
  (define g (zmat->matrix (gpu-matmul (zmat-of-integers f rows-a)
                                      (zmat-of-integers f rows-b))))
  (mat=? g (cpu-product f rows-a rows-b)))

(define availability-tests
  (test-suite
   "the driver is reachable from Racket"
   (check-true (cuda-available?) "nvcuda.dll loads")
   (check-true (> (cuda-device-count) 0) "at least one device")
   (let-values ([(maj min) (cuda-compute-capability 0)])
     (check-true (>= maj 5) (format "compute capability sm_~a~a" maj min)))))

(define agreement-tests
  (test-suite
   "GPU equals pure Racket, exactly"
   (for ([n (in-list '(8 12 24))])
     (define f (make-field n))
     (for ([dims (in-list '((1 1 1) (3 4 5) (16 16 16) (17 19 23) (40 40 40)))])
       (define r (first dims)) (define k (second dims)) (define c (third dims))
       (define A (random-int-matrix f r k -50 50))
       (define B (random-int-matrix f k c -50 50))
       (check-true (agree? f A B)
                   (format "Q(zeta_~a) ~ax~a @ ~ax~a exact" n r k k c))))))

(define identity-tests
  (test-suite
   "algebraic sanity through the GPU path"
   (let* ([f (make-field 24)]
          [n 12]
          [I (for/list ([i (in-range n)])
               (for/list ([j (in-range n)])
                 (if (= i j) (cyc-one f) (cyc-zero f))))]
          [A (random-int-matrix f n n -30 30)])
     (check-true (mat=? (zmat->matrix (gpu-matmul (zmat-of-integers f A)
                                                  (zmat-of-integers f I)))
                        (mat f A))
                 "A I = A on the device")
     (check-true (mat=? (zmat->matrix (gpu-matmul (zmat-of-integers f I)
                                                  (zmat-of-integers f A)))
                        (mat f A))
                 "I A = A on the device"))))

(define zeta-tests
  (test-suite
   "the reduction table is doing real work"
   ;; zeta^(d) .. zeta^(2d-2) all fold back through Phi_n; a 1x1 product of
   ;; high powers exercises every row of R.
   (for ([n (in-list '(8 12 24))])
     (define f (make-field n))
     (define d (field-degree f))
     (for* ([i (in-range d)] [j (in-range d)])
       (define A (list (list (cyc-basis f i))))
       (define B (list (list (cyc-basis f j))))
       (define g (zmat->matrix (gpu-matmul (zmat-of-integers f A)
                                           (zmat-of-integers f B))))
       (check-true (cyc=? (mat-ref g 0 0) (cyc* (cyc-basis f i) (cyc-basis f j)))
                   (format "Q(zeta_~a): zeta^~a * zeta^~a reduced correctly" n i j))))))

(define overflow-tests
  (test-suite
   "int64 is a wall, not a rounding error"
   (let* ([f (make-field 24)]
          [n 64]
          [big (expt 2 40)]
          [A (for/list ([_ (in-range n)])
               (for/list ([_ (in-range n)])
                 (make-cyc f (for/list ([_ (in-range (field-degree f))]) big))))])
     (check-exn exn:fail?
                (lambda () (gpu-matmul (zmat-of-integers f A) (zmat-of-integers f A)))
                "a product that would wrap is refused, not wrapped"))
   (let ([f (make-field 8)])
     (check-exn exn:fail?
                (lambda ()
                  (zmat-of-integers f (list (list (make-cyc f (list 1/2))))))
                "a non-integer coefficient is refused by the Z[zeta_n] path"))))

(define mub-through-gpu-tests
  (test-suite
   "dimension 6 MUBs, verified with the GPU doing the multiplication"
   ;; The MUB matrices have denominators, so clear them: for bases B, C over a
   ;; common denominator q, |<b,c>|^2 = 1/6 iff the integer matrices satisfy
   ;; the same relation scaled by q^2. Here we check unitarity in the form
   ;; (q B)(q B)^H = q^2 I, entirely on the device.
   (let* ([ms (mubs-d6)]
          [F (make-field 24)])
     (for ([p (in-list ms)])
       (define M (cdr p))
       ;; common denominator of every coefficient
       (define q
         (for*/fold ([l 1]) ([i (in-range 6)] [j (in-range 6)]
                             [x (in-vector (cyc-coeffs (mat-ref M i j)))])
           (lcm l (denominator x))))
       (define scaled
         (for/list ([i (in-range 6)])
           (for/list ([j (in-range 6)])
             (cyc-scale (mat-ref M i j) q))))
       (define adj
         (for/list ([i (in-range 6)])
           (for/list ([j (in-range 6)])
             (cyc-scale (cyc-conjugate (mat-ref M j i)) q))))
       (define prod (zmat->matrix (gpu-matmul (zmat-of-integers F scaled)
                                              (zmat-of-integers F adj))))
       (define want
         (mat F (for/list ([i (in-range 6)])
                  (for/list ([j (in-range 6)])
                    (if (= i j) (cyc-rational F (* q q)) (cyc-zero F))))))
       (check-true (mat=? prod want)
                   (format "~a: (qB)(qB)^H = q^2 I on the GPU, q = ~a" (car p) q))))))

(module+ test
  (gpu-init!)
  (void
   (run-tests
    (test-suite "gpu"
                availability-tests
                agreement-tests
                identity-tests
                zeta-tests
                overflow-tests
                mub-through-gpu-tests)))
  (gpu-shutdown!))
