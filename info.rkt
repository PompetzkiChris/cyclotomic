#lang info

(define collection "cyclotomic")
(define pkg-desc
  "Exact arithmetic in cyclotomic fields Q(zeta_n), exact matrices, mutually-unbiased-basis verification, and exact Z[zeta_n] linear algebra on NVIDIA GPUs through the CUDA driver API. No floating point.")
(define version "0.1")
(define pkg-authors '("cpompetzki"))
(define license 'MIT)
(define deps '("base" "math-lib"))
(define build-deps '("rackunit-lib" "racket-doc" "scribble-lib"))
(define test-omit-paths '("tools" "refcheck"))
(define compile-omit-paths '("tools" "refcheck"))
