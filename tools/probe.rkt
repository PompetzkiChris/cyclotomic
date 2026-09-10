#lang racket/base
;; Does Racket actually reach the GPU? Report what the driver says.

(require "../cuda/driver.rkt")

(printf "nvcuda.dll found : ~a\n" (cuda-available?))
(cuda-init!)
(define v (cuda-driver-version))
(printf "driver version   : ~a (CUDA ~a.~a)\n" v (quotient v 1000)
        (quotient (modulo v 1000) 10))
(printf "devices          : ~a\n" (cuda-device-count))

(for ([i (in-range (cuda-device-count))])
  (define-values (maj min) (cuda-compute-capability i))
  (printf "\ndevice ~a: ~a\n" i (cuda-device-name i))
  (printf "  compute capability : sm_~a~a\n" maj min)
  (printf "  multiprocessors    : ~a\n" (cuda-device-attribute i ATTR-MULTIPROCESSOR-COUNT))
  (printf "  max threads/block  : ~a\n" (cuda-device-attribute i ATTR-MAX-THREADS-PER-BLOCK))
  (printf "  SM clock           : ~a MHz\n"
          (quotient (cuda-device-attribute i ATTR-CLOCK-RATE) 1000))
  (printf "  memory bus width   : ~a bit\n" (cuda-device-attribute i ATTR-BUS-WIDTH))
  (printf "  memory clock       : ~a MHz\n"
          (quotient (cuda-device-attribute i ATTR-MEMORY-CLOCK-RATE) 1000)))

(define ctx (make-context 0))
(define-values (free total) (cuda-mem-info))
(printf "\ncontext created. VRAM ~a.~a GB free of ~a.~a GB\n"
        (quotient free (expt 2 30))
        (quotient (* 10 (modulo free (expt 2 30))) (expt 2 30))
        (quotient total (expt 2 30))
        (quotient (* 10 (modulo total (expt 2 30))) (expt 2 30)))

;; round-trip some exact integers through device memory
(define n 1024)
(define nbytes (* n 8))
(define src (make-bytes nbytes))
(for ([i (in-range n)])
  (integer->integer-bytes (- (* i i) 500000) 8 #t #f src (* i 8)))
(define d (device-alloc nbytes))
(copy-to-device! d src)
(define back (make-bytes nbytes))
(copy-from-device! back d)
(device-free! d)
(printf "host->device->host round trip of ~a int64s: ~a\n"
        n (if (equal? src back) "IDENTICAL" "CORRUPTED"))

(context-destroy! ctx)
(printf "context destroyed cleanly.\n")
