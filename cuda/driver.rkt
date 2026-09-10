#lang racket/base
;; Racket bindings to the CUDA Driver API (nvcuda.dll).
;;
;; The driver API rather than the runtime API, deliberately: it is a stable C
;; ABI, it needs no host compiler at run time, and it loads PTX directly. That
;; means Racket can own the whole pipeline -- allocate, upload, launch, read
;; back -- with no Python and no C shim in the path.
;;
;; Every entry point is checked. A CUDA error raises a Racket exception
;; carrying the driver's own message, so a failure is never silent.

;; No contract-out here: ffi/unsafe and racket/contract both export `->`, and
;; `_fun` needs its own. This layer is unsafe by construction anyway -- the
;; contracts live on the pure mathematics in cyclotomic/, where they mean
;; something. What this module guarantees instead is that every driver call is
;; checked and every failure raises.
(require ffi/unsafe
         ffi/unsafe/define)

(provide cuda-available? cuda-init!
         cuda-device-count cuda-device-name cuda-device-attribute
         cuda-compute-capability cuda-driver-version
         make-context cuda-context? context-destroy!
         cuda-mem-info
         device-alloc device-free!
         call-with-device-buffer call-with-device-buffers
         copy-to-device! copy-from-device! device-memset-32!
         load-ptx unload-module! cuda-module? module-function cuda-function?
         release-sync-event!
         launch! synchronize! synchronize-blocking! current-gpu-wait
         ;; streams, pinned host memory, async transfers
         make-stream stream? stream-destroy! stream-synchronize!
         pinned-alloc pinned-free! pinned-ptr pinned-bytes pinned?
         call-with-pinned
         copy-to-device/async! copy-from-device/async!
         ;; launch configuration from the device, not from a guess
         max-potential-block-size
         ATTR-MULTIPROCESSOR-COUNT
         ATTR-CLOCK-RATE
         ATTR-MEMORY-CLOCK-RATE
         ATTR-BUS-WIDTH
         ATTR-MAX-THREADS-PER-BLOCK
         ATTR-COMPUTE-CAPABILITY-MAJOR
         ATTR-COMPUTE-CAPABILITY-MINOR)

;; --------------------------------------------------------------- the library

(define nvcuda
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "nvcuda" '(#f))))

(define (cuda-available?) (and nvcuda #t))

(define-ffi-definer define-cu (or nvcuda (ffi-lib #f))
  #:default-make-fail make-not-available)

;; --------------------------------------------------------------- basic types

(define _CUresult _int32)
(define _CUdevice _int32)
(define _CUdeviceptr _uint64)
(define _CUcontext (_cpointer/null 'CUctx))
(define _CUmodule (_cpointer/null 'CUmod))
(define _CUfunction (_cpointer/null 'CUfunc))
(define _CUstream (_cpointer/null 'CUstream))

;; ------------------------------------------------------------ error handling

(define-cu cuGetErrorString
  (_fun _CUresult (out : (_ptr o _string/utf-8)) -> (r : _CUresult)
        -> (if (zero? r) out "unknown CUDA error")))

(define (check who r)
  (unless (zero? r)
    (error who "CUDA error ~a: ~a" r
           (with-handlers ([exn:fail? (lambda (_) "<no message>")])
             (cuGetErrorString r))))
  (void))

;; ------------------------------------------------------------------ bindings

(define-cu cuInit (_fun _uint32 -> _CUresult))
(define-cu cuDriverGetVersion (_fun (out : (_ptr o _int32)) -> (r : _CUresult)
                                    -> (values r out)))
(define-cu cuDeviceGetCount (_fun (out : (_ptr o _int32)) -> (r : _CUresult)
                                  -> (values r out)))
(define-cu cuDeviceGet (_fun (out : (_ptr o _CUdevice)) _int32 -> (r : _CUresult)
                             -> (values r out)))
(define-cu cuDeviceGetName
  (_fun (buf : _bytes) (len : _int32 = (bytes-length buf)) _CUdevice -> _CUresult))
(define-cu cuDeviceGetAttribute
  (_fun (out : (_ptr o _int32)) _int32 _CUdevice -> (r : _CUresult) -> (values r out)))

;; The primary context, not a fresh one: it is what every other CUDA library on
;; the process uses, so retaining it interoperates instead of competing.
(define-cu cuDevicePrimaryCtxRetain
  (_fun (out : (_ptr o _CUcontext)) _CUdevice -> (r : _CUresult) -> (values r out)))
(define-cu cuDevicePrimaryCtxRelease_v2 (_fun _CUdevice -> _CUresult))
(define-cu cuCtxSetCurrent (_fun _CUcontext -> _CUresult))
(define-cu cuCtxSynchronize (_fun #:blocking? #t -> _CUresult))

;; Events, so a completed kernel can be waited on by POLLING rather than by
;; blocking. Racket CS runs every green thread on one OS thread, so a blocking
;; driver call freezes the whole runtime no matter what #:blocking? says.
;; Polling with (sleep 0) between queries keeps the scheduler alive.
(define _CUevent (_cpointer/null 'CUevent))
(define CUDA_ERROR_NOT_READY 600)
(define-cu cuEventCreate
  (_fun (out : (_ptr o _CUevent)) _uint32 -> (r : _CUresult) -> (values r out)))
(define-cu cuEventRecord (_fun _CUevent _CUstream -> _CUresult))
(define-cu cuEventQuery (_fun _CUevent -> _CUresult))
(define-cu cuEventDestroy_v2 (_fun _CUevent -> _CUresult))
(define-cu cuMemGetInfo_v2
  (_fun (free : (_ptr o _size)) (total : (_ptr o _size)) -> (r : _CUresult)
        -> (values r free total)))

(define-cu cuMemAlloc_v2
  (_fun (out : (_ptr o _CUdeviceptr)) _size -> (r : _CUresult) -> (values r out)))
(define-cu cuMemFree_v2 (_fun _CUdeviceptr -> _CUresult))

;; #:blocking? #t keeps the Racket scheduler alive across these, which in turn
;; means the GC may run during the call -- so the host pointer handed over must
;; be immobile. copy-to-device! / copy-from-device! stage through an
;; 'atomic-interior buffer for exactly that reason; passing a plain byte string
;; here would be a moving target.
(define-cu cuMemcpyHtoD_v2 (_fun #:blocking? #t _CUdeviceptr _pointer _size -> _CUresult))
(define-cu cuMemcpyDtoH_v2 (_fun #:blocking? #t _pointer _CUdeviceptr _size -> _CUresult))
(define-cu cuMemsetD32_v2 (_fun _CUdeviceptr _uint32 _size -> _CUresult))

(define-cu cuModuleLoadDataEx
  (_fun #:blocking? #t
        (out : (_ptr o _CUmodule)) _pointer _uint32 _pointer _pointer
        -> (r : _CUresult) -> (values r out)))
(define-cu cuModuleUnload (_fun _CUmodule -> _CUresult))
(define-cu cuModuleGetFunction
  (_fun (out : (_ptr o _CUfunction)) _CUmodule _string/utf-8 -> (r : _CUresult)
        -> (values r out)))

(define-cu cuStreamCreate
  (_fun (out : (_ptr o _CUstream)) _uint32 -> (r : _CUresult) -> (values r out)))
(define-cu cuStreamDestroy_v2 (_fun _CUstream -> _CUresult))
(define-cu cuStreamSynchronize (_fun #:blocking? #t _CUstream -> _CUresult))
(define-cu cuStreamQuery (_fun _CUstream -> _CUresult))

;; Page-locked host memory. A pageable buffer forces the driver to stage the
;; copy through its own pinned bounce buffer, which costs roughly half the
;; achievable bandwidth and rules out a genuinely asynchronous transfer.
(define-cu cuMemHostAlloc
  (_fun (out : (_ptr o _pointer)) _size _uint32 -> (r : _CUresult) -> (values r out)))
(define-cu cuMemFreeHost (_fun _pointer -> _CUresult))
(define-cu cuMemcpyHtoDAsync_v2
  (_fun _CUdeviceptr _pointer _size _CUstream -> _CUresult))
(define-cu cuMemcpyDtoHAsync_v2
  (_fun _pointer _CUdeviceptr _size _CUstream -> _CUresult))

(define-cu cuOccupancyMaxPotentialBlockSize
  (_fun (grid : (_ptr o _int32)) (block : (_ptr o _int32))
        _CUfunction _pointer _size _int32
        -> (r : _CUresult) -> (values r grid block)))

(define-cu cuLaunchKernel
  (_fun #:blocking? #t
        _CUfunction
        _uint32 _uint32 _uint32          ; grid
        _uint32 _uint32 _uint32          ; block
        _uint32                          ; shared bytes
        _CUstream
        _pointer                         ; kernelParams
        _pointer                         ; extra
        -> _CUresult))

;; ------------------------------------------------------------ device queries

(define ATTR-MAX-THREADS-PER-BLOCK 1)
(define ATTR-CLOCK-RATE 13)
(define ATTR-MULTIPROCESSOR-COUNT 16)
(define ATTR-MEMORY-CLOCK-RATE 36)
(define ATTR-BUS-WIDTH 37)
(define ATTR-COMPUTE-CAPABILITY-MAJOR 75)
(define ATTR-COMPUTE-CAPABILITY-MINOR 76)

(define initialised? #f)

(define (cuda-init!)
  (unless initialised?
    (unless nvcuda
      (error 'cuda-init! "nvcuda.dll not found; is an NVIDIA driver installed?"))
    (check 'cuInit (cuInit 0))
    (set! initialised? #t))
  (void))

(define (cuda-driver-version)
  (cuda-init!)
  (define-values (r v) (cuDriverGetVersion))
  (check 'cuDriverGetVersion r)
  v)

(define (cuda-device-count)
  (cuda-init!)
  (define-values (r n) (cuDeviceGetCount))
  (check 'cuDeviceGetCount r)
  n)

(define (device-handle ordinal)
  (cuda-init!)
  (define-values (r d) (cuDeviceGet ordinal))
  (check 'cuDeviceGet r)
  d)

(define (cuda-device-name ordinal)
  (define buf (make-bytes 256 0))
  (check 'cuDeviceGetName (cuDeviceGetName buf (device-handle ordinal)))
  (define z (or (for/first ([i (in-range (bytes-length buf))]
                            #:when (zero? (bytes-ref buf i))) i)
                (bytes-length buf)))
  (bytes->string/utf-8 (subbytes buf 0 z)))

(define (cuda-device-attribute ordinal attr)
  (define-values (r v) (cuDeviceGetAttribute attr (device-handle ordinal)))
  (check 'cuDeviceGetAttribute r)
  v)

(define (cuda-compute-capability ordinal)
  (values (cuda-device-attribute ordinal ATTR-COMPUTE-CAPABILITY-MAJOR)
          (cuda-device-attribute ordinal ATTR-COMPUTE-CAPABILITY-MINOR)))

;; ---------------------------------------------------------------- contexts

(struct cuda-context (ptr device) #:transparent)

(define (make-context ordinal)
  (define dev (device-handle ordinal))
  (define-values (r ctx) (cuDevicePrimaryCtxRetain dev))
  (check 'cuDevicePrimaryCtxRetain r)
  (check 'cuCtxSetCurrent (cuCtxSetCurrent ctx))
  (cuda-context ctx dev))

(define (context-destroy! c)
  (check 'cuDevicePrimaryCtxRelease
         (cuDevicePrimaryCtxRelease_v2 (cuda-context-device c))))

;; How to wait for the device. There is a real trade-off and it is the caller's
;; to make, so it is a parameter rather than a hardcoded choice:
;;
;;   'poll  (default) event-poll with (sleep 0) between queries. Other Racket
;;          threads keep running. Costs roughly 30% on a short kernel.
;;   'block cuCtxSynchronize. Maximum throughput, and it freezes every green
;;          thread in the process until the device is done.
(define current-gpu-wait (make-parameter 'poll))

(define (synchronize!)
  (case (current-gpu-wait)
    [(block) (synchronize-blocking!)]
    [else (synchronize-polling!)]))

;; Wait for the device by polling an event, yielding to the Racket scheduler
;; between polls, so other threads keep running while the GPU works.
(define cached-event #f)

(define (sync-event)
  (or cached-event
      (let-values ([(r ev) (cuEventCreate 2)])   ; CU_EVENT_DISABLE_TIMING
        (check 'cuEventCreate r)
        (set! cached-event ev)
        ev)))

(define (release-sync-event!)
  (when cached-event
    (with-handlers ([exn:fail? void]) (void (cuEventDestroy_v2 cached-event)))
    (set! cached-event #f)))

;; One event, reused. Creating and destroying one per product is measurable
;; when a chain issues thousands of syncs.
(define (synchronize-polling!)
  (define ev (sync-event))
  (let ()
     (check 'cuEventRecord (cuEventRecord ev #f))
     (let loop ([spins 0])
       (define q (cuEventQuery ev))
       (cond
         [(zero? q) (void)]
         [(= q CUDA_ERROR_NOT_READY)
          ;; (sleep 0) yields to the Racket scheduler and returns as soon as it
          ;; comes back, so other threads run without adding latency here. Only
          ;; a genuinely long wait escalates to a real sleep, which stops this
          ;; from spinning a core for seconds on a big kernel.
          (if (< spins 200000) (sleep 0) (sleep 1/2000))
          (loop (add1 spins))]
         [else (check 'cuEventQuery q)]))))

;; Hard blocking sync, for teardown paths where yielding is pointless.
(define (synchronize-blocking!)
  (check 'cuCtxSynchronize (cuCtxSynchronize)))

(define (cuda-mem-info)
  (define-values (r free total) (cuMemGetInfo_v2))
  (check 'cuMemGetInfo r)
  (values free total))

;; ------------------------------------------------------------------- memory

(define (device-alloc nbytes)
  (define-values (r p) (cuMemAlloc_v2 nbytes))
  (check 'cuMemAlloc r)
  p)

(define (device-free! p) (check 'cuMemFree (cuMemFree_v2 p)))

;; Allocate, run body, free -- even if body raises or escapes. Device memory is
;; not GC-managed, so an early exit without this leaks VRAM until the process
;; dies.
(define (call-with-device-buffer nbytes proc)
  (define p (device-alloc (max 1 nbytes)))
  (dynamic-wind void
                (lambda () (proc p))
                (lambda () (device-free! p))))

(define (call-with-device-buffers sizes proc)
  (let loop ([todo sizes] [got '()])
    (cond [(null? todo) (apply proc (reverse got))]
          [else (call-with-device-buffer
                 (car todo)
                 (lambda (p) (loop (cdr todo) (cons p got))))])))

;; Stage through an immobile buffer: the memcpy calls are #:blocking?, so the
;; GC may move a plain byte string out from under the driver mid-copy.
;;
;; 'atomic-interior memory is GC-managed and immobile. It must NOT be passed to
;; `free` -- that is a heap corruption, not a leak. It is reclaimed when
;; unreachable, and stays put in the meantime, which is exactly what is wanted.
(define (copy-to-device! dptr bs)
  (define n (bytes-length bs))
  (define buf (malloc n 'atomic-interior))
  (memcpy buf bs n)
  (check 'cuMemcpyHtoD (cuMemcpyHtoD_v2 dptr buf n)))

(define (copy-from-device! bs dptr)
  (define n (bytes-length bs))
  (define buf (malloc n 'atomic-interior))
  (check 'cuMemcpyDtoH (cuMemcpyDtoH_v2 buf dptr n))
  (memcpy bs buf n))


;; ------------------------------------------------------------------ streams

(struct stream (ptr) #:transparent)

(define (make-stream)
  (define-values (r s) (cuStreamCreate 1))   ; CU_STREAM_NON_BLOCKING
  (check 'cuStreamCreate r)
  (stream s))

(define (stream-destroy! s) (check 'cuStreamDestroy (cuStreamDestroy_v2 (stream-ptr s))))

;; Poll rather than block, for the same reason synchronize! does.
(define (stream-synchronize! s)
  (let loop ([spins 0])
    (define q (cuStreamQuery (stream-ptr s)))
    (cond
      [(zero? q) (void)]
      [(= q CUDA_ERROR_NOT_READY)
       (if (< spins 200000) (sleep 0) (sleep 1/2000))
       (loop (add1 spins))]
      [else (check 'cuStreamQuery q)])))

;; ------------------------------------------------------- pinned host memory

(struct pinned (ptr size) #:transparent
  #:constructor-name make-pinned-record)

(define (pinned-alloc nbytes)
  (define-values (r p) (cuMemHostAlloc (max 1 nbytes) 0))
  (check 'cuMemHostAlloc r)
  (make-pinned-record p nbytes))

(define (pinned-free! h) (check 'cuMemFreeHost (cuMemFreeHost (pinned-ptr h))))

(define (pinned-bytes h [n #f])
  (define k (or n (pinned-size h)))
  (define bs (make-bytes k))
  (memcpy bs (pinned-ptr h) k)
  bs)

(define (call-with-pinned nbytes proc)
  (define h (pinned-alloc nbytes))
  (dynamic-wind void (lambda () (proc h)) (lambda () (pinned-free! h))))

(define (copy-to-device/async! dptr h nbytes s)
  (check 'cuMemcpyHtoDAsync
         (cuMemcpyHtoDAsync_v2 dptr (pinned-ptr h) nbytes (stream-ptr s))))

(define (copy-from-device/async! h dptr nbytes s)
  (check 'cuMemcpyDtoHAsync
         (cuMemcpyDtoHAsync_v2 (pinned-ptr h) dptr nbytes (stream-ptr s))))

;; Ask the driver what block size keeps this kernel occupied, instead of
;; hardcoding 256 and hoping.
(define (max-potential-block-size fn [dynamic-shared 0] [block-limit 0])
  (define-values (r grid block)
    (cuOccupancyMaxPotentialBlockSize (cuda-function-ptr fn) #f dynamic-shared block-limit))
  (check 'cuOccupancyMaxPotentialBlockSize r)
  (values grid block))

(define (device-memset-32! dptr value count)
  (check 'cuMemsetD32 (cuMemsetD32_v2 dptr value count)))

;; ------------------------------------------------------------------ modules

(struct cuda-module (ptr) #:transparent)
(struct cuda-function (ptr) #:transparent)

;; JIT option codes
(define CU_JIT_INFO_LOG_BUFFER 3)
(define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4)
(define CU_JIT_ERROR_LOG_BUFFER 5)
(define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6)

;; Load PTX with the JIT logs wired up, so a bad module says what the JIT
;; objected to instead of just "invalid image".
(define (load-ptx src)
  (define image
    (cond [(bytes? src) src]
          [else (call-with-input-file src (lambda (in) (port->bytes* in)))]))
  ;; Same refusal as compile-cuda, so PTX from any source -- a file, another
  ;; toolchain, a string -- cannot bring floating point into this process.
  (assert-no-float-image! image)
  (define nlog 8192)
  (define img  (malloc (add1 (bytes-length image)) 'atomic-interior))
  (memcpy img image (bytes-length image))
  (ptr-set! img _byte (bytes-length image) 0)         ; NUL-terminate
  (define infolog (malloc nlog 'atomic-interior))
  (define errlog  (malloc nlog 'atomic-interior))
  (define opts (malloc _int32 4 'raw))
  (define vals (malloc _pointer 4 'raw))
  (dynamic-wind
   void
   (lambda ()
     (ptr-set! opts _int32 0 CU_JIT_INFO_LOG_BUFFER)
     (ptr-set! opts _int32 1 CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
     (ptr-set! opts _int32 2 CU_JIT_ERROR_LOG_BUFFER)
     (ptr-set! opts _int32 3 CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
     (ptr-set! vals _pointer 0 infolog)
     (ptr-set! vals _pointer 1 (cast nlog _intptr _pointer))
     (ptr-set! vals _pointer 2 errlog)
     (ptr-set! vals _pointer 3 (cast nlog _intptr _pointer))
     (define-values (r m) (cuModuleLoadDataEx img 4 opts vals))
     (unless (zero? r)
       (error 'load-ptx "CUDA error ~a loading PTX: ~a\nJIT error log:\n~a"
              r
              (with-handlers ([exn:fail? (lambda (_) "<no message>")])
                (cuGetErrorString r))
              (cstr errlog nlog)))
     (cuda-module m))
   ;; only opts/vals are 'raw; img/infolog/errlog are 'atomic-interior and are
   ;; GC-managed, so freeing them would corrupt the heap.
   (lambda () (free opts) (free vals))))

;; Duplicated deliberately rather than required from nvrtc.rkt: the driver must
;; be able to refuse float PTX even where NVRTC is absent.
(define ptx-float-rx
  #px"[.](f16x2|bf16x2|f16|bf16|f32|f64|tf32)\\b|%f[0-9]|%fd[0-9]")

(define (assert-no-float-image! image)
  (define txt (bytes->string/utf-8 image #\?))
  (define hits (regexp-match* ptx-float-rx txt))
  (unless (null? hits)
    (error 'load-ptx
           "refusing floating point: this PTX contains ~a float construct~a (~a). This package is exact only."
           (length hits) (if (= 1 (length hits)) "" "s")
           (car hits))))

(define (cstr p n)
  (define bs (make-bytes n))
  (memcpy bs p n)
  (define z (or (for/first ([i (in-range n)] #:when (zero? (bytes-ref bs i))) i) n))
  (bytes->string/utf-8 (subbytes bs 0 z) #\?))

(define (port->bytes* in)
  (let loop ([acc '()])
    (define b (read-bytes 65536 in))
    (if (eof-object? b)
        (apply bytes-append (reverse acc))
        (loop (cons b acc)))))

(define (unload-module! m)
  (check 'cuModuleUnload (cuModuleUnload (cuda-module-ptr m))))

(define (module-function m name)
  (define-values (r f) (cuModuleGetFunction (cuda-module-ptr m) name))
  (check 'cuModuleGetFunction r)
  (cuda-function f))

;; ------------------------------------------------------------------- launch

;; args are (cons tag value) with tag in '(u64 i32 u32); u64 covers device
;; pointers. Each value is boxed into its own cell and an array of pointers to
;; those cells is what cuLaunchKernel expects.
(define (launch! fn grid block args #:shared [shared 0])
  (define n (length args))
  (define cells '())
  (define arr (malloc _pointer (max n 1) 'raw))
  (define r
    (dynamic-wind
     void
     (lambda ()
       ;; built here, not before the wind, so a failure part-way through still
       ;; frees whatever was allocated
       (set! cells
             (for/list ([a (in-list args)])
               (define tag (car a))
               (define v (cdr a))
               (define ty (case tag [(u64) _uint64] [(i32) _int32] [(u32) _uint32]))
               (define c (malloc ty 'raw))
               (ptr-set! c ty v)
               c))
       (for ([c (in-list cells)] [i (in-naturals)])
         (ptr-set! arr _pointer i c))
       (cuLaunchKernel (cuda-function-ptr fn)
                       (car grid) (cadr grid) (caddr grid)
                       (car block) (cadr block) (caddr block)
                       shared #f arr #f))
     (lambda ()
       (for ([c (in-list cells)]) (free c))
       (free arr))))
  (check 'cuLaunchKernel r))
