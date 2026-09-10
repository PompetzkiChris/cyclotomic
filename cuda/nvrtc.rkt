#lang racket/base
;; NVRTC: compile CUDA C++ from a Racket string, at run time, to PTX.
;;
;; Without this the package can only run kernels I shipped. With it, a Racket
;; program can write a kernel, compile it for the device actually present, and
;; launch it -- no nvcc, no build step, no host compiler.
;;
;;   (define ptx (compile-cuda #<<EOF
;;   extern "C" __global__ void scale(long long* v, long long s, long long n) {
;;       long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
;;       if (i < n) v[i] *= s;
;;   }
;;   EOF
;;                 #:arch "compute_120"))
;;   (define mod (load-ptx ptx))
;;
;; A compile failure raises with NVRTC's own log, so the error names the line
;; and the reason rather than just failing.

(require ffi/unsafe
         ffi/unsafe/define
         racket/list
         racket/string)

(provide nvrtc-available?
         nvrtc-version
         compile-cuda
         compile-cuda/log
         device-arch-string
         exn:fail:cuda-compile?
         exn:fail:cuda-compile-log
         exn:fail:cuda-float?
         exn:fail:cuda-float-hits
         ptx-float-hits
         assert-no-float!
         source-float-hits
         strip-noncode
         preprocessor-lines)


;; Built with build-path rather than written as a string literal, so no one has
;; to think about backslash escaping.
(define default-cuda-root
  (let ([p (build-path "C:" "Program Files" "NVIDIA GPU Computing Toolkit"
                       "CUDA" "v13.4")])
    (and (directory-exists? p) (path->string p))))

;; NVRTC's file name carries a major version that does not track the toolkit
;; version, and on Windows it lives in the toolkit's bin directory rather than
;; anywhere the loader searches by default. So: look for it, in the places it
;; actually is, instead of hoping PATH is right.
(define (nvrtc-candidates)
  (define roots
    (filter values
            (list (getenv "CUDA_PATH")
                  (getenv "CUDA_HOME")
                  default-cuda-root)))
  (define dirs
    (append*
     (for/list ([r (in-list roots)])
       (list (build-path r "bin" "x64") (build-path r "bin")))))
  (append
   ;; whatever the loader can already see
   (list "nvrtc64_130_0" "nvrtc64_120_0" "nvrtc64_112_0" "nvrtc")
   ;; then every nvrtc64_*.dll under a toolkit bin, newest name last
   (for*/list ([d (in-list dirs)]
               #:when (directory-exists? d)
               [f (in-list (sort (map path->string (directory-list d)) string<?))]
               #:when (regexp-match? #rx"^nvrtc64_.*[.]dll$" f)
               #:unless (regexp-match? #rx"builtins" f))
     (path->string (build-path d f)))))


;; nvrtc64_*.dll loads nvrtc-builtins64_*.dll from beside itself, and the loader
;; will not find that if we opened nvrtc by absolute path from a directory it
;; does not search. Put the toolkit bin directories on PATH first.
(define (add-toolkit-dirs-to-path!)
  (define roots
    (filter values (list (getenv "CUDA_PATH")
                         (getenv "CUDA_HOME")
                         default-cuda-root)))
  (define dirs
    (for*/list ([r (in-list roots)]
                [sub (in-list (list (build-path r "bin" "x64") (build-path r "bin")))]
                #:when (directory-exists? sub))
      (path->string sub)))
  (unless (null? dirs)
    (define cur (or (getenv "PATH") ""))
    (define missing (filter (lambda (d) (not (regexp-match? (regexp (regexp-quote d)) cur))) dirs))
    (unless (null? missing)
      (putenv "PATH" (string-append (string-join missing ";") ";" cur)))))

(define nvrtc-lib
  (begin
    (add-toolkit-dirs-to-path!)
    (for/or ([c (in-list (nvrtc-candidates))])
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (ffi-lib c)))))

(define (nvrtc-available?) (and nvrtc-lib #t))

(define-ffi-definer define-nv (or nvrtc-lib (ffi-lib #f))
  #:default-make-fail make-not-available)

(define _nvrtcResult _int32)
(define _nvrtcProgram (_cpointer/null 'nvrtcProgram))

(define-nv nvrtcGetErrorString (_fun _nvrtcResult -> _string/utf-8))
(define-nv nvrtcVersion
  (_fun (maj : (_ptr o _int32)) (min : (_ptr o _int32)) -> (r : _nvrtcResult)
        -> (values r maj min)))
(define-nv nvrtcCreateProgram
  (_fun (out : (_ptr o _nvrtcProgram))
        _string/utf-8 _string/utf-8 _int32 _pointer _pointer
        -> (r : _nvrtcResult) -> (values r out)))
(define-nv nvrtcDestroyProgram
  (_fun (p : (_ptr io _nvrtcProgram)) -> (r : _nvrtcResult) -> r))
(define-nv nvrtcCompileProgram
  (_fun #:blocking? #t _nvrtcProgram _int32 _pointer -> _nvrtcResult))
(define-nv nvrtcGetProgramLogSize
  (_fun _nvrtcProgram (n : (_ptr o _size)) -> (r : _nvrtcResult) -> (values r n)))
(define-nv nvrtcGetProgramLog
  (_fun _nvrtcProgram _pointer -> _nvrtcResult))
(define-nv nvrtcGetPTXSize
  (_fun _nvrtcProgram (n : (_ptr o _size)) -> (r : _nvrtcResult) -> (values r n)))
(define-nv nvrtcGetPTX
  (_fun _nvrtcProgram _pointer -> _nvrtcResult))

;; A compile error carries the log, so it can be inspected rather than parsed
;; back out of a message.
(struct exn:fail:cuda-compile exn:fail (log) #:transparent)

;; ---------------------------------------------------------------------------
;; No floating point, structurally.
;;
;; NVRTC will happily compile whatever CUDA C++ it is handed, floats included,
;; so "this package does not use floating point" would stop being a property of
;; the package and start being a property of what the caller happened to write.
;;
;; The check is on the ARTIFACT, not the intent: after compiling, the generated
;; PTX is scanned for float types and float registers, and if any appear the
;; PTX is not returned. Macros, typedefs, templates, __half, intrinsics,
;; literals -- none of it matters, because whatever the source did, the compiled
;; code either contains a float instruction or it does not.
;;
;; There is deliberately no escape hatch.
;; ---------------------------------------------------------------------------
(struct exn:fail:cuda-float exn:fail (hits) #:transparent)

;; PTX spells every float type and register this way; erring toward refusal.
(define float-rx
  #px"[.](f16x2|bf16x2|f16|bf16|f32|f64|tf32)\\b|%f[0-9]|%fd[0-9]|\\b0[dDfF][0-9A-Fa-f]{8}")


;; The source layer, airtight rather than advisory.
;;
;; The PTX check alone guarantees only that no float INSTRUCTION executes. It
;; does not guarantee no float was ever involved, because NVRTC constant-folds
;; a float expression whose operands are compile-time known and emits an
;; integer. For the stronger claim, no float may reach the compiler at all.
;;
;; Two things could hide one from a scan of the text:
;;
;;   The preprocessor. A #define could rename float to anything and an #include
;;   could drag in a header full of them, so preprocessor directives are
;;   refused outright. The source handed in IS the translation unit that gets
;;   compiled. NVRTC has no standard headers to include anyway.
;;
;;   Spellings that do not literally say "float". Every float-producing name in
;;   CUDA carries one of a small set of substrings -- including the conversion
;;   intrinsics that manufacture a float out of an integer bit pattern, such as
;;   __int_as_float, __uint_as_float and __int2half_rn. Those substrings are
;;   rejected case-insensitively, along with every way of writing a float
;;   literal, hex-float notation included.
;;
;; Comments and string literals are stripped first so prose cannot trip it.
;;
;; A float therefore cannot be named, written as a literal, manufactured from
;; an integer, or smuggled in behind a macro. With the PTX check downstream, no
;; float is involved at any stage.

;; Remove comments and string/char literals so their contents are not scanned.
(define (strip-noncode src)
  (define n (string-length src))
  (define out (open-output-string))
  (define (at i) (and (< i n) (string-ref src i)))
  (let loop ([i 0] [mode 'code])
    (cond
      [(>= i n) (void)]
      [(eq? mode 'code)
       (cond
         [(and (eqv? (at i) #\/) (eqv? (at (add1 i)) #\*)) (loop (+ i 2) 'block)]
         [(and (eqv? (at i) #\/) (eqv? (at (add1 i)) #\/)) (loop (+ i 2) 'line)]
         [(eqv? (at i) #\") (write-char #\space out) (loop (add1 i) 'str)]
         [(eqv? (at i) #\') (write-char #\space out) (loop (add1 i) 'chr)]
         [else (write-char (at i) out) (loop (add1 i) 'code)])]
      [(eq? mode 'block)
       (cond
         [(and (eqv? (at i) #\*) (eqv? (at (add1 i)) #\/))
          (write-char #\space out)
          (loop (+ i 2) 'code)]
         [else
          (when (eqv? (at i) #\newline) (write-char #\newline out))
          (loop (add1 i) 'block)])]
      [(eq? mode 'line)
       (cond
         [(eqv? (at i) #\newline) (write-char #\newline out) (loop (add1 i) 'code)]
         [else (loop (add1 i) 'line)])]
      [(eq? mode 'str)
       (cond
         [(eqv? (at i) #\\) (loop (+ i 2) 'str)]
         [(eqv? (at i) #\") (loop (add1 i) 'code)]
         [else (loop (add1 i) 'str)])]
      [else
       (cond
         [(eqv? (at i) #\\) (loop (+ i 2) 'chr)]
         [(eqv? (at i) #\') (loop (add1 i) 'code)]
         [else (loop (add1 i) 'chr)])]))
  (get-output-string out))

(define float-word-rx #px"(?i:float|double|half|bfloat|fp16|fp8|tf32|_Float|__fp16)")

(define float-literal-rx
  #px"[0-9]+[.][0-9]*|[.][0-9]+|[0-9]+[eE][-+]?[0-9]+|0[xX][0-9A-Fa-f]*[pP][-+]?[0-9]+")

(define (preprocessor-lines src)
  (for/list ([line (in-list (regexp-split #rx"\n" src))]
             [i (in-naturals 1)]
             #:when (regexp-match? #px"^[ \t]*#" line))
    (cons i line)))

(define (source-float-hits src)
  (define code (strip-noncode src))
  (append (regexp-match* float-word-rx code)
          (regexp-match* float-literal-rx code)))

(define (ptx-float-hits ptx)
  (define txt (if (bytes? ptx) (bytes->string/utf-8 ptx #\?) ptx))
  (map (lambda (m) (if (bytes? m) (bytes->string/utf-8 m #\?) m))
       (regexp-match* float-rx txt)))

;; Raise unless the image is free of floating point. Used by compile-cuda and
;; by load-ptx, so neither door is open.
(define (assert-no-float! ptx #:who [who 'assert-no-float!])
  (define hits (ptx-float-hits ptx))
  (unless (null? hits)
    (define tally (make-hash))
    (for ([h (in-list hits)]) (hash-update! tally h add1 0))
    (raise (exn:fail:cuda-float
            (format "~a: refusing floating point. The compiled PTX contains ~a float construct~a: ~a. This package is exact only; there is no option to permit this."
                    who (length hits) (if (= 1 (length hits)) "" "s")
                    (string-join
                     (for/list ([(k v) (in-hash tally)]) (format "~a x~a" k v))
                     ", "))
            (current-continuation-marks)
            hits)))
  (void))

(define (check who r)
  (unless (zero? r)
    (error who "NVRTC error ~a: ~a" r
           (with-handlers ([exn:fail? (lambda (_) "<no message>")])
             (nvrtcGetErrorString r))))
  (void))

(define (nvrtc-version)
  (unless nvrtc-lib (error 'nvrtc-version "NVRTC not found"))
  (define-values (r maj min) (nvrtcVersion))
  (check 'nvrtcVersion r)
  (values maj min))

;; "compute_120" for an sm_120 device. Virtual arch, so the driver JITs the
;; final SASS for whatever is actually installed.
(define (device-arch-string major minor)
  (format "compute_~a~a" major minor))

(define (cstr-array strs)
  ;; NULL-free array of char* that stays alive for the call
  (define ptrs (for/list ([s (in-list strs)])
                 (define b (string->bytes/utf-8 s))
                 (define p (malloc (add1 (bytes-length b)) 'atomic-interior))
                 (memcpy p b (bytes-length b))
                 (ptr-set! p _byte (bytes-length b) 0)
                 p))
  (define arr (malloc _pointer (max 1 (length ptrs)) 'atomic-interior))
  (for ([p (in-list ptrs)] [i (in-naturals)])
    (ptr-set! arr _pointer i p))
  (values arr ptrs))

;; Compile and return (values ptx-bytes log-string). Raises on failure with
;; the log attached.
(define (compile-cuda/log src
                          #:name [name "kernel.cu"]
                          #:arch [arch "compute_120"]
                          #:options [extra '()])
  (unless nvrtc-lib (error 'compile-cuda "NVRTC not found; is the CUDA toolkit installed?"))
  ;; The preprocessor could hide a float behind a name, so it is refused.
  (let ([pp (preprocessor-lines src)])
    (unless (null? pp)
      (raise (exn:fail:cuda-float
              (format "compile-cuda: preprocessor directives are not allowed. A #define could rename float and an #include could bring in a header full of them, either of which would put a float past the source check. Line ~a: ~a"
                      (car (car pp)) (string-trim (cdr (car pp))))
              (current-continuation-marks)
              (map cdr pp)))))
  ;; reject on sight before compiling, so a float expression cannot be quietly
  ;; folded into an acceptable-looking integer
  (let ([shits (source-float-hits src)])
    (unless (null? shits)
      (raise (exn:fail:cuda-float
              (format "compile-cuda: refusing floating point. The source mentions ~a: ~a. This package is exact only; there is no option to permit this."
                      (if (= 1 (length shits)) "a float construct" "float constructs")
                      (string-join (remove-duplicates shits) ", "))
              (current-continuation-marks)
              shits))))
  (define-values (r prog) (nvrtcCreateProgram src name 0 #f #f))
  (check 'nvrtcCreateProgram r)
  (define opts (append (list (string-append "--gpu-architecture=" arch)
                             "--std=c++17")
                       extra))
  (define-values (optarr keep) (cstr-array opts))
  (define cres (nvrtcCompileProgram prog (length opts) optarr))
  ;; the log is worth having even on success: it carries warnings
  (define-values (lr logn) (nvrtcGetProgramLogSize prog))
  (check 'nvrtcGetProgramLogSize lr)
  (define logbuf (malloc (max 1 logn) 'atomic-interior))
  (check 'nvrtcGetProgramLog (nvrtcGetProgramLog prog logbuf))
  (define log
    (let ([bs (make-bytes (max 1 logn))])
      (memcpy bs logbuf (max 1 logn))
      (define z (or (for/first ([i (in-range (bytes-length bs))]
                                #:when (zero? (bytes-ref bs i))) i)
                    (bytes-length bs)))
      (bytes->string/utf-8 (subbytes bs 0 z) #\?)))
  (cond
    [(not (zero? cres))
     (define msg
       (format "compile-cuda: NVRTC failed (~a: ~a)\n~a"
               cres
               (with-handlers ([exn:fail? (lambda (_) "?")]) (nvrtcGetErrorString cres))
               log))
     (nvrtcDestroyProgram prog)
     (raise (exn:fail:cuda-compile msg (current-continuation-marks) log))]
    [else
     (define-values (pr ptxn) (nvrtcGetPTXSize prog))
     (check 'nvrtcGetPTXSize pr)
     (define buf (malloc ptxn 'atomic-interior))
     (check 'nvrtcGetPTX (nvrtcGetPTX prog buf))
     (define out (make-bytes ptxn))
     (memcpy out buf ptxn)
     (nvrtcDestroyProgram prog)
     ;; NVRTC returns a NUL-terminated image; trim so load-ptx sees clean bytes
     (define z (or (for/first ([i (in-range (bytes-length out))]
                               #:when (zero? (bytes-ref out i))) i)
                   (bytes-length out)))
     (define ptx (subbytes out 0 z))
     ;; the artifact decides, not the source
     (assert-no-float! ptx #:who 'compile-cuda)
     (values ptx log)]))

(define (compile-cuda src
                      #:name [name "kernel.cu"]
                      #:arch [arch "compute_120"]
                      #:options [extra '()])
  (define-values (ptx log) (compile-cuda/log src #:name name #:arch arch #:options extra))
  ptx)
