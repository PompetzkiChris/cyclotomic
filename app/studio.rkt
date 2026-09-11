#lang racket/gui
;; MUB EXECUTIONER -- pick a dimension. It NAMES the families that build its
;; mutually unbiased bases and KILLS every pair (each overlap verified EXACTLY
;; 1/d) while pinning the whole machine to the wall and auto-optimizing the
;; rest: the RTX 5090 blazing on big exact Z[zeta_24] products with VRAM filled
;; to the free limit, every core of the 9950X3D burning exact integer
;; arithmetic, and results streamed to disk. No floating point anywhere.
;;
;; TO PAUSE: press Ctrl+Alt+Del (or click away). The run idles and RESUMES when
;; you return -- the core-burners are killed so the machine truly calms.

(require "../field.rkt" "../matrix.rkt" "../mub.rkt"
         "../exact-io.rkt" "../mub-general.rkt"
         "../cuda/driver.rkt" "../cuda/gpu.rkt" "../cuda/accel.rkt"
         (only-in math/number-theory factorize)
         racket/string racket/place)

(define gpu-ok?
  (with-handlers ([(lambda (_) #t) (lambda (e) #f)])
    (gpu-init!) (gpu-accel-install!) (and (current-gpu-ready?) #t)))

(define disk-file "C:\\ClaudeOutput\\math-results\\executioner_stream.bin")
(define log-file  "C:\\ClaudeOutput\\math-results\\executioner_kills.log")
(define hog-bytes (* 4096 4096 8 8))   ; one resident 4096^2 x 8-plane matrix = 1 GiB

;; Build each dimension's optimal MUB set ONCE and cache it. The matrices never
;; change, so a march that loops 2->16 forever must not reconstruct the
;; Galois-ring / finite-field sets every pass -- that was the bulk of the
;; per-pass allocation. 15 immutable sets held is bounded and tiny.
(define mub-cache (make-hasheqv))
(define (build-mubs d) (hash-ref! mub-cache d (lambda () (mubs-optimal d))))
(define (pp-str fs)
  (string-join (for/list ([f (in-list fs)])
                 (if (= 1 (cadr f)) (format "~a" (car f)) (format "~a^~a" (car f) (cadr f)))) " x "))
(define (families-of d)
  (define fs (factorize d))
  (if (= 1 (length fs))
      (list (format "d = ~a = ~a   PRIME POWER -> complete set of ~a MUBs (solved)"
                    d (pp-str fs) (optimal-count d)))
      (cons (format "d = ~a = ~a   composite -> ~a MUBs (tensor bound; a 4th+ is open)"
                    d (pp-str fs) (optimal-count d))
            (for/list ([f (in-list fs)])
              (define q (expt (car f) (cadr f)))
              (format "   family: prime power ~a -> its complete set of ~a" q (add1 q))))))

;; auto-optimize VRAM: as many resident matrices as fit, minus headroom for the
;; running product and driver overhead
(define (auto-nhog)
  (with-handlers ([(lambda (_) #t) (lambda (e) 3)])
    (define-values (free total) (cuda-mem-info))
    (max 1 (min 20 (- (quotient free hog-bytes) 2)))))

;; ---- palette --------------------------------------------------------------
(define col-bg (make-object color% 18 19 24))
(define col-pend (make-object color% 90 94 104))
(define col-kill (make-object color% 220 60 60))
(define col-live (make-object color% 46 200 110))
(define col-warn (make-object color% 240 180 70))
(define f-mono (make-object font% 13 'modern 'normal 'normal))
(define f-huge (make-object font% 24 'modern 'normal 'bold))
(define f-lbl  (make-object font% 13 'default 'normal 'bold))

;; ---- state (declared before the frame so on-activate can see it) ----------
(define running #f)
(define paused #f)
(define burners '())
(define hogs '())
(define kills-done 0)
(define kt-sum 0)
(define bytes-written 0)
(define (mb n) (quotient n 1048576))

;; ---- window (pauses on deactivate) ----------------------------------------
(define exec-frame%
  (class frame%
    (super-new)
    (define/override (on-subwindow-char r e)
      (define k (send e get-key-code))
      (cond [(and running (memv k (list #\p #\P #\space)))
             (set-paused! (not paused)) #t]
            [else (super on-subwindow-char r e)]))))
(define frame (new exec-frame% [label "MUB EXECUTIONER -- RTX 5090 + 9950X3D"] [width 1080] [height 840]))
(define root (new vertical-panel% [parent frame] [spacing 4] [border 10] [alignment '(left top)]))
(new message% [parent root] [font f-lbl]
     [label "MUB EXECUTIONER    name the families, kill every pair, pin the machine to the wall    (exact, no float)"])
(new message% [parent root] [font f-lbl] [auto-resize #t]
     [label "TO PAUSE: press  P  (toggles).   TO STOP: Task Manager -> End task (kills the app and every core-burner)."])

(define ctl (new horizontal-panel% [parent root] [stretchable-height #f] [spacing 10] [alignment '(left center)]))
(new message% [parent ctl] [label "Dimension:"])
(define dim-choice (new choice% [parent ctl] [label ""] [choices (for/list ([d (in-range 2 17)]) (number->string d))] [selection 0]))
(define exec-btn
  (new button% [parent ctl] [label "EXECUTE"] [enabled gpu-ok?]
       [callback (lambda (b e)
                   (unless running
                     (send exec-btn enable #f)
                     (send exec-btn set-label "RUNNING -- press P to pause")
                     (start-execute (string->number (send dim-choice get-string-selection)))))]))

(define fam (new text-field% [parent root] [label ""] [style '(multiple)] [min-height 78] [enabled #f]))

;; ---- kill list ------------------------------------------------------------
(define targets '())
(define d-now 12)
(define (draw-kills c dc)
  (send dc set-background col-bg) (send dc clear)
  (send dc set-font f-mono)
  (send dc set-text-foreground (if paused col-warn col-live))
  (send dc draw-text (format "FAMILY  d = ~a~a" d-now (if paused "     [PAUSED]" "")) 14 6)
  (define y 30)
  (for ([t (in-list targets)])
    (define st (vector-ref t 3))
    (send dc set-text-foreground (case st [(pending) col-pend] [(live) col-live] [(killed) col-kill]))
    (send dc draw-text
          (format "~a  ~a~a"
                  (case st [(pending) "  ...  "] [(live) " >>>>> "] [(killed) "KILLED "])
                  (vector-ref t 0)
                  (if (eq? st 'killed) (format "   |<.,.>|^2 = ~a  [exact]" (vector-ref t 4)) ""))
          14 y)
    (set! y (+ y 22))))
(define kill-canvas (new canvas% [parent root] [min-height 210] [paint-callback draw-kills]))

(define score (new message% [parent root] [font f-huge] [auto-resize #t] [label "0 killed"]))
(define eta   (new message% [parent root] [font f-lbl] [auto-resize #t] [label "family:  -"]))
(define meter (new message% [parent root] [font f-lbl] [auto-resize #t] [label "GPU: -   CPU: -   RAM: -   DISK: -"]))

(define (mmss ms) (let ([s (quotient (max 0 ms) 1000)])
  (format "~a:~a~a" (quotient s 60) (if (< (modulo s 60) 10) "0" "") (modulo s 60))))

(define (targets-for d)
  (define set (build-mubs d))
  (define nb (length set))
  (for*/list ([i (in-range nb)] [j (in-range (add1 i) nb)])
    (vector (format "~a  x  ~a" (car (list-ref set i)) (car (list-ref set j))) i j 'pending #f)))

;; ---- pause / resume -------------------------------------------------------
(define (spawn-burner)
  (place ch (let loop ([a (expt 7 30000)])
              (loop (modulo (* a a) (add1 (expt 2 120000)))))))
(define (set-paused! p)
  (cond
    [(and p (not paused))
     (set! paused #t)
     (let ([old burners]) (set! burners '())
       (thread (lambda () (for ([b (in-list old)]) (with-handlers ([(lambda (_) #t) void]) (place-kill b))))))]
    [(and (not p) paused running)
     (set! paused #f)
     (thread (lambda () (set! burners (for/list ([_ (in-range (processor-count))]) (spawn-burner)))))])
  (send kill-canvas refresh))
(define (wait-while-paused) (let w () (when (and running paused) (sleep 0.1) (w))))

;; ---- execute --------------------------------------------------------------
(define (start-execute d)
  (set! running #t) (set! paused #f) (set! kills-done 0) (set! kt-sum 0) (set! bytes-written 0)
  (set! d-now d) (set! targets (targets-for d))
  (define ALL (list 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16))
  (define start-idx (or (index-of ALL d) 0))
  (define march (append (list-tail ALL start-idx) (take ALL start-idx)))
  (define nhog (auto-nhog))
  (send fam set-value
        (string-append
         (apply string-append (map (lambda (s) (string-append s "\n")) (families-of d)))
         (format "\ntensor bases:  ~a\n" (map car (build-mubs d)))
         (format "targets: ~a pairs, each |<Bi,Bj>|^2 must be exactly 1/~a\n" (length targets) d)
         (format "auto-optimized: ~a resident 4096^2 matrices in VRAM (~~~a GB), ~a CPU cores burning\n"
                 nhog (quotient (* nhog hog-bytes) 1073741824) (processor-count))))
  (send kill-canvas refresh)
  (thread (lambda ()
    (define F (make-field 24))
    (define blaze (zmat-build F 4096 4096 (lambda (i j t) (- (random 64) 32))))
    (define bd (zmat->dmat blaze))
    (set! hogs (cons bd (filter values
                                (for/list ([_ (in-range (sub1 nhog))])
                                  (with-handlers ([(lambda (_) #t) (lambda (e) #f)]) (zmat->dmat blaze))))))
    (set! burners (for/list ([_ (in-range (processor-count))]) (spawn-burner)))
    ;; GPU + disk, serial on the one CUDA context. The snapshot copies the
    ;; device result into ONE preallocated host buffer -- not a fresh 1 GB
    ;; byte string each time -- so committed memory stays flat, and a periodic
    ;; major GC reclaims the per-pass matrices the march builds.
    (define snap-buf (make-bytes (* (zmat-rows blaze) (zmat-cols blaze) (field-degree F) 8)))
    (thread (lambda ()
      (let loop ([i 0])
        (when running
          (cond
            [paused (sleep 0.1) (loop i)]
            [else
             (with-handlers ([(lambda (_) #t) (lambda (e) (void))])
               (dmat-free! (dmat* bd bd #:audit? #f)))
             (when (and running (not paused) (zero? (modulo i 20)))
               (with-handlers ([(lambda (_) #t) (lambda (e) (void))])
                 (copy-from-device! snap-buf (dmat-ptr bd))     ; reuse one buffer
                 (call-with-output-file disk-file #:exists 'replace (lambda (o) (write-bytes snap-buf o)))
                 (set! bytes-written (+ bytes-written (bytes-length snap-buf))))
               (collect-garbage 'incremental))
             (loop (add1 i))])))))
    ;; MARCH UPWARD, unbounded: d, d+1, d+2, ...  computing the OPTIMAL known MUB
    ;; set for each dimension and killing every pair. No loop back, no stop at 16
    ;; -- it climbs forever until you pause (P) or End-task it.
    (let mdim ([dd d])
      (when running
        (with-handlers
          ([(lambda (_) #t)
            (lambda (e)
              (queue-callback (lambda ()
                (send eta set-label (format "d=~a: construction unavailable, skipped   ·   ~a exact so far" dd kills-done)))))])
          (define set (build-mubs dd))                 ; mubs-optimal dd, built once
          (define names (map car set))
          (define mats  (map cdr set))
          (define nb (length set))
          (define tg (for*/list ([i (in-range nb)] [j (in-range (add1 i) nb)])
                       (vector (format "~a  x  ~a" (list-ref names i) (list-ref names j)) i j 'pending #f)))
          (queue-callback (lambda ()
            (set! d-now dd) (set! targets tg)
            (send fam set-value
                  (string-append
                   (apply string-append (map (lambda (s) (string-append s "\n")) (families-of dd)))
                   (format "\ntargets: ~a pairs, each |<Bi,Bj>|^2 must be exactly 1/~a\n" (length tg) dd)))
            (send kill-canvas refresh)))
          (for ([t (in-list tg)] [k (in-naturals)] #:break (not running))
            (wait-while-paused)
            (when running
              (queue-callback (lambda () (vector-set! t 3 'live) (send kill-canvas refresh)))
              (define kt0 (now-ms))
              (define v (parameterize ([current-mat*-hook #f])
                          (unbiasedness (list-ref mats (vector-ref t 1)) (list-ref mats (vector-ref t 2)))))
              (sleep 0.5)
              (set! kills-done (add1 kills-done)) (set! kt-sum (+ kt-sum (- (now-ms) kt0)))
              (with-handlers ([(lambda (_) #t) void])
                (call-with-output-file log-file #:exists 'append
                  (lambda (o) (fprintf o "d=~a  ~a  |<.,.>|^2=~a  ~a\n"
                                       dd (vector-ref t 0) v (if (equal? v (/ 1 dd)) "OK" "FAIL")))))
              (define avg (quotient kt-sum kills-done))
              (define left (- (length tg) (add1 k)))
              (queue-callback (lambda ()
                (vector-set! t 3 'killed)
                (vector-set! t 4 (if (equal? v (/ 1 dd)) (format "1/~a" dd) (format "~a NOT 1/~a" v dd)))
                (send score set-label (format "~a killed" kills-done))
                (send eta set-label (format "MARCH  d=~a:  ~a left   ETA ~a   ·   ~a exact, climbing (P=pause)"
                                            dd left (mmss (* left avg)) kills-done))
                (send kill-canvas refresh)))))
          ;; reclaim this family before the next dimension -> flat memory
          (collect-garbage))
        (mdim (add1 dd)))))))

;; ---- meters ---------------------------------------------------------------
(define (sh cmd) (with-handlers ([(lambda (_) #t) (lambda (e) "")])
                   (string-trim (with-output-to-string (lambda () (system cmd))))))
(define (num s) (let ([m (regexp-match* #px"[0-9]+" s)]) (if (pair? m) (car m) "?")))
(define (poll)
  (define g (sh "nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv,noheader"))
  (define c (num (sh "powershell -NoProfile -Command \"(Get-CimInstance Win32_Processor).LoadPercentage\"")))
  (define r (num (sh "powershell -NoProfile -Command \"$m=Get-CimInstance Win32_OperatingSystem; [int](($m.TotalVisibleMemorySize-$m.FreePhysicalMemory)/1048576)\"")))
  (send meter set-label (format "GPU: ~a   |   CPU: ~a %   RAM: ~a GB   DISK: ~a MB~a"
                                g c r (mb bytes-written) (if paused "   [PAUSED]" ""))))
(when gpu-ok? (new timer% [interval 900] [notify-callback poll]))

(send frame show #t)
(send fam set-value "starting the upward march automatically...\n")

;; Auto-start the march on launch (climbs from d=2 upward, unbounded). The
;; EXECUTE button still works, but nothing has to be clicked.
(when gpu-ok?
  (queue-callback
   (lambda ()
     (unless running
       (send exec-btn enable #f)
       (send exec-btn set-label "RUNNING -- press P to pause")
       (start-execute 2)))))
