#lang racket/base
;; Exact timing and exact number formatting.
;;
;; The mathematics in this package never touched a float, but the timing and
;; the reporting did: current-inexact-milliseconds returns a flonum, and every
;; "x.y GB" and "n Giga-ops/s" was computed in floating point. That makes the
;; claim "no floating point" true only of the parts one happens to care about,
;; which is not what the claim says.
;;
;; So: clocks read exact integers, ratios stay exact rationals, and decimals
;; are rendered by integer arithmetic. No flonum is constructed anywhere.

(require racket/contract)

(provide
 (contract-out
  [now-ms      (-> exact-integer?)]
  [elapsed-ms  (-> exact-integer? exact-integer?)]
  [dec         (->* (rational?) (exact-nonnegative-integer?) string?)]
  [ratio       (-> rational? rational? string?)]
  [bytes->gb   (-> exact-nonnegative-integer? string?)]
  [bytes->mb   (-> exact-nonnegative-integer? string?)]))

;; Exact integer milliseconds. current-milliseconds is exact; its inexact
;; cousin is the one that hands back a flonum.
(define (now-ms) (current-milliseconds))
(define (elapsed-ms t0) (- (current-milliseconds) t0))

;; Render an exact rational with `places` decimals, by integer arithmetic only.
;; No division to a flonum, no real->decimal-string.
(define (dec q [places 1])
  (unless (exact? q)
    (raise-argument-error 'dec "exact rational (this package makes no flonums)" q))
  (define scale (expt 10 places))
  (define neg? (negative? q))
  (define scaled (round (* (abs q) scale)))     ; exact
  (define whole (quotient scaled scale))
  (define frac (remainder scaled scale))
  (define fs (number->string frac))
  (define padded (string-append (make-string (max 0 (- places (string-length fs))) #\0) fs))
  (string-append (if neg? "-" "")
                 (number->string whole)
                 (if (zero? places) "" (string-append "." padded))))

;; a/b as a decimal string, exactly; b = 0 renders as "-"
(define (ratio a b)
  (if (zero? b) "-" (dec (/ a b) 1)))

(define (bytes->gb b) (dec (/ b (expt 2 30)) 1))
(define (bytes->mb b) (dec (/ b (expt 2 20)) 1))
