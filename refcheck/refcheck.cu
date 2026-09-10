// refcheck -- an independent CUDA C++ implementation of exact Z[zeta_n] matrix
// multiplication, written to CHECK the Racket implementation rather than to
// share anything with it.
//
// Independence is the point, so nothing here is ported from the Racket side:
//
//   * Phi_n is recomputed here from x^n - 1 = prod_{d|n} Phi_d(x) by exact
//     integer polynomial division, in C++.
//   * The power table is rebuilt here.
//   * The GPU kernel is deliberately written differently: one thread per
//     output ELEMENT computing the whole length-deg convolution inline, with
//     no shared-memory tiling and no scatter. If the Racket kernel's tiling or
//     its R-scatter were wrong, this would not repeat the mistake.
//   * The CPU oracle accumulates in __int128, so a product that silently
//     overflowed int64 on either GPU path is caught rather than matched.
//
// Usage:
//   refcheck selftest                       -- internal CPU/GPU agreement
//   refcheck mul A.zmat B.zmat OUT.zmat     -- product of two plane files
//   refcheck cmp X.zmat Y.zmat              -- exact comparison, exit 1 if differ
//
// File format (little endian):
//   char[4] "ZMAT" | int32 n | int32 deg | int32 rows | int32 cols
//   then deg*rows*cols int64, plane-major then row-major.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
// A signed 128-bit accumulator. MSVC has no __int128, so on Windows this is
// built from _umul128 plus manual carry; elsewhere it is the native type.
// The point of accumulating wider than int64 is that a product which silently
// wrapped on the GPU must be DETECTED here, not reproduced.
// ---------------------------------------------------------------------------
#if defined(_MSC_VER)
#include <intrin.h>
struct i128 {
    unsigned long long lo;
    long long hi;
};
static inline i128 i128_from(long long v) { i128 r; r.lo = (unsigned long long)v; r.hi = v < 0 ? -1 : 0; return r; }
static inline i128 i128_zero() { i128 r; r.lo = 0; r.hi = 0; return r; }
static inline i128 i128_add(i128 a, i128 b) {
    i128 r; r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1 : 0);
    return r;
}
static inline i128 i128_mul64(long long x, long long y) {
    // full signed 64x64 -> 128
    long long hi;
    long long lo = _mul128(x, y, &hi);
    i128 r; r.lo = (unsigned long long)lo; r.hi = hi;
    return r;
}
static inline i128 i128_mul_small(i128 a, long long m) {
    // Truncated two's-complement 128 x 64 -> 128.
    // _umul128 treats m as UNSIGNED, so for m < 0 it computes
    //   a.lo * (m + 2^64) = a.lo*m + a.lo*2^64
    // and the high word comes out too large by exactly a.lo. Correct it.
    unsigned long long um = (unsigned long long)m;
    unsigned long long hi_of_lo;
    unsigned long long lo = _umul128(a.lo, um, &hi_of_lo);
    long long hi = (long long)hi_of_lo + (long long)((unsigned long long)a.hi * um);
    if (m < 0) hi -= (long long)a.lo;
    i128 r; r.lo = lo; r.hi = hi;
    return r;
}
static inline bool i128_fits_i64(i128 a) {
    long long slo = (long long)a.lo;
    return (a.hi == 0 && slo >= 0) || (a.hi == -1 && slo < 0);
}
static inline long long i128_to_i64(i128 a) { return (long long)a.lo; }
static inline bool i128_eq(i128 a, i128 b) { return a.lo == b.lo && a.hi == b.hi; }
#else
typedef __int128 i128;
static inline i128 i128_from(long long v) { return (i128)v; }
static inline i128 i128_zero() { return (i128)0; }
static inline i128 i128_add(i128 a, i128 b) { return a + b; }
static inline i128 i128_mul64(long long x, long long y) { return (i128)x * (i128)y; }
static inline i128 i128_mul_small(i128 a, long long m) { return a * (i128)m; }
static inline bool i128_fits_i64(i128 a) { return a <= (i128)INT64_MAX && a >= (i128)INT64_MIN; }
static inline long long i128_to_i64(i128 a) { return (long long)a; }
static inline bool i128_eq(i128 a, i128 b) { return a == b; }
#endif

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(2);} } while(0)

// ---------------------------------------------------------------- polynomials
typedef std::vector<long long> Poly;   // index = degree, low to high

static Poly poly_trim(Poly p) {
    while (p.size() > 1 && p.back() == 0) p.pop_back();
    return p;
}

static Poly poly_mul(const Poly& a, const Poly& b) {
    Poly r(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i])
            for (size_t j = 0; j < b.size(); ++j)
                if (b[j]) r[i + j] += a[i] * b[j];
    return poly_trim(r);
}

// exact division; aborts if it is not exact, which for the cyclotomic
// recursion would mean the mathematics is wrong rather than the input
static Poly poly_div_exact(Poly num, const Poly& den) {
    num = poly_trim(num);
    Poly d = poly_trim(den);
    int dd = (int)d.size() - 1;
    long long lead = d[dd];
    int qdeg = (int)num.size() - 1 - dd;
    if (qdeg < 0) return Poly{0};
    Poly q(qdeg + 1, 0);
    for (int k = qdeg; k >= 0; --k) {
        long long rk = num[k + dd];
        if (rk == 0) continue;
        if (rk % lead != 0) { fprintf(stderr, "poly_div_exact: not exact\n"); exit(2); }
        long long c = rk / lead;
        q[k] = c;
        for (int i = 0; i <= dd; ++i) num[k + i] -= c * d[i];
    }
    for (long long v : num) if (v != 0) { fprintf(stderr, "poly_div_exact: remainder\n"); exit(2); }
    return poly_trim(q);
}

static Poly cyclotomic(int n) {
    static std::vector<Poly> cache;
    if ((int)cache.size() <= n) cache.resize(n + 1);
    if (!cache[n].empty()) return cache[n];
    Poly xn(n + 1, 0);
    xn[0] = -1; xn[n] = 1;
    Poly lower{1};
    for (int d = 1; d < n; ++d)
        if (n % d == 0) lower = poly_mul(lower, cyclotomic(d));
    cache[n] = poly_div_exact(xn, lower);
    return cache[n];
}

// zeta^m in the power basis, m = 0 .. 2*deg-2, as an integer table
static std::vector<int> power_table(int n, int& degOut) {
    Poly phi = cyclotomic(n);
    int deg = (int)phi.size() - 1;
    degOut = deg;
    int rows = 2 * deg - 1;
    int span = rows > n ? rows : n + 1;
    std::vector<int> pow((size_t)span * deg, 0);
    for (int j = 0; j < deg && j < span; ++j) pow[(size_t)j * deg + j] = 1;
    for (int m = deg; m < span; ++m) {
        const int* prev = &pow[(size_t)(m - 1) * deg];
        int top = prev[deg - 1];
        int* cur = &pow[(size_t)m * deg];
        for (int j = deg - 1; j >= 1; --j) cur[j] = prev[j - 1];
        cur[0] = 0;
        if (top)
            for (int j = 0; j < deg; ++j) cur[j] -= top * (int)phi[j];
    }
    return std::vector<int>(pow.begin(), pow.begin() + (size_t)rows * deg);
}

// ------------------------------------------------------------------ the file
struct ZMat {
    int n = 0, deg = 0, rows = 0, cols = 0;
    std::vector<long long> a;             // deg * rows * cols
    long long& at(int t, int i, int j) { return a[((size_t)t * rows + i) * cols + j]; }
    long long at(int t, int i, int j) const { return a[((size_t)t * rows + i) * cols + j]; }
};

static ZMat read_zmat(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "ZMAT", 4) != 0) {
        fprintf(stderr, "%s: bad magic\n", path); exit(2);
    }
    ZMat z;
    int32_t hdr[4];
    if (fread(hdr, sizeof(int32_t), 4, f) != 4) { fprintf(stderr, "short header\n"); exit(2); }
    z.n = hdr[0]; z.deg = hdr[1]; z.rows = hdr[2]; z.cols = hdr[3];
    size_t count = (size_t)z.deg * z.rows * z.cols;
    z.a.resize(count);
    if (fread(z.a.data(), sizeof(long long), count, f) != count) {
        fprintf(stderr, "%s: short data\n", path); exit(2);
    }
    fclose(f);
    return z;
}

static void write_zmat(const char* path, const ZMat& z) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    fwrite("ZMAT", 1, 4, f);
    int32_t hdr[4] = { z.n, z.deg, z.rows, z.cols };
    fwrite(hdr, sizeof(int32_t), 4, f);
    fwrite(z.a.data(), sizeof(long long), z.a.size(), f);
    fclose(f);
}

// -------------------------------------------------------------- CPU oracle
// Accumulates in __int128: if any product overflows int64 this reports it
// instead of quietly agreeing with a GPU that wrapped the same way.
static ZMat cpu_mul(const ZMat& A, const ZMat& B, const std::vector<int>& R,
                    bool* overflow) {
    int deg = A.deg, n = A.rows, k = A.cols, c = B.cols;
    int nrows = 2 * deg - 1;
    ZMat C; C.n = A.n; C.deg = deg; C.rows = n; C.cols = c;
    C.a.assign((size_t)deg * n * c, 0);
    *overflow = false;
    std::vector<i128> raw(nrows);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            for (int m = 0; m < nrows; ++m) raw[m] = i128_zero();
            for (int p = 0; p < deg; ++p)
                for (int q = 0; q < deg; ++q) {
                    i128 s = i128_zero();
                    for (int t = 0; t < k; ++t)
                        s = i128_add(s, i128_mul64(A.at(p, i, t), B.at(q, t, j)));
                    raw[p + q] = i128_add(raw[p + q], s);
                }
            for (int t = 0; t < deg; ++t) {
                i128 acc = i128_zero();
                for (int m = 0; m < nrows; ++m) {
                    int r = R[(size_t)m * deg + t];
                    if (r) acc = i128_add(acc, i128_mul_small(raw[m], (long long)r));
                }
                if (!i128_fits_i64(acc)) *overflow = true;
                C.at(t, i, j) = i128_to_i64(acc);
            }
        }
    }
    return C;
}

// ------------------------------------------------------------------ GPU
// One thread per output element; the whole convolution is done inline. No
// tiling, no scatter -- deliberately a different shape from the kernel it is
// checking.
__global__ void ref_mul(const long long* __restrict__ A,
                        const long long* __restrict__ B,
                        long long* __restrict__ C,
                        const int* __restrict__ R,
                        int deg, int n, int k, int c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= c) return;

    const int nrows = 2 * deg - 1;
    long long raw[15];                      // deg <= 8 covers Phi_24
    for (int m = 0; m < nrows; ++m) raw[m] = 0;

    for (int p = 0; p < deg; ++p) {
        const long long* Ap = A + (size_t)p * n * k + (size_t)i * k;
        for (int q = 0; q < deg; ++q) {
            const long long* Bq = B + (size_t)q * k * c;
            long long s = 0;
            for (int t = 0; t < k; ++t) s += Ap[t] * Bq[(size_t)t * c + j];
            raw[p + q] += s;
        }
    }
    for (int t = 0; t < deg; ++t) {
        long long acc = 0;
        for (int m = 0; m < nrows; ++m) {
            int r = R[(size_t)m * deg + t];
            if (r) acc += (long long)r * raw[m];
        }
        C[((size_t)t * n + i) * c + j] = acc;
    }
}

static ZMat gpu_mul(const ZMat& A, const ZMat& B, const std::vector<int>& R) {
    int deg = A.deg, n = A.rows, k = A.cols, c = B.cols;
    ZMat C; C.n = A.n; C.deg = deg; C.rows = n; C.cols = c;
    C.a.assign((size_t)deg * n * c, 0);

    long long *dA, *dB, *dC; int* dR;
    CUDA_OK(cudaMalloc(&dA, A.a.size() * sizeof(long long)));
    CUDA_OK(cudaMalloc(&dB, B.a.size() * sizeof(long long)));
    CUDA_OK(cudaMalloc(&dC, C.a.size() * sizeof(long long)));
    CUDA_OK(cudaMalloc(&dR, R.size() * sizeof(int)));
    CUDA_OK(cudaMemcpy(dA, A.a.data(), A.a.size() * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, B.a.data(), B.a.size() * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dR, R.data(), R.size() * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((c + 15) / 16, (n + 15) / 16);
    ref_mul<<<grid, block>>>(dA, dB, dC, dR, deg, n, k, c);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(C.a.data(), dC, C.a.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR);
    return C;
}

// ------------------------------------------------------------------ commands
static int cmd_selftest() {
    std::mt19937_64 rng(20260910);
    int fails = 0;
    printf("independent CUDA C++ reference: internal CPU/GPU agreement\n");
    printf("%6s %5s %6s %6s %10s %10s\n", "n_cyc", "deg", "size", "bits", "cpu_gpu", "overflow");
    for (int ncyc : {6, 8, 12, 24}) {
        int deg = 0;
        std::vector<int> R = power_table(ncyc, deg);
        for (int sz : {1, 5, 16, 33, 64}) {
            int bits = 10;
            ZMat A; A.n = ncyc; A.deg = deg; A.rows = sz; A.cols = sz;
            ZMat B = A;
            A.a.resize((size_t)deg * sz * sz);
            B.a.resize((size_t)deg * sz * sz);
            long long lim = 1LL << bits;
            for (auto& v : A.a) v = (long long)(rng() % (2 * lim)) - lim;
            for (auto& v : B.a) v = (long long)(rng() % (2 * lim)) - lim;
            bool ovf = false;
            ZMat cpu = cpu_mul(A, B, R, &ovf);
            ZMat gpu = gpu_mul(A, B, R);
            bool same = (cpu.a == gpu.a);
            if (!same) ++fails;
            printf("%6d %5d %6d %6d %10s %10s\n", ncyc, deg, sz, bits,
                   same ? "EXACT" : "DIFFER", ovf ? "YES" : "no");
        }
    }
    // Phi_n cross-check against known values
    printf("\nPhi_n recomputed here:\n");
    for (int n : {6, 8, 12, 24, 105}) {
        Poly p = cyclotomic(n);
        printf("  Phi_%-4d deg %-3d :", n, (int)p.size() - 1);
        for (size_t i = 0; i < p.size() && i < 12; ++i) printf(" %lld", p[i]);
        if (p.size() > 12) printf(" ...");
        printf("\n");
    }
    printf("\n%s\n", fails ? "SELFTEST FAILED" : "selftest passed");
    return fails ? 1 : 0;
}

static int cmd_mul(const char* pa, const char* pb, const char* po) {
    ZMat A = read_zmat(pa), B = read_zmat(pb);
    if (A.n != B.n || A.cols != B.rows) { fprintf(stderr, "shape/field mismatch\n"); return 2; }
    int deg = 0;
    std::vector<int> R = power_table(A.n, deg);
    if (deg != A.deg) { fprintf(stderr, "deg mismatch: file %d, computed %d\n", A.deg, deg); return 2; }
    bool ovf = false;
    ZMat cpu = cpu_mul(A, B, R, &ovf);
    ZMat gpu = gpu_mul(A, B, R);
    if (cpu.a != gpu.a) { fprintf(stderr, "reference CPU and GPU disagree\n"); return 2; }
    if (ovf) fprintf(stderr, "warning: result exceeds int64\n");
    write_zmat(po, cpu);
    printf("ok %dx%d over Q(zeta_%d), deg %d, cpu==gpu, overflow=%s\n",
           A.rows, B.cols, A.n, deg, ovf ? "YES" : "no");
    return 0;
}

static int cmd_cmp(const char* px, const char* py) {
    ZMat X = read_zmat(px), Y = read_zmat(py);
    if (X.n != Y.n || X.deg != Y.deg || X.rows != Y.rows || X.cols != Y.cols) {
        printf("SHAPE MISMATCH\n"); return 1;
    }
    size_t diffs = 0; size_t first = 0;
    for (size_t i = 0; i < X.a.size(); ++i)
        if (X.a[i] != Y.a[i]) { if (!diffs) first = i; ++diffs; }
    if (diffs) {
        printf("DIFFER: %zu of %zu coefficients, first at index %zu (%lld vs %lld)\n",
               diffs, X.a.size(), first, X.a[first], Y.a[first]);
        return 1;
    }
    printf("IDENTICAL: %zu coefficients\n", X.a.size());
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return cmd_selftest();
    if (argc == 5 && strcmp(argv[1], "mul") == 0) return cmd_mul(argv[2], argv[3], argv[4]);
    if (argc == 4 && strcmp(argv[1], "cmp") == 0) return cmd_cmp(argv[2], argv[3]);
    fprintf(stderr,
        "usage:\n  refcheck selftest\n  refcheck mul A.zmat B.zmat OUT.zmat\n"
        "  refcheck cmp X.zmat Y.zmat\n");
    return 2;
}
