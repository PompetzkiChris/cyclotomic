// Exact integer kernels for Z[zeta_n] linear algebra.
//
// Everything is 64-bit integer arithmetic. There is no floating point in this
// file, and none in the PTX it compiles to -- which is checkable, and is
// checked by the Racket side.
//
// A matrix over Z[zeta_n] is deg = phi(n) integer planes, each n x c, laid out
// contiguously with stride planeStride = n*c.
//
// The product needs the raw convolution
//     raw[m] = sum_{i+j=m} A_i B_j
// folded back into the power basis through the integer table R, where
//     zeta^m = sum_t R[m][t] zeta^t   (mod Phi_n).
//
// plane_madd does one (i, j) pair: it computes P = A_i B_j tile by tile and
// immediately scatters R[m][t] * P into every output plane t. No raw plane is
// ever materialised, so the working set is just A + B + C.

#define TILE 16

extern "C" __global__ void plane_madd(
    const long long* __restrict__ A,   // one plane, n x k
    const long long* __restrict__ B,   // one plane, k x c
    long long*       __restrict__ C,   // deg planes, each n x c
    const int*       __restrict__ Rrow,// deg reduction coefficients for this m
    int n, int k, int c, int deg,
    long long planeStride)
{
    __shared__ long long As[TILE][TILE];
    __shared__ long long Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    long long acc = 0;

    const int tiles = (k + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int aCol = t * TILE + threadIdx.x;
        const int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < n && aCol < k) ? A[(long long)row * k + aCol] : 0LL;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < k && col < c) ? B[(long long)bRow * c + col] : 0LL;

        __syncthreads();

        #pragma unroll
        for (int u = 0; u < TILE; ++u) {
            acc += As[threadIdx.y][u] * Bs[u][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < c) {
        const long long idx = (long long)row * c + col;
        for (int t = 0; t < deg; ++t) {
            const int r = Rrow[t];
            if (r != 0) {
                C[t * planeStride + idx] += (long long)r * acc;
            }
        }
    }
}

// Zero a device buffer of 64-bit words.
extern "C" __global__ void zero_i64(long long* __restrict__ p, long long count)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < count; i += stride) p[i] = 0LL;
}

// Largest absolute value in a buffer, for overflow auditing after the fact.
// Block-level reduction into out[blockIdx.x]; the host finishes the max.
extern "C" __global__ void absmax_i64(const long long* __restrict__ p,
                                      long long count,
                                      long long* __restrict__ out)
{
    __shared__ long long s[256];
    long long best = 0;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < count; i += stride) {
        long long v = p[i];
        if (v < 0) v = -v;
        if (v > best) best = v;
    }
    s[threadIdx.x] = best;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            long long o = s[threadIdx.x + half];
            if (o > s[threadIdx.x]) s[threadIdx.x] = o;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// ===========================================================================
// Fused product: ONE launch for the whole field product, not deg^2 of them.
//
// plane_madd above is correct but reads badly. Launched deg^2 times, it pulls
// every A plane and every B plane out of global memory deg times over, and
// pays deg^2 launches. For Q(zeta_24), deg = 8: 64 launches and 8x the traffic
// each operand actually needs.
//
// This version stages ALL deg planes of both operands into shared memory once
// per k-tile, and each thread accumulates the entire length-(2deg-1) raw
// convolution in registers before folding it through R exactly once. Global
// traffic drops by a factor of deg, and the launch count drops to one.
//
// Shared memory: 2 * deg * FT * FT * 8 bytes. deg <= 8, FT = 16  ->  32 KB.
// Registers: 2*deg-1 = 15 int64 accumulators per thread.
//
// The arithmetic is identical to plane_madd's, and gpu-tests.rkt checks the
// two against each other as well as against pure Racket and the independent
// CUDA C++ reference.
// ===========================================================================

#define FT     16     // tile edge
#define MAXDEG  8     // phi(n) supported; Phi_24 has degree 8
#define MAXRAW 15     // 2*MAXDEG - 1

extern "C" __global__ void fused_madd(
    const long long* __restrict__ A,    // deg planes, each n x k
    const long long* __restrict__ B,    // deg planes, each k x c
    long long*       __restrict__ C,    // deg planes, each n x c
    const int*       __restrict__ R,    // (2*deg-1) x deg reduction table
    int n, int k, int c, int deg)
{
    __shared__ long long As[MAXDEG][FT][FT];
    __shared__ long long Bs[MAXDEG][FT][FT + 1];   // +1 kills bank conflicts

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * FT + ty;
    const int col = blockIdx.x * FT + tx;
    const int nraw = 2 * deg - 1;

    long long raw[MAXRAW];
    #pragma unroll
    for (int m = 0; m < MAXRAW; ++m) raw[m] = 0LL;

    const long long aPlane = (long long)n * k;
    const long long bPlane = (long long)k * c;

    const int tiles = (k + FT - 1) / FT;
    for (int t = 0; t < tiles; ++t) {
        const int aCol = t * FT + tx;
        const int bRow = t * FT + ty;

        for (int p = 0; p < deg; ++p) {
            As[p][ty][tx] = (row < n && aCol < k)
                ? A[(long long)p * aPlane + (long long)row * k + aCol] : 0LL;
            Bs[p][ty][tx] = (bRow < k && col < c)
                ? B[(long long)p * bPlane + (long long)bRow * c + col] : 0LL;
        }
        __syncthreads();

        for (int u = 0; u < FT; ++u) {
            for (int p = 0; p < deg; ++p) {
                const long long a = As[p][ty][u];
                if (a == 0LL) continue;              // sparse planes are common
                for (int q = 0; q < deg; ++q) {
                    raw[p + q] += a * Bs[q][u][tx];
                }
            }
        }
        __syncthreads();
    }

    if (row < n && col < c) {
        const long long idx = (long long)row * c + col;
        const long long cPlane = (long long)n * c;
        for (int tt = 0; tt < deg; ++tt) {
            long long acc = 0LL;
            for (int m = 0; m < nraw; ++m) {
                const int r = R[m * deg + tt];
                if (r) acc += (long long)r * raw[m];
            }
            C[(long long)tt * cPlane + idx] = acc;
        }
    }
}

// ===========================================================================
// Register-blocked plane product.
//
// fused_madd above cuts global traffic by a factor of deg and is measurably
// SLOWER for it: 32 KB of shared memory per block collapses occupancy, and on
// this device that costs more than the traffic it saves. Measured 0.3-0.6x.
// It is kept because it is exact and the two kernels check each other, but it
// is not the fast path.
//
// The actual win is arithmetic intensity. plane_madd gives each thread one
// output element, so every multiply-add needs two shared-memory reads. Here
// each thread owns a TM x TN block of outputs, so TM+TN shared reads feed
// TM*TN multiply-adds -- eight times fewer reads per unit of arithmetic at
// 4x4, with only 16 int64 accumulators in registers.
//
//   block  16 x 16 threads = 256
//   tile   BM x BN = 64 x 64 outputs, BK = 16 deep
//   shared As[BK][BM] + Bs[BK][BN] = 16 KB, leaving occupancy intact
//
// As is stored transposed so the inner loop reads it with stride 1.
// ===========================================================================

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

extern "C" __global__ void plane_madd_rb(
    const long long* __restrict__ A,   // one plane, n x k
    const long long* __restrict__ B,   // one plane, k x c
    long long*       __restrict__ C,   // deg planes, each n x c
    const int*       __restrict__ Rrow,// deg reduction coefficients for this m
    int n, int k, int c, int deg,
    long long planeStride)
{
    __shared__ long long As[BK][BM];
    __shared__ long long Bs[BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // each thread owns a TM x TN block
    const int tRow = (tid / (BN / TN)) * TM;      // 0,4,...,60
    const int tCol = (tid % (BN / TN)) * TN;

    long long acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0LL;

    // loader indices: 256 threads move 64x16 = 1024 elements, 4 each
    const int aRowL = tid / BK;            // 0..15  -> which of 16 rows per pass
    const int aColL = tid % BK;            // 0..15
    const int bRowL = tid / BN;            // 0..3
    const int bColL = tid % BN;            // 0..63

    for (int t0 = 0; t0 < k; t0 += BK) {
        #pragma unroll
        for (int s = 0; s < BM; s += 16) {
            const int r = rowBase + aRowL + s;
            const int cc = t0 + aColL;
            As[aColL][aRowL + s] =
                (r < n && cc < k) ? A[(long long)r * k + cc] : 0LL;
        }
        #pragma unroll
        for (int s = 0; s < BK; s += 4) {
            const int r = t0 + bRowL + s;
            const int cc = colBase + bColL;
            Bs[bRowL + s][bColL] =
                (r < k && cc < c) ? B[(long long)r * c + cc] : 0LL;
        }
        __syncthreads();

        #pragma unroll
        for (int u = 0; u < BK; ++u) {
            long long a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[u][tRow + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[u][tCol + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = rowBase + tRow + i;
        if (r >= n) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int cc = colBase + tCol + j;
            if (cc >= c) continue;
            const long long v = acc[i][j];
            if (v == 0LL) continue;
            const long long idx = (long long)r * c + cc;
            for (int tt = 0; tt < deg; ++tt) {
                const int rr = Rrow[tt];
                if (rr) C[tt * planeStride + idx] += (long long)rr * v;
            }
        }
    }
}

// ===========================================================================
// Narrow operands, wide accumulate.
//
// Register blocking bought almost nothing (1.0-1.1x), which says the kernel is
// not starved of memory -- it is starved of integer throughput. There is no
// native 64x64 integer multiply on this hardware; mul.lo.s64 is synthesised
// from several 32-bit multiplies and adds.
//
// But mul.wide.s32 -- 32 x 32 -> 64 -- IS a single instruction. Every operand
// here is already bounds-checked before launch, so whenever |A|,|B| < 2^31 the
// planes can be narrowed to int32 and the product done with one IMAD per
// multiply-add while accumulating in full 64-bit width. Nothing is rounded or
// truncated: the accumulator stays int64 and the same bound that authorised
// the launch still holds.
//
// Halving the operand width also halves the shared-memory tiles, which lifts
// occupancy on top of the instruction win.
// ===========================================================================

extern "C" __global__ void narrow_i64_i32(
    const long long* __restrict__ src, int* __restrict__ dst, long long count)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < count; i += stride) dst[i] = (int)src[i];
}

extern "C" __global__ void plane_madd_w32(
    const int* __restrict__ A,         // one plane, n x k, |A| < 2^31
    const int* __restrict__ B,         // one plane, k x c, |B| < 2^31
    long long* __restrict__ C,         // deg planes, each n x c
    const int* __restrict__ Rrow,
    int n, int k, int c, int deg,
    long long planeStride)
{
    __shared__ int As[BK][BM];
    __shared__ int Bs[BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    const int tRow = (tid / (BN / TN)) * TM;
    const int tCol = (tid % (BN / TN)) * TN;

    long long acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0LL;

    const int aRowL = tid / BK;
    const int aColL = tid % BK;
    const int bRowL = tid / BN;
    const int bColL = tid % BN;

    for (int t0 = 0; t0 < k; t0 += BK) {
        #pragma unroll
        for (int s = 0; s < BM; s += 16) {
            const int r = rowBase + aRowL + s;
            const int cc = t0 + aColL;
            As[aColL][aRowL + s] = (r < n && cc < k) ? A[(long long)r * k + cc] : 0;
        }
        #pragma unroll
        for (int s = 0; s < BK; s += 4) {
            const int r = t0 + bRowL + s;
            const int cc = colBase + bColL;
            Bs[bRowL + s][bColL] = (r < k && cc < c) ? B[(long long)r * c + cc] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int u = 0; u < BK; ++u) {
            int a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[u][tRow + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[u][tCol + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += (long long)a[i] * (long long)b[j];   // mul.wide.s32
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = rowBase + tRow + i;
        if (r >= n) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int cc = colBase + tCol + j;
            if (cc >= c) continue;
            const long long v = acc[i][j];
            if (v == 0LL) continue;
            const long long idx = (long long)r * c + cc;
            for (int tt = 0; tt < deg; ++tt) {
                const int rr = Rrow[tt];
                if (rr) C[tt * planeStride + idx] += (long long)rr * v;
            }
        }
    }
}
