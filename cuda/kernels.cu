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

// ===========================================================================
// ULTRA: one launch, narrow operands, wide accumulate, C written exactly once.
//
// Measured against the four kernels above on this device -- RTX 5090, sm_120,
// 170 SMs, 1536 threads and 64K registers per SM, 100 KB of shared memory per
// SM in carve-outs of {0,8,16,32,64,100} KB, 96 MiB of L2 -- every one of them
// leaves the same three things on the table:
//
//   1. deg^2 launches. Q(zeta_24) pays 64 of them, and each one re-reads a
//      whole A plane and a whole B plane out of L2.
//   2. C is accumulated with +=, so every launch reads C back and writes it
//      again. At n = 1024 that is more traffic than the operands.
//   3. A 64x64 output tile gives 256 blocks at n = 1024. The device has 170
//      SMs and room for 24 blocks on each: the grid cannot fill it.
//
// This kernel fixes all three at once. The field product is the convolution
//     raw[m] = sum_{p+q=m} A_p B_q ,  out_t = sum_m R[m][t] raw[m] ,
// and the convolution itself supplies the arithmetic intensity: one output
// element per thread already does DEG*DEG multiply-accumulates against only
// 2*DEG shared-memory reads, so DEG/2 MACs per read with no output blocking
// at all. That is what makes a 16x16 tile worth using -- and a 16x16 tile at
// n = 1024 is 4096 blocks, which does fill the device.
//
// The price is registers: raw[] is 2*DEG-1 accumulators of 64 bits, so 30 of
// them at DEG = 8. That is why the tile is 16x16 and one element per thread
// rather than 64x64 and sixteen: sixteen outputs would want 480 registers and
// the hardware caps a thread at 255.
//
// DEG is a template parameter, not an argument. phi(n) is known on the host,
// and compiling one kernel per degree is what makes every loop here unroll,
// every raw[] index a constant, and every accumulator a register instead of a
// spill. A degree with no instantiation falls back to the w32 path above.
//
// Exactness is unchanged and is the reason for the shape of it:
//   - operands are int32 only because the host already proved |A|,|B| < 2^31
//     from the coefficients it wrote; narrowing is a representation change,
//     not a rounding
//   - every product is mul.wide.s32, 32x32 -> 64, and every accumulator is
//     int64, so no product is ever truncated
//   - R is applied once, at the end, in int64
//   - there is no floating-point type, literal, or intrinsic in any of it, and
//     the PTX is scanned for float instructions before it is allowed to load
// ===========================================================================

#define UT 16     // output tile edge: 16x16 outputs per block
#define UK 16     // k-depth staged per step

// Shared memory is indexed by hand rather than with a 3-D array so the two
// padding strides can be chosen against the 32 banks:
//   S  = DEG+1  -- plane stride. Odd for even DEG, so the 16 threads of a tile
//                  row hit 16 distinct banks when they read their own planes.
//   KS = UT*S+1 -- k stride. Coprime to 32, so the 16 threads that each stage
//                  a different k index also hit distinct banks. Without the
//                  +1 the stride is a multiple of 16 and the store collapses
//                  onto two banks.
// A is stored [u][row][p] and B as [u][col][q], which makes the compute-loop
// read of A a 2-address broadcast and the read of B stride-S.

template<int DEG>
__device__ __forceinline__ void fused_w32_core(
    const int*       __restrict__ A,   // DEG planes, each n x k, |A| < 2^31
    const int*       __restrict__ B,   // DEG planes, each k x c, |B| < 2^31
    long long*       __restrict__ C,   // DEG planes, each n x c
    const int*       __restrict__ R,   // (2*DEG-1) x DEG reduction table
    int n, int k, int c)
{
    const int NRAW = 2 * DEG - 1;
    const int S    = DEG + 1;
    const int KS   = UT * S + 1;

    __shared__ int As[UK * KS];
    __shared__ int Bs[UK * KS];
    __shared__ int Rs[(2 * DEG - 1) * DEG];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * UT + ty;
    const int col = blockIdx.x * UT + tx;
    const int tid = ty * UT + tx;

    // R is the same for every block and is tiny; stage it once.
    if (tid < NRAW * DEG) Rs[tid] = R[tid];

    long long raw[2 * DEG - 1];
    #pragma unroll
    for (int m = 0; m < NRAW; ++m) raw[m] = 0LL;

    const long long aPlane = (long long)n * k;
    const long long bPlane = (long long)k * c;

    for (int t0 = 0; t0 < k; t0 += UK) {
        const int aCol = t0 + tx;
        const int bRow = t0 + ty;
        const bool aOk = (row < n) && (aCol < k);
        const bool bOk = (bRow < k) && (col < c);

        const int* ap = A + (long long)row * k + aCol;
        const int* bp = B + (long long)bRow * c + col;

        #pragma unroll
        for (int p = 0; p < DEG; ++p) {
            As[tx * KS + ty * S + p] = aOk ? ap[p * aPlane] : 0;
            Bs[ty * KS + tx * S + p] = bOk ? bp[p * bPlane] : 0;
        }
        __syncthreads();

        #pragma unroll 4
        for (int u = 0; u < UK; ++u) {
            int a[DEG], b[DEG];
            #pragma unroll
            for (int p = 0; p < DEG; ++p) a[p] = As[u * KS + ty * S + p];
            #pragma unroll
            for (int q = 0; q < DEG; ++q) b[q] = Bs[u * KS + tx * S + q];
            #pragma unroll
            for (int p = 0; p < DEG; ++p) {
                #pragma unroll
                for (int q = 0; q < DEG; ++q)
                    raw[p + q] += (long long)a[p] * (long long)b[q];  // mul.wide.s32
            }
        }
        __syncthreads();
    }

    if (row < n && col < c) {
        const long long idx = (long long)row * c + col;
        const long long cPlane = (long long)n * c;
        #pragma unroll
        for (int t = 0; t < DEG; ++t) {
            long long acc = 0LL;
            #pragma unroll
            for (int m = 0; m < NRAW; ++m) {
                const int r = Rs[m * DEG + t];
                if (r) acc += (long long)r * raw[m];
            }
            C[t * cPlane + idx] = acc;         // written once, never read back
        }
    }
}

#define ULTRA_KERNEL(DEG)                                                     \
  extern "C" __global__ __launch_bounds__(UT * UT)                            \
  void fused_w32_d##DEG(const int* __restrict__ A, const int* __restrict__ B, \
                        long long* __restrict__ C, const int* __restrict__ R, \
                        int n, int k, int c)                                  \
  { fused_w32_core<DEG>(A, B, C, R, n, k, c); }

ULTRA_KERNEL(1)    // Q(zeta_1), Q(zeta_2)
ULTRA_KERNEL(2)    // Q(zeta_3), Q(zeta_4), Q(zeta_6)
ULTRA_KERNEL(4)    // Q(zeta_8), Q(zeta_12)
ULTRA_KERNEL(6)    // Q(zeta_7), Q(zeta_9), Q(zeta_14), Q(zeta_18)
ULTRA_KERNEL(8)    // Q(zeta_24), Q(zeta_15), Q(zeta_16), Q(zeta_20), Q(zeta_30)
ULTRA_KERNEL(10)
ULTRA_KERNEL(12)
ULTRA_KERNEL(16)

// ===========================================================================
// ULTRA-K: the same product with 3^L multiplies instead of 4^L.
//
// Measured on this device, fused_w32_d8 above costs about 0.55 ms per plane
// product at 2048x2048 on top of 10 ms of fixed cost, and that cost is flat
// against unroll factor and occupancy. Cutting the multiply count is the only
// thing that moves it -- so cut the multiply count.
//
// Karatsuba does a length-2H convolution with three length-H ones:
//     Z0 = a0 b0,  Z2 = a1 b1,  Z1 = (a0+a1)(b0+b1) - Z0 - Z2
// and recursively deg = 2^L costs 3^L multiplies: 27 instead of 64 at deg 8,
// 9 instead of 16 at deg 4, 81 instead of 256 at deg 16. Exactly -- every
// coefficient of the result is the integer it would have been, because every
// intermediate is an integer sum of integers. There is nothing here to round.
//
// Only the forming of the 3^L products is done on the device. The recombination
// is linear, so Racket collapses it -- together with the Phi_n reduction --
// into one deg x 3^L integer matrix K with out_t = sum_j K[t][j] P_j, derived
// in karatsuba.rkt and checked there against cyc* . The device applies K once
// per output element and unwinds no recursion of its own.
//
// The price is range, and it is paid in a checked certificate rather than
// hoped for: each multiplicand is a sum of up to 2^L coefficients, so the host
// must see 2^L max|A| and 2^L max|B| inside int32, and the output bound becomes
// rowsum(K) * 4^L * k * maxA * maxB, with rowsum computed exactly (18 for
// Q(zeta_24), not estimated). When that does not fit, the selector falls back
// to fused_w32 above, which has the smaller bound and the larger multiply
// count. Nothing is ever truncated to make it fit.
// ===========================================================================

template<int L> struct Pow3 { enum { v = 3 * Pow3<L-1>::v }; };
template<>      struct Pow3<0> { enum { v = 1 }; };

// Forms the 3^L products in the order karatsuba.rkt's kara-mac does: low half,
// high half, sum half. Every index is a compile-time constant after unrolling,
// so P[] and the partial sums stay in registers.
template<int L> struct Kmac {
    __device__ __forceinline__ static void go(const int* a, const int* b, long long* P)
    {
        const int H = 1 << (L - 1);
        const int M = Pow3<L-1>::v;
        int as[H], bs[H];
        #pragma unroll
        for (int z = 0; z < H; ++z) { as[z] = a[z] + a[z+H]; bs[z] = b[z] + b[z+H]; }
        Kmac<L-1>::go(a,   b,   P);
        Kmac<L-1>::go(a+H, b+H, P + M);
        Kmac<L-1>::go(as,  bs,  P + 2*M);
    }
};
template<> struct Kmac<0> {
    __device__ __forceinline__ static void go(const int* a, const int* b, long long* P)
    {
        P[0] += (long long)a[0] * (long long)b[0];     // mul.wide.s32
    }
};

template<int L>
__device__ __forceinline__ void kara_w32_core(
    const int*       __restrict__ A,   // DEG planes, each n x k
    const int*       __restrict__ B,   // DEG planes, each k x c
    long long*       __restrict__ C,   // DEG planes, each n x c
    const int*       __restrict__ K,   // DEG x 3^L output matrix
    int n, int k, int c)
{
    const int DEG = 1 << L;
    const int NP  = Pow3<L>::v;
    const int S   = DEG + 1;
    const int KS  = UT * S + 1;

    __shared__ int As[UK * KS];
    __shared__ int Bs[UK * KS];
    __shared__ int Ks[DEG * Pow3<L>::v];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * UT + ty;
    const int col = blockIdx.x * UT + tx;
    const int tid = ty * UT + tx;

    for (int z = tid; z < DEG * NP; z += UT * UT) Ks[z] = K[z];

    long long P[Pow3<L>::v];
    #pragma unroll
    for (int j = 0; j < NP; ++j) P[j] = 0LL;

    const long long aPlane = (long long)n * k;
    const long long bPlane = (long long)k * c;

    for (int t0 = 0; t0 < k; t0 += UK) {
        const int aCol = t0 + tx;
        const int bRow = t0 + ty;
        const bool aOk = (row < n) && (aCol < k);
        const bool bOk = (bRow < k) && (col < c);

        const int* ap = A + (long long)row * k + aCol;
        const int* bp = B + (long long)bRow * c + col;

        #pragma unroll
        for (int p = 0; p < DEG; ++p) {
            As[tx * KS + ty * S + p] = aOk ? ap[p * aPlane] : 0;
            Bs[ty * KS + tx * S + p] = bOk ? bp[p * bPlane] : 0;
        }
        __syncthreads();

        #pragma unroll 4
        for (int u = 0; u < UK; ++u) {
            int a[DEG], b[DEG];
            #pragma unroll
            for (int p = 0; p < DEG; ++p) a[p] = As[u * KS + ty * S + p];
            #pragma unroll
            for (int q = 0; q < DEG; ++q) b[q] = Bs[u * KS + tx * S + q];
            Kmac<L>::go(a, b, P);
        }
        __syncthreads();
    }

    if (row < n && col < c) {
        const long long idx = (long long)row * c + col;
        const long long cPlane = (long long)n * c;
        #pragma unroll
        for (int t = 0; t < DEG; ++t) {
            long long acc = 0LL;
            #pragma unroll
            for (int j = 0; j < NP; ++j) {
                const int w = Ks[t * NP + j];
                if (w) acc += (long long)w * P[j];
            }
            C[t * cPlane + idx] = acc;
        }
    }
}

#define KARA_KERNEL(L)                                                        \
  extern "C" __global__ __launch_bounds__(UT * UT)                            \
  void kara_w32_l##L(const int* __restrict__ A, const int* __restrict__ B,    \
                     long long* __restrict__ C, const int* __restrict__ K,    \
                     int n, int k, int c)                                     \
  { kara_w32_core<L>(A, B, C, K, n, k, c); }

KARA_KERNEL(1)   // deg 2  : 3 multiplies, not 4
KARA_KERNEL(2)   // deg 4  : 9, not 16
KARA_KERNEL(3)   // deg 8  : 27, not 64
KARA_KERNEL(4)   // deg 16 : 81, not 256
