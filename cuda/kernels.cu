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
