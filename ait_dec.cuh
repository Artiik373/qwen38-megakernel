// AI+ (llamAmpere AIT3 / AIT4 tile layout, AIT5_U unit layout) weight decoders for the megakernel. The tensors stay
// in their GGUF bytes in VRAM (no repack); these functions decode them. Formats: llamAmpere ggml/src/ggml-ait.h
// (types 111/112: 32 rows x 512 K chunks, rate k = 3/4) and ggml-ai4.h (type 114: 16 rows x 128 K units, rate k = 5).
// Level = 1MAD of a 16-bit window of a tail-biting bit stream: ((bytesum(state * mul) + 2) >> 2) - 128.
// Weight in the rotated basis: level * row scale / 64 (* column scale, tile types only); activations go through the
// same rotation (sign flip, FWHT256, / 16, then * column scale) so that y = W x = W' x'.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#define AITD_MUL3 0x87D8D12Du
#define AITD_MUL4 0xAA73876Bu // AIT4 and AIT5_U

__host__ __device__ __forceinline__ int aitd_level(uint32_t st, uint32_t mul) {
    const uint32_t x = st * mul;
    const int b = (int) ((x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24));
    return ((b + 2) >> 2) - 128;
}

// tile types: bytes of one 32 x 512 chunk
__host__ __device__ __forceinline__ int64_t aitd_chunk_bytes(int k) { return 32 * (64 * (int64_t) k + 2); }

// the 8 levels a lane holds of tile t (0..63) of a chunk: weight j of lane l sits at tile row 16 (j / 4) + l / 4 + 8 ((j / 2) % 2),
// tile col 2 (l % 4) + j % 2 (the m16n8k32 A fragment slots)
__device__ __forceinline__ void aitd_tile_lane8(const uint8_t * chunk, int t, int l, int k, uint32_t mul, int (&lev)[8]) {
    const uint8_t * grp = chunk + (int64_t) (t >> 2) * 128 * k + 4 * (t & 3);
    const int nw = 8 * k;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int pos = (8 * l + j) * k % (256 * k);
        const int wi = pos >> 5, sh = pos & 31;
        const uint32_t lo = *(const uint32_t *) (grp + 16 * wi), hi = *(const uint32_t *) (grp + 16 * ((wi + 1) % nw));
        const uint32_t v = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
        lev[j] = aitd_level(v & 0xFFFFu, mul);
    }
}

// row scale (fp16) of tile-type row m (of chunk row tile m / 32) for K column kcol, column scale of kcol
__device__ __forceinline__ float aitd_row_scale(const uint8_t * base, int64_t K, int k, int64_t m, int64_t kcol) {
    const int64_t S = K / 512, rt = m >> 5;
    const int rr = (int) (m & 31);
    const int64_t ks = 2 * (kcol >> 10) + (rr >> 4);
    return __half2float(*(const __half *) (base + (rt * S + ks) * aitd_chunk_bytes(k) + 2048 * (int64_t) k + 2 * (rr & 15)));
}
__device__ __forceinline__ float aitd_col_scale(const uint8_t * base, int64_t K, int k, int64_t kcol) {
    const int64_t S = K / 512, ks = kcol >> 9;
    const int c = (int) (kcol & 511);
    return __half2float(*(const __half *) (base + ((int64_t) (c >> 4) * S + ks) * aitd_chunk_bytes(k) + 2048 * (int64_t) k + 32 + 2 * (c & 15)));
}

// AIT5_U (unit) types: slab of 16 rows = K / 128 units of (32 B fp16 row scales | 16 tiles of 16 k bytes)
__host__ __device__ __forceinline__ int64_t ai5d_unit_bytes(int k) { return 32 + 16 * 16 * (int64_t) k; }

// level of weight n (0..127) of a unit tile (tw = the tile's 4 k words)
__device__ __forceinline__ int ai5d_level(const uint32_t * tw, int k, uint32_t mul, int n) {
    const int nb = 128 * k, nwd = 4 * k;
    int b0 = k * n + k - 16;
    if (b0 < 0) b0 += nb;
    const int w = b0 >> 5, sh = b0 & 31;
    const uint64_t c = ((uint64_t) tw[w] << 32) | tw[(w + 1) % nwd];
    return aitd_level((uint32_t) ((c >> (48 - sh)) & 0xFFFFu), mul);
}

// activation rotation of one 256 block by one warp (lane holds elements 8 lane .. 8 lane + 7): x' = H256 (sign . x) / 16.
// The column scale (tile types) is applied by the caller: x'[k] *= cs[k].
__device__ __constant__ uint32_t aitd_sign_bits[8] = {
    0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5ced2a1bu, 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au};

__device__ __forceinline__ void aitd_rot256(float (&v)[8], int lane) {
    const uint32_t sw = aitd_sign_bits[lane >> 2] >> ((lane & 3) * 8);
#pragma unroll
    for (int q = 0; q < 8; q++) if ((sw >> q) & 1) v[q] = -v[q];
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int q = 0; q < 8; q++) if (!(q & h)) { const float a = v[q], b = v[q + h]; v[q] = a + b; v[q + h] = a - b; }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool up = lane & m;
#pragma unroll
        for (int q = 0; q < 8; q++) {
            const float o = __shfl_xor_sync(0xffffffffu, v[q], m);
            v[q] = up ? o - v[q] : v[q] + o;
        }
    }
#pragma unroll
    for (int q = 0; q < 8; q++) v[q] *= 0.0625f;
}

// ---- decode GEMV on fp16 tensor cores (tile types AIT3 / AIT4) ------------------------------------------------------
// y[m][n] = sum_k w'[m][k] x'[n][k], w' = level * rs / 64 * cs (rotated basis), x' = rot(x) (aitd_rot256), n < NT tokens (<= 8).
// A fragment (m16n8k16, f16): a0 = (row g, k 2 tq, 2 tq + 1), a1 = row g + 8, a2 / a3 = the same for k + 8. A weight tile (8 K
// columns) is exactly 8 of those k slots, so MMA m of a 32-K group takes tiles 2 m and 2 m + 1: no permutation of x'.
// A = fp16(level * cs): the integer level is exact in fp16, one rounding by the product. The row scale / 64 multiplies a
// 1024-K block's fp32 sum (rs is constant over it).
// tile16 = row tile (16 rows: rows 32 rt + 16 u .. + 15). cs: compact K halves (aitd_cs_extract). x: [NT][K] halves.
__device__ __forceinline__ void aitd_mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// the 4 levels (weights j = 4 u .. 4 u + 3 of lane l) of tile i of the 16-byte-word group at grp (4 tiles interleaved per
// word), as ints
__device__ __forceinline__ void aitd_lev4(const uint8_t * grp, int i, int l, int u, int k, uint32_t mul, int (&lev)[4]) {
    const int nw = 8 * k;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int pos = (8 * l + 4 * u + j) * k % (256 * k);
        const int wi = pos >> 5, sh = pos & 31;
        const uint32_t lo = *(const uint32_t *) (grp + 16 * wi + 4 * i), hi = *(const uint32_t *) (grp + 16 * ((wi + 1) % nw) + 4 * i);
        lev[j] = aitd_level((sh ? ((lo >> sh) | (hi << (32 - sh))) : lo) & 0xFFFFu, mul);
    }
}

// ---- fast decoders (same values as aitd_level / aitd_lev4 / ai5d_level; tests: aitwv_test, aitpf_test) ----------------
// level of a 16-bit window: the byte sum of window * mul is one dp4a
__device__ __forceinline__ int aitd_lv(uint32_t w16, uint32_t mul) {
    return (int) ((__dp4a(w16 * mul, 0x01010101u, 0u) + 2u) >> 2) - 128;
}

// lev4 with the rate K a compile time constant: the lane's 4 weights are consecutive in the stream, so one 64-bit window
// (two words) serves all four (sh + 3 K + 16 <= 59 bits); the stream wraps (tail-biting) only through the second word
template <int K>
__device__ __forceinline__ void aitd_lev4t(const uint8_t * grp, int i, int l, int u, uint32_t mul, int (&lev)[4]) {
    constexpr int NW = 8 * K;
    const int pos = (8 * l + 4 * u) * K;
    const int wi = pos >> 5, sh = pos & 31;
    const int wi1 = wi + 1 == NW ? 0 : wi + 1;
    const uint32_t lo = *(const uint32_t *) (grp + 16 * wi + 4 * i), hi = *(const uint32_t *) (grp + 16 * wi1 + 4 * i);
    const uint64_t c = (((uint64_t) hi << 32) | lo) >> sh;
#pragma unroll
    for (int j = 0; j < 4; j++) lev[j] = aitd_lv((uint32_t) (c >> (j * K)) & 0xFFFFu, mul);
}

// the 4 levels of one lane from its two stream words (lo, hi: the lane's window words of one tile)
template <int K>
__device__ __forceinline__ void aitd_lev4w(uint32_t lo, uint32_t hi, int sh, uint32_t mul, int (&lev)[4]) {
    const uint64_t c = (((uint64_t) hi << 32) | lo) >> sh;
#pragma unroll
    for (int j = 0; j < 4; j++) lev[j] = aitd_lv((uint32_t) (c >> (j * K)) & 0xFFFFu, mul);
}

// the 4 levels of one lane from the window word clo (= the stream bits at the lane's position, >= 28 valid) as exact fp16:
// h[0] = (level 0, level 1), h[1] = (level 2, level 3). level + 128 = (byte sum + 2) >> 2 < 256 goes into the mantissa of
// 1024 (0x6400 | v), minus 1152 gives the level exactly
template <int K>
__device__ __forceinline__ void aitd_lev2h(uint32_t clo, uint32_t mul, uint32_t (&h)[2]) {
    uint32_t s[4];
#pragma unroll
    for (int j = 0; j < 4; j++) s[j] = __dp4a(((clo >> (j * K)) & 0xFFFFu) * mul, 0x01010101u, 2u);
#pragma unroll
    for (int p = 0; p < 2; p++) {
        const uint32_t v = (((s[2 * p + 1] << 16) | s[2 * p]) >> 2) & 0x03FF03FFu | 0x64006400u;
        const __half2 hv = __hsub2(*reinterpret_cast<const __half2 *>(&v), __half2half2(__ushort_as_half(0x6480)));
        h[p] = *reinterpret_cast<const uint32_t *>(&hv);
    }
}

struct Ai5Pos { int w, w1, sh; };
// exact fp16 pair (level n, level n + 1) of a unit tile: the stream is circular, so weight n + 1 is the window 5 bits on
__device__ __forceinline__ uint32_t ai5d_lev2h(const uint32_t * tw, const Ai5Pos & p, uint32_t mul) {
    const uint64_t c = ((uint64_t) tw[p.w] << 32) | tw[p.w1];
    const uint32_t s0 = __dp4a(((uint32_t) (c >> (48 - p.sh)) & 0xFFFFu) * mul, 0x01010101u, 2u);
    const uint32_t s1 = __dp4a(((uint32_t) (c >> (43 - p.sh)) & 0xFFFFu) * mul, 0x01010101u, 2u);
    const uint32_t v = (((s1 << 16) | s0) >> 2) & 0x03FF03FFu | 0x64006400u;
    const __half2 h = __hsub2(*reinterpret_cast<const __half2 *>(&v), __half2half2(__ushort_as_half(0x6480)));
    return *reinterpret_cast<const uint32_t *>(&h);
}

// AIT5_U: the word / shift of weight n of a unit tile (lane constants, hoisted by the caller) and its level
// (Ai5Pos is declared above ai5d_lev2h)
__device__ __forceinline__ Ai5Pos ai5d_pos(int n) {
    int b0 = 5 * n + 5 - 16;
    if (b0 < 0) b0 += 640;
    const int w = b0 >> 5;
    return Ai5Pos{w, w + 1 == 20 ? 0 : w + 1, b0 & 31};
}
__device__ __forceinline__ int ai5d_lv(const uint32_t * tw, const Ai5Pos & p, uint32_t mul) {
    const uint64_t c = ((uint64_t) tw[p.w] << 32) | tw[p.w1];
    return aitd_lv((uint32_t) (c >> (48 - p.sh)) & 0xFFFFu, mul);
}

template <int NT>
__device__ __forceinline__ void aitd_gemv_tile(const uint8_t * base, int64_t K, int k, uint32_t mul, int tile16, const __half * cs,
                                               const __half * x, float (&out)[NT > 1 ? 4 : 4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int64_t S = K / 512;
    const int rt = tile16 >> 1, u = tile16 & 1;
    float tot[4] = {0.f, 0.f, 0.f, 0.f};
    for (int kb = 0; kb < K / 1024; kb++) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
        for (int h = 0; h < 2; h++) { // the two 512-K chunks of the block
            const int64_t ks = 2 * kb + h;
            const uint8_t * chunk = base + ((int64_t) rt * S + ks) * aitd_chunk_bytes(k);
#pragma unroll 1
            for (int G = 0; G < 16; G++) { // 4 tiles = 32 K
                const uint8_t * grp = chunk + (int64_t) G * 128 * k;
                const int64_t k0 = 512 * ks + 32 * G;
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    uint32_t a[4];
                    int lv0[4], lv1[4];
                    aitd_lev4(grp, 2 * m, lane, u, k, mul, lv0);
                    aitd_lev4(grp, 2 * m + 1, lane, u, k, mul, lv1);
                    // levels j = 0,1: row g (h8 = 0), ce 0,1; j = 2,3: row g + 8
                    const int64_t kc0 = k0 + 16 * m + 2 * tq, kc1 = kc0 + 8;
                    const __half2 c0 = *(const __half2 *) (cs + kc0), c1 = *(const __half2 *) (cs + kc1);
                    const __half2 p0 = __floats2half2_rn((float) lv0[0], (float) lv0[1]), p1 = __floats2half2_rn((float) lv0[2], (float) lv0[3]);
                    const __half2 p2 = __floats2half2_rn((float) lv1[0], (float) lv1[1]), p3 = __floats2half2_rn((float) lv1[2], (float) lv1[3]);
                    a[0] = *(const uint32_t *) &(const __half2 &) __hmul2(p0, c0);
                    a[1] = *(const uint32_t *) &(const __half2 &) __hmul2(p1, c0);
                    a[2] = *(const uint32_t *) &(const __half2 &) __hmul2(p2, c1);
                    a[3] = *(const uint32_t *) &(const __half2 &) __hmul2(p3, c1);
                    const int tok = g < NT ? g : 0;
                    uint32_t b0 = *(const uint32_t *) (x + (int64_t) tok * K + kc0), b1 = *(const uint32_t *) (x + (int64_t) tok * K + kc1);
                    if (g >= NT) b0 = b1 = 0;
                    aitd_mma16816(c, a, b0, b1);
                }
            }
        }
        // rows (u, g) and (u, g + 8): the scale slots of the 1024-K block (chunk 2 kb + u)
        const uint8_t * sc = base + ((int64_t) rt * S + 2 * kb + u) * aitd_chunk_bytes(k) + 2048 * (int64_t) k;
        const float r0 = __half2float(*(const __half *) (sc + 2 * g)) * (1.f / 64.f);
        const float r1 = __half2float(*(const __half *) (sc + 2 * (g + 8))) * (1.f / 64.f);
        tot[0] += r0 * c[0]; tot[1] += r0 * c[1]; tot[2] += r1 * c[2]; tot[3] += r1 * c[3];
    }
    // C fragment: c0 = (row g, token 2 tq), c1 = (row g, token 2 tq + 1), c2 / c3 = row g + 8
    out[0] = tot[0]; out[1] = tot[1]; out[2] = tot[2]; out[3] = tot[3];
}

// compact column scales (K halves) of a tile-type tensor
__global__ void aitd_cs_extract(const uint8_t * base, __half * cs, int64_t K, int k) {
    const int64_t kc = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (kc < K) cs[kc] = *(const __half *) (base + ((int64_t) ((kc & 511) >> 4) * (K / 512) + (kc >> 9)) * aitd_chunk_bytes(k) + 2048 * (int64_t) k + 32 + 2 * (kc & 15));
}

// AIT5_U (unit layout, no column scale): tile16 = slab = 16 rows. A = fp16 level (exact), the row's fp16 scale (per 128-K unit)
// multiplies the unit's fp32 sum. Weight n of a unit tile = row n / 8, column n % 8.
template <int NT>
__device__ __forceinline__ void ai5d_gemv_unit(const uint8_t * base, int64_t K, uint32_t mul, int slab, const __half * x, float (&out)[4]) {
    constexpr int k = 5;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    float tot[4] = {0.f, 0.f, 0.f, 0.f};
    const uint8_t * sb = base + (int64_t) slab * (K / 128) * ai5d_unit_bytes(k);
#pragma unroll 1
    for (int un = 0; un < K / 128; un++) {
        const uint8_t * blk = sb + (int64_t) un * ai5d_unit_bytes(k);
        const uint32_t * tw = (const uint32_t *) (blk + 32);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
        for (int m = 0; m < 8; m++) { // tiles 2 m, 2 m + 1
            const uint32_t * t0 = tw + (2 * m) * 4 * k, * t1 = t0 + 4 * k;
            const int n0 = 8 * g + 2 * tq, n1 = n0 + 64;
            uint32_t a[4];
            a[0] = *(const uint32_t *) &(const __half2 &) __floats2half2_rn((float) ai5d_level(t0, k, mul, n0), (float) ai5d_level(t0, k, mul, n0 + 1));
            a[1] = *(const uint32_t *) &(const __half2 &) __floats2half2_rn((float) ai5d_level(t0, k, mul, n1), (float) ai5d_level(t0, k, mul, n1 + 1));
            a[2] = *(const uint32_t *) &(const __half2 &) __floats2half2_rn((float) ai5d_level(t1, k, mul, n0), (float) ai5d_level(t1, k, mul, n0 + 1));
            a[3] = *(const uint32_t *) &(const __half2 &) __floats2half2_rn((float) ai5d_level(t1, k, mul, n1), (float) ai5d_level(t1, k, mul, n1 + 1));
            const int64_t kc0 = 128 * (int64_t) un + 16 * m + 2 * tq;
            const int tok = g < NT ? g : 0;
            uint32_t b0 = *(const uint32_t *) (x + (int64_t) tok * K + kc0), b1 = *(const uint32_t *) (x + (int64_t) tok * K + kc0 + 8);
            if (g >= NT) b0 = b1 = 0;
            aitd_mma16816(c, a, b0, b1);
        }
        const float r0 = __half2float(((const __half *) blk)[g]), r1 = __half2float(((const __half *) blk)[g + 8]);
        tot[0] += r0 * c[0]; tot[1] += r0 * c[1]; tot[2] += r1 * c[2]; tot[3] += r1 * c[3];
    }
    out[0] = tot[0]; out[1] = tot[1]; out[2] = tot[2]; out[3] = tot[3];
}

// ---- int8 tensor-core helpers (decode ait_wv.cuh and prefill ait_prefill.cu) ----
// weights through L1 (ld.global.ca): the megakernel is built with -dlcm=cg, which makes plain loads bypass L1; the two tile warps of a
// 32-row chunk and the overlapping lo / hi words of a lane read the same lines
__device__ __forceinline__ uint32_t aq_ld32(const uint32_t * p) { uint32_t v; asm("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ float aq_ldh(const void * p) { unsigned short v; asm("ld.global.ca.u16 %0, [%1];" : "=h"(v) : "l"(p)); return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint2 aq_ld64(const void * p) { uint2 v; asm("ld.global.ca.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v; }
__device__ __forceinline__ uint4 aq_ld128(const void * p) { uint4 v; asm("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ void aq_mma(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 4 dp4a results (0x40 per byte: 64 * bytesum + 128 - 128 * 256, see AQ_ACC0) -> 4 int8 levels: byte j = byte 1 of s_j
__device__ __forceinline__ uint32_t aq_pack4(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    return __byte_perm(__byte_perm(s0, s1, 0x0051), __byte_perm(s2, s3, 0x5100), 0x7610);
}
#define AQ_ACC0 0xFFFF8080u // 128 - 128 * 256: dp4a with 0x40 per byte = 64 * bytesum; the level (bytesum + 2) >> 2 - 128 is then byte 1
template <int K>
__device__ __forceinline__ void aq_dec_tile(const uint4 lo4, const uint4 hi4, int wsh, uint32_t (&a)[4]) {
    constexpr uint32_t mul = K == 3 ? AITD_MUL3 : AITD_MUL4;
    const uint32_t los[4] = {lo4.x, lo4.y, lo4.z, lo4.w}, his[4] = {hi4.x, hi4.y, hi4.z, hi4.w};
    uint32_t s[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t w = __funnelshift_r(los[i], his[i], wsh);
#pragma unroll
        for (int j = 0; j < 4; j++) s[i][j] = __dp4a(((w >> (j * K)) & 0xFFFFu) * mul, 0x40404040u, AQ_ACC0);
    }
    a[0] = aq_pack4(s[0][0], s[0][1], s[1][0], s[1][1]);
    a[1] = aq_pack4(s[0][2], s[0][3], s[1][2], s[1][3]);
    a[2] = aq_pack4(s[2][0], s[2][1], s[3][0], s[3][1]);
    a[3] = aq_pack4(s[2][2], s[2][3], s[3][2], s[3][3]);
}
__device__ __forceinline__ void aq_s2(const uint32_t * tw, const Ai5Pos & p, uint32_t & s0, uint32_t & s1) {
    const uint64_t c = ((uint64_t) aq_ld32(tw + p.w) << 32) | aq_ld32(tw + p.w1);
    s0 = __dp4a(((uint32_t) (c >> (48 - p.sh)) & 0xFFFFu) * AITD_MUL4, 0x40404040u, AQ_ACC0);
    s1 = __dp4a(((uint32_t) (c >> (43 - p.sh)) & 0xFFFFu) * AITD_MUL4, 0x40404040u, AQ_ACC0);
}
// AIT5_U: the 32-K block G (0..3) of the unit at tw: a[0..1] = K 16 (2G) .., a[2..3] = K 16 (2G + 1) ..
__device__ __forceinline__ void aq_dec_u(const uint32_t * tw, int G, const Ai5Pos & pa, const Ai5Pos & pb, uint32_t (&a)[4]) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const uint32_t * t0 = tw + 2 * (2 * G + h) * 20, * t1 = t0 + 20;
        uint32_t x0, x1, x2, x3;
        // (t0: K 0..7 of the 16, t1: K 8..15; pa: row g, pb: row g + 8)
        aq_s2(t0, pa, x0, x1); aq_s2(t1, pa, x2, x3);
        a[2 * h] = aq_pack4(x0, x1, x2, x3);
        aq_s2(t0, pb, x0, x1); aq_s2(t1, pb, x2, x3);
        a[2 * h + 1] = aq_pack4(x0, x1, x2, x3);
    }
}
