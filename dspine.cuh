// dspine.cuh: DSpine drafter kernels and their launchers (declared in dspine.h). Included by qwen38_mk.cu at global scope,
// after its anonymous namespace. Block rows are at most BMAX = 16, so every GEMM here is a skinny one: a warp per output
// row, the 16 accumulators in registers, the weights read once from DRAM, X from L2.
#include <algorithm>
#include <climits>
#pragma once
#include "dspine.h"

namespace dsp {

__device__ __forceinline__ float hf(const uint8_t * p) { return __half2float(*reinterpret_cast<const __half *>(p)); }

// Q4_K / Q6_K scale and min (ggml get_scale_min_k4)
__device__ __forceinline__ void sm_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// element e of a row of type ty (codebook and lm_head rows: a few rows per draft pass)
__device__ float deq(int ty, const uint8_t * row, int e) {
    switch (ty) {
        case TY_F32: return reinterpret_cast<const float *>(row)[e];
        case TY_BF16: return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(row)[e]);
        case TY_Q8_0: {
            const uint8_t * b = row + 34 * (e >> 5);
            return hf(b) * (float) reinterpret_cast<const int8_t *>(b + 2)[e & 31];
        }
        case TY_Q4K: {
            const uint8_t * b = row + 144 * (e >> 8);
            const int r = e & 255, j = r >> 6, rr = r & 63, l = rr & 31;
            int sc, mn;
            sm_k4(2 * j + (rr >> 5), b + 4, sc, mn);
            const int by = b[16 + 32 * j + l], q = rr < 32 ? (by & 15) : (by >> 4);
            return hf(b) * sc * q - hf(b + 2) * mn;
        }
        default: { // TY_Q6K
            const uint8_t * b = row + 210 * (e >> 8);
            const int r = e & 255, h = r >> 7, rr = r & 127, l = rr & 31, quad = rr >> 5;
            const int qlv = b[64 * h + l + (quad & 1) * 32], low = quad < 2 ? (qlv & 15) : (qlv >> 4);
            const int hi = (b[128 + 32 * h + l] >> (2 * quad)) & 3;
            const int q = (low | (hi << 4)) - 32;
            return hf(b + 208) * (float) reinterpret_cast<const int8_t *>(b + 192)[8 * h + (l >> 4) + 2 * quad] * q;
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// sum over the block (blockDim.x a multiple of 32, <= 1024); red: 32 floats of shared memory; safe to call again
__device__ __forceinline__ float block_sum(float v, float * red) {
    v = warp_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    float s = 0.f;
    for (int i = 0; i < nw; i++) s += red[i];
    return s;
}

// ---- Q4_K on tensor cores: Y[m][n] (+)= sum_k W[n][k] X[m][k], m < 16 (one grid.y step), K a multiple of 256 ----
// A warp owns 8 weight rows (the MMA's n) and all 16 X rows (its m): the X fragments are loaded once per 8 weight rows
// (the scalar kernel loaded them once per weight row: it was bound by those loads, 26 ms per draft). Operands are fp16 (the
// dequantised weight rounds once, the sums are fp32). The k index inside an MMA is arbitrary as long as A and B agree: lane
// t of a row's 4 lanes takes the 4 consecutive weights of its own bytes (8 bytes of a 64-group per lane: word w, nibble nib),
// so every weight load is one aligned uint2 and the X fragment is two float4.
__device__ __forceinline__ unsigned pk2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}
__device__ __forceinline__ void mma_f16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// the 4 warps' K slices of the same 8 rows: summed by warp 0 (C: rows g8 / g8 + 8 of X, columns 2t, 2t + 1 = weight rows n0 + 2t, + 1)
__device__ __forceinline__ void tc_reduce_store(const float (&acc)[4], int ks, int lane, int g8, int t, int n0, int N, int Mc, float * Y, int ldy, int add) {
    __shared__ float red[4][32][4];
#pragma unroll
    for (int i = 0; i < 4; i++) red[ks][lane][i] = acc[i];
    __syncthreads();
    if (ks != 0) return;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float v = red[0][lane][i] + red[1][lane][i] + red[2][lane][i] + red[3][lane][i];
        const int m = g8 + 8 * (i >> 1), n = n0 + 2 * t + (i & 1);
        if (m < Mc && n < N) Y[(size_t) m * ldy + n] = add ? Y[(size_t) m * ldy + n] + v : v;
    }
}
__global__ void __launch_bounds__(128) k_gemm_q4k_tc(const uint8_t * __restrict__ W, size_t rb, int K, int N, const float * __restrict__ X,
                                                     int ldx, float * __restrict__ Y, int ldy, int M, int add) {
    const int lane = threadIdx.x & 31, g8 = lane >> 2, t = lane & 3, ks = threadIdx.x >> 5; // ks: the warp's K slice
    const int n0 = blockIdx.x * 8;
    if (n0 >= N) return; // (the whole block: uniform before the barrier below)
    const int m0 = blockIdx.y * 16, Mc = min(16, M - m0);
    X += (size_t) m0 * ldx;
    Y += (size_t) m0 * ldy;
    const uint8_t * row = W + (size_t) min(n0 + g8, N - 1) * rb; // this lane's weight row (rows past N: a valid row, never stored)
    const float * const x0 = X + (size_t) g8 * ldx + 8 * t;       // X rows g8 and g8 + 8 (rows >= Mc read as 0)
    const float * const x1 = X + (size_t) (g8 + 8) * ldx + 8 * t;
    const bool v0 = g8 < Mc, v1 = g8 + 8 < Mc;
    const int nb = K >> 8;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    uint4 h, hn;
    uint2 q[4], qn[4];
    auto ld = [&](int b, uint4 & hv, uint2(&qv)[4]) {
        const uint8_t * bp = row + (size_t) 144 * b;
        hv = *reinterpret_cast<const uint4 *>(bp); // d, dmin, 12 scale bytes
#pragma unroll
        for (int j = 0; j < 4; j++) qv[j] = *reinterpret_cast<const uint2 *>(bp + 16 + 32 * j + 8 * t);
    };
    const int bs = ks * nb / 4, be = (ks + 1) * nb / 4; // this warp's blocks of the row's K
    if (bs < be) ld(bs, h, q);
    for (int b = bs; b < be; b++) {
        if (b + 1 < be) ld(b + 1, hn, qn);
        const float d = __half2float(__ushort_as_half((unsigned short) (h.x & 65535u)));
        const float dm = __half2float(__ushort_as_half((unsigned short) (h.x >> 16)));
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float am[2], bm[2]; // lo / hi nibble: d * scale, dmin * min (ggml get_scale_min_k4 of index 2j, 2j + 1)
#pragma unroll
            for (int nib = 0; nib < 2; nib++) {
                const int id = 2 * j + nib, sh = 8 * (id & 3);
                const uint32_t s0 = h.y >> sh, s1 = h.z >> sh, s2 = h.w >> sh;
                float sc, mn;
                if (id < 4) {
                    sc = (float) (s0 & 63);
                    mn = (float) (s1 & 63);
                } else {
                    sc = (float) ((s2 & 15) | (((s0 >> 6) & 3) << 4));
                    mn = (float) (((s2 >> 4) & 15) | (((s1 >> 6) & 3) << 4));
                }
                am[nib] = d * sc;
                bm[nib] = dm * mn;
            }
#pragma unroll
            for (int w = 0; w < 2; w++) {
                const uint32_t wd = w ? q[j].y : q[j].x;
#pragma unroll
                for (int nib = 0; nib < 2; nib++) {
                    const int k = b * 256 + 64 * j + 4 * w + 32 * nib; // + 8t in x0 / x1
                    const float4 xa = v0 ? *reinterpret_cast<const float4 *>(x0 + k) : make_float4(0.f, 0.f, 0.f, 0.f);
                    const float4 xb = v1 ? *reinterpret_cast<const float4 *>(x1 + k) : make_float4(0.f, 0.f, 0.f, 0.f);
                    const unsigned a[4] = {pk2(xa.x, xa.y), pk2(xb.x, xb.y), pk2(xa.z, xa.w), pk2(xb.z, xb.w)};
                    float v[4];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const uint32_t by = (wd >> (8 * i)) & 255u;
                        v[i] = am[nib] * (float) (nib ? by >> 4 : by & 15u) - bm[nib];
                    }
                    mma_f16(acc, a, pk2(v[0], v[1]), pk2(v[2], v[3]));
                }
            }
        }
        h = hn;
#pragma unroll
        for (int j = 0; j < 4; j++) q[j] = qn[j];
    }
    tc_reduce_store(acc, ks, lane, g8, t, n0, N, Mc, Y, ldy, add);
}

// ---- Q6_K / BF16 on tensor cores: the same skeleton and k permutation as k_gemm_q4k_tc ----
// BF16: K a multiple of 64, rows 16-byte aligned (a lane loads 8 weights at a time). Q6_K: K a multiple of 256 (210-byte blocks: 2-byte loads).
template <int TY>
__global__ void __launch_bounds__(128) k_gemm_tc2(const uint8_t * __restrict__ W, size_t rb, int K, int N, const float * __restrict__ X,
                                                  int ldx, float * __restrict__ Y, int ldy, int M, int add) {
    const int lane = threadIdx.x & 31, g8 = lane >> 2, t = lane & 3, ks = threadIdx.x >> 5;
    const int n0 = blockIdx.x * 8;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * 16, Mc = min(16, M - m0);
    X += (size_t) m0 * ldx;
    Y += (size_t) m0 * ldy;
    const uint8_t * row = W + (size_t) min(n0 + g8, N - 1) * rb;
    const float * const x0 = X + (size_t) g8 * ldx;
    const float * const x1 = X + (size_t) (g8 + 8) * ldx;
    const bool v0 = g8 < Mc, v1 = g8 + 8 < Mc;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    // one MMA: this lane's 4 weights w are at k + 8t .. + 3 of the row
    auto mma4 = [&](int k, float w0, float w1, float w2, float w3) {
        const float4 xa = v0 ? *reinterpret_cast<const float4 *>(x0 + k + 8 * t) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 xb = v1 ? *reinterpret_cast<const float4 *>(x1 + k + 8 * t) : make_float4(0.f, 0.f, 0.f, 0.f);
        const unsigned a[4] = {pk2(xa.x, xa.y), pk2(xb.x, xb.y), pk2(xa.z, xa.w), pk2(xb.z, xb.w)};
        mma_f16(acc, a, pk2(w0, w1), pk2(w2, w3));
    };
    if constexpr (TY == TY_BF16) {
        const int nc = K >> 6, cs = ks * nc / 4, ce = (ks + 1) * nc / 4;
        for (int c = cs; c < ce; c++) {
            const int k = c * 64;
            const uint4 u[2] = {*reinterpret_cast<const uint4 *>(row + 2 * (k + 8 * t)), *reinterpret_cast<const uint4 *>(row + 2 * (k + 32 + 8 * t))};
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const unsigned wd[4] = {u[i].x, u[i].y, u[i].z, u[i].w};
#pragma unroll
                for (int w = 0; w < 2; w++)
                    mma4(k + 32 * i + 4 * w, __uint_as_float(wd[2 * w] << 16), __uint_as_float(wd[2 * w] & 0xffff0000u),
                         __uint_as_float(wd[2 * w + 1] << 16), __uint_as_float(wd[2 * w + 1] & 0xffff0000u));
            }
        }
    } else { // TY_Q6K
        const int nb = K >> 8, bs = ks * nb / 4, be = (ks + 1) * nb / 4;
        for (int b = bs; b < be; b++) {
            const uint8_t * bp = row + (size_t) 210 * b;
            const float d = hf(bp + 208);
#pragma unroll 1
            for (int h = 0; h < 2; h++) {
                uint16_t ql[2][4], qh[4]; // ql[0]: bytes 8t.. of ql[64h ..], ql[1]: of ql[64h + 32 ..], qh: of qh[32h ..]
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    ql[0][i] = *reinterpret_cast<const uint16_t *>(bp + 64 * h + 8 * t + 2 * i);
                    ql[1][i] = *reinterpret_cast<const uint16_t *>(bp + 64 * h + 32 + 8 * t + 2 * i);
                    qh[i] = *reinterpret_cast<const uint16_t *>(bp + 128 + 32 * h + 8 * t + 2 * i);
                }
#pragma unroll
                for (int gi = 0; gi < 4; gi++) {
                    const float ds = d * (float) reinterpret_cast<const int8_t *>(bp + 192)[8 * h + 2 * gi + (t >> 1)];
#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        float v[4];
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int e = 4 * w + i, sh = 8 * (e & 1);
                            const int lo = (ql[gi & 1][e >> 1] >> sh) & 255, hb = (qh[e >> 1] >> sh) & 255;
                            v[i] = ds * (float) (((gi < 2 ? lo & 15 : lo >> 4) | (((hb >> (2 * gi)) & 3) << 4)) - 32);
                        }
                        mma4(256 * b + 128 * h + 32 * gi + 4 * w, v[0], v[1], v[2], v[3]);
                    }
                }
            }
        }
    }
    tc_reduce_store(acc, ks, lane, g8, t, n0, N, Mc, Y, ldy, add);
}

// ---- Y[m][n] (+)= sum_k W[n][k] X[m][k], m < M (16 rows per grid.y step) ----
template <int TY>
__global__ void __launch_bounds__(256) k_gemm(const uint8_t * __restrict__ W, size_t rb, int K, int N, const float * __restrict__ X,
                                              int ldx, float * __restrict__ Y, int ldy, int M, int add) {
    const int lane = threadIdx.x & 31, n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= N) return; // (whole warps: no block barrier below)
    const int m0 = blockIdx.y * 16, Mc = min(16, M - m0);
    X += (size_t) m0 * ldx;
    Y += (size_t) m0 * ldy;
    const uint8_t * row = W + (size_t) n * rb;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.f;
    if constexpr (TY == TY_Q4K) {
        const int j = lane >> 3, c = lane & 7; // 64-value group j, 4 bytes c of its 32
        for (int b = 0; b < (K >> 8); b++) {
            const uint8_t * bp = row + 144 * b;
            int s1, m1, s2, m2;
            sm_k4(2 * j, bp + 4, s1, m1);
            sm_k4(2 * j + 1, bp + 4, s2, m2);
            const float d = hf(bp), dm = hf(bp + 2);
            const float a1 = d * s1, b1 = dm * m1, a2 = d * s2, b2 = dm * m2;
            const uint32_t w = *reinterpret_cast<const uint32_t *>(bp + 16 + 32 * j + 4 * c);
            float lo[4], hi[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                lo[i] = a1 * (float) ((w >> (8 * i)) & 15) - b1;
                hi[i] = a2 * (float) ((w >> (8 * i + 4)) & 15) - b2;
            }
            const int k0 = b * 256 + 64 * j + 4 * c;
#pragma unroll
            for (int m = 0; m < 16; m++) {
                if (m < Mc) {
                    const float4 xl = *reinterpret_cast<const float4 *>(X + (size_t) m * ldx + k0);
                    const float4 xh = *reinterpret_cast<const float4 *>(X + (size_t) m * ldx + k0 + 32);
                    acc[m] += lo[0] * xl.x + lo[1] * xl.y + lo[2] * xl.z + lo[3] * xl.w + hi[0] * xh.x + hi[1] * xh.y + hi[2] * xh.z + hi[3] * xh.w;
                }
            }
        }
    } else if constexpr (TY == TY_Q6K) {
        const int h = lane >> 4, c = lane & 15; // half h of the block, values l = 2c, 2c + 1 of its 32
        for (int b = 0; b < (K >> 8); b++) {
            const uint8_t * bp = row + 210 * b;
            const float d = hf(bp + 208);
            const int8_t * sc = reinterpret_cast<const int8_t *>(bp + 192) + 8 * h;
            const uint16_t a0 = *reinterpret_cast<const uint16_t *>(bp + 64 * h + 2 * c);
            const uint16_t a1 = *reinterpret_cast<const uint16_t *>(bp + 64 * h + 2 * c + 32);
            const uint16_t qq = *reinterpret_cast<const uint16_t *>(bp + 128 + 32 * h + 2 * c);
            const int is = c >> 3;
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int b0 = (a0 >> (8 * u)) & 255, b1 = (a1 >> (8 * u)) & 255, qh = (qq >> (8 * u)) & 255;
                const float v1 = d * sc[is] * (float) (((b0 & 15) | ((qh & 3) << 4)) - 32);
                const float v2 = d * sc[is + 2] * (float) (((b1 & 15) | (((qh >> 2) & 3) << 4)) - 32);
                const float v3 = d * sc[is + 4] * (float) (((b0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                const float v4 = d * sc[is + 6] * (float) (((b1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
                const int k = b * 256 + 128 * h + 2 * c + u;
#pragma unroll
                for (int m = 0; m < 16; m++)
                    if (m < Mc) {
                        const float * xr = X + (size_t) m * ldx + k;
                        acc[m] += v1 * xr[0] + v2 * xr[32] + v3 * xr[64] + v4 * xr[96];
                    }
            }
        }
    } else if constexpr (TY == TY_Q8_0) {
        for (int b = 0; b < (K >> 5); b++) {
            const uint8_t * bp = row + 34 * b;
            const float v = hf(bp) * (float) reinterpret_cast<const int8_t *>(bp + 2)[lane];
            const int k = b * 32 + lane;
#pragma unroll
            for (int m = 0; m < 16; m++)
                if (m < Mc) acc[m] += v * X[(size_t) m * ldx + k];
        }
    } else { // TY_BF16
        for (int k = lane * 2; k < K; k += 64) {
            const __nv_bfloat162 w2 = *reinterpret_cast<const __nv_bfloat162 *>(row + 2 * k);
            const float w0 = __low2float(w2), w1 = __high2float(w2);
#pragma unroll
            for (int m = 0; m < 16; m++)
                if (m < Mc) {
                    const float2 xv = *reinterpret_cast<const float2 *>(X + (size_t) m * ldx + k);
                    acc[m] += w0 * xv.x + w1 * xv.y;
                }
        }
    }
#pragma unroll
    for (int m = 0; m < 16; m++) {
        const float v = warp_sum(acc[m]);
        if (lane == 0 && m < Mc) Y[(size_t) m * ldy + n] = add ? Y[(size_t) m * ldy + n] + v : v;
    }
}

void gemm(const Mat & w, const float * X, int ldx, float * Y, int ldy, int M, bool add, cudaStream_t st) {
    const bool al = ldx % 4 == 0 && ((uintptr_t) X & 15) == 0;
    const dim3 gt((w.N + 7) / 8, (M + 15) / 16);
    if (w.type == TY_Q4K && w.K % 256 == 0 && w.rb % 16 == 0 && al && ((uintptr_t) w.w & 15) == 0) {
        k_gemm_q4k_tc<<<gt, 128, 0, st>>>(w.w, w.rb, w.K, w.N, X, ldx, Y, ldy, M, add);
        return;
    }
    if (w.type == TY_Q6K && w.K % 256 == 0 && al) {
        k_gemm_tc2<TY_Q6K><<<gt, 128, 0, st>>>(w.w, w.rb, w.K, w.N, X, ldx, Y, ldy, M, add);
        return;
    }
    if (w.type == TY_BF16 && w.K % 64 == 0 && w.rb % 16 == 0 && al && ((uintptr_t) w.w & 15) == 0) {
        k_gemm_tc2<TY_BF16><<<gt, 128, 0, st>>>(w.w, w.rb, w.K, w.N, X, ldx, Y, ldy, M, add);
        return;
    }
    const dim3 g((w.N + 7) / 8, (M + 15) / 16);
    switch (w.type) {
        case TY_Q4K: k_gemm<TY_Q4K><<<g, 256, 0, st>>>(w.w, w.rb, w.K, w.N, X, ldx, Y, ldy, M, add); break;
        case TY_Q6K: k_gemm<TY_Q6K><<<g, 256, 0, st>>>(w.w, w.rb, w.K, w.N, X, ldx, Y, ldy, M, add); break;
        case TY_Q8_0: k_gemm<TY_Q8_0><<<g, 256, 0, st>>>(w.w, w.rb, w.K, w.N, X, ldx, Y, ldy, M, add); break;
        default: k_gemm<TY_BF16><<<g, 256, 0, st>>>(w.w, w.rb, w.K, w.N, X, ldx, Y, ldy, M, add); break;
    }
}

// ---- RMS norm of rows (n values, row stride ld): y = x / sqrt(mean(x^2) + eps) * w (w null: 1); rms_out[row] = sqrt(mean(x^2)).
// y may be x
__global__ void __launch_bounds__(256) k_rms(const float * x, const float * w, float * y, float * rms_out, int n, int ld, float eps) {
    __shared__ float red[32];
    const float * xs = x + (size_t) blockIdx.x * ld;
    float * ys = y + (size_t) blockIdx.x * ld;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xs[i] * xs[i];
    s = block_sum(s, red);
    const float inv = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) ys[i] = xs[i] * inv * (w ? w[i] : 1.f);
    if (rms_out && threadIdx.x == 0) rms_out[blockIdx.x] = sqrtf(s / n);
}
void rms(const float * x, const float * w, float * y, float * rms_out, int rows, int n, int ld, float eps, cudaStream_t st) {
    k_rms<<<rows, 256, 0, st>>>(x, w, y, rms_out, n, ld, eps);
}

__global__ void k_bf2f(const __nv_bfloat16 * src, float * dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = __bfloat162float(src[i]);
}
void bf2f(const __nv_bfloat16 * src, float * dst, size_t n, cudaStream_t st) {
    k_bf2f<<<(unsigned) std::min<size_t>((n + 255) / 256, 4096), 256, 0, st>>>(src, dst, n);
}

__global__ void k_stage(const float * x, __nv_bfloat16 * dst, int j, int T) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < (size_t) T * D; i += (size_t) gridDim.x * blockDim.x) {
        const size_t t = i / D, e = i - t * D;
        dst[(t * NF + j) * D + e] = __float2bfloat16_rn(x[i]);
    }
}
void stage(const float * x, __nv_bfloat16 * dst, int j, int T, cudaStream_t st) {
    k_stage<<<(unsigned) std::min<size_t>(((size_t) T * D + 255) / 256, 4096), 256, 0, st>>>(x, dst, j, T);
}

// ---- head norm + NeoX RoPE (all HD = 128 dims rotated: pairs (i, i + 64), theta^(-i / 64)) ----
// thread i < 128 holds element i of one head in s[]; w: the head's norm weight; pos: its position
__device__ __forceinline__ void head_norm_rope(float * s, float v, const float * w, int pos, float theta, float eps, float * red) {
    const int i = threadIdx.x;
    const float ss = block_sum(v * v, red);
    s[i] = v * rsqrtf(ss / HD + eps) * w[i];
    __syncthreads();
    if (i < HD / 2) {
        const float inv = powf(theta, -(float) i / (HD / 2));
        float sn, cs;
        sincosf((float) pos * inv, &sn, &cs);
        const float a = s[i], b = s[i + HD / 2];
        s[i] = a * cs - b * sn;
        s[i + HD / 2] = b * cs + a * sn;
    }
    __syncthreads();
}

// context K / V of M rows at positions pos0 + m into the ring (one layer): K normed and rotated, V as is
__global__ void __launch_bounds__(HD) k_inject_kv(const float * kb, const float * vb, const float * kn, __half * kr, __half * vr, int pos0, float theta, float eps) {
    __shared__ float s[HD], red[32];
    const int m = blockIdx.x, h = blockIdx.y, i = threadIdx.x, pos = pos0 + m;
    head_norm_rope(s, kb[(size_t) m * KD + h * HD + i], kn, pos, theta, eps, red);
    const size_t o = ((size_t) (pos & (WIN - 1)) * NKV + h) * HD + i;
    kr[o] = __float2half_rn(s[i]);
    vr[o] = __float2half_rn(vb[(size_t) m * KD + h * HD + i]);
}
void inject_kv(const float * kb, const float * vb, const float * kn, __half * kr, __half * vr, int pos0, int M, float theta, float eps, cudaStream_t st) {
    k_inject_kv<<<dim3(M, NKV), HD, 0, st>>>(kb, vb, kn, kr, vr, pos0, theta, eps);
}

__global__ void k_set_tags(int * tags, int pos0, int M) {
    if (threadIdx.x < M) tags[(pos0 + threadIdx.x) & (WIN - 1)] = pos0 + threadIdx.x;
}
void set_tags(int * tags, int pos0, int M, cudaStream_t st) { k_set_tags<<<1, 32, 0, st>>>(tags, pos0, M); }

// the block's q (NH heads) and k (NKV heads): norm (q_norm / k_norm) and rotation at position pos0 + m, in place
__global__ void __launch_bounds__(HD) k_qk_prep(float * q, float * kb, const float * qn, const float * kn, int pos0, float theta, float eps) {
    __shared__ float s[HD], red[32];
    const int m = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    float * p = h < NH ? q + (size_t) m * QD + h * HD : kb + (size_t) m * KD + (h - NH) * HD;
    head_norm_rope(s, p[i], h < NH ? qn : kn, pos0 + m, theta, eps, red);
    p[i] = s[i];
}
void qk_prep(float * q, float * kb, const float * qn, const float * kn, int pos0, int M, float theta, float eps, cudaStream_t st) {
    k_qk_prep<<<dim3(M, NH + NKV), HD, 0, st>>>(q, kb, qn, kn, pos0, theta, eps);
}

// ---- attention of block row m (position n + m), head h over: the ring's positions in (n + m - WIN, n) whose tag matches,
// and the block's rows 0 .. m (causal). GQA 4:1. Warp w takes every 4th key; online softmax, merged through shared memory.
__global__ void __launch_bounds__(128) k_attn(const float * q, const float * kb, const float * vb, const __half * kr, const __half * vr,
                                              const int * tags, int n, float * out) {
    __shared__ float s_m[4], s_l[4], s_a[4][HD];
    const int m = blockIdx.x, h = blockIdx.y, hk = h >> 2, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int qpos = n + m;
    const float4 qq = *reinterpret_cast<const float4 *>(q + (size_t) m * QD + h * HD + lane * 4);
    const float scale = rsqrtf((float) HD);
    float mx = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    auto step = [&](float dot, float v0, float v1, float v2, float v3) {
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s), c = __expf(mx - mn), p = __expf(s - mn);
        l = l * c + p;
        a0 = a0 * c + p * v0;
        a1 = a1 * c + p * v1;
        a2 = a2 * c + p * v2;
        a3 = a3 * c + p * v3;
        mx = mn;
    };
    for (int p = max(0, qpos - WIN + 1) + warp; p < n; p += 4) {
        const int slot = p & (WIN - 1);
        if (tags[slot] != p) continue;
        const size_t o = ((size_t) slot * NKV + hk) * HD + lane * 4;
        const uint2 kk = *reinterpret_cast<const uint2 *>(kr + o), vv = *reinterpret_cast<const uint2 *>(vr + o);
        const float2 k01 = __half22float2(*reinterpret_cast<const __half2 *>(&kk.x)), k23 = __half22float2(*reinterpret_cast<const __half2 *>(&kk.y));
        const float2 v01 = __half22float2(*reinterpret_cast<const __half2 *>(&vv.x)), v23 = __half22float2(*reinterpret_cast<const __half2 *>(&vv.y));
        step(qq.x * k01.x + qq.y * k01.y + qq.z * k23.x + qq.w * k23.y, v01.x, v01.y, v23.x, v23.y);
    }
    for (int j = warp; j <= m; j += 4) {
        const float4 k = *reinterpret_cast<const float4 *>(kb + (size_t) j * KD + hk * HD + lane * 4);
        const float4 v = *reinterpret_cast<const float4 *>(vb + (size_t) j * KD + hk * HD + lane * 4);
        step(qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w, v.x, v.y, v.z, v.w);
    }
    s_a[warp][lane * 4] = a0;
    s_a[warp][lane * 4 + 1] = a1;
    s_a[warp][lane * 4 + 2] = a2;
    s_a[warp][lane * 4 + 3] = a3;
    if (lane == 0) {
        s_m[warp] = mx;
        s_l[warp] = l;
    }
    __syncthreads();
    const float M = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
    float tl = 0.f, ta = 0.f;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const float e = __expf(s_m[w] - M); // (a warp without a key: -inf, weight 0)
        tl += s_l[w] * e;
        ta += s_a[w][threadIdx.x] * e;
    }
    out[(size_t) m * QD + h * HD + threadIdx.x] = ta / tl;
}
void attn(const float * q, const float * kb, const float * vb, const __half * kr, const __half * vr, const int * tags, int n, int M, float * out, cudaStream_t st) {
    k_attn<<<dim3(M, NH), 128, 0, st>>>(q, kb, vb, kr, vr, tags, n, out);
}

__global__ void k_swiglu(float * g, const float * u, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float a = g[i];
        g[i] = a / (1.f + __expf(-a)) * u[i];
    }
}
void swiglu(float * g, const float * u, size_t n, cudaStream_t st) {
    k_swiglu<<<(unsigned) std::min<size_t>((n + 255) / 256, 4096), 256, 0, st>>>(g, u, n);
}

// ---- DSpine bridge ----
// codes: row 0 = the anchor token's codebook row x sqrt(CODE); rows t >= 1 = sqrt(CODE) * craw[t] / max(|craw[t]|, eps)
__global__ void __launch_bounds__(256) k_codes(const float * craw, const uint8_t * cb, int cty, size_t crb, const int * toks, float * out, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const float sq = sqrtf((float) CODE);
    if (t == 0) {
        const uint8_t * row = cb + (size_t) toks[0] * crb;
        for (int e = threadIdx.x; e < CODE; e += blockDim.x) out[e] = sq * deq(cty, row, e);
        return;
    }
    float s = 0.f;
    for (int e = threadIdx.x; e < CODE; e += blockDim.x) s += craw[(size_t) t * CODE + e] * craw[(size_t) t * CODE + e];
    s = block_sum(s, red);
    const float f = sq / fmaxf(sqrtf(s), eps);
    for (int e = threadIdx.x; e < CODE; e += blockDim.x) out[(size_t) t * CODE + e] = craw[(size_t) t * CODE + e] * f;
}
void codes(const float * craw, const Mat & cb, const int * toks, int M, float * out, float eps, cudaStream_t st) {
    k_codes<<<M, 256, 0, st>>>(craw, cb.w, cb.type, cb.rb, toks, out, eps);
}

// msg[t - 1] = sigmoid(q[t] + s[t - 1] + bias) * v[t - 1], t = 1 .. M - 1
__global__ void k_msg(const float * q, const float * s, const float * v, const float * bias, float * out) {
    const int t = blockIdx.x + 1;
    for (int r = threadIdx.x; r < RANK; r += blockDim.x) {
        const float g = 1.f / (1.f + __expf(-(q[(size_t) t * RANK + r] + s[(size_t) (t - 1) * RANK + r] + bias[r])));
        out[(size_t) (t - 1) * RANK + r] = g * v[(size_t) (t - 1) * RANK + r];
    }
}
void msg(const float * q, const float * s, const float * v, const float * bias, int M, float * out, cudaStream_t st) {
    k_msg<<<M - 1, 256, 0, st>>>(q, s, v, bias, out);
}

// x[t] += delta[t - 1] * rms[t] (t >= 1); raw[t] = x[t] before the write
__global__ void k_apply(float * x, const float * delta, const float * rms_, float * raw) {
    const int t = blockIdx.x;
    for (int e = threadIdx.x; e < D; e += blockDim.x) {
        const float v = x[(size_t) t * D + e];
        raw[(size_t) t * D + e] = v;
        if (t > 0) x[(size_t) t * D + e] = v + delta[(size_t) (t - 1) * D + e] * rms_[t];
    }
}
void apply(float * x, const float * delta, const float * rms_, int M, float * raw, cudaStream_t st) { k_apply<<<M, 256, 0, st>>>(x, delta, rms_, raw); }

// ---- top-16 per row of the draft logits (descending, ties: lower id first). Block per row; each thread keeps the best of
// its strided slice and rescans it only when it won a round. Writes -inf over the winners.
__global__ void __launch_bounds__(1024) k_topk(float * logits, int V, int * cand) {
    __shared__ float sv[32];
    __shared__ int si[32], win;
    float * L = logits + (size_t) blockIdx.x * V;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float lm = -INFINITY;
    int li = INT_MAX;
    auto scan = [&] {
        lm = -INFINITY;
        li = INT_MAX;
        for (int i = tid; i < V; i += 1024)
            if (L[i] > lm) {
                lm = L[i];
                li = i;
            }
    };
    scan();
    for (int r = 0; r < TOPK; r++) {
        float v = lm;
        int ix = li;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
            if (ov > v || (ov == v && oi < ix)) v = ov, ix = oi;
        }
        if (lane == 0) sv[warp] = v, si[warp] = ix;
        __syncthreads();
        if (warp == 0) {
            v = sv[lane];
            ix = si[lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
                if (ov > v || (ov == v && oi < ix)) v = ov, ix = oi;
            }
            if (lane == 0) {
                cand[(size_t) blockIdx.x * TOPK + r] = ix;
                win = ix;
            }
        }
        __syncthreads();
        if ((win & 1023) == tid) {
            L[win] = -INFINITY;
            scan();
        }
        __syncthreads();
    }
}
void topk(float * logits, int V, int rows, int * cand, cudaStream_t st) { k_topk<<<rows, 1024, 0, st>>>(logits, V, cand); }

// ---- lattice ----
// row (t - 1) * TOPK + j: the predecessor code of position t: the anchor's codebook row for t = 1, else candidate j of row t - 1
__global__ void __launch_bounds__(256) k_pred_codes(const int * cand, const int * toks, const uint8_t * cb, int cty, size_t crb, float * out) {
    const int r = blockIdx.x, t = r / TOPK + 1, j = r % TOPK;
    const int tok = t == 1 ? toks[0] : cand[(size_t) (t - 1) * TOPK + j];
    const uint8_t * row = cb + (size_t) tok * crb;
    const float sq = sqrtf((float) CODE);
    for (int e = threadIdx.x; e < CODE; e += blockDim.x) out[(size_t) r * CODE + e] = sq * deq(cty, row, e);
}
void pred_codes(const int * cand, const int * toks, const Mat & cb, int M, float * out, cudaStream_t st) {
    k_pred_codes<<<(M - 1) * TOPK, 256, 0, st>>>(cand, toks, cb.w, cb.type, cb.rb, out);
}

// msg = sigmoid(Wp pred + (Wg c_t) + bias) * (Wv pred)
__global__ void k_lat_msg(const float * gp, const float * pv, const float * qst, const float * bias, float * out) {
    const int r = blockIdx.x, t = r / TOPK + 1;
    for (int k = threadIdx.x; k < RANK; k += blockDim.x) {
        const float g = 1.f / (1.f + __expf(-(gp[(size_t) r * RANK + k] + qst[(size_t) t * RANK + k] + bias[k])));
        out[(size_t) r * RANK + k] = g * pv[(size_t) r * RANK + k];
    }
}
void lat_msg(const float * gp, const float * pv, const float * qst, const float * bias, int M, float * out, cudaStream_t st) {
    k_lat_msg<<<(M - 1) * TOPK, 256, 0, st>>>(gp, pv, qst, bias, out);
}

// refined = output_norm(raw[t] + delta * rms[t])
__global__ void __launch_bounds__(256) k_refine(const float * raw, const float * rms_, const float * delta, const float * outn, float eps, float * out) {
    __shared__ float red[32];
    constexpr int PT = D / 256;
    const int r = blockIdx.x, t = r / TOPK + 1;
    float y[PT];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < PT; i++) {
        const int e = threadIdx.x + 256 * i;
        y[i] = raw[(size_t) t * D + e] + delta[(size_t) r * D + e] * rms_[t];
        s += y[i] * y[i];
    }
    s = block_sum(s, red);
    const float inv = rsqrtf(s / D + eps);
#pragma unroll
    for (int i = 0; i < PT; i++) {
        const int e = threadIdx.x + 256 * i;
        out[(size_t) r * D + e] = y[i] * inv * outn[e];
    }
}
void refine(const float * raw, const float * rms_, const float * delta, const float * outn, int M, float eps, float * out, cudaStream_t st) {
    k_refine<<<(M - 1) * TOPK, 256, 0, st>>>(raw, rms_, delta, outn, eps, out);
}

// table[t - 1][j * TOPK + k] = lm_head_row(candidate k of row t) . refined(t, predecessor j)   (predecessor-major)
__global__ void __launch_bounds__(256) k_table(const uint8_t * hrows, size_t hrb, int hty, const float * refined, float * out) {
    __shared__ float hs[D];
    const int r = blockIdx.x, t1 = r / TOPK, k = r % TOPK, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint8_t * row = hrows + (size_t) r * hrb;
    for (int e = threadIdx.x; e < D; e += blockDim.x) hs[e] = deq(hty, row, e);
    __syncthreads();
    for (int j = warp; j < TOPK; j += 8) {
        const float * rf = refined + (size_t) (t1 * TOPK + j) * D;
        float s = 0.f;
        for (int e = lane; e < D; e += 32) s += hs[e] * rf[e];
        s = warp_sum(s);
        if (lane == 0) out[(size_t) t1 * (TOPK * TOPK) + j * TOPK + k] = s;
    }
}
void table(const uint8_t * hrows, size_t hrb, int hty, const float * refined, int M, float * out, cudaStream_t st) {
    k_table<<<(M - 1) * TOPK, 256, 0, st>>>(hrows, hrb, hty, refined, out);
}

// greedy walk of the table: predecessor 0; position i takes the best candidate k given the previous pick
__global__ void k_walk(const int * cand, const float * table_, int M, int * drafts) {
    if (threadIdx.x) return;
    int pred = 0;
    for (int i = 1; i < M; i++) {
        float best = -INFINITY;
        int bk = 0;
        for (int k = 0; k < TOPK; k++) {
            const float s = table_[(size_t) (i - 1) * (TOPK * TOPK) + pred * TOPK + k];
            if (s > best) best = s, bk = k;
        }
        drafts[i - 1] = cand[(size_t) i * TOPK + bk];
        pred = bk;
    }
}
void walk(const int * cand, const float * table_, int M, int * drafts, cudaStream_t st) { k_walk<<<1, 32, 0, st>>>(cand, table_, M, drafts); }

} // namespace dsp
