#include <cstdlib>
#include "ait_prefill.h"
// AI+ (AIT3 / AIT4 / AIT5_U) batched prefill GEMM: y[tok][row] (+)= W[row] . x[tok]. Its own translation unit: the decode
// core (ait_acc, ait_wv.cuh) needs the wv shims below, nothing else of the megakernel.
//   1. ait_rot_rows: x (fp16, K per row) -> the rotated rows (sign flip, FWHT256, / 16) with row stride AIT_XLD
//   2. ait_gemm_k: a block of 8 warps = 8 row tiles of 16 rows, 64 tokens (8 m16n8 groups); each warp decodes its tile's
//      weights once per 64 tokens (fp16 tensor cores, the row scale folded into the fp32 sum) and stores / adds y
// The activations are rounded once (to fp16 after the rotation); the decode pass's int8 round trip does not apply here.
#define MK_SHARED_NS
#include "mk_shared.cuh"
#include "dspine.h"

__constant__ Bufs cBt; // (ait_wv.cuh's decode-pass users; unused here)
#define cB cBt
template <int T, bool X16> struct WvCfg {
    static constexpr int TPG = X16 ? 4 : 8;
    static constexpr int NG = (T + TPG - 1) / TPG;
};
struct WvXS {
    const int8_t * xa;
    const int8_t * xb;
    int ld;
    const float4 * md;
    const float2 * mx;
    const float4 * mx4;
    const __half * xr = nullptr;
};
template <int NG> struct WvAcc { float v[NG][4]; };
#include "ait_wv.cuh"

namespace {

constexpr int AP_TOK = 64; // tokens per block
constexpr int AP_WARPS = 8;

// one warp per (token, 256-K block): 8 elements per lane
__global__ void __launch_bounds__(256) ait_rot_rows(const __half * __restrict__ x, int K, int ntok, __half * __restrict__ xr) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nb = K >> 8;
    for (int job = blockIdx.x * 8 + warp; job < ntok * nb; job += gridDim.x * 8) {
        const int t = job / nb, b = job - t * nb;
        const int e0 = 256 * b + 8 * lane;
        float v[8]; // (x rows are in the prefill's hperm order within each 16-group: element e sits at slot hperm(e))
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e = e0 + i, lo = e & 15;
            const int slot = (((lo >> 1) & 1) << 2 | ((lo >> 2) & 3)) << 1 | (lo & 1);
            v[i] = __half2float(x[(int64_t) t * K + (e & ~15) + slot]);
        }
        aitd_rot256(v, lane);
        __half2 o[4];
#pragma unroll
        for (int i = 0; i < 4; i++) o[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
        *reinterpret_cast<uint4 *>(xr + (int64_t) t * AIT_XLD + e0) = *reinterpret_cast<uint4 *>(o);
    }
}

__global__ void __launch_bounds__(256) ait_gemm_k(AitMat m, int ntile, const __half * __restrict__ xr, int ntok, float * __restrict__ y, int ldy, int add) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int tile = blockIdx.x * AP_WARPS + warp;
    if (tile >= ntile) return;
    const int t0 = blockIdx.y * AP_TOK, nt = min(AP_TOK, ntok - t0);
    float tot[AP_TOK / 8][4];
    ait_acc<AP_TOK / 8>(m, tile, 0, m.K >> 8, xr + (int64_t) t0 * AIT_XLD, nt, tot);
#pragma unroll
    for (int n = 0; n < AP_TOK / 8; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int tok = t0 + 8 * n + 2 * tq + (c & 1);
            const int row = 16 * tile + g + 8 * (c >> 1);
            if (tok < ntok) {
                float * p = y + (int64_t) tok * ldy + row;
                *p = add ? *p + tot[n][c] : tot[n][c];
            }
        }
}

// ---- int8 tensor-core path -----------------------------------------------------------------------------------------
// A = the trellis levels as int8 (exact, no column scale), B = the rotated activations times the column scale, quantised to
// int8 per (token, 256-K block). mma.m16n8k32.s8 with the K order permuted so that a lane's decoded weights are its A bytes:
// lane (g, tq) holds, per 32-K block, the columns {2tq, 2tq + 1, 8 + 2tq, 9 + 2tq} (a0 / a1: rows g / g + 8) and
// {16 + 2tq, 17 + 2tq, 24 + 2tq, 25 + 2tq} (a2 / a3). The B operand is stored in the same order.
// B layout (ait_quant_rows writes it, ait_gemm_q copies it to shared memory unchanged):
//   XQ[((blk256 * NTG + tok / 8) * 8 + G) * 256 + (tok % 8) * 32 + 8 tq + 4 s + 2 h + e]  (G: 32-K block of the 256 block;
//   s: 16-K half, h: 8-K half, e: pair element), NTG = ntokpad / 8; SA[blk256 * ntokpad + tok] = absmax / 127.
constexpr int AQ_TOK = 128;    // tokens per block (16 n groups)
constexpr int AQ_SLICE = AQ_TOK * 256; // bytes of one 256-K slice of B

__global__ void __launch_bounds__(256) ait_quant_rows(const __half * __restrict__ x, const __half * __restrict__ cs, int K, int ntok, int ntokpad,
                                                      int8_t * __restrict__ XQ, float * __restrict__ SA) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nb = K >> 8, ntg = ntokpad >> 3;
    for (int job = blockIdx.x * 8 + warp; job < ntokpad * nb; job += gridDim.x * 8) {
        const int t = job / nb, b = job - t * nb;
        const int e0 = 256 * b + 8 * lane;
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e = e0 + i, lo = e & 15;
            const int slot = (((lo >> 1) & 1) << 2 | ((lo >> 2) & 3)) << 1 | (lo & 1);
            v[i] = t < ntok ? __half2float(x[(int64_t) t * K + (e & ~15) + slot]) : 0.f;
        }
        aitd_rot256(v, lane);
        if (cs) {
            const uint4 cr = *reinterpret_cast<const uint4 *>(cs + e0);
            const __half * ch = reinterpret_cast<const __half *>(&cr);
#pragma unroll
            for (int i = 0; i < 8; i++) v[i] *= __half2float(ch[i]);
        }
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) a = fmaxf(a, fabsf(v[i]));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        const float id = a > 0.f ? 127.f / a : 0.f;
        if (lane == 0) SA[(int64_t) b * ntokpad + t] = a * (1.f / 127.f);
        int8_t * dst = XQ + ((int64_t) (b * ntg + (t >> 3)) * 8 + (lane >> 2)) * 256 + (t & 7) * 32 + 4 * ((lane >> 1) & 1) + 2 * (lane & 1);
#pragma unroll
        for (int mq = 0; mq < 4; mq++) {
            const uint32_t lo = (uint8_t) (int8_t) __float2int_rn(v[2 * mq] * id), hi = (uint8_t) (int8_t) __float2int_rn(v[2 * mq + 1] * id);
            *reinterpret_cast<uint16_t *>(dst + 8 * mq) = (uint16_t) (lo | (hi << 8));
        }
    }
}

__device__ __forceinline__ void aq_cp16(void * smem, const void * gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t) __cvta_generic_to_shared(smem)), "l"(gmem));
}

#ifndef AQ_TPW
#define AQ_TPW 1 // 16-row tiles per warp (each B fragment from shared memory feeds that many MMAs)
#endif
#ifndef AQ_NTG
#define AQ_NTG 16 // 8-token groups per block (block tokens = 8 * AQ_NTG), all in each warp's registers
#endif
#ifndef AQ_PF5
#define AQ_PF5 2 // AIT5_U: 128-K units ahead whose weights are prefetched to L1
#endif
#ifndef AQ_MINB
#define AQ_MINB 1 // blocks per SM the register budget is set for
#endif
constexpr int AQ_BT = 8 * AQ_NTG;
constexpr int AQ_SMEM_USE = 2 * AQ_BT * 256 + 2 * AQ_BT * 4;
static_assert(AQ_TOK % AQ_BT == 0, "token padding");
// KT: 3 / 4 (AIT3 / AIT4 tile types), 5 (AIT5_U). Block (bx, by) = 8 * TPW tiles of 16 rows x (128 / TPW) tokens; 1-D grid, token
// blocks fastest (the weights of a row group are read by the concurrently running token blocks).
template <int KT, int TPW>
__global__ void __launch_bounds__(256, AQ_MINB) ait_gemm_q(AitMat m, int ntile, const int8_t * __restrict__ XQ, const float * __restrict__ SA, int ntok, int ntokpad,
                                                     float * __restrict__ y, int ldy, int add) {
    constexpr int BT = AQ_BT, NTG = AQ_NTG, SLICE = BT * 256;
    extern __shared__ __align__(16) uint8_t aq_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, tid = threadIdx.x;
    const int nby = ntokpad / BT, by = blockIdx.x % nby, bx = blockIdx.x / nby;
    const int tile0 = (bx * 8 + warp) * TPW, t0 = by * BT;
    const int K = m.K, NS = K >> 8, ntg = ntokpad >> 3;
    float * const sS = reinterpret_cast<float *>(aq_sm + 2 * SLICE); // [2][BT]
    auto issue = [&](int s, int buf) {
        const uint8_t * src = reinterpret_cast<const uint8_t *>(XQ) + ((int64_t) s * ntg + (t0 >> 3)) * 2048;
#pragma unroll
        for (int i = 0; i < SLICE / 16 / 256; i++) aq_cp16(aq_sm + buf * SLICE + (tid + 256 * i) * 16, src + (tid + 256 * i) * 16);
        if (tid < BT / 4) aq_cp16(sS + buf * BT + 4 * tid, SA + (int64_t) s * ntokpad + t0 + 4 * tid);
        asm volatile("cp.async.commit_group;");
    };
    float tot[TPW][NTG][4];
    int c[TPW][NTG][4];
#pragma unroll
    for (int j = 0; j < TPW; j++)
#pragma unroll
        for (int n = 0; n < NTG; n++) tot[j][n][0] = tot[j][n][1] = tot[j][n][2] = tot[j][n][3] = 0.f;
#pragma unroll
    for (int j = 0; j < TPW; j++)
#pragma unroll
        for (int n = 0; n < NTG; n++) c[j][n][0] = c[j][n][1] = c[j][n][2] = c[j][n][3] = 0;
    // lane constants of the weight stream (tile = tile0 + j)
    const int S = K / 512;
    const int64_t cb = aitd_chunk_bytes(KT == 5 ? 4 : KT);
    int rt[TPW], u[TPW], wi[TPW], wsh[TPW], wi1[TPW];
    bool live[TPW];
    const uint8_t * sbase5[TPW];
#pragma unroll
    for (int j = 0; j < TPW; j++) {
        const int tile = tile0 + j;
        live[j] = tile < ntile;
        rt[j] = tile >> 1;
        u[j] = tile & 1;
        const int wpos = (8 * lane + 4 * u[j]) * (KT == 5 ? 4 : KT);
        wi[j] = wpos >> 5;
        wsh[j] = wpos & 31;
        wi1[j] = wi[j] + 1 == 8 * KT ? 0 : wi[j] + 1;
        sbase5[j] = m.w + (int64_t) tile * (K / 128) * ai5d_unit_bytes(5);
    }
    const Ai5Pos pa = ai5d_pos(8 * g + 2 * tq), pb = ai5d_pos(8 * g + 2 * tq + 64);
    auto flush = [&](int j, const float r0, const float r1, const float * ss) {
#pragma unroll
        for (int n = 0; n < NTG; n++) {
            const float2 sa = *reinterpret_cast<const float2 *>(ss + 8 * n + 2 * tq);
            tot[j][n][0] += (float) c[j][n][0] * (r0 * sa.x); tot[j][n][1] += (float) c[j][n][1] * (r0 * sa.y);
            tot[j][n][2] += (float) c[j][n][2] * (r1 * sa.x); tot[j][n][3] += (float) c[j][n][3] * (r1 * sa.y);
            c[j][n][0] = c[j][n][1] = c[j][n][2] = c[j][n][3] = 0;
        }
    };
    auto mmas = [&](const uint8_t * sb, int G, const uint32_t (&a)[TPW][4]) {
#pragma unroll
        for (int n = 0; n < NTG; n++) {
            const uint2 b = *reinterpret_cast<const uint2 *>(sb + (n * 8 + G) * 256 + lane * 8);
#pragma unroll
            for (int j = 0; j < TPW; j++) aq_mma(c[j][n], a[j], b.x, b.y);
        }
    };
    issue(0, 0);
    float r0[TPW], r1[TPW];
    uint4 lo4[TPW], hi4[TPW];
#pragma unroll
    for (int j = 0; j < TPW; j++) {
        r0[j] = r1[j] = 0.f;
        lo4[j] = hi4[j] = make_uint4(0, 0, 0, 0);
        if constexpr (KT != 5) {
            if (live[j]) {
                const uint8_t * chunk = m.w + (int64_t) rt[j] * S * cb;
                lo4[j] = *reinterpret_cast<const uint4 *>(chunk + 16 * wi[j]);
                hi4[j] = *reinterpret_cast<const uint4 *>(chunk + 16 * wi1[j]);
                const uint8_t * sc = m.w + ((int64_t) rt[j] * S + u[j]) * cb + 2048 * (int64_t) KT;
                r0[j] = __half2float(*reinterpret_cast<const __half *>(sc + 2 * g)) * (1.f / 64.f);
                r1[j] = __half2float(*reinterpret_cast<const __half *>(sc + 2 * (g + 8))) * (1.f / 64.f);
            }
        }
    }
    for (int s = 0; s < NS; s++) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (s + 1 < NS) issue(s + 1, (s + 1) & 1);
        const uint8_t * sb = aq_sm + (s & 1) * SLICE;
        const float * ss = sS + (s & 1) * BT;
        if constexpr (KT != 5) {
#pragma unroll
            for (int G = 0; G < 8; G++) {
                uint32_t a[TPW][4];
#pragma unroll
                for (int j = 0; j < TPW; j++) {
                    if (!live[j]) { a[j][0] = a[j][1] = a[j][2] = a[j][3] = 0; continue; }
                    aq_dec_tile<KT>(lo4[j], hi4[j], wsh[j], a[j]);
                    // the next group's words (the next slice's first group after the last one: loaded before the barrier)
                    const int sn = G < 7 ? s : s + 1, Gn = G < 7 ? G + 1 : 0;
                    if (sn < NS) {
                        const uint8_t * grp = m.w + ((int64_t) rt[j] * S + (sn >> 1)) * cb + (int64_t) (sn & 1) * 8 * 128 * KT + (int64_t) Gn * 128 * KT;
                        lo4[j] = *reinterpret_cast<const uint4 *>(grp + 16 * wi[j]);
                        hi4[j] = *reinterpret_cast<const uint4 *>(grp + 16 * wi1[j]);
                    }
                }
                mmas(sb, G, a);
            }
#pragma unroll
            for (int j = 0; j < TPW; j++) flush(j, r0[j], r1[j], ss);
#pragma unroll
            for (int j = 0; j < TPW; j++) { // (the scales of the next slice: read here, off the next barrier's critical path)
                if (live[j] && s + 1 < NS && ((s + 1) & 3) == 0) {
                    const int sn = s + 1;
                    const uint8_t * sc = m.w + ((int64_t) rt[j] * S + 2 * (sn >> 2) + u[j]) * cb + 2048 * (int64_t) KT;
                    r0[j] = __half2float(*reinterpret_cast<const __half *>(sc + 2 * g)) * (1.f / 64.f);
                    r1[j] = __half2float(*reinterpret_cast<const __half *>(sc + 2 * (g + 8))) * (1.f / 64.f);
                }
            }
        } else {
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const uint32_t * tw[TPW];
#pragma unroll
                for (int j = 0; j < TPW; j++) {
                    if (AQ_PF5 > 0 && live[j] && lane < (ai5d_unit_bytes(5) + 127) / 128 && 2 * s + hh + AQ_PF5 < 2 * NS)
                        asm volatile("prefetch.global.L1 [%0];" ::"l"(sbase5[j] + (int64_t) (2 * s + hh + AQ_PF5) * ai5d_unit_bytes(5) + 128 * lane));
                    const uint8_t * blk = sbase5[j] + (int64_t) (2 * s + hh) * ai5d_unit_bytes(5);
                    tw[j] = reinterpret_cast<const uint32_t *>(blk + 32);
                    r0[j] = __half2float(reinterpret_cast<const __half *>(blk)[g]);
                    r1[j] = __half2float(reinterpret_cast<const __half *>(blk)[g + 8]);
                }
#pragma unroll
                for (int G = 0; G < 4; G++) {
                    uint32_t a[TPW][4];
#pragma unroll
                    for (int j = 0; j < TPW; j++) {
                        if (!live[j]) { a[j][0] = a[j][1] = a[j][2] = a[j][3] = 0; continue; }
                        aq_dec_u(tw[j], G, pa, pb, a[j]);
                    }
                    mmas(sb, hh * 4 + G, a);
                }
#pragma unroll
                for (int j = 0; j < TPW; j++) flush(j, r0[j], r1[j], ss); // (the row scale changes per 128-K unit)
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; j++) {
        if (!live[j]) continue;
        const int tile = tile0 + j;
#pragma unroll
        for (int n = 0; n < NTG; n++)
#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int tok = t0 + 8 * n + 2 * tq + (cc & 1);
                const int row = 16 * tile + g + 8 * (cc >> 1);
                if (tok < ntok) {
                    float * p = y + (int64_t) tok * ldy + row;
                    *p = add ? *p + tot[j][n][cc] : tot[j][n][cc];
                }
            }
    }
}
} // namespace

// y[tok * ldy + row] (+)= sum_k W[row][k] x[tok * K + k], tokens 0 .. ntok - 1, in pieces of AIT_PF_PIECE tokens.
// xrot: scratch of AIT_PF_PIECE * AIT_XLD halves.
void ait_prefill_gemm(const void * w, const void * cs, int type, int K, int N, const void * x, int ntok, void * xrot, float * y, int ldy,
                      bool add, cudaStream_t st) {
    const int ntile = N / 16;
    const AitMat m{static_cast<const uint8_t *>(w), static_cast<const __half *>(cs), type, K};
    static const bool fp16 = getenv("MK_AIT_FP16") && atoi(getenv("MK_AIT_FP16")) != 0;
    if (!fp16) { // int8 tensor cores (ait_gemm_q); MK_AIT_FP16=1: the fp16 path below
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute(ait_gemm_q<3, AQ_TPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, AQ_SMEM_USE);
            cudaFuncSetAttribute(ait_gemm_q<4, AQ_TPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, AQ_SMEM_USE);
            cudaFuncSetAttribute(ait_gemm_q<5, AQ_TPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, AQ_SMEM_USE);
            attr = true;
        }
        for (int p0 = 0; p0 < ntok; p0 += AIT_PF_PIECE) {
            const int n = ntok - p0 < AIT_PF_PIECE ? ntok - p0 : AIT_PF_PIECE;
            const int npad = (n + AQ_TOK - 1) / AQ_TOK * AQ_TOK;
            int8_t * xq = static_cast<int8_t *>(xrot);
            float * sa = reinterpret_cast<float *>(static_cast<uint8_t *>(xrot) + (((size_t) npad * K + 255) & ~(size_t) 255));
            const int jobs = (npad * (K >> 8) + 7) / 8;
            ait_quant_rows<<<jobs > 4096 ? 4096 : jobs, 256, 0, st>>>(static_cast<const __half *>(x) + (int64_t) p0 * K, static_cast<const __half *>(cs), K, n, npad, xq, sa);
            const int nblk = ((ntile + 8 * AQ_TPW - 1) / (8 * AQ_TPW)) * (npad / AQ_BT);
            float * yy = y + (int64_t) p0 * ldy;
            if (type == T_AIT3) ait_gemm_q<3, AQ_TPW><<<nblk, 256, AQ_SMEM_USE, st>>>(m, ntile, xq, sa, n, npad, yy, ldy, add ? 1 : 0);
            else if (type == T_AIT4) ait_gemm_q<4, AQ_TPW><<<nblk, 256, AQ_SMEM_USE, st>>>(m, ntile, xq, sa, n, npad, yy, ldy, add ? 1 : 0);
            else ait_gemm_q<5, AQ_TPW><<<nblk, 256, AQ_SMEM_USE, st>>>(m, ntile, xq, sa, n, npad, yy, ldy, add ? 1 : 0);
        }
        return;
    }
    for (int p0 = 0; p0 < ntok; p0 += AIT_PF_PIECE) {
        const int n = ntok - p0 < AIT_PF_PIECE ? ntok - p0 : AIT_PF_PIECE;
        const int jobs = (n * (K >> 8) + 7) / 8;
        ait_rot_rows<<<jobs > 4096 ? 4096 : jobs, 256, 0, st>>>(static_cast<const __half *>(x) + (int64_t) p0 * K, K, n,
                                                                  static_cast<__half *>(xrot));
        ait_gemm_k<<<dim3((ntile + AP_WARPS - 1) / AP_WARPS, (n + AP_TOK - 1) / AP_TOK), 256, 0, st>>>(m, ntile, static_cast<const __half *>(xrot), n,
                                                                                                      y + (int64_t) p0 * ldy, ldy, add ? 1 : 0);
    }
}
