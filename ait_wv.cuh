#pragma once
#include "ait_core.cuh"
#include <type_traits>

// super-blocks [sb0, sb0 + nsb) of 16-row tile `tile` of an AI+ matrix: the fp32 sums of T tokens (NGM groups of 8; the
// m16n8 C layout) of the rotated activations xr (row stride AIT_XLD halves) into tot
#ifndef AIT_PFG
#define AIT_PFG 0 // 32-K groups ahead whose weights the int8 tile path prefetches to L2 (0: none)
#endif
#ifndef AIT_PFD
#define AIT_PFD 0 // super-blocks (256 K) ahead whose weights ait_acc prefetches to L2 (0: none)
#endif
__device__ __forceinline__ void ait_pfl2(const void * p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
template <int NGM>
__device__ __forceinline__ void ait_acc(const AitMat m, int tile, int sb0, int nsb, const __half * const xr, int T, float (&tot)[NGM][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int n = 0; n < NGM; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) tot[n][c] = 0.f;
    const int64_t K = m.K;
    if (m.type == T_AIT5U) {
        const uint8_t * const sbase = m.w + (int64_t) tile * (K / 128) * ai5d_unit_bytes(5);
        const Ai5Pos pa = ai5d_pos(8 * g + 2 * tq), pb = ai5d_pos(8 * g + 2 * tq + 64); // (lane constants)
#pragma unroll 1
        for (int un = 2 * sb0; un < 2 * (sb0 + nsb); un++) {
            if constexpr (AIT_PFD > 0) {
                if (un + 2 * AIT_PFD < (int) (K >> 7) && 128 * lane < ai5d_unit_bytes(5))
                    ait_pfl2(sbase + (int64_t) (un + 2 * AIT_PFD) * ai5d_unit_bytes(5) + 128 * lane);
            }
            const uint8_t * blk = sbase + (int64_t) un * ai5d_unit_bytes(5);
            const uint32_t * tw = reinterpret_cast<const uint32_t *>(blk + 32);
            float c[NGM][4];
#pragma unroll
            for (int n = 0; n < NGM; n++)
#pragma unroll
                for (int i = 0; i < 4; i++) c[n][i] = 0.f;
#pragma unroll 2
            for (int mm = 0; mm < 8; mm++) {
                const uint32_t * t0 = tw + (2 * mm) * 20, * t1 = t0 + 20;
                const int n0 = 8 * g + 2 * tq, n1 = n0 + 64;
                uint32_t a[4];
                a[0] = ai5d_lev2h(t0, pa, AITD_MUL4);
                a[1] = ai5d_lev2h(t0, pb, AITD_MUL4);
                a[2] = ai5d_lev2h(t1, pa, AITD_MUL4);
                a[3] = ai5d_lev2h(t1, pb, AITD_MUL4);
                const int64_t kc0 = 128 * (int64_t) un + 16 * mm + 2 * tq;
#pragma unroll
                for (int n = 0; n < NGM; n++) {
                    const int tok = 8 * n + g;
                    uint32_t b0 = 0, b1 = 0;
                    if (tok < T) {
                        b0 = *reinterpret_cast<const uint32_t *>(xr + (int64_t) tok * AIT_XLD + kc0);
                        b1 = *reinterpret_cast<const uint32_t *>(xr + (int64_t) tok * AIT_XLD + kc0 + 8);
                    }
                    aitd_mma16816(c[n], a, b0, b1);
                }
            }
            const float r0 = __half2float(reinterpret_cast<const __half *>(blk)[g]);
            const float r1 = __half2float(reinterpret_cast<const __half *>(blk)[g + 8]);
#pragma unroll
            for (int n = 0; n < NGM; n++) {
                tot[n][0] += r0 * c[n][0]; tot[n][1] += r0 * c[n][1];
                tot[n][2] += r1 * c[n][2]; tot[n][3] += r1 * c[n][3];
            }
        }
    } else {
        auto run = [&](auto KC) {
        constexpr int k = decltype(KC)::value; // (AIT3: 3, AIT4: 4: a compile time constant, the stream positions fold)
        constexpr uint32_t mul = k == 3 ? AITD_MUL3 : AITD_MUL4;
        const int64_t S = K / 512;
        const int rt = tile >> 1, u = tile & 1;
        const int64_t cb = aitd_chunk_bytes(k);
        const int wpos = (8 * lane + 4 * u) * k, wi = wpos >> 5, wsh = wpos & 31, wi1 = wi + 1 == 8 * k ? 0 : wi + 1; // (lane constants)
#pragma unroll 1
        for (int sb = sb0; sb < sb0 + nsb; sb++) {
            const uint8_t * chunk = m.w + ((int64_t) rt * S + (sb >> 1)) * cb + (int64_t) (sb & 1) * 8 * 128 * k;
            if constexpr (AIT_PFD > 0) {
                if (sb + AIT_PFD < (int) (K >> 8)) {
                    const int sp = sb + AIT_PFD;
                    if (128 * lane < 1024 * k) ait_pfl2(m.w + ((int64_t) rt * S + (sp >> 1)) * cb + (int64_t) (sp & 1) * 8 * 128 * k + 128 * lane);
                }
            }
            float c[NGM][4];
#pragma unroll
            for (int n = 0; n < NGM; n++)
#pragma unroll
                for (int i = 0; i < 4; i++) c[n][i] = 0.f;
#pragma unroll 2
            for (int G = 0; G < 8; G++) { // 4 tiles = 32 K
                const uint8_t * grp = chunk + (int64_t) G * 128 * k;
                const uint4 lo4 = *reinterpret_cast<const uint4 *>(grp + 16 * wi), hi4 = *reinterpret_cast<const uint4 *>(grp + 16 * wi1);
                const uint32_t los[4] = {lo4.x, lo4.y, lo4.z, lo4.w}, his[4] = {hi4.x, hi4.y, hi4.z, hi4.w};
#pragma unroll
                for (int mm = 0; mm < 2; mm++) {
                    uint32_t lh0[2], lh1[2];
                    aitd_lev2h<k>(__funnelshift_r(los[2 * mm], his[2 * mm], wsh), mul, lh0);
                    aitd_lev2h<k>(__funnelshift_r(los[2 * mm + 1], his[2 * mm + 1], wsh), mul, lh1);
                    const int64_t kc0 = 256 * (int64_t) sb + 32 * G + 16 * mm + 2 * tq, kc1 = kc0 + 8;
                    const __half2 c0 = *reinterpret_cast<const __half2 *>(m.d + kc0), c1 = *reinterpret_cast<const __half2 *>(m.d + kc1);
                    const __half2 p0 = __hmul2(*reinterpret_cast<const __half2 *>(&lh0[0]), c0);
                    const __half2 p1 = __hmul2(*reinterpret_cast<const __half2 *>(&lh0[1]), c0);
                    const __half2 p2 = __hmul2(*reinterpret_cast<const __half2 *>(&lh1[0]), c1);
                    const __half2 p3 = __hmul2(*reinterpret_cast<const __half2 *>(&lh1[1]), c1);
                    uint32_t a[4];
                    a[0] = *reinterpret_cast<const uint32_t *>(&p0);
                    a[1] = *reinterpret_cast<const uint32_t *>(&p1);
                    a[2] = *reinterpret_cast<const uint32_t *>(&p2);
                    a[3] = *reinterpret_cast<const uint32_t *>(&p3);
#pragma unroll
                    for (int n = 0; n < NGM; n++) {
                        const int tok = 8 * n + g;
                        uint32_t b0 = 0, b1 = 0;
                        if (tok < T) {
                            b0 = *reinterpret_cast<const uint32_t *>(xr + (int64_t) tok * AIT_XLD + kc0);
                            b1 = *reinterpret_cast<const uint32_t *>(xr + (int64_t) tok * AIT_XLD + kc1);
                        }
                        aitd_mma16816(c[n], a, b0, b1);
                    }
                }
            }
            // rows (u, g) and (u, g + 8): the scale slots of the 1024-K block (chunk 2 (sb / 4) + u)
            const uint8_t * sc = m.w + ((int64_t) rt * S + 2 * (sb >> 2) + u) * cb + 2048 * (int64_t) k;
            const float r0 = __half2float(*reinterpret_cast<const __half *>(sc + 2 * g)) * (1.f / 64.f);
            const float r1 = __half2float(*reinterpret_cast<const __half *>(sc + 2 * (g + 8))) * (1.f / 64.f);
#pragma unroll
            for (int n = 0; n < NGM; n++) {
                tot[n][0] += r0 * c[n][0]; tot[n][1] += r0 * c[n][1];
                tot[n][2] += r1 * c[n][2]; tot[n][3] += r1 * c[n][3];
            }
        }
        };
        if (m.type == T_AIT3) run(std::integral_constant<int, 3>{});
        else run(std::integral_constant<int, 4>{});
    }
}

// The int8 version of ait_acc: xq / sa are one slot of the scratch (ait_rot_fill: layout above, column scale folded in).
// The sums are exact int32 per 256-K super-block (AIT5_U: per 128-K unit), then scaled by the row scale and the token's scale.
template <int NGM>
__device__ __forceinline__ void ait_acc_q(const AitMat m, int tile, int sb0, int nsb, const int8_t * const xq, const float * const sa, float (&tot)[NGM][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int n = 0; n < NGM; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) tot[n][c] = 0.f;
    const int64_t K = m.K;
    auto flush = [&](int (&c)[NGM][4], const float r0, const float r1, int sb) {
#pragma unroll
        for (int n = 0; n < NGM; n++) {
            const uint2 sv = aq_ld64(sa + sb * 16 + 8 * n + 2 * tq); const float2 s = make_float2(__uint_as_float(sv.x), __uint_as_float(sv.y));
            tot[n][0] += (float) c[n][0] * (r0 * s.x); tot[n][1] += (float) c[n][1] * (r0 * s.y);
            tot[n][2] += (float) c[n][2] * (r1 * s.x); tot[n][3] += (float) c[n][3] * (r1 * s.y);
            c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0;
        }
    };
    auto mmas = [&](int (&c)[NGM][4], const int8_t * sbp, int G, const uint32_t (&a)[4]) {
#pragma unroll
        for (int n = 0; n < NGM; n++) {
            const uint2 b = aq_ld64(sbp + (n * 8 + G) * 256 + lane * 8);
            aq_mma(c[n], a, b.x, b.y);
        }
    };
    int c[NGM][4];
#pragma unroll
    for (int n = 0; n < NGM; n++) c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0;
    if (m.type == T_AIT5U) {
        const uint8_t * const sbase = m.w + (int64_t) tile * (K / 128) * ai5d_unit_bytes(5);
        const Ai5Pos pa = ai5d_pos(8 * g + 2 * tq), pb = ai5d_pos(8 * g + 2 * tq + 64);
#pragma unroll 1
        for (int un = 2 * sb0; un < 2 * (sb0 + nsb); un++) {
            if constexpr (AIT_PFD > 0) {
                if (un + 2 * AIT_PFD < (int) (K >> 7) && 128 * lane < ai5d_unit_bytes(5))
                    ait_pfl2(sbase + (int64_t) (un + 2 * AIT_PFD) * ai5d_unit_bytes(5) + 128 * lane);
            }
            const uint8_t * blk = sbase + (int64_t) un * ai5d_unit_bytes(5);
            const uint32_t * tw = reinterpret_cast<const uint32_t *>(blk + 32);
            const int sb = un >> 1, hh = un & 1;
            const int8_t * const sbp = xq + (int64_t) sb * 4096;
#pragma unroll
            for (int G = 0; G < 4; G++) {
                uint32_t a[4];
                aq_dec_u(tw, G, pa, pb, a);
                mmas(c, sbp, hh * 4 + G, a);
            }
            const float r0 = aq_ldh(blk + 2 * g);
            const float r1 = aq_ldh(blk + 2 * (g + 8));
            flush(c, r0, r1, sb);
        }
    } else {
        auto run = [&](auto KC) {
            constexpr int k = decltype(KC)::value;
            const int64_t S = K / 512;
            const int rt = tile >> 1, u = tile & 1;
            const int64_t cb = aitd_chunk_bytes(k);
            const int wpos = (8 * lane + 4 * u) * k, wi = wpos >> 5, wsh = wpos & 31, wi1 = wi + 1 == 8 * k ? 0 : wi + 1;
            const uint8_t * const wbase = m.w + (int64_t) rt * S * cb;
            const int sbe = sb0 + nsb;
            uint4 lo4 = aq_ld128(wbase + (int64_t) (sb0 >> 1) * cb + (int64_t) (sb0 & 1) * 8 * 128 * k + 16 * wi);
            uint4 hi4 = aq_ld128(wbase + (int64_t) (sb0 >> 1) * cb + (int64_t) (sb0 & 1) * 8 * 128 * k + 16 * wi1);
#pragma unroll 1
            for (int sb = sb0; sb < sbe; sb++) {
                const uint8_t * chunk = wbase + (int64_t) (sb >> 1) * cb + (int64_t) (sb & 1) * 8 * 128 * k;
                const uint8_t * sc = wbase + (int64_t) (2 * (sb >> 2) + u) * cb + 2048 * (int64_t) k;
                const float r0 = aq_ldh(sc + 2 * g) * (1.f / 64.f);
                const float r1 = aq_ldh(sc + 2 * (g + 8)) * (1.f / 64.f);
                const int8_t * const sbp = xq + (int64_t) sb * 4096;
                const int sbn = sb + 1 < sbe ? sb + 1 : sb; // (the last super-block re-reads its own G 0)
                const uint8_t * const nchunk = wbase + (int64_t) (sbn >> 1) * cb + (int64_t) (sbn & 1) * 8 * 128 * k;
#pragma unroll
                for (int G = 0; G < 8; G++) {
                    uint32_t a[4];
                    aq_dec_tile<k>(lo4, hi4, wsh, a);
                    {
                        const uint8_t * grp = G < 7 ? chunk + (int64_t) (G + 1) * 128 * k : nchunk;
                        lo4 = aq_ld128(grp + 16 * wi);
                        hi4 = aq_ld128(grp + 16 * wi1);
                    }
                    if constexpr (AIT_PFG > 0) {
                        const int gi = 8 * sb + G + AIT_PFG; // the flat group index (16 groups per 512-K chunk)
                        if (lane < k && gi < 8 * (int) (K >> 8))
                            ait_pfl2(wbase + (int64_t) (gi >> 4) * cb + (int64_t) (gi & 15) * 128 * k + 128 * lane);
                    }
                    mmas(c, sbp, G, a);
                }
                flush(c, r0, r1, sb);
            }
        };
        if (m.type == T_AIT3) run(std::integral_constant<int, 3>{});
        else run(std::integral_constant<int, 4>{});
    }
}
// AI+ (AIT3 / AIT4 / AIT5_U) matrices inside the wv phases. Included by wv.cuh (anonymous namespace, after WvXS / WvAcc).
// An AI+ Mat has fm = 1 and goes through wv_tile_any like the wv types, but reads rotated fp16 activations instead of the
// int8 units: wv_fill (ait = true) rotates the units it staged ([u0, u0 + US) of T rows; x' = H256 (sign . x) / 16 per 256 block)
// into the block's own scratch in global memory (cB.xrot, MAXD rows of AIT_XLD halves, indexed by absolute K: only this
// block writes and reads it, so a __syncthreads orders it), and WvXS::xr points there. The int8 view stays for the dp4a items
// of the same phase (alpha / beta). The tile function is the verified fp16 m16n8k16 decode of ait_dec.cuh (tests
// ait_test / gemv_test / gemv5_test): A = fp16(level * column scale) (AIT5_U: level), B = x', fp32 sums, folded with the row
// scale per 256-K super-block (tile types: per 1024 K, so the same scale; AIT5_U: per 128-K unit). Same fragment as the
// wv types: v[n][c] = row 16 tile + g + 8 (c >> 1), token TPG n + 2 tq + (c & 1) (X16: the split layout, valid in tq < 2).
// The normed consumers get the same result as the int8 path up to the activation rounding: the RMS scale stays outside (s_rs).
static_assert(AIT_XLD == FF, "AIT_XLD is the widest K (ffn_down input)");

__device__ __forceinline__ bool ait_type(int t) { return t == T_AIT3 || t == T_AIT4 || t == T_AIT5U; }
__device__ __forceinline__ const __half * ait_xr() { return cB.xrot + (size_t) blockIdx.x * ((size_t) MAXD * AIT_XLD); }

// The int8 decode path (llamAmpere style): B = the rotated activations times the matrix's column scale, quantised to int8 per
// (token, 256-K block); A = the trellis levels as int8 (aq_dec_*, ait_dec.cuh); mma.m16n8k32.s8. Up to 3 matrices that read the
// same activations (different column scales) each get a slot in the block's scratch (cB.xrot):
//   header (AitHdr, 64 B) | XQ slot i at 64 + i * 98304 (24 blocks of 256 K x 4096 B; ntg = 2: 16 token slots) | SA slots.
// XQ[((blk * 2 + tok / 8) * 8 + G) * 256 + (tok % 8) * 32 + 8 tq + 4 s + 2 h + e]; SA[blk * 16 + tok] = absmax / 127.
struct AitSet {
    const __half * d[3] = {nullptr, nullptr, nullptr}; // column scales (nullptr: AIT5_U, none)
    int n = 0;
};
struct AitHdr {
    const __half * d[3];
    int n;
    int pad[7];
};
constexpr int AIT_XQ_SLOT = 98304, AIT_SA_SLOT = 4352, AIT_XQ_BASE = 64, AIT_SA_BASE = 64 + 3 * 98304;
__device__ __forceinline__ void ait_add(AitSet & s, const Mat & m) {
    if (!ait_type(m.type) || s.n >= 3) return;
    s.d[s.n++] = m.type == T_AIT5U ? nullptr : m.d;
}
__device__ __forceinline__ AitSet ait_set(const Mat & a) { AitSet s; ait_add(s, a); return s; }
__device__ __forceinline__ AitSet ait_set(const Mat & a, const Mat & b) { AitSet s; ait_add(s, a); ait_add(s, b); return s; }
__device__ __forceinline__ AitSet ait_set(const Mat & a, const Mat & b, const Mat & c) { AitSet s; ait_add(s, a); ait_add(s, b); ait_add(s, c); return s; }
__device__ __forceinline__ const int8_t * ait_xq(const AitHdr * h, int slot) { return reinterpret_cast<const int8_t *>(h) + AIT_XQ_BASE + slot * AIT_XQ_SLOT; }
__device__ __forceinline__ const float * ait_sa(const AitHdr * h, int slot) { return reinterpret_cast<const float *>(reinterpret_cast<const int8_t *>(h) + AIT_SA_BASE + slot * AIT_SA_SLOT); }

// units [u0, u0 + US) (32 elements each; US a multiple of 8) of T rows of q -> rotated, scaled, int8 in the block's scratch
__device__ __noinline__ void ait_rot_fill(const QBuf & q, int T, int u0, int US, const AitSet ts) {
    int8_t * const base = reinterpret_cast<int8_t *>(cB.xrot + (size_t) blockIdx.x * ((size_t) MAXD * AIT_XLD));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nb = US >> 3;
    if (threadIdx.x == 0) {
        AitHdr * const h = reinterpret_cast<AitHdr *>(base);
        h->d[0] = ts.d[0]; h->d[1] = ts.d[1]; h->d[2] = ts.d[2]; h->n = ts.n;
    }
    for (int job = warp; job < T * nb; job += NW) {
        const int t = job / nb, b = job - t * nb;
        const int e0 = 32 * u0 + 256 * b + 8 * lane; // 8 consecutive elements: one 16-chunk half
        const int c = e0 >> 4, byte = e0 & 15;
        const int gi = t * q.U + 8 * (c >> 4) + 2 * ((c >> 2) & 3) + (c & 1), part = (c >> 1) & 1;
        const int2 raw = __ldcg(reinterpret_cast<const int2 *>((part ? q.xb : q.xa) + (size_t) gi * 16 + byte));
        const float d = __half2float(__ldcg(reinterpret_cast<const __half *>(q.meta + gi) + part));
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float) (int8_t) ((i < 4 ? raw.x : raw.y) >> (8 * (i & 3))) * d;
        aitd_rot256(v, lane);
        const int ba = e0 >> 8;
#pragma unroll
        for (int s = 0; s < 3; s++) {
            if (s >= ts.n) break;
            float w[8];
            if (ts.d[s]) {
                const uint4 cr = *reinterpret_cast<const uint4 *>(ts.d[s] + e0);
                const __half * ch = reinterpret_cast<const __half *>(&cr);
#pragma unroll
                for (int i = 0; i < 8; i++) w[i] = v[i] * __half2float(ch[i]);
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) w[i] = v[i];
            }
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) a = fmaxf(a, fabsf(w[i]));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float id = a > 0.f ? 127.f / a : 0.f;
            if (lane == 0) reinterpret_cast<float *>(base + AIT_SA_BASE + s * AIT_SA_SLOT)[ba * 16 + t] = a * (1.f / 127.f);
            int8_t * dst = base + AIT_XQ_BASE + s * AIT_XQ_SLOT + ((ba * 2 + (t >> 3)) * 8 + (lane >> 2)) * 256 + (t & 7) * 32 + 4 * ((lane >> 1) & 1) + 2 * (lane & 1);
#pragma unroll
            for (int mq = 0; mq < 4; mq++) {
                const uint32_t lo = (uint8_t) (int8_t) __float2int_rn(w[2 * mq] * id), hi = (uint8_t) (int8_t) __float2int_rn(w[2 * mq + 1] * id);
                *reinterpret_cast<uint16_t *>(dst + 8 * mq) = (uint16_t) (lo | (hi << 8));
            }
        }
    }
    __syncthreads();
}


// super-blocks [sb0, sb0 + nsb) of 16-row tile `tile` of an AI+ matrix (the decode pass: T <= 16 tokens, WvAcc layout)
template <int T, bool X16>
__device__ __noinline__ WvAcc<WvCfg<T, X16>::NG> ait_tile(const Mat m, int tile, int sb0, int nsb, const WvXS xs) {
    constexpr int NGM = (T + 7) / 8; // m16n8 groups of 8 tokens
    constexpr int NG = WvCfg<T, X16>::NG;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    float tot[NGM][4];
    {
        const AitHdr * const hh = reinterpret_cast<const AitHdr *>(ait_xr());
        const __half * const dk = m.type == T_AIT5U ? nullptr : m.d;
        int slot = 0;
        if (hh->d[1] == dk && hh->n > 1) slot = 1;
        if (hh->d[2] == dk && hh->n > 2) slot = 2;
        if (hh->d[0] == dk) slot = 0;
        const int8_t * const xq = ait_xq(hh, slot);
        ait_acc_q<NGM>(AitMat{m.w, m.d, m.type, m.K}, tile, sb0, nsb, xq, ait_sa(hh, slot), tot);
    }
    WvAcc<NG> out;
    if constexpr (!X16) {
#pragma unroll
        for (int n = 0; n < NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) out.v[n][c] = tot[n][c];
    } else {
        // split layout: group n holds tokens 4 n + 2 tq + (c & 1) in the threads tq < 2; my group n >> 1 holds them in the
        // threads tq + 2 (n & 1)
#pragma unroll
        for (int n = 0; n < NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++)
                out.v[n][c] = __shfl_sync(0xffffffffu, tot[n >> 1][c], (lane & ~3) | ((tq + 2 * (n & 1)) & 3));
    }
    return out;
}
