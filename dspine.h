// dspine.h: DSpine speculative drafter (Qwen3.8-27B-DSpine-b16, arXiv 2609.36173): shapes, weight views and the
// launchers of dspine.cuh (kernels, built into qwen38_mk.cu's TU). Host orchestration: dspine.cu.
//
// The drafter is plain CUDA kernels on stream 0 next to the megakernel; the megakernel only copies the residual
// stream of 5 target layers into a bf16 buffer (qwen38_mk.cu: dsp_cap) and the prefill does the same (pk: dsp::stage).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace dsp {

constexpr int D = 5120;          // draft and target width
constexpr int NLY = 5;           // draft layers
constexpr int NH = 32, NKV = 8, HD = 128;
constexpr int QD = NH * HD, KD = NKV * HD; // 4096, 1024
constexpr int FF = 17408;
constexpr int CODE = 1024, RANK = 512; // DSpine code bridge
constexpr int NF = 5;            // target layers fused by fc
constexpr int FD = NF * D;       // fc input width
constexpr int WIN = 8192;        // sliding window (a power of two: ring slot = position & (WIN - 1))
constexpr int BMAX = 16;         // block size (rows of one draft pass)
constexpr int TOPK = 16;         // lattice width (dflash.selector_top_k)
constexpr int LROWS = (BMAX - 1) * TOPK; // lattice rows (predecessor candidates x positions)

// ggml type ids of the types the kernels read (gguf::GType values)
enum { TY_F32 = 0, TY_Q8_0 = 8, TY_Q4K = 12, TY_Q6K = 14, TY_BF16 = 30 };

struct Mat { // a row-major GGUF matrix on the device: row n (n < N) holds K values, rb bytes
    const uint8_t * w = nullptr;
    int type = 0, K = 0, N = 0;
    size_t rb = 0;
};
struct Layer {
    Mat q, k, v, o, gate, up, down, rd, wr; // rd: dspine_read [CODE x D], wr: dspine_write [D x RANK]
    const float *an = nullptr, *qn = nullptr, *kn = nullptr, *fn = nullptr; // attn_norm, q_norm, k_norm, ffn_norm
};
struct Weights {
    Mat fc, code, gate, pgate, prev; // fc [D x FD]; dspine_codebook [vocab x CODE]; dspine_gate / pred_gate / prev [RANK x CODE]
    const float *encn = nullptr, *outn = nullptr, *gbias = nullptr; // enc.output_norm, output_norm, dspine_gate.bias
    Layer L[NLY];
    float eps = 1e-6f, theta = 1e7f;
};

// device scratch (all fp32 unless noted); rows <= BMAX except the lattice (LROWS)
struct Scratch {
    float *feat, *g;                 // [BMAX][FD] bf16 -> f32 features, [BMAX][D] fc output
    float *kb, *vb;                  // [BMAX][KD]
    float *x, *xn, *q, *att;         // [BMAX][D], [BMAX][D], [BMAX][QD], [BMAX][QD]
    float *ffg, *ffu;                // [BMAX][FF]
    float *codes, *bq, *bs, *bv, *bmsg; // [BMAX][CODE], [BMAX][RANK] x4
    float *bdelta, *raw, *rms;       // [BMAX][D], [BMAX][D] (last layer's pre-write state), [BMAX]
    int *toks, *cand, *drafts;       // [BMAX], [BMAX][TOPK], [BMAX]
    float *pc, *gp, *pv, *lmsg;      // lattice: [LROWS][CODE], [LROWS][RANK] x3
    float *ldelta, *refined;         // [LROWS][D] x2
    float *table;                    // [BMAX - 1][TOPK * TOPK]
    uint8_t * hrows;                 // [LROWS][hrb] raw lm_head rows of the candidates
};

// ---- kernels' launchers (dspine.cuh) ----
void gemm(const Mat & w, const float * X, int ldx, float * Y, int ldy, int M, bool add, cudaStream_t st = 0);
void rms(const float * x, const float * w, float * y, float * rms_out, int rows, int n, int ld, float eps, cudaStream_t st = 0);
void bf2f(const __nv_bfloat16 * src, float * dst, size_t n, cudaStream_t st = 0);
void stage(const float * x, __nv_bfloat16 * dst, int j, int T, cudaStream_t st = 0); // x rows [T][D] -> dst[t][j][:]
void inject_kv(const float * kb, const float * vb, const float * kn, __half * kr, __half * vr, int pos0, int M, float theta, float eps, cudaStream_t st);
void set_tags(int * tags, int pos0, int M, cudaStream_t st);
void qk_prep(float * q, float * kb, const float * qn, const float * kn, int pos0, int M, float theta, float eps, cudaStream_t st);
void attn(const float * q, const float * kb, const float * vb, const __half * kr, const __half * vr, const int * tags, int n, int M, float * out, cudaStream_t st);
void swiglu(float * g, const float * u, size_t n, cudaStream_t st);
void codes(const float * craw, const Mat & codebook, const int * toks, int M, float * out, float eps, cudaStream_t st);
void msg(const float * q, const float * s, const float * v, const float * bias, int M, float * out, cudaStream_t st);
void apply(float * x, const float * delta, const float * rms, int M, float * raw, cudaStream_t st);
void topk(float * logits, int V, int rows, int * cand, cudaStream_t st); // destroys logits
void pred_codes(const int * cand, const int * toks, const Mat & codebook, int M, float * out, cudaStream_t st);
void lat_msg(const float * gp, const float * pv, const float * qst, const float * bias, int M, float * out, cudaStream_t st);
void refine(const float * raw, const float * rms, const float * delta, const float * outn, int M, float eps, float * out, cudaStream_t st);
void table(const uint8_t * hrows, size_t hrb, int hty, const float * refined, int M, float * out, cudaStream_t st);
void walk(const int * cand, const float * table, int M, int * drafts, cudaStream_t st);

// the megakernel's capture target (qwen38_mk.cu): rows of the residual stream after the target layers lay[0 .. n)
// go to buf[row][j][D] (bf16); n = 0: off
cudaError_t set_capture(__nv_bfloat16 * buf, const int * lay, int n);

} // namespace dsp
