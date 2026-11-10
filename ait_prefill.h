// AI+ batched prefill GEMM (ait_prefill.cu)
#pragma once
#include <cuda_runtime.h>

constexpr int AIT_PF_PIECE = 512; // tokens per piece; the rotated scratch holds this many rows of 17408 halves

// y[tok * ldy + row] (+)= sum_k W[row][k] x[tok * K + k] for tokens 0 .. ntok - 1. w / cs: the Mat's GGUF bytes and column
// scales (cs: null for AIT5_U); type: the T_AIT* enum value; xrot: AIT_PF_PIECE * 17408 halves of scratch.
void ait_prefill_gemm(const void * w, const void * cs, int type, int K, int N, const void * x, int ntok, void * xrot, float * y, int ldy,
                      bool add, cudaStream_t st);
