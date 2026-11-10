// AI+ weight tile core (llamAmpere AIT3 / AIT4 / AIT5_U): fp16 tensor-core decode shared by the decode pass (ait_wv.cuh)
// and the batched prefill (ait_prefill.cuh). The activations are the block-rotated fp16 rows (sign flip, FWHT256, / 16).
#pragma once
#include "ait_dec.cuh"
#include "mk_shared.cuh"

constexpr int AIT_XLD = 17408; // halves per token row of the rotated activations (FF; the widest K)

struct AitMat {
    const uint8_t * w; // the GGUF bytes
    const __half * d;  // tile types: K fp16 column scales (loader); AIT5_U: unused
    int type;          // T_AIT3 / T_AIT4 / T_AIT5U
    int K;
};

