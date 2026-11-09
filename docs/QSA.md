# QSA (Qwen Sparse Attention)

Status: **loading only.** The indexer tensors and parameters are read and uploaded; attention runs dense on every
layer (the output is then the reference's `LLAMA_QSA_DENSE=1`, and identical to QSA below 2048 context tokens). Block
selection in decode and prefill is NOT implemented: it touches every attention kernel variant of the megakernel.
Verified on an RTX 3090: the QSA GGUF loads (every low-bit row repack self-check passes) and decodes coherently, `-d 0`
and `-d 7` ids identical (48-token haiku prompt only; no long-context or reference-logit comparison yet).

Model: `Qwen3.8-27B-QSA-GSQ-RCO-*.gguf` (reference engine: kuraneko1/Qwen3.8-27B-QSA-llama.cpp, `build_qsa_mask` in
`src/models/qwen35.cpp`). Its weight types need the native low-bit kernels (see ENGINE.md type list).

## Metadata and tensors
- `qwen35.attention.indexer.head_count` (4), `.key_length` (128), `.top_k` (2048); `qwen35.attention.compress_ratios`
  (int32 x 65: 4 on the 16 attention layers, 0 elsewhere, 0 on the MTP layer). `MKEngine::Impl::qsa_params` refuses any other layout.
- Per QSA attention layer, F32: `indexer.q_proj.weight` [5120 -> 4x128], `indexer.k_proj.weight` [5120 -> 128],
  `indexer.q_norm.weight`, `indexer.k_norm.weight` (pooled key), `indexer.k_norm.pre_weight` (raw key). Norm weights are
  stored 1+w. Uploaded unchanged (`LayerW::ix_*`, about 13 MB per layer, 210 MB total). A layer without them stays dense.

## Selection (per attention layer, per query at position p; dense when n_kv <= top_k)
1. raw key per token `k = RMSNorm(k_proj(h), pre_weight)`; pool every r=4 consecutive keys by mean, `RMSNorm(., k_norm)`,
   RoPE at the block's first position (n_rot 64, base 1e7).
2. `q = RoPE(RMSNorm(q_proj(h) per head, q_norm), p)`.
3. score(b) = `sum_heads relu(pooled_b . q_h) / sqrt(128)`; blocks not fully visible (`(b+1)*4-1 > p`) are excluded.
4. top `ceil(top_k / 4)` = 512 blocks, plus the query's own block; expand to tokens; causal mask; ordinary gated attention
   over those tokens. Ties: higher score, then lower block index (must not depend on the verify depth T).
`bench/qsa_ref.py` is the CPU reference for this.

## Planned cache
Pooled block keys only (about 14 MB per layer in F16 at 55k blocks) plus the last <= 3 raw keys of the open block; not
per-token raw keys. KV stays fully allocated: QSA cuts attention compute and KV reads, not KV size.
