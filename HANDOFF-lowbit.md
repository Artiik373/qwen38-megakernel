# Handoff: native low-bit weight types for qwen38-megakernel

## Goal
These two models must load and run with exact kernels:
- `~/models/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf`, which currently fails with "unsupported ggml type 18" (IQ3_XXS) at `blk.0.ffn_down`.
- `~/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, which currently fails with "unsupported ggml type 22" (IQ2_S).

User decision: native kernels, exact against ggml dequantization, and no large VRAM growth. Do not re-quantize lossily, and do not widen everything to Q6_K. An exact re-encode into an existing type is allowed only where it is lossless and small (IQ1_M → Q6_K).

## Repository state
- Repo `~/qwen38-megakernel`, branch `main`, HEAD b3b76ab "Merge upstream origin/main (to cdf7622)". Not pushed; the user pushes with `git push fork main`.
- Uncommitted WebP vision work. Keep it and commit it separately:
  - `vision/mk_vision.cpp`, `vision/mk_vision.h` (dlopen of `libwebp.so.7`)
  - `docs/SERVER.md`
  - `bench/wv_ref.cuh`
- Leave the stashes alone: "pre-merge wv_ref.cuh regen" and "exl3-wip".
- Commit identity: `git -c user.name=artii -c user.email=artii@DESKTOP-PQ32M5K.localdomain commit ...`
- Build: `make -j4 LLAMA=llama.cpp qwen38_mk libqwen38mk.so libqwen38vision.so`. It takes about 14 minutes, so use a long timeout. The vision lib still needs a rebuild to pick up WebP.
- Before GPU runs, check `nvidia-smi`; the user may be running `mk_server`.
- Regression baseline, using `~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf`:
  - Command: `-p "<prompt>" -n 128 --kv q8 --draft-vocab draft_vocab_56k.bin --ids`.
  - Runs `-d 0` and `-d 7` must give identical ids. Last measured: 41.2 and 199.5 tok/s.
- Follow DOX: update the type lists in `docs/ENGINE.md`, `MK_TYPES` in `docs/BUILDING.md`, and `docs/CONFIGURATION.md` (also check it for stale `MK_K9` / `MK_WPF` rows). Regenerate `bench/wv_ref.cuh` with `python3 bench/wv_extract.py`.

## Tensor census (unsupported types only)
The census script is at `/tmp/census.py`. Run it as `.venv/bin/python /tmp/census.py <gguf>` from the repo root.

**IQ4_XS-3.84bpw**
| Type | Tensors |
|---|---|
| IQ3_XXS | ~100: `ffn_down` ×27, `gate`/`up`, `qkv`, `attn_output`, `ssm_out`, `ssm_alpha`; plus the MTP block |
| IQ2_XXS | `ffn_gate` ×2, `ffn_up` ×1 |
| IQ2_XS | `ffn_gate` ×1 |
| Q2_K | `ssm_beta` ×1 |
| IQ4_XS | `token_embd` |

**Swift**
| Type | Tensors |
|---|---|
| IQ2_S | `ffn_down` ×4, `gate`/`up`/`attn` ×12, and `token_embd` |
| IQ2_XS | `ffn_down` ×3, plus 6 others |
| IQ2_XXS | 5 |
| Q2_K | `ffn_down` ×1, `attn_gate` ×4, `attn_q` ×6, `qkv` ×1, `ffn_gate` ×1 |
| IQ1_M | `ffn_gate` ×1 |
| BF16 | `ssm_alpha` / `ssm_beta` ×96 |
| IQ3_XXS | many |
| Q4_K | `output.weight` |

The MTP block (`blk.64`) is always re-quantized to Q4_K, so the new types only need adding to `requant_type()` in `engine_impl.h` for it.

## Architecture facts (verified by reading the code)

**Routing**
- A matrix runs on one kernel family at every T: either wv tensor-core tiles or dp4a rows. Paths therefore do not have to be bit-identical to each other.
- Main-layer `ffn_down` must be in the wv layout (`loader.cu` ~1022, "the wide pass needs every ffn_down in the wv layout").
- The hybrid types (Q3K, IQ3S) use the wv layout only for `ffn_down` (`wv_lay && wv_fd`). Their other main-layer matrices stay on dp4a through `gemv_pairs_dp`, `rows16_dp`, `rows16_dp_x8` and `resid_dp` in `wv.cuh` (~871-911, ~1890). New types follow the same pattern.

**dp4a unit model** (`qwen38_mk.cu` ~485-690)
- Unit `u = lane + 32i` covers 32 weights: block `b = u>>3`, `p = (u>>1)&3`, `hh = u&1`.
- A is elements `16hh..+15` of sub-block `2p`; B is the same elements of sub-block `2p+1`.
- `accR(fA, gA, fB, gB, A[4], B[4])` computes `acc += fA·dx·dot(A) + fB·dx·dot(B) − gA·sqA − gB·sqB`.
- Pattern to copy: `rows_iq3s` (~889-940).
- Dispatch sites:
  - `mrows` / `gemv_pairs` switch (~947-1010)
  - `resid32` (~1338)
  - `rows16` (~1424)
  - BS tables (~1774, ~6447)
  - `MKL_MMH` instantiations (~6619)

**wv tiles** (`wv.cuh`)
- `WV<T>` struct provides `load`, `scales`, `frag`/`wsc`, or for MIN 2 `frag2`/`wsc2` with `tq`/`h`.
- `wv_sb` (~608) branches on `MIN`: 0 means no min, 1 means `−gA·mx`, 2 means offset 32.
- S16 (per-16 scales) is X16 (split MMA) only. `ffn_down` is always X16, and in X16 `mx` carries the per-chunk `dx·qsum`.
- Per-16 scales with MIN 0/1 is not supported yet. Add a branch to `wv_sb`: `else if constexpr (W::S16)` calls `frag2` and `wsc2(s, k, h, fA, gA)`.
- Switch sites that need new cases:
  - `wv_type`, `wv_s16`, `wv_min2` (~38-42, ~211)
  - `wv_tile_any` (~825)
  - `wv_ubytes` (~848)
  - `wv_tile_any_st` (~1750)
  - `wv_ck_x16_any` (~2330)
- Tile-major meta: `wv_tmeta(m, tile, row, sb, MB)`.
- Host relayout lives in `wv_host.cuh` (BS tables).

**Prefill** (`prefill_gemm.cuh`)
- `HalfW` stage = (sb, p). The `TIQ3S` paths to model on:
  - `WT`/`HT` tables (~64, ~126)
  - `chunk_src` (~67)
  - `esrc`/`vsrc` for LAY (~354)
  - `load` (~447)
  - `scales` (~496)
  - `frag` (~586); also add the new types to the `OFF` table at ~522.
- Prefill is fp16, so it is approximate. It only needs to be deterministic.
- Use bytes `128 ± g` with `OFF = −1152`, as TQ80/IQ4 do.
- Dispatch is in `prefill_host.cu` (~74, ~89, ~342, ~355). `pf_mbl` lives in `prefill_gemm.cuh` (~148), and `MK_GLU_MIX_PAIRS` in `mk_launch.h`.

**Loader** (`loader.cu`)
- `want()` (~332): add IQ1_M to the exact Q6_K re-encode path (`to_q6`), next to Q3K/IQ3S.
- `up_mat` native repack (~540-566) is the model for new repacks. It runs a per-row self-check against `to_float` and has a `wv_place` branch.
- `needq` list (~925), `tname` (~898), `dp4a_need` (~1047).
- lm_head check (~1073).
- `token_embd` check (~1137).
- Grid recovery from ggml: `iq3s_grid_host()` (~206), uploaded with `set_g_iq3s_grid`.

**Shared definitions**
- `gguf.h`: type enum and sizes. Add IQ2_XXS=16 (66 B), IQ2_XS=17 (74 B), IQ3_XXS=18 (98 B), IQ2_S=22 (82 B), IQ1_M=29 (56 B), Q2_K=10 (84 B), all with 256-weight blocks.
- `mk_shared.cuh`: the `T_*` constants, `MK_TY_*` (~254-297), `mk_built`, the abi bits (~584), `wv_fm`, and `TQ*` (~562).
- `Makefile`: `MK_TYPES_ALL` and the `subst` chain.

**Embedding** (`qwen38_mk.cu` ~1069-1151)
- `embd_rb`, `dequant_embd`, `embed_elem_other`, `embed_row_other`.
- `prefill.cuh` ~52-71 and ~1029; `prefill_host.cu` ~523 and ~726.
- `ModelW::embd_t` comment at `mk_shared.cuh:339`.

**lm_head** (`qwen38_mk.cu` ~4486-4532)
- `rows_t<..., T_Q4K>` already exists for the non-H16 path (dq4).
- For H16, `rows_q4k` must be templated on `S` (SX/SXH), as `rows_q6k` is.
- `MK_DRAFT_Q4` (`loader.cu` ~1114) must accept a Q4_K head, or skip it.

**Byte packing rule**
- `base + (g ^ mk) − mk` in 32-bit arithmetic is correct only if every result byte is in [0, 255].
- Signed s8 values: `((0x80808080 + (g ^ mk) − mk) ^ 0x80808080)`.

## Chosen designs (exact by construction; checked per row against ggml `to_float` with memcmp, also accepting −0 vs +0)

1. **IQ3_XXS → `T_IQ3XXS = 18`**
   - Values: 96 B per block, laid out as IQ3_S:
     - Index bytes at `8u'`: A[4] then B[4]. Index `c` covers weights `4c..4c+3` of the half.
     - Signs at `64 + 4u'`: byte `m`, bit `c` is A element `4c+m`; bit `4+c` is B.
   - Meta: 8 B per block: `scales[4]` (byte `p`: nibble for sub-block `2p` | `2p+1` << 4), then fp16 d, then pad.
   - Grid: extend `g_iq3s_grid` to 768 entries, with XXS at 512+idx. Recover it with ggml (d=1, ls=0 gives y = g/4).
   - Weight: `w = (d·0.25)·(2ls+1)·g·sign`, MIN 0, signed bytes, per-32 scales (S16 false). No shared-memory grid copy (GS).
   - Signs: `ksigns(i) = i | (popcount(i)&1)<<7`, 7 bits per 8 weights, taken from `aux32 >> 7l`. The scale is `aux32 >> 28`.
2. **IQ2_XXS / IQ2_XS / IQ2_S → one kernel type `T_IQ2S = 22`**
   - Values: 64 B per block. At `8u'`: A index lows [2] (index 0 = elements 0-7 of the half, index 1 = 8-15), B index lows [2], A signs u16, B signs u16.
   - Meta: 20 B per block, read with 4-byte loads:
     - `qh[8]`: byte `u'`, bits `2c..` (c = A0, A1, B0, B1)
     - `sc[8]`: byte `u'` = A-chunk ls | B-chunk ls << 4. XXS repeats its per-32 scale in both nibbles.
     - fp16 d
     - base (grid base >> 8: 0 for S, 4 for XS, 6 for XXS)
     - pad
   - Union grid `g_iq2_grid[1792]` (8 B each): S at 0, XS at 1024, XXS at 1536. Index is `lo | (qh + base) << 8`.
   - Weight: `w = (d·0.125)·(2ls+1)·g·sign`, with g in {8, 25, 43}. MIN 0, signed bytes, S16 true.
   - ggml layouts, verified:
     - XXS: 66 B. `aux8[l]` is the index; signs come from `aux32[1] >> 7l`; the scale is `aux32[1] >> 28`.
     - XS: 74 B. Index is `qs[4ib+l] & 511`, signs are `>> 9`, scale is `scales[ib] >> 4·(l/2)`.
     - S: 82 B. Index is `qs[l] | (qh[ib] << (8−2l) & 0x300)`, signs are `qs[32+..]`, scales as XS.
3. **Q2_K → `T_Q2K = 10`**
   - Values: 64 B per block. At `8u'`: A word, then B word. Byte `m`, bits `2j` hold element `4j+m`.
   - Meta: 20 B per block: ggml `scales[16]` (chunk `e>>4`; low nibble sc, high nibble m), fp16 d, fp16 dmin.
   - Weight: `fA = d·sc`, `gA = dmin·m` per 16, MIN 1, S16. This needs the new MIN-1 + S16 branch in `wv_sb`.
   - In prefill, scales become `a = (S_lo, S_hi)` and `b = (−M_lo, −M_hi)`, and `frag` broadcasts by half (kk).
   - ggml element `e`: q = `(qs[32(e>>7) + (e&31)] >> 2((e>>5)&3)) & 3`.
4. **IQ1_M → exact Q6_K re-encode in `want()`**
   - `q = 8g ± 1`, `sc = 2ls+1`, `d6 = d/8`. The existing full-row check catches any subnormal edge case.
5. **BF16 alpha/beta:** already go through `want(..., T_F32)`. Verify only.
6. **Embedding:** add `dequant_iq4xs_e` and `dequant_iq2s_e`, using the IQ2_S part of the union grid (base 0). Both must equal ggml's values; add the cases to every embed switch.
7. **Q4_K lm_head:** as described in the lm_head section above.

## Order of work
1. `gguf.h`, `mk_shared.cuh` and Makefile type plumbing.
2. IQ3_XXS end to end: loader repack and check, dp4a rows, WV struct, prefill, IQ4_XS embed. Build and get the IQ4_XS model loading. Fix later blockers one at a time as they surface (IQ2_XXS/XS, then Q2_K).
3. The IQ2 family, then Q2_K (including the `wv_sb` branch), IQ1_M, the IQ2_S embed, and the Q4_K lm_head. Then the Swift model.
4. Verify:
   - Both models load and produce coherent text.
   - d0 and d7 `--ids` are identical on both models.
   - UD-Q4_K_XL regression ids are unchanged.
5. Docs, regenerate `wv_ref.cuh`, commit.

Expected VRAM growth over the file sizes: roughly 100–300 MB in total. Mention it in the final report.
