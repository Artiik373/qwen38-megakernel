# Engine notes

How the engine works inside. This document began as open-jet's megakernel README (this repository started as that
directory, history included) and has been updated as the engine changed. Measured speed is in the
[README](../README.md#numbers); numbers quoted here carry their date.

A specialised CUDA inference engine for Qwen3.8-27B `Q4_K_M` and unsloth's `UD-Q4_K_XL` (llama.cpp arch `qwen35`:
48 Gated-DeltaNet + 16 gated-attention layers, one MTP block). It is tuned and tested on an RTX 3090 (sm_86, 82 SMs);
`make` also builds for other 24 GB GPUs (sm_89, sm_120), which are untested.

## Model files

Which model file runs on it (details in [The GGUF](#the-gguf)):

| Model file | Runs on the engine |
|---|---|
| unsloth's Qwen3.8-27B `UD-Q4_K_XL` | yes (tuned on it) |
| unsloth's Qwen3.8-27B `Q4_K_M` | yes |
| another Qwen3.8-27B `Q4_K_M` | when every tensor has a type listed under [The GGUF](#the-gguf); untested |
| any other quant | refused by the loader, which names the first tensor it cannot read (use llama.cpp) |

## How it works

Decoding runs an entire speculative cycle in one persistent cooperative kernel launch:

```
main model over T = 1 + n_draft tokens (up to 8)  ->  lm_head + verification on device (greedy, or exact
speculative sampling)  ->  MTP catch-up rows for the accepted tokens  ->  up to n_draft MTP draft steps
```

The ~340 phases of a cycle (projections, DeltaNet core, attention, FFN, lm_head, MTP) are separated by a software grid
barrier; the host only reads back the accepted tokens. Prompts go through batched tensor-core kernels instead (see
[Prefill](#prefill)).

- **Wide verify** (`MK_WV=1 MK_MAXT=8`, the served build): the main pass's matrix products run as int8 tensor-core
  GEMMs (`mma.m16n8k32`, tokens on N; `wv.cuh`) on a weight layout made at load (fragment-ordered tiles, values
  unchanged), so a pass over 8 rows costs little more than a pass over 1. Every output is the same sequence of integer
  sums and float FMAs at any T. The lean build (`make MKFLAGS=`, `MK_MAXT=5`) reads the weights with dp4a GEMVs and
  verifies at most 5 rows (4 drafts).
- **Draft depth:** up to `-d` drafts per cycle. An on-device gate stops drafting early when the running product of the
  MTP head's own top-1 probabilities falls below a threshold (0.25-0.4, by draft index; at depth > 5 the deeper drafts
  have their own gates). With sampling, the drafts are sampled too. `MK_POLICY` overrides the defaults (a
  `key=value,...` list, documented at `MKEngine::set_policy`).
- **Greedy output does not depend on the draft depth:** the attention split-K partition depends only on the context
  length, and the verify GEMMs compute every row the same way at any T. With sampling, the token distribution is the
  one of sampling without drafts.

Speculation in numbers (2026-10-16, commit 7d67c5d, CLI, UD-Q4_K_XL, q8_0 KV; tokens / total time):

| prompts | drafts | tok/s | draft acceptance | tokens / cycle | ms / cycle |
|---|---|---|---|---|---|
| 4 short chat prompts, greedy | 0 | 44.9 | - | 1.00 | 22.3 |
| | 2 | 99.2 | 76.9% | 2.38 | 24.0 |
| | 7 | 128.3 | 50.9% | 3.43 | 26.7 |
| the same, sampled, thinking on (8 runs) | 7 | 124.0 | 51.5% | 3.30 | 26.6 |
| a 38k-token prompt, greedy | 0 | 41.7 | - | 1.00 | 24.0 |
| | 7 | 113.6 | 48.4% | 3.40 | 29.9 |
| copying text verbatim, short / at 38k | 7 | 257.7 / 235.3 | 95.1% / 97.9% | 7.51 | 29.1 / 31.9 |

A cycle that verifies 8 rows took ~29.1 ms against 22.3 ms for 1 row (short context), so the gain comes from accepted
drafts. Without drafts, llama.cpp b11548 decoded the same prompts at 42.0 tok/s (short) and 35.1 tok/s (38k) in that
session.

## Correctness

The output is not byte-identical to llama.cpp's: the arithmetic is ordered and rounded differently, so where two tokens
are close the other one can win, and a generation then continues differently. Measured against llama.cpp's own
next-token distributions (same GGUF; reference: llama.cpp b10246 generating token by token with its q8_0 KV cache;
13 sequences of this repo's code and docs, 13,299 positions scored after a 1024-token prompt each), UD-Q4_K_XL:

| against llama.cpp generating token by token | top token agrees | KL divergence (mean, nats) | perplexity (reference 2.8099) |
|---|---|---|---|
| llama.cpp, batched instead (2026-10-11) | 98.489% | 0.00166 | 2.8116 |
| this engine, prefill kernels then the decode kernel (`mixed`), q8_0 KV | 98.737% | 0.00119 | 2.8085 |
| this engine, prefill kernels only, q8_0 KV | 98.744% | 0.00124 | 2.8073 |

The engine's rows were measured on 2026-10-17 (Linux, the served flags with K5; the `mixed` report has been the same
since 2026-10-15, and the native Windows build of 147054b gives it byte for byte). The engine is at least as close to
llama.cpp's token-by-token run as llama.cpp's own batched run is. Most top-token disagreements are near-ties: the
median lead of llama.cpp's token over the engine's is 0.043 nats (llama.cpp batched: 0.051), and 1 of 168 exceeds 0.5
nats.

At long context (same reference setup, q8_0 KV): 99.121% top-token agreement and KL 0.00219 over 4,094 positions of 2
sequences at ~32k (2026-10-17), and 99.267% / 0.00163 over 4,095 positions of one 64k sequence (2026-10-15, 7d67c5d).
The plain Q4_K_M was last scored on 2026-10-13, on a wide-verify build with up to 4 drafts: 98.887% / 0.00127
(llama.cpp batched on that file: 98.489% / 0.00184). To reproduce:

```sh
make tools/llama_ref
tools/llama_ref -m model.gguf -f code.txt -f prose.txt --score-from 1024 -b 1 --ref-out llama.ref
tools/llama_ref -m model.gguf --kld llama.ref                            # llama.cpp batched: the noise floor
./qwen38_mk -m model.gguf --kld llama.ref --score-path mixed --kv q8     # the megakernel
```

`llama_ref` teacher-forces 2048-token chunks of the text (`--chunk`) and stores, per scored position, the 256 most
likely tokens (`logit_stats.h`); `--kld` compares an engine with such a file (`--ref-out` also writes its own).
`--score-path` selects the megakernel's batched prefill kernels (`prefill`, the default), the decode kernel (`decode`),
or the prefill kernels for the unscored prompt and the decode kernel for the scored positions (`mixed`, what a
generation runs).

Speculative verification is exact: greedy output is the main model's greedy choice at every draft depth, and with
sampling a draft is accepted with probability p(draft) and a rejected one is resampled from the rest of the
distribution, so the sampled distribution is the same as without drafts (chi-square and PIT tests).

The random numbers come from a counter-based generator keyed by the seed and the position: a row's acceptance and
resampling numbers by its position, a sampled draft's by the position it drafts. Each position is decided once, and a
position drafted again after a rejection before it draws the same number, but that number is independent of
everything that decided the tokens before it, so the output stays exact (2026-10-17: a CPU replica of the loop,
`tests/spec_rng_sim.py`, 396k re-drafted positions; the PIT test on 13,401 re-drafted rows of the engine at 7
drafts). Two opt-in `MK_POLICY` settings were not: with `blk=1` (block verification), which reads every row's number
and leaves the rows after its cut to the next cycle, the acceptance numbers and the drafts are now keyed by the cycle's
start position as well (position-only keys made the next cycle reuse numbers conditioned on the last one: biased in
the replica; with the new keys 72k rows of the engine pass the PIT test); and `chain=2` with sampled drafts runs
without drop gates (dropping a draft on its own probability selects which drafts get verified: biased).

**unsloth UD-Q4_K_XL** (`Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17.6 GB: Q4_K / Q5_K / Q6_K / Q8_0 / IQ4_XS per tensor plus a
few Q3_K, IQ3_S and IQ4_NL) puts its extra bits where the decode spends its time: its FFN gate / up average 5.1 bits
per weight against Q4_K's 4.5, so a cycle reads more weight bytes than with the plain Q4_K_M. Its MTP block (Q6_K in
the file) is re-quantized to Q4_K at load ([The GGUF](#the-gguf)). In one session on 2026-10-14 the plain Q4_K_M
decoded faster at most context lengths (README).

## Build / run

The served configuration and its build are in the [README](../README.md#build); the lean engine (up to 4 drafts)
builds in a few minutes with `make MKFLAGS=`. The CLI:

```sh
./qwen38_mk -m ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf -p "Write quicksort in C" -n 512 \
            -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin --think --temp 1.0 --top-p 0.95 --top-k 20
```

CLI flags: `-d` drafts per cycle (default 2; 0-7 with the served build, 0-4 with the lean one; more than 4 needs
`--kv q8`), `-c` KV capacity, `--kv f16|q8|auto` (KV cache format; auto: q8_0 only when fp16 does not fit, or when
`MK_KV` is set), `--think [xhigh|medium|low]` (reasoning mode; the chat template's default effort is xhigh),
`--temp` / `--top-p` / `--top-k` / `--min-p` / `--seed` (sampling; default greedy), `--presence-penalty` /
`--frequency-penalty` (on the generated tokens), `-p text` / `-f file` (the prompt), `-n` tokens to generate, `--raw`
(no chat template), `--ids`, `--quiet`, `--prompt-ids f` (prompt as token ids), `--turn2 text` / `--turn2-file f` (a
second user turn appended to the conversation), `--score ids.txt` (teacher-forced agreement), `--kld ref`
(`--ref-out`, `--score-path`, `--kld-images`: see [Correctness](#correctness)), `--state-dir dir` (post-prefill state
files: the first prompt's prefill is restored from there when a file of the same prompt, binary, model and settings
exists, else saved), `--digest` (a hash of the state after the first prefill), `--batch jobs.txt` (one job per line,
one model load).

Environment variables for runs (read at load):
- `MK_KV`, `MK_KV_WIN`: low-bit KV cache ([KV cache](#kv-cache)).
- `MK_PF_CAP=n`: prefill chunk tokens (128-2048 in steps of 128; default 2048, the launchers use 1024 above 96k
  context).
- `MK_EXACT=1`: exact reuse (the server sets it with `--exact-reuse`; see [Server](#server)).
- `MK_CK_SLOTS=n`: DeltaNet checkpoint slots in pinned host memory (default 8; the CLI keeps 1 unless set).
- `MK_KEEP_WARM_S=s`: keep-warm window after each server request (default 120, 0 = off; [Server](#server)).
- `MK_WATCHDOG=s`: seconds a decode launch may take before the engine stops with an error (default 60; prefill chunks
  and checkpoint copies get at least 120 s; 0 = plain blocking waits). A GPU fault the CUDA runtime does not report
  becomes an error message and exit code 3 instead of a hang.
- `MK_POLICY`: the draft-depth policy ([How it works](#how-it-works)).
- `MK_STATE_WARM_MS=ms`: GPU warm-up before decoding from a restored state file (`--state-dir`; default 1000, 0 = off).

Measurement and A/B switches: `MK_PROFILE=1` (per-phase timing, including a "min-composed" estimate that filters out
time-slicing by other GPU users; `MK_PROFILE_SEQ=1` per barrier position), `MK_PF_PROFILE=1` (per-kernel prefill
timing), `MK_PF_GAPS=1` (GPU idle time per prefill), `MK_PF_SYNC=1` (the previous prefill flow: wait for each chunk
before queuing the next), `MK_PREFILL=mk` (prefill through the decode megakernel instead), `MK_ATTN_PV16=0` /
`MK_ATTN_SPLIT=0` (prefill attention with fp32-accumulated P V / without the tail split; with both, bit-identical to
`MK_ATTN=1`, the original prefill attention kernel), `MK_ATTN_S8=0` (q8 cache: Q K^T on fp16 tensor cores instead of
int8; `=2`: one Q scale per row instead of per 32 dims), `MK_DATTN=0` (q8 cache: the previous, unpipelined decode
attention; same output), `MK_MTP_Q4=0` (the MTP block in its stored types, see [The GGUF](#the-gguf)),
`MK_MTP_WIN=n` (the MTP block's decode attention sees the last n positions only; output stays exact, acceptance can
change), `MK_REENC3=1` / `MK_IQ4NL_Q8=1` / `MK_AB_F32=1` / `MK_PF_GLU_MIX=0` (the earlier load and prefill paths
named in [The GGUF](#the-gguf)), `MK_MEMLOG=1` (device memory per load step). Compile-time options for tests and
analysis: `-DMK_SPEC_LOG=1` (with `MK_SPEC_LOG_FILE=path`: one JSON line per decode cycle), `-DMK_TEST_NAN=1` (with
`MK_TEST_NAN_AT=n`: poisons the first attention layer's K scales before decode launch n; the engine must stop with an
error and the server answer HTTP 500, or an error event in a started stream, and recover), `-DMK_KV_EMU=1` (scores
lower-bit KV formats per layer by rounding the q8 cache), `-DMK_PF_INFCHK=1` (counts, per prefill GEMM and layer, the
fp16 partial sums that overflow to inf / NaN, non-finite GEMM outputs, and the fp16 GEMM inputs the +-65504 clamps
saturate or turn from NaN into -65504; printed after each batched prefill; the served build compiles none of it. On
2026-10-17 none occurred on 38k- and 101.6k-token prompts).

### The GGUF

The kernels take the weight type as a template argument and dispatch on each matrix's type at run time
(`gemv_pairs_any`, `resid_any`, `ffn_gu`, prefill `gemm`), so every projection of the trunk and the MTP block can be
any of:

| type | how the kernels read it |
|---|---|
| Q4_K, Q5_K | as stored; in the wide-verify build the main-pass matrices are re-laid-out at load into fragment-ordered tiles (values unchanged) |
| Q6_K, Q8_0 | re-laid-out at load (values unchanged) |
| IQ4_XS | re-laid-out at load in pairs of 256-weight blocks, 272 B per pair (both block headers, then each block's index bytes in Q4_K order); 4-bit indices mapped to int8 weights with `__byte_perm` |
| Q3_K, IQ3_S | re-laid-out at load (values + scales streams, 114 / 112 B per 256 weights); the GEMV and the prefill GEMM feed exactly the integers and scales of the Q6_K re-encode they replace (`MK_REENC3=1`): bit-identical results at ~0.54x the bytes |
| IQ4_NL | re-laid-out at load into 144-byte blocks of 256 weights (the eight fp16 scales, then the index bytes); bit-identical to the Q8_0 re-encode it replaces (`MK_IQ4NL_Q8=1`) at 0.53x the bytes |

`token_embd` must be Q4_K (it stays in mapped pinned host memory, not in VRAM) and `output` Q6_K. Q8_0 `ssm_alpha` /
`ssm_beta` (UD) stay int8 with their block scales and the kernels apply the norm weight on the fly; other types are
dequantized to F32 at load and folded with the norm weight into fp32 rows (the same values; `MK_AB_F32=1` forces this
path). The exact re-encodes keep every weight's value (both formats store a small integer times the same fp16 `d` and an
integer sub-block scale), and the loader checks each re-encoded or re-laid-out row bit for bit against ggml's
dequantization of the original. A file with any other type is refused, naming the first tensor that does not match.
Which files that covers: [Model files](#model-files).

The engine reads the file with its own parser (`gguf.h`, mmap), independently of libllama's loader (which reads only
the vocabulary, in parallel), so the parser checks every read against the file size: version 2 / 3, the kv and tensor
counts, string and array lengths (no nested arrays, as ggml), duplicate keys or tensor names, 1-4 dimensions of at
most 2^63 elements, the type and whole blocks per row, `general.alignment` (a power of two up to 1 MiB), and each
tensor's data (aligned, inside the file); sizes are computed without overflow. A malformed file is refused with an
error naming the field. `tests/gguf_fuzz.cpp` (`make tests/gguf_fuzz`) feeds it malformed and randomly corrupted
files under AddressSanitizer / UBSan; UD-Q4_K_XL, UD-Q4_K_M and Q4_K_M parse to the same tensors and keys as before
the checks (2026-10-17).

The MTP draft block (`blk.64.*`, `nextn.eh_proj`) is the one lossy exception: its matrices stored wider than Q4_K
(Q5_K, Q6_K, Q8_0: all of them in UD-Q4_K_XL, `eh_proj` / `attn_v` / `ffn_down` in Q4_K_M) are re-quantized to Q4_K at
load (`ggml_quantize_chunk`, ~2.5 s of CPU for UD-Q4_K_XL, on a background thread during the load), so every MTP pass
streams less. Only the drafts change: they are verified by the unchanged main model, so greedy output and the sampled
output distribution stay exact. When it went in (2026-10-12, before wide verify) it saved 107 MiB of VRAM on
UD-Q4_K_XL, sped greedy decode up 1.5% and moved acceptance by under a point. `MK_MTP_Q4=0` keeps the stored types
(`MK_MTP_Q4=ab` loads both and the policy key `mtpq=0|1` switches between them, for A/B tests).

The draft head gets the same treatment (`MK_DRAFT_Q4`, on by default; `MK_DRAFT_Q4=0` turns it off and saves its 90 MiB
of VRAM): the 32,768 rows of the draft vocabulary are re-quantized from `output`'s Q6_K to a Q4_K copy at load (0.8 s),
which every draft pass reads instead of the shared lm_head rows (0.69x the bytes, contiguous). Tokens added to the draft
vocabulary on the fly still come from lm_head. Greedy output and the sampled distribution are unchanged (the main head
verifies); the policy key `dq4=0|1` switches it at run time. Measured on 2026-10-17: +2.1% greedy decode on the short
bench prompts (4 x 512 tokens, 7 drafts; cycles and accepted drafts identical), no measurable change at 38k and 101.6k.

Gate and up projections of different types (25 of the 65 layers of UD-Q4_K_XL) are read each in its own type over the
shared activations in decode. In prefill one mixed-type fused GEMM computes both (warps 0-3 gate rows, warps 4-7 up
rows) with the same products, sums and `silu(gate) * up` as two GEMMs plus a GLU kernel, so it is bit-identical to them
without the fp32 round trip (`MK_PF_GLU_MIX=0`: the two-GEMM path). unsloth's Q4_K_M has no mixed pairs. A plain
`llama-quantize ... Q4_K_M` stores `ssm_out`, `ssm_alpha`, `ssm_beta` and `nextn.eh_proj` as Q4_K: alpha / beta are
dequantized and the other two run in their stored Q4_K.

To build a file with unsloth's mix from the BF16 GGUF (unsloth's imatrix optional):

```sh
llama-quantize --imatrix imatrix_unsloth.gguf \
    --tensor-type ssm_out=q5_k --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 \
    --tensor-type nextn.eh_proj=q8_0 \
    Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
```

Benchmarks: `bench/bench.py` (decode on the 4 chat prompts, CLI; `--model` for another GGUF), `bench/think_bench.py`
(reasoning mode) and `bench/code_bench.py` (greedy decode, context sweep, second turn), the scripts upstream open-jet
measures with; `bench/*_bench.cu` are GEMV / GEMM / attention microbenchmarks (`attn_bench.cu` predates the current
prefill attention and no longer builds against `prefill_attn.cuh`).

## Server

`server/mk_server.py` serves the engine (through `libqwen38mk.so`, the C API in `capi.cpp`) with an OpenAI-compatible
`/v1/chat/completions`, streaming or not, plus `/health`, `/v1/models` and `/props`. `scripts/serve.sh` /
`scripts\serve.bat` start it with our settings (README); directly:

```sh
python3 server/mk_server.py -m ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf -c 120000 --kv q8 -d 7 --exact-reuse \
    --mmproj ~/models/mmproj-BF16.gguf
```

Options (defaults in brackets): `--host` [127.0.0.1], `--port` [18080], `-c` context [32768], `-d` drafts [4],
`--kv auto|f16|q8` [auto: q8_0 only when fp16 does not fit], `--draft-vocab` [draft_vocab_32k.bin], `--exact-reuse`
[off; the launchers turn it on], `--ck-every n` (a checkpoint every n prompt tokens [4096], 0 = off), `--ck-msgs 1|0`
(a checkpoint at each message start [1]), `--mmproj` (images, below), `--vision-backend auto|cpu|mtmd` [auto],
`--vision-threads` [0 = all], `--image-min-tokens` / `--image-max-tokens` [the mmproj's], `--image-cache` [8; images
kept between requests].

- Prompts are rendered with the chat template stored in the GGUF (jinja2): `chat_template_kwargs`
  `enable_thinking` / `preserve_thinking` / `reasoning_effort`, OpenAI `tools`. Reasoning is streamed as
  `reasoning_content`, Qwen's XML tool calls are returned as structured `tool_calls`. A call counts only with both
  closing tags (`</function>`, `</tool_call>`): one cut by `max_tokens` (or left open) is returned as text at the end
  of `content`, and the reply ends with `finish_reason` "length" (complete calls before it are still returned).
- Successive requests of a conversation only prefill what is new: the prompt is matched against the engine's sequence
  (prompts + emitted tokens). When the new prompt diverges inside what the engine already processed (an edited message,
  a re-rendered thinking block or tool call), the engine rewinds to the last DeltaNet checkpoint at or before the
  divergence. Checkpoints are saved inside the batched prefill (at the generation prompt, at each message start and
  every `--ck-every` tokens) into a pool of `MK_CK_SLOTS` slots in pinned host memory. Anything else starts over with a
  batched prefill.
- **Exact reuse** (`--exact-reuse`, sets `MK_EXACT=1`): the server never appends to rows the decode kernels wrote; it
  restores the last checkpoint at or before the divergence and prefills from there. The prefill computes rows that do
  not depend on where its chunks start (attention over absolute key segments, checkpoints on the absolute 32-token scan
  grid), so a reused request answers bit for bit like the same request sent fresh. The cost is the previous reply's
  re-prefill on a follow-up turn.
- **Keep-warm:** after each request (and at startup) `capi.cpp` runs a one-block, 32-thread polling kernel for
  `MK_KEEP_WARM_S` seconds (default 120, 0 = off), each launch ending itself after 50 ms; the next request stops it and
  waits a few microseconds. The driver keys the P-state to utilization, so only a continuous load holds full clocks.
  Measured on 2026-10-17 (RTX 3090): ~165 W while warm vs 30-35 W idle; from idle the clocks take 2.2-2.7 s to come up.
  `MK_KEEP_WARM_LOAD=blocks,threads,on_ms,off_ms,spin` changes the load (for measurements).
- Sampling: temperature [0.8], top_p [0.95], top_k (1..32; absent or <= 0 means 32, the engine's maximum; above 32 is
  clamped with a one-time warning), min_p [0.05], seed, presence_penalty / frequency_penalty (-2..2, vLLM's semantics:
  the tokens generated in this response); repetition_penalty is ignored (with a warning). One request at a time.
- Errors: a CUDA error or non-finite logits make the engine throw; the server answers HTTP 500 and keeps serving. In a
  stream that has started (status 200 sent), the error is the last event, OpenAI style:
  `data: {"error": {"message": ..., "type": "server_error", "code": 500}}`, then `data: [DONE]`. `/slots` requests get
  501.
- Limits (environment, read at startup): a request body needs a Content-Length (411 without one, also for chunked
  bodies; 400 when malformed) of at most `MK_MAX_BODY_MB` MB (default 128; 413, refused before the body is read, also
  on `Expect: 100-continue`). At most `MK_MAX_IMAGES` images per request (default 32; 400), and no more images are
  decoded once the prompt cannot fit the context. The image library refuses an image of more than
  `MK_IMAGE_MAX_PIXELS` pixels (default 100,000,000) from its header, before decoding it. Measured on 2026-10-17
  without the limit: a 446 KB PNG declaring 12000 x 12000 pixels took 39 s and 0.9 GB of server memory (decode, then
  4096 tokens encoded); a 20000 x 20000 one asks stb_image for ~2.4 GB.
- Disconnects: before the prefill and before each decode cycle the server checks whether the client closed its
  connection, and stops (a prefill that is running finishes first). The engine keeps what it generated, as after a
  reply cut by `max_tokens`, so the next request reuses or rewinds it as usual.
- Images: see [Images](#images-vision).

## KV cache

fp16, or q8_0 (per 32 values, fp16 scale) when fp16 does not fit the context or `--kv q8` asks for it. On top of the
q8_0 cache, `MK_KV` sets a format per attention layer and K / V: a comma-separated list of `<K|V|*><slot>=<4..8>[r]`,
slot 0-15 = the 16 attention layers, 16 = the MTP block, `*` or empty = all slots, `r` = stored Hadamard-rotated.
`MK_KV_WIN=W` keeps the most recent >= W positions of every low-bit side as q8_0 in a ring as well, so recent tokens
are read at 8 bits. The decode attention expands low-bit tiles to int8 in shared memory and runs the same pipelined
kernel as the q8_0 cache; the prefill attention reads them too. With exact reuse, each checkpoint also holds the window
rings as of its position.

Each layer's cache and the RoPE table hold 2 x `MK_MAXT` rows (16 in the served build, ~0.6 MB in all) past the
context: with more than 5 drafts, the last cycles before "context full" write the MTP block's draft rows up to 4
positions past it (since 2026-10-17; before, those rows overwrote the first rows of the next head or of the next buffer,
layer 3's cache, and a reused prefix then decoded differently; the server, which stops 16 tokens before the end, never
got there, a CLI or C API run decoding to "context full" did).

Presets (opt-in): D' = `MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048` (the README has its accuracy, VRAM and the
262,144-token context), D = `MK_KV='K*=6,V*=5' MK_KV_WIN=2048`. In the per-layer scans made for the low-bit formats, K
was ~2.3x as sensitive as V, and no setting that saved 0.5 GB or more stayed within the KL-divergence noise of the
q8_0 cache, so the presets are opt-in.

## Images (vision)

With `--mmproj <mmproj GGUF>` (projector `qwen3vl_merger`, e.g. unsloth's `mmproj-BF16.gguf`) the server accepts
OpenAI `image_url` parts (base64 `data:` URLs) and `/props` reports `modalities.vision`. The vision tower runs on the
CPU (no GPU memory: the 120k-context VRAM is unchanged); the engine takes its output rows in the batched prefill.

```sh
make vision        # libqwen38vision.so (+ tools/mkv_cli, tools/vision_ref); `make` builds it when llama.cpp has libmtmd
python3 server/mk_server.py -m Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj mmproj-BF16.gguf -c 120000 --kv q8 -d 7
```

- **Prompt, as llama-server builds it:** a message with images is flattened llama-server's way (text parts joined by
  newlines, none next to an image), each image becomes `<|vision_start|>`, its rows, `<|vision_end|>`, and decoding /
  preprocessing (stb_image; smart resize to multiples of 32 px within 8 .. 4096 tokens, the padded bilinear resize,
  normalisation) are llama.cpp's bit for bit: same token counts and pixels as `llama-server --mmproj`
  (`--image-min-tokens` / `--image-max-tokens` set the limits; fewer tokens encode faster).
- **Engine** (`MKImage`, `MKEngine::prefill(tokens, cks, ck_exact, images)`, `mk_prefill_img`): an image's positions
  take its fp32 rows instead of token embeddings (`pk_embed_img`; the MTP block takes them too, as llama.cpp does) and
  llama.cpp's Qwen-VL M-RoPE positions (IMROPE: t / row / column per frequency, the text after an image continuing at
  `+ max(nx, ny)`): rows of the RoPE table are rewritten for the image spans (gathered from the plain table), so no
  attention kernel changes and a text-only prompt runs exactly what it ran before (KLD logs byte-identical).
  Checkpoints, rewinds and exact reuse work across images; a rewind into an image re-feeds it.
- **Server reuse:** an image's placeholder tokens are all alike, so the reused prefix also ends at the first image
  that is not the same file (content hash) at the same place; decoded / encoded images are cached (`--image-cache`
  images between requests; a request holds every image it references until it ends, so it may use more).
- **Encoder** (`vision/mk_vision.h`, `--vision-backend auto|cpu|mtmd`): `cpu` = `vision/vit_cpu.cpp`, the tower in fp32
  with AVX2 / FMA kernels (packed 6x16 FMA GEMM micro-kernel, attention on the same kernel with an online softmax,
  exact tanh-GELU, weights widened from the file's type, a pool of pinned workers that take tasks dynamically;
  x86-64 with AVX2 + FMA: Zen 2-5, Haswell+; results independent of the thread count); `mtmd` = llama.cpp's libmtmd
  encoder (any CPU); `auto` = `cpu` where it runs.

Accuracy against an fp64 forward of the checkpoint (`vision/vit_ref.py`, 5 images of 196 .. 1024 tokens; 2026-10-16):
relative L2 error of the rows, lowest row cosine:

| encoder | rel. error | min row cosine |
|---|---|---|
| vit_cpu, BF16 mmproj | 4e-6 .. 4e-5 | 1.000000 |
| libmtmd, BF16 mmproj (BF16 GEMM inputs, fp16 K / V, table GELU) | 4e-3 .. 2.7e-2 | 0.903 |
| libmtmd, Q8_0 mmproj | 1.3e-2 .. 8.4e-2 | 0.889 |

At the output (megakernel, q8 KV, teacher-forced on llama.cpp's distributions given the fp64 rows; `tools/vision_ref`,
`qwen38_mk --kld --kld-images`): mean KL with the fp64 rows 0.0006 .. 0.0016 (the text-only level, 0.0012), with
vit_cpu's rows the same, with libmtmd's BF16 rows up to 2.5x that, with Q8_0 rows up to 7x. The BF16 mmproj is the
one to use: vit_cpu widens either file to fp32, so a Q8_0 file is no faster, only less accurate.

Encode time, Ryzen 9 5900XT (Zen 3, 16 cores / 32 threads), BF16 mmproj, steady state (2026-10-16):

| image | tokens | vit_cpu | libmtmd |
|---|---|---|---|
| 448 x 448 | 196 | 0.51 s | 0.80 s |
| 640 x 480 | 300 | 0.78 s | 1.29 s |
| 1024 x 1024 | 1024 | 3.56 s | 6.07 s |

The first encode after the server's load costs ~0.5 s more (the weights streamed in again); the server absorbs it with
a warm-up encode at startup. A new image after the CPU has been idle takes ~1.2-1.5 s for 300 tokens while the cores
ramp up.

## How it gets its speed

- **Weights stay in their GGUF k-quant encoding** (Q4_K / Q5_K / Q6_K / Q8_0 / IQ4_XS, plus Q3_K / IQ3_S / IQ4_NL),
  read once per cycle regardless of T. Q6_K, Q8_0, IQ4_XS, IQ4_NL, Q3_K and IQ3_S are re-laid-out at load time (values
  unchanged) so every lane does aligned 16/8-byte loads; IQ4_XS indices become int8 weights through two `__byte_perm`
  table lookups per 4 bytes (llama.cpp's `get_int_from_table_16`).
- **Wide verify** (served build): the main-pass phases run int8 tensor-core GEMMs over the fragment-ordered weights
  (every lane load one 16-byte LDG of contiguous bytes; integer to float without I2F through a magic-number add), so
  the cost of a verify pass grows little from 1 to 8 rows.
- **dp4a GEMV with shared activations** (the lean build, and the matrices wide verify does not take): activations are
  int8 (per-16/32 scales, as llama.cpp's q8_1), each warp streams 2-4 weight rows at once and reuses every
  shared-memory activation load across them, so T = 4 costs almost the same memory time as T = 1. Weight loads bypass
  L1 and are marked L2 evict-first.
- **No launches, few round trips**: activations are re-quantised by the producing phase (including the next RMS norm's
  weight), so consumers only async-copy (`cp.async`) them.
- **Gated-DeltaNet rollback by lazy replay**: the recurrent state is committed only through the surely-accepted token;
  accepted drafts are replayed from a small (k, v, g, beta) ring at the start of the next cycle, so rejected drafts cost
  no state snapshots or extra state traffic.
- **Split barrier before the DeltaNet core** (`MK_K5=1`, served build): the 48 DeltaNet heads go to the first 48 blocks
  that finish the DeltaNet projections; they load the head's state and replay its accepted drafts before waiting for
  the rest. Bitwise exact; decode +1.1% at 38k and +0.75% at 101.6k context (A-B-A, 2026-10-17).
- **Attention**: split-K flash decoding with tensor-core MMAs (Q.K^T and P.V in fp16 with fp32 accumulation, online
  softmax) over 32-position K/V tiles; the chunk boundaries depend on the context length only, so greedy output does
  not depend on the draft depth. With the q8 cache (`attn_core_q8`) each block runs a three-deep software pipeline with
  one barrier per tile (S of tile j, softmax of tile j - 1 and P V of tile j - 2 on specialised warps), the raw q8 K
  and V tiles stream through cp.async rings and both MMA operands are converted from int8 in registers. Same
  arithmetic as the simple loop (bit-identical output), about twice its KV bandwidth at 100K context; when it went in
  (2026-10-12) decode gained 18.7% at 101.6K depth and 7.3% at 38K. Chunks of a single tile (context <= 640) keep the
  simple loop, which has no pipeline to fill; low-bit `MK_KV` slots run the pipelined kernel too (`attn_core_q8m`).
  At T > 5 (up to 48 query rows) one pass over K / V serves all rows (`attn_core_w48`).
- **MTP drafting on device**, with the draft lm_head restricted to the 32,768 most frequent tokens
  (`draft_vocab_32k.bin`, built by `tools/vocab_freq` from local code + docs; covers ~99.4% of tokens), plus the tokens
  of the current sequence that are missing from it, added on the fly (identifiers of pasted code). Verification always
  uses the full vocabulary, so this affects only the acceptance rate, never the output.
- **Sampling** keeps the 32 best logits of each verified row; a draft is accepted with probability p(draft) and a
  rejected one is resampled from the rest of the distribution, which keeps the sampled distribution exact.
- **L2 warm-up**: before each barrier, blocks prefetch (evict-last) what their next latency-bound phase reads first
  (DeltaNet state, conv weights, norm weights, RoPE rows, first K/V tile).

## Prefill

A prompt (all but its last token) is processed in chunks of up to 2048 tokens (`MK_PF_CAP`) by ordinary kernel
launches (`prefill.cuh`), which leave exactly the state the megakernel expects (trunk and MTP K/V caches, committed
DeltaNet state, conv ring, hidden-state history); the last prompt token then goes through the megakernel, which
produces the first token and the first MTP drafts. The same path appends to a running conversation
(`MKEngine::prefill` on a non-empty sequence, e.g. the next user turn): it starts at the committed DeltaNet position
and also re-feeds the last emitted token and any accepted drafts not yet folded into the recurrent state; the output is
token-identical to processing the whole conversation as one fresh prompt. The host queues each chunk's work before it
waits for the previous chunk, so the GPU does not idle between chunks (`MK_PF_SYNC=1`: the previous flow); on Windows
those waits use a high-resolution waitable timer, since a plain sleep ends at the next timer tick (15.6 ms by default).

- **GEMMs on fp16 tensor cores straight from the k-quants** (`prefill_gemm.cuh`): 128 x 128 tiles (narrower token
  tiles for short prompts), software-pipelined through cp.async stages; weights stay in their GGUF blocks and are
  dequantised to fp16 in registers (exact small integers via the `0x6400` trick, then one fp16 scale and min per
  sub-block); activations are fp16 with the elements of each 16-group permuted so that a 4-byte weight word is directly
  an MMA A fragment. Products accumulate in fp16 over k = 128, then in fp32. Measured inside a 101.6k prefill: 89-105
  TFLOPS, 108-117 in isolation (2026-10-12). FFN gate and up share one GEMM whose epilogue writes `silu(gate) * up` as
  the down projection's fp16 input.
- **DeltaNet in chunked (WY) form on tensor cores**: per 32-token chunk, `A = tril(beta K K^T)` with the gate decays,
  `T = (I + A)^-1` (one warp, forward substitution), then `U = T beta V - T beta gamma (K S)`, `O = gamma Q S + P U` and
  `S <- gamma_last S + K^T (decay U)`. Value columns are independent, so a block owns (head, 64 columns) and each warp
  keeps its 128 x 8 slice of the state in fp32 MMA accumulators, turning it (and K S, U) into fp16 MMA operands in
  registers with `movmatrix`; gates are carried as log g, so tiny or zero gates are exact.
- **Attention**: causal flash attention over the KV cache on tensor cores (`prefill_attn.cuh`). A block takes 16 query
  positions of one KV head (6 q heads x 16 rows); two warps per q head split S by key positions and P V by head-dim
  halves, so 12 warps keep 64 fp32 accumulators each and K / V of the next phase load during the MMAs. P V accumulates
  in fp16 per 32-position tile, then in fp32. The q-tiles of the last, partial wave are split along the keys and merged.
  When this kernel went in (2026-10-11) the prefill of a 101.6K-token prompt went from 127.9 s to 90.3 s. With the q8
  cache (default), Q K^T runs on int8 tensor cores straight from the cache (`pk_attn8`): Q is quantized per 32 dims
  like the cache, each 32-dim block's int32 partial is scaled by q scale x k scale, the raw q8 K / V tiles stream
  through a cp.async ring and V is converted for the next tile while the current one computes (2026-10-12: 59 -> 84
  TFLOPS-equivalent in the microbenchmark at 100K; attention of a 101.6K-token prompt 38.2 s -> 25.8 s, no measurable
  KLD change). With `MK_EXACT=1` the attention works on absolute key segments, so a row does not depend on where its
  chunk starts.

## Limitations

- Specialised to this model's shapes (27B) and the weight types under [The GGUF](#the-gguf) (tested: unsloth's Q4_K_M
  and UD-Q4_K_XL); the loader rejects any other model or type. Tuned and tested on sm_86 (RTX 3090); built for sm_86,
  sm_89 and sm_120 GPUs with 24 GB or more. `make` builds for the installed GPU (`make ARCH=sm_89` to override).
- More than 4 drafts needs the served build and the q8 KV cache (`--kv q8`; low-bit `MK_KV` presets are fine).
- The decode kernel runs one block per SM and needs at least 80 SMs for more than 4 drafts (50 for up to 4): the
  engine refuses a smaller GPU at load, naming the requirement (an RTX 3090 has 82). On a GPU with more than 128 SMs it
  runs 128 blocks.
- Sampling supports temperature, top-k (<= 32), top-p and min-p (llama.cpp's order), and presence / frequency
  penalties on the generated tokens (applied first; greedy too; exact under speculative verification: a verify row also
  counts the drafts before it); no repetition penalty.
- Text and images (no video or audio; images as base64 `data:` URLs; Qwen-VL projectors only); one sequence: a
  conversation is reused while it grows (or diverges after a checkpoint); editing an earlier part, e.g. context
  compaction, re-prefills from the last checkpoint before the edit. One request at a time; no slot save / restore.

## Files

- `qwen38_mk.cu` / `qwen38_mk.h` - loader, repacking, the megakernel and the `MKEngine` API
- `wv.cuh` / `wv_host.cuh` - wide verify: the int8 tensor-core verify GEMM and its load-time weight re-layout
- `prefill.cuh` / `prefill_gemm.cuh` / `prefill_attn.cuh` - batched prefill kernels, their quantised fp16 GEMM and
  the prefill attention
- `loader.cuh` - the host side of the model load (read-ahead, reusable buffers); `gguf.h` - minimal mmap GGUF reader
- `capi.cpp` - C API (engine + libllama tokenizer + keep-warm) for `libqwen38mk.so`
- `server/mk_server.py` - OpenAI-compatible HTTP server; `scripts/serve.sh` / `scripts/serve.bat` - the launchers
- `main.cu` - CLI (tokenization via libllama)
- `compat_win32.h` / `.cpp`, `vision/win32.h` / `.cpp`, `build.ps1` - the native Windows build
- `draft_vocab_32k.bin` - MTP draft-head token subset (int32 ids)
- `tools/vocab_freq.cpp` - builds a draft vocabulary from a list of text files
- `tools/llama_ref.cpp` / `logit_stats.h` - llama.cpp's next-token distributions and the comparison with them (KL
  divergence, perplexity, top-token agreement)
- `vision/mk_vision.{h,cpp}` - the CPU vision encoder's C API (`libqwen38vision.so`): image decoding, backends
- `vision/vit_cpu.{h,cpp}` - the vision tower in fp32 with AVX2 / FMA kernels (the `cpu` backend)
- `vision/mkv_cli.cpp` - encodes images from the command line (grid, time, rows)
- `vision/vit_check.cpp` - test: vit_cpu against libmtmd (pixels, rows, times); `vision/vit_ref.py` - the fp64 reference
- `tools/vision_ref.cpp` - llama.cpp's distributions for an image prompt + the image rows it fed (`--kld-images`)
- `bench/` - benchmarks (see [Build / run](#build--run)); `tests/` - server tests (no GPU needed)
