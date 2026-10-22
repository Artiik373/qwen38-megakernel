# Benchmarks

Every number here was measured on one machine: an RTX 3090 (24 GB, 480 W limit), a Ryzen 9 5900XT and 64 GB of RAM,
Windows 11 with WSL2, CUDA 13.2, with unsloth's `Qwen3.8-27B-UD-Q4_K_XL.gguf` unless a section says otherwise. Each
result carries its date and the commit it was measured on.

How to read them:
- **Sessions.** The same binary runs several percent faster or slower from one session to the next with GPU temperature
  and clocks (up to ~11% between sessions on 2026-10-12, with the room and the fan speed). Absolute numbers are only
  compared within one session; changes made after a full run are given as A-B-A deltas (old build, new build, old build
  again in one hold), never added onto older absolutes.
- **Rates.** "tok/s" for decode is generated tokens over decode time, as the server reports it (`timings`). The
  short-prompt rows are means of the per-request rates, as the client reports them; token-weighted rates (total tokens
  over total time) are given next to them.
- **Comparisons with other engines** are limited to what we ran ourselves on this machine: llama.cpp b11548 (tuned) and
  upstream open-jet 2993f47.

Contents: [the current release](#the-current-release) - [full run 2026-10-20](#full-run-2026-10-20) -
[exactness](#exactness) - [KV cache presets](#kv-cache-presets) - [the conversation cache](#the-conversation-cache) -
[keep-warm](#keep-warm) - [Q4_K_M and upstream's scripts](#q4_k_m-and-upstreams-scripts) - [images and
video](#images-and-video) - [earlier full runs](#earlier-full-runs) - [reproducing](#reproducing).

## The current release

The current release is ef4a6dd (2026-10-21). Its GPU code is that of 7322fbd, the build of the last full run: ptxas
reports the same 740 functions, none changed. What changed since are the server and the engine's host code (the
`/v1/responses` API, pinned prefix checkpoints, the side-request and parking rules), measured with recorded client
traffic under [the conversation cache](#the-conversation-cache). The release checks of ef4a6dd (native Windows,
2026-10-21) gave the same token ids at 7 and 0 drafts, the same accuracy report and the same server replies as the Linux
references, and the same results as the 7322fbd Linux build in the side-request, parking and title scenarios under q8_0
and preset D.

## Full run 2026-10-20

7322fbd on Linux (WSL2), the server at 120,000 tokens of context in the configuration served that day (wide verify, 7
drafts, exact reuse, side requests and parked conversations on, prefill chunk 1024), one session from 19:17 to 21:18
with the GPU fans at 100%, after the card had new thermal paste and pads that afternoon. The llama.cpp column is
b11548's `llama-server`, tuned (q8_0 KV, 4 MTP drafts), driven by the same client with the same prompts and model file:
the best value of each row over three sessions (2026-10-12 and 10-13; the questions on 10-13).

| | llama.cpp (10-12 / 10-13) | this, q8_0 KV | this, preset D |
|---|---|---|---|
| greedy decode, short prompts (4 x 512 tokens), mean of request rates | 82.7 tok/s | 155.1 | **155.5** |
| the same, token-weighted | 76.2 | 131.7 | **132.1** |
| sampled decode, thinking on (8 runs), mean of request rates | 65.8 tok/s | 136.5 | **136.8** |
| the same, token-weighted | 63.9 | 130.4 | **130.8** |
| decode at 11.6k context (200 tokens) | 69.3 tok/s | **137.3** | 136.1 |
| decode at 38k context (256 tokens) | 62.3 tok/s | **117.9** | 101.9 |
| decode at 101.6k context (256 tokens) | 53.4 tok/s | **108.7** | 98.6 |
| read an 11.6k-token prompt | 8.24 s | **5.98 s** | 6.09 s |
| read a 38k-token prompt | 29.6 s | **23.2 s** | 24.1 s |
| read a 101.6k-token prompt from scratch | 101.0 s | **74.5 s** | 77.6 s |
| first token, short prompt | 124 ms | 66 ms | **65 ms** |
| follow-up turn in an 11.8k-token chat, prompt time | 323 ms | **178 ms** | 185 ms |
| edit an earlier message in an 11.7k-token chat, prompt time | 329 ms | **76 ms** | 76 ms |
| edit the end of a 9.1k-token system prompt, prompt time | 956 ms | **89 ms** | 97 ms |
| VRAM, whole card, server idle | 22.8 GB | 21.1 GB | **20.0 GB** |
| 282 benchmark questions: right, total time | 229, 2,727 s | 230, **1,681 s** | 230, 1,722 s |

Notes:
- **The follow-up turn** re-reads the previous reply, because exact reuse is on (`MK_EXACT_REUSE=0`: faster
  follow-ups, not bit-identical to a fresh request; on 2026-10-14 the same row took ~80 ms without exact reuse).
- **The 101.6k-token prompt from scratch** was read on a server started with `--side-max 0 --park-mib 0` (21:15 and
  21:17). With those features on, the short request the client sends to clear the cache runs beside the cached
  conversation and keeps it, so the client's own "from scratch" request was served from the cache.
- **Preset D's long-context decode rows** are not like-for-like: from 2,048 tokens on, the replies differ from q8_0's,
  and with them the share of accepted drafts. The server's console lines give tokens per engine cycle for each reply,
  so the time per cycle can be compared (reply time / (reply tokens / tokens per cycle)):

  | context | q8_0: tokens per cycle, ms per cycle | preset D: tokens per cycle, ms per cycle | D per cycle |
  |---|---|---|---|
  | 11.6k (200 tokens) | 3.75, 27.4 | 3.83, 28.2 | +2.8% |
  | 38k (256 tokens) | 3.53, 29.9 | 2.98, 29.2 | -2.4% |
  | 101.6k (256 tokens) | 3.61, 33.3 | 3.40, 34.5 | +3.8% |

  One reply each; the console prints seconds to 0.01 s, so each derived value is good to about 0.5%.
- **Preset D's edit rows** come from a re-run at 21:06: a browser opened by accident during the first pass overlapped
  them. The same applies to the timing of its GSM8K and MMLU-Pro sets (1,723 s in total with the affected timings);
  answers are greedy and did not change.
- **The questions**, per set, were answered the same way by q8_0 and preset D: GSM8K 50 / 50, MMLU-Pro 47 / 70, ARC-C
  47 / 50, MATH-500 29 / 40 (and 9 / 10 with thinking on), HumanEval 37 / 41, NIAH 9 / 9, LongBench v2 2 / 12. The
  question client gives LongBench v2 answers 128 tokens, which cut 8 of its 12 answers in both engines on 2026-10-14;
  the 262k runs below give them 1,024. The total time includes the side requests and parks the questions cause beside
  the cached long conversation (the 2026-10-15 run had no host store).
- **GPU state** (nvidia-smi at the start / end of each pass): q8_0 51 / 68 C, SM 2040 / 1950 MHz; preset D 59 / 72 C,
  1995 / 1950 MHz; memory 9851 MHz and fans 99-100% throughout.
- **Against the previous full run** (2026-10-15, 7d67c5d): decode +1.2% to +4.8%, the 101.6k prompt from scratch 77.7 ->
  74.5 s, the questions 1,739 -> 1,681 s. The engine changes in between were measured one by one (A-B-A, see below);
  the repaste lowered the memory junction from 103.7 to 91.0 C under the same decode load and raised the SM clock by
  ~36 MHz, so part of the session-to-session difference is thermal.

### Preset D at the full context

The same session, a server at 262,144 tokens with preset D (20:21-21:05):

| | |
|---|---|
| prompt | 260,896 tokens, read from scratch in 326 s |
| decode at that depth | 66 tok/s (a summary of up to 512 tokens, no thinking) |
| follow-up turn, prompt time | 1.28 s |
| VRAM, whole card, after the prompt | 23.5 GB |
| needle in a haystack (4 needles, asks one), at 190k and the longest length that fits | 4 / 4 |
| LongBench v2 at that length (6 questions, 1,024 answer tokens) | 4 / 6 |

Natively on Windows, with the desktop on the same card, a 240k-context server with preset D found a needle at 227,170
tokens (prompt read in 275.7 s) with a peak of 24,116 of 24,576 MiB (2026-10-18, 282cd55).

### How the run was measured

- **Client:** a Python script (not part of this repository; the one of the 2026-10-14 and 10-15 runs, which sends the
  requests of the 2026-10-12 / 13 llama.cpp sessions plus a clock warm-up), against `/v1/chat/completions` without
  streaming; it reads the server's `timings` (prompt and decode time) and `usage` (cached tokens). One warm-up request,
  then ~3 s of decode so the timed requests start at full clocks.
- **Short prompts:** four chat prompts (a Python prime check with doctests, how a hash map works, a 300-word story, a
  Flask-to-FastAPI migration plan): greedy without thinking, 512 tokens each; then sampled with thinking (temperature
  1.0, top_p 0.95, top_k 20) with seeds 11 and 22, up to 768 tokens each. First token: three one-word questions,
  `max_tokens` 1.
- **Long context:** an 11.6k-token excerpt of a 38k-token document with a summary request (greedy, thinking on, 200
  tokens) and a follow-up question; the whole 38k-token document, then a 101.6k-token one, each with a 256-token answer
  (greedy, no thinking). The 101.6k prompt reuses the 38k one's prefix in the main pass; the from-scratch row is the
  separate run above.
- **Edits:** a three-turn chat over 30k characters of the document, then turn 2's user message edited; a 9.1k-token
  system prompt, then its last word changed. Prompt time of the edited request.
- **VRAM:** `nvidia-smi` memory used for the whole card (other processes included).
- **Questions:** 282 items, greedy, zero-shot, the same prompts for both engines, thinking off except the 10 MATH-500
  items with thinking; scored offline. The llama.cpp numbers come from the same client on 2026-10-13.

## Exactness

### Speculation

- **Greedy output does not depend on the draft depth.** The attention's split-K partition depends only on the context
  length, and the verify GEMMs compute every row the same way whatever the number of rows, so the same prompt gives the
  same token ids at 0, 4 and 7 drafts. Every release is checked this way on four prompts (the 2026-10-21 release:
  identical).
- **Sampling keeps the distribution of sampling without drafts.** A draft is accepted with probability p(draft) and a
  rejected one resampled from the rest of the distribution. Checked with chi-square and PIT tests; the counter-based
  random numbers, keyed by seed and position, were checked with a CPU replica of the loop (396k re-drafted positions)
  and a PIT test on 13,401 re-drafted rows of the engine at 7 drafts (KS p = 0.70; 2026-10-17). Two opt-in policy
  settings that were biased were fixed or restricted then ([ENGINE.md](ENGINE.md#correctness)).
- **The MTP layer only drafts.** A model file without one runs at depth 0, and its greedy output equals the drafted
  engine's bit for bit (UD-Q4_K_XL, 2026-10-17).

### Exact reuse

With `--exact-reuse` (`MK_EXACT=1`; the launchers set it), a reused request is answered bit for bit like the same
request sent fresh. The test is a state digest after the prompt: every K / V row, every row of the 8-bit window ring,
the DeltaNet and convolution state, compared with a fresh prefill of the whole prompt.

| case | result | date |
|---|---|---|
| follow-up turns, edits, abandoned replies, 38k context, q8_0 and preset D', several chunk sizes | digest bitwise equal to fresh | 2026-10-14 |
| sampled follow-up / edit / restored park with a fixed seed (the draft vocabulary rebuilt as a fresh prefill builds it) | 9 / 9 equal to fresh (before the change, 5 of 9 differed) | 2026-10-20 |
| a side request beside a cached conversation, a park and restore | digest after the side request / restore bitwise equal to before; the next replies equal to the conversation alone on a fresh server, greedy and sampled (8 / 8) | 2026-10-19 |
| a new session resumed at a pinned checkpoint (live, parked, after a side request, after the conversation grew to 63k) | replies equal to the same request on a fresh server, q8_0 and preset D, greedy and sampled | 2026-10-21 |

What it costs: a follow-up turn re-reads the previous reply (11.6k-token chat, 7 drafts: 167 instead of 78 ms; with a
thinking reply kept in the history: 469 instead of 86 ms; 2026-10-14). Edits cost the same either way. A fresh
98k-token prompt took 1.3% longer, and restoring a checkpoint and reading 32 / 128 / 512 tokens after it took 2.6% and
1.6% less and 0.7% more time (A-B-A, q8_0, 2026-10-14). The accuracy report is the same within noise: with exact reuse
98.662% / 0.00118 on the short set and 99.121% / 0.00231 at 32k, without it 98.654% / 0.00116 and 99.023% / 0.00232 on
the build it was added to (2026-10-14). The llama.cpp comparison below was measured without exact reuse (the CLI's
default); the launchers' server runs with it.

### Against llama.cpp

The engine is not bit-identical to llama.cpp: the arithmetic is ordered and rounded differently, so where two tokens are
close the other one can win. It is scored against llama.cpp's own next-token distributions: llama.cpp b10246 generating
token by token with its q8_0 KV cache, on 13 sequences of this repository's code and docs (13,299 positions scored after
a 1024-token prompt each), with the 256 most likely tokens stored per position (`tools/llama_ref`, `--kld`).

| against llama.cpp generating token by token | top token agrees | KL divergence (mean, nats) | perplexity (reference 2.8099) |
|---|---|---|---|
| llama.cpp, batched instead (2026-10-11) | 98.489% | 0.00166 | 2.8116 |
| this engine, prefill then decode kernels (`mixed`), q8_0 KV (2026-10-17) | 98.737% | 0.00119 | 2.8085 |
| this engine, prefill kernels only, q8_0 KV (2026-10-17) | 98.744% | 0.00124 | 2.8073 |

The `mixed` report has been the same since 2026-10-15, still at 7322fbd on 2026-10-20, and the native Windows builds
give it byte for byte. Most disagreements are near-ties: the median lead of llama.cpp's token over the engine's is 0.043
nats (llama.cpp batched: 0.051), and 1 of 168 exceeds 0.5 nats. At long context (same reference setup, q8_0 KV):
99.121% / 0.00219 over 4,094 positions of 2 sequences at ~32k (2026-10-17), and 99.267% / 0.00163 over 4,095 positions
of one 64k sequence (2026-10-15, 7d67c5d).

### Windows and Linux

The Windows build (MSVC + nvcc) against the Linux build (g++ + nvcc) of the same source (fdad907), on the same model
file, 2026-10-16:

| check | Windows vs Linux |
|---|---|
| greedy token ids, 7 and 0 drafts | identical |
| `--kld` report against llama.cpp (13,299 positions) | identical |
| server replies: text, thinking, an image, an image follow-up turn | identical |
| ptxas registers and spills, every function | identical |
| decode at 38k context | 173.0 vs 173.1 tok/s |
| prefill of a 38k-token prompt | 22.3 vs 22.5 s |
| decode, chat prompts 2-4 (512 tokens, 7 drafts) | within 1.5% |
| image encode, 196 / 300 / 1024 tokens | 0.59 / 0.81 / 3.56 s vs 0.50 / 0.80 / 3.95 s |

The first chat prompt after the model load decoded faster on Windows (220 vs 132-145 tok/s, the same tokens), likely
because the GPU's clocks ramp up faster there. Every release since has passed the same identity checks natively; the
2026-10-21 release also replayed the side-request and parking scenarios under q8_0 and preset D and the title scenario
under preset D (all identical to Linux).

The image encoder's rows differed between the two builds at fp32 rounding level (relative 2e-6 to 1.3e-5, below either
build's error against an fp64 forward of the vision tower): its attention took e^x from the C library, and glibc's and
Microsoft's `expf` differ in the last bit on 193,924 of the 2^32 inputs. Since 2026-10-18 the encoder computes it with
glibc's algorithm: the Linux rows are unchanged and the Windows rows equal them bit for bit (57 images: stills and every
frame pair of 4 video clips, at 7 / 8 / 30 / 32 threads).

## KV cache presets

`MK_KV` stores K and V below 8 bits; `MK_KV_WIN=2048` keeps the most recent 2,048 positions in an 8-bit ring next to the
low-bit cache. Measured with a rounding emulator first (2026-10-12 / 13): the low-bit loss of K and V sits almost
entirely in the most recent ~0.5-2k positions, so with the window 5-bit V and 6-bit K cost little, and rotation of K
or V buys nothing under it. While a whole sequence fits in the window, every attention read comes from the 8-bit ring,
and the output is q8_0's: the short-prompt accuracy report below is byte-identical to q8_0's, and on 2026-10-20 the
first 17 replies of the run (short prompts, greedy and sampled) had the same lengths, tokens per cycle and accepted
drafts under both caches.

Accuracy against llama.cpp (the reference above), top-token agreement / mean KL in nats. An fp16 cache, more precise
than q8_0, marks the floor of the measurement:

| KV cache | short prompts (13,299 positions) | ~32k (4,094) | 64k (4,095) |
|---|---|---|---|
| fp16 (2026-10-12 / 10-13) | 98.774% / 0.00113 | 99.096% / 0.00226 | 0.00161 |
| q8_0 | 98.737% / 0.00119 (2026-10-17, 7322fbd 10-20) | 99.121% / 0.00219 (10-17) | 99.267% / 0.00163 (10-15) |
| D: `K*=6,V*=5` + window | = q8_0 (2026-10-20, 7322fbd) | 98.950% / 0.00236 (10-20, 7322fbd) | 99.243% / 0.00160 (10-13) |
| D': D + 8-bit MTP layer | = q8_0 (10-15, 7d67c5d) | 0.00236 (10-15) | 0.00161 (10-15) |
| C: `V*=5` + window | = q8_0 (10-13) | 99.072% / 0.00228 (10-13) | 99.316% / 0.00159 (10-13) |
| 4-bit everywhere: `*=4`, no window | 0.00547 (10-15) | | |

The MTP layer's cache only feeds the drafts, so D and D' score the same wherever both were measured. On the 2026-10-12
/ 13 builds of the fp16 row, q8_0 read 0.00113 on the short set and 0.00219 at 32k: fp16 was no better than q8_0 (the
reference itself was generated with a q8_0 cache). The values of the short set moved by up to 0.00006 between builds as
the kernels changed, which is why each cell carries its date.

Memory and speed, preset D against q8_0:

| | q8_0 | preset D | measured |
|---|---|---|---|
| VRAM, idle server at 120k context | 21.1 GB | 20.0 GB | 2026-10-20, 7322fbd |
| KV cache at 120k context | | 1,132 MiB less | 2026-10-13 |
| the full 262,144-token context | | 23.5 GB | 2026-10-20, 7322fbd |
| reading an 11.6k / 38k / 101.6k-token prompt | 5.98 / 23.2 / 74.5 s | 6.09 / 24.1 / 77.6 s | 2026-10-20, 7322fbd |
| decode time per engine cycle, 11.6k / 38k / 101.6k | | +2.8% / -2.4% / +3.8% | 2026-10-20, 7322fbd (derived, above) |
| 282 questions | 230 right | 230 right, the same per set | 2026-10-20, 7322fbd |

Other presets (2026-10-13, measured as KV memory saved at 120k context against q8_0, KL relative to q8_0 on the short /
32k / 64k sets, decode time per cycle and prefill rate at 38k / 101.6k context):

| preset | spec | KV saved at 120k | KL vs q8_0 | decode per cycle | prefill rate |
|---|---|---|---|---|---|
| C | `MK_KV='V*=5' MK_KV_WIN=2048` | 691 MiB | = / +4% / +2% | +1.2% / +3.7% | -1.3% / -2.7% |
| D | `MK_KV='K*=6,V*=5' MK_KV_WIN=2048` | 1,132 MiB | = / +1% / +5% | +2.9% / +5.0% | -3.3% / -6.7% |
| E | `MK_KV='K*=6r,V*=4r' MK_KV_WIN=2048` | 1,381 MiB | +2% / +5% / +10% | +2.4% / +4.9% | -2.0% / -4.2% |

Later the same day a decode attention that serves 6-8 verified rows in one pass cut preset D's time per cycle at
101.6k context from 44.0 to 39.4 ms (q8_0: 41.2 to 38.0 ms), and on 2026-10-14 compile-time 6-bit K / 5-bit V prefill
kernels cut the prefill time of D' by 2.5% / 1.3% at 101.6k / 38k; the 2026-10-20 numbers above include both. D' (the
MTP layer at 8 bits, 70 MB more) decoded 1.2% / 2.9% faster than D at 101.6k / 38k on 2026-10-13. Other formats tried
in the emulator (3-bit, codebooks, per-row scales, random signs, angle coding) scored worse than the plain
per-32-value grid.

## The conversation cache

How it works: [SERVER.md](SERVER.md). Costs (WSL2, q8_0 unless noted, host memory already allocated, 2026-10-19):

| operation | 12k tokens | 38k tokens | 101.6k tokens | 120k tokens |
|---|---|---|---|---|
| park a conversation in host memory | 31 ms | 62 ms (D: 63 ms) | 151 ms | 177 ms (4.99 GB) |
| restore it | 29 ms | 65 ms (D: 74-92 ms) | 152 ms | 178 ms |
| read it again instead | 7.4 s | 23.9 s | 78.7 s | 98.4 s |

- The first park of a large conversation pays for page-locked host memory at ~1.1 s per GB (5.7 s for 120k tokens); it
  is kept for reuse until the server exits.
- A side request (a client's title or summary request beside a long conversation) moves only the rows it overwrites:
  12 ms each way for a 2.7k-token side prompt, 20 / 19 ms for 8k (preset D: 17-22 ms), whatever the length of the cached
  conversation (2026-10-19). Its own conversation is parked afterwards: 10-15 ms for a title, ~10 ms back, 15-21 ms for
  ~400 tokens (2026-10-20).
- Two OpenCode 2.0.32 sessions working at the same time on native Windows (230k context, preset D) finished in 110 s
  with ~72k prompt tokens read, against 242 s and ~232k tokens with the host store off (2026-10-19).

### Pinned prefix checkpoints

A new session of a client that keeps its system prompt and tools resumes after them. Measured with each client's
captured first requests, preset D, 131,072 context, native Windows, 2026-10-21: the server before the change (538d441
with 7322fbd's engine) against the change (merged as 2e659e0 + 4eb7b51). "Live": a 52-55k-token conversation of the
same client is cached; "parked": it was parked by another 9.8k-token conversation. Times are the client's wall time,
including the park of the long conversation and ~2.5-3.5 s of first-use host-memory allocation.

| new session (prompt tokens) | | before: cached / time | after: cached / time |
|---|---|---|---|
| Claude Code, default settings (11,850) | live | 0 / 10.12 s | 8,288 / 5.92 s |
| | parked | 0 / 8.67 s | 8,288 / 3.98 s |
| Claude Code with its cache-friendly settings (11,809) | live | 0 / 9.96 s | 9,024 / 5.11 s |
| | parked | 0 / 8.28 s | 9,024 / 3.27 s |
| Codex CLI, `/v1/responses` (8,847) | live | 0 / 8.01 s | 8,672 / 3.44 s |
| | parked | 32 / 6.64 s | 8,672 / 2.00 s |
| Crush, another directory (11,168) | live | 0 / 9.30 s | 10,304 / 4.11 s |
| | parked | 32 / 7.67 s | 10,304 / 2.12 s |

### Replayed client traffic

The requests 9 clients sent during their test drives on 2026-10-21 were replayed request by request against a native
Windows server (preset D, 230k context): the release (ef4a6dd) against the server before its pinned checkpoints and
side-request rules (538d441, the same engine). The replay sends the recorded requests back to back; the server samples
its own replies, so reply lengths differ slightly from the live runs. 2026-10-21, 23:01-00:31.

| | before (538d441) | ef4a6dd |
|---|---|---|
| 9 single-client sequences: time reading prompts | 885 s | 800 s |
| tokens read | 1,267k | 1,150k |
| first turns of new agent sessions (50 requests) | 173 s | 120 s |
| DeepSeek Harness / Crush / Qwen Code / Pi / Goose / Claude Code first turns | 29.8 / 30.5 / 62.3 / 8.4 / 5.7 / 27.9 s | 11.3 / 20.2 / 47.2 / 5.2 / 3.1 / 24.3 s |
| Open WebUI first turns | 92 s | 75 s |
| Claude Code + Open WebUI interleaved: tokens read, cache misses | 508k, 28 | 489k, 10 |
| DeepSeek Harness + Aider interleaved: tokens read, cache misses | 266k, 32 | 218k, 7 |

What it costs: every parked conversation now carries its pinned checkpoint (+288 MiB with preset D), so fewer fit in
the host store (Claude Code's sequence held 4 instead of 6, and two of its requests cached 4-6k fewer tokens). In the
Open WebUI sequences, the default title request on a long chat parked the chat with low priority and the next request
evicted it: one turn read 46.8k tokens again (33.7 s instead of 0.5 s). With Open WebUI's title and tags templates set
as in [CLIENTS.md](CLIENTS.md) (quoting only the end of the chat), the title request is short and this does not
happen.

`--park-min` (among its uses: conversations of fewer tokens are not parked; [SERVER.md](SERVER.md)), both interleaved
sequences, ef4a6dd, fixed seeds:

| `--park-min` | parks | GiB copied | real conversations evicted | tokens read | time reading prompts |
|---|---|---|---|---|---|
| 1024 (the default) | 68 | 106.5 | 15 | 706,771 | 489 s |
| 4096 | 51 | 81.4 | 8 | 709,651 | 489 s |

4096 halved the evictions of real conversations and copied a quarter less for the same prompt time; the default stays
1024.

## Keep-warm

After each request the engine keeps the GPU's clocks up for `MK_KEEP_WARM_S` seconds (default 120). On native Windows
at 147054b, a 255-token reply after 65 s of idle started in ~65 ms instead of 391-397 ms and decoded at 80.2 instead of
61-63 tok/s (2026-10-17). Holding the clocks costs power: 164-173 W instead of 30-35 W at idle on this card
(2026-10-17).

## Q4_K_M and upstream's scripts

unsloth's plain `Qwen3.8-27B-Q4_K_M.gguf` runs on the same engine. Its last full run was on 2026-10-14 (b3be928, one
session with UD-Q4_K_XL, q8_0 KV, 7 drafts, 120k context, WSL2, fans at 96%):

| | UD-Q4_K_XL | Q4_K_M |
|---|---|---|
| greedy decode, short prompts | 144.7 tok/s | 148.1 tok/s |
| decode at 11.6k / 38k / 101.6k context | 128.6 / 109.9 / 102.0 | 145.7 / 109.4 / 106.0 |
| read a 101.6k-token prompt from scratch | 79.0 s | 77.6 s |
| VRAM, whole card | 21.1 GB | 20.7 GB |
| 282 questions right | 230 | 231 |

The two files disagreed on 15 of the 282 answers and had the same perplexity. Accuracy against llama.cpp on Q4_K_M:
98.887% / 0.00127 (a wide-verify build with up to 4 drafts, 2026-10-13; llama.cpp batched on that file: 98.489% /
0.00184).

Upstream open-jet's own benchmark scripts (`bench/code_bench.py`, `bench/think_bench.py`, best of 2, greedy code runs,
the thinking runs at temperature 1.0 with seeds 1000 / 1001), on Q4_K_M: this engine at b3be928 with 7 drafts and the
q8_0 cache (2026-10-14) against upstream's engine at open-jet 2993f47 with its 4 drafts on the same card (2026-10-13):

| test | open-jet 2993f47 | this engine |
|---|---|---|
| writing code (no context) | 133.8 tok/s | 173.3 tok/s |
| 30k tokens of code in context | 87.4 | 117.5 |
| short prompts (mean of 4) | 118.0 | 149.2 |
| reasoning mode (4 prompts x 2 seeds, 1,024 tokens) | 89.1 | 125.4 |
| prefill, 29.6k-token prompt | 1,527 tok/s | 1,811 tok/s |

## Images and video

On the Ryzen 9 5900XT (32 threads):
- **Images** (2026-10-16): a 196 / 300 / 1024-token image encodes in 0.51 / 0.78 / 3.56 s back to back; llama.cpp's
  libmtmd on the same CPU took 0.80 / 1.29 / 6.07 s. A new image after the CPU has been idle takes ~1.2-1.5 s for 300
  tokens while the cores ramp up. The encoder's rows are within 4e-6 to 4e-5 (relative) of an fp64 forward of the
  vision tower (libmtmd: 4e-3 to 2.7e-2). On six image prompts, the mean KL against llama.cpp fed the same image rows is
  0.0006-0.0059 (text only: 0.0012). Prompt tokens equal llama-server's on 7 of 7 image prompts.
- **Video** (2026-10-18; frames sampled at 4 fps as llama.cpp does, two consecutive frames per grid of tokens): a 640 x
  360 clip costs ~450 prompt tokens per second of video (228 at 2 fps) and its frame pairs encode in 0.56 s each, about
  real time at 4 fps; 320 x 240: 167 tokens per second, 0.22 s per pair; 1280 x 720: 1,861 tokens per second, ~3.2 s
  per pair. The prompt token counts equal llama-server's on four clips, and the frame-pair rows are within 5e-6 to
  1.1e-5 of an fp64 forward (libmtmd's: 5.5e-3 to 2.2e-2).

The image and video encoder runs on the CPU, so it takes no VRAM: with an image projector loaded the idle server's
VRAM at 120k context did not grow (2026-10-16).

## Earlier full runs

The 2026-10-14 run with charts and raw data, and the gates behind each change up to 2026-10-16, are in the open-jet
fork's [perf docs](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf/perf), where this engine was
developed until it moved here.

### 2026-10-15 (7d67c5d)

The same client and settings as 2026-10-20 (WSL2, 120k context, 7 drafts, exact reuse), q8_0 and preset D', without the
host store. The llama.cpp column is the one above.

| | this, q8_0 KV | this, preset D' |
|---|---|---|
| greedy decode, short prompts (mean / token-weighted) | 151.3 / 129.0 tok/s | 150.2 / 128.2 |
| sampled decode, thinking on (mean / token-weighted) | 134.9 / 124.4 tok/s | 134.1 / 123.8 |
| decode at 11.6k / 38k / 101.6k context | 135.1 / 114.3 / 105.4 tok/s | 131.3 / 101.7 / 96.8 |
| read an 11.6k / 38k / 101.6k-token prompt | 6.15 / 22.7 / 77.7 s | 6.32 / 23.6 / 82.2 s |
| first token, short prompt | 68 ms | 70 ms |
| follow-up turn (11.8k chat) / edit a message / edit the system prompt, prompt time | 182 / 75 / 90 ms | 190 / 79 / 100 ms |
| VRAM, whole card, server idle | 21.3 GB | 20.3 GB |
| 282 questions | 230 right, 1,739 s | 230 right, 1,775 s |

Preset D' at 262,144 tokens: a 260,896-token prompt in 334 s, decode at 63 tok/s at that depth, a follow-up turn in 1.3
s, 23.4 GB.

### 2026-10-14 (b3be928)

WSL2, fans at 96%, UD-Q4_K_XL and Q4_K_M, each with q8_0 and preset D'. Rows as above.

| | UD, q8_0 | UD, D' | Q4_K_M, q8_0 | Q4_K_M, D' |
|---|---|---|---|---|
| greedy decode, short prompts, tok/s | 144.7 | 145.0 | 148.1 | 147.7 |
| sampled decode, thinking on, tok/s | 129.8 | 128.9 | 136.2 | 138.7 |
| decode at 11.6k / 38k / 101.6k, tok/s | 128.6 / 109.9 / 102.0 | 125.9 / 97.4 / 94.0 | 145.7 / 109.4 / 106.0 | 142.2 / 108.7 / 99.8 |
| first token, ms | 71 | 69 | 70 | 70 |
| follow-up turn, prompt time, ms | 183 | 191 | 181 | 186 |
| read an 11.6k / 38k / 101.6k-token prompt, s | 6.24 / 23.0 / 79.0 | 6.45 / 24.0 / 83.1 | 6.13 / 22.7 / 77.6 | 6.24 / 23.3 / 81.2 |
| edit a message / the system prompt, ms | 81 / 92 | 79 / 101 | 74 / 86 | 77 / 95 |
| VRAM, whole card, GB | 21.1 | 20.1 | 20.7 | 19.7 |

Preset D' gave the same answer as the 8-bit cache on all 282 questions with both files. At 262,144 tokens with D', a
261k-token prompt took 337 s, decode ran at 62 tok/s and a follow-up turn started in 2.2 s; needle retrieval was 8 of 8
at 190k and 261k on both files.

### Changes between the 2026-10-15 and 2026-10-20 runs

Each measured A-B-A in one hold against the build before it:

| change | effect | measured |
|---|---|---|
| split barrier before the DeltaNet core (`-DMK_K5=1`, the served build) | decode +1.1% at 38k, +0.75% at 101.6k (at the edge of noise) | Linux, 2026-10-17 |
| the next prefill chunk queued while the current one runs | GPU idle in a 38k prefill 11.8 -> 0.19 ms; prefill time unchanged | native, 147054b, 2026-10-17 |
| draft head reads a Q4_K copy of its rows (`MK_DRAFT_Q4`, default on) | greedy short prompts +2.1%; no change at 38k / 101.6k | 2026-10-17 |
| per-chunk waits before ffn_down instead of a grid barrier (`MK_C1`) | +0.41% at 38k, +0.59% at 101.6k, +0.44% short | Linux, 2026-10-20 |
| control block in mapped host memory, barrier counts carried across launches (`MK_EGRESS`) | +1.0% at 38k, +0.97% short, +0.8% at 101.6k (Linux, two holds: +0.7% to +1.3%) | native, 2026-10-20 |

Greedy token ids and the accuracy report are unchanged by all of them. The draft head's copy changes which drafts are
proposed, so a sampled reply with a fixed seed can differ from the one before it; its distribution is the same.

## Reproducing

- **Server benchmarks:** [bench/suite](../bench/suite/README.md) has the clients behind the server tables (speed, the
  282 questions, the from-scratch prefill, the 262,144-token run), the scorer and the summary script. The datasets are
  not in this repository: `python -I bench/suite/fetch.py` downloads them and checks every item against the hashes of
  the items these runs used (`bench/suite/items.json`). HumanEval answers are executed only with `score.py
  --exec-humaneval`. Re-scoring the saved 2026-10-20 answers with the suite's scorer gives the same 230 right answers
  for q8_0 and preset D, item for item, and its summary script gives the same table.
- **Speed, CLI:** `python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin --model <gguf>` (the four
  chat prompts through the CLI; it runs the CLI's default KV cache, so at most 4 drafts); `bench/code_bench.py` and
  `bench/think_bench.py` are upstream's scripts; `bench/*.cu` are kernel microbenchmarks.
- **Accuracy against llama.cpp:**

  ```sh
  make tools/llama_ref
  tools/llama_ref -m model.gguf -f code.txt -f prose.txt --score-from 1024 -b 1 --ref-out llama.ref
  tools/llama_ref -m model.gguf --kld llama.ref                            # llama.cpp batched, for scale
  ./qwen38_mk -m model.gguf --kld llama.ref --score-path mixed --kv q8     # this engine
  ```

- **Draft-depth independence:** `./qwen38_mk -m <gguf> -p "..." -n 256 -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin
  --ids`, then the same with `-d 4` and `-d 0`: the ids must be identical.

More on each check: [BUILDING.md](BUILDING.md#checking-a-build).
