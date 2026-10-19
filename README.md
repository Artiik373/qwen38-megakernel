# qwen38-megakernel

The development home of a single-GPU engine for **Qwen3.8-27B on one RTX 3090 (24 GB)** and the OpenAI-compatible
server around it, with image and video input. It contains:

- **The engine**, a CUDA megakernel tuned on an RTX 3090. Each speculative decode cycle is one persistent cooperative
  kernel launch: the model's own MTP head drafts up to 7 tokens, and the main model verifies all 8 rows in one pass,
  with its matrix products on int8 tensor cores ("wide verify"). Prompts go through a separate batched tensor-core
  prefill. Weights stay in their GGUF k-quant encoding: unsloth's `UD-Q4_K_XL` and `Q4_K_M` files run as they are.
- **Exact output.** Greedy output is the same at every draft depth, so speculation never changes it, and sampled output
  has the same distribution as without drafts. With exact reuse (the launcher's default), a follow-up turn or an edited
  conversation answers bit for bit like the same request sent fresh.
- **The server** (`server/mk_server.py`): `/v1/chat/completions` (streaming, tools, `reasoning_content`, stop
  strings), the Anthropic Messages API (`/v1/messages`, streaming, `/v1/messages/count_tokens`: Anthropic SDK clients
  such as Claude Code), llama-server's `/health`, `/props`, `/v1/models`, `/slots`, `/metrics`, `/tokenize`,
  `/detokenize`, `/apply-template`. Follow-up turns and edits reuse the processed prefix through DeltaNet checkpoints. After each
  request the engine keeps the GPU's clocks up for a while (keep-warm), so a reply after a pause starts at full speed.
- **The vision encoder** turns OpenAI `image_url` parts and video clips (llama-server's `input_video`) into model input.
  It runs **on the CPU** (fp32 AVX2 / FMA), so images and video cost no VRAM and the full context stays available.
- **KV cache:** q8_0 by default. Opt-in near-lossless presets with 6-bit K and 5-bit V free about 1 GB at 120k context
  and fit the model's full 262,144-token context on 24 GB.

The same sources build natively on Windows (MSVC + nvcc) and on Linux / WSL2, and the two builds give the same output,
bit for bit ([Windows](#windows)).

## Numbers

One RTX 3090 (24 GB) and a Ryzen 9 5900XT, unsloth's UD-Q4_K_XL, the server at 120k context, one request at a time,
greedy unless noted. The same binary can run several percent faster or slower from one session to the next (GPU
temperature and clocks), so every number below carries its date, and changes made after the last full run are given as
A-B-A deltas against the build before them, not added onto its numbers.

### Last full run: 2026-10-15

This engine at commit 7d67c5d on Linux (WSL2), in the configuration served that day (wide verify, 7 drafts, exact
reuse), one session. The llama.cpp column is b11548's `llama-server`, tuned (q8_0 KV, 4 MTP drafts), with the same
client, prompts and model file: the best value of each row over three earlier sessions (2026-10-12 and 2026-10-13).

| | llama.cpp (10-12 / 10-13) | this, q8_0 KV | this, preset D' |
|---|---|---|---|
| greedy decode, short prompts (4 x 512 tokens) | 82.7 tok/s | **151.3** | 150.2 |
| sampled decode, thinking on (8 runs) | 65.8 tok/s | **134.9** | 134.1 |
| decode at 11.6k context | 69.3 tok/s | **135.1** | 131.3 |
| decode at 38k context | 62.3 tok/s | **114.3** | 101.7 |
| decode at 101.6k context | 53.4 tok/s | **105.4** | 96.8 |
| read an 11.6k / 38k / 101.6k-token prompt | 8.24 / 29.6 / 101.0 s | **6.15 / 22.7 / 77.7 s** | 6.32 / 23.6 / 82.2 s |
| first token, short prompt | 124 ms | **68 ms** | 70 ms |
| follow-up turn in an 11.8k chat, prompt time | 323 ms | **182 ms** | 190 ms |
| edit an earlier message in an 11.7k chat | 329 ms | **75 ms** | 79 ms |
| edit the end of a 9.1k system prompt | 956 ms | **90 ms** | 100 ms |
| VRAM, whole card, server idle | 22.8 GB | 21.3 GB | **20.3 GB** |
| 282 benchmark questions (GSM8K, MMLU-Pro, ARC-C, MATH-500, HumanEval, NIAH, LongBench v2) | 229 right, 2,727 s (10-13) | 230 right, **1,739 s** | 230 right, 1,775 s |

- The two short-prompt rows are means of the per-request rates, as the client reports them, for both engines. Weighted
  by tokens (total tokens / total time), the same commit decoded these prompts at 128.3 tok/s greedy and 124.0 sampled
  (CLI, 2026-10-16); llama.cpp's best session decoded them at 76.2 and 63.9 (2026-10-12).
- The follow-up turn re-processes the previous reply because exact reuse is on (`MK_EXACT_REUSE=0` turns it off: faster
  follow-up turns, not bit-identical to a fresh request).
- With preset D' the reply texts, and with them the draft acceptance, can differ from the q8_0 run, so its long-context
  decode rows are not a like-for-like comparison.

**Since then** (up to 147054b, 2026-10-17), every change kept the text output bit-identical (greedy ids at 0, 4 and 7
drafts, the KL-divergence report and the server replies, checked at each merge). Speed changes, each measured A-B-A in
one session:
- **K5**, a split grid barrier between the DeltaNet projections and the DeltaNet core: decode +1.1% at 38k and +0.75% at
  101.6k context (2026-10-17, Linux; at the edge of the run-to-run noise).
- **Prefill one chunk ahead**: the host queues a prompt chunk before it waits for the previous one. On native Windows
  the GPU's idle time during a 38k-token prefill went from 11.8 to 0.19 ms (2026-10-17); prefill time did not change
  measurably.
- **Keep-warm**, new: on native Windows at 147054b, a 255-token reply after 65 s of idle started in ~65 ms instead of
  391-397 ms and decoded at 80.2 instead of 61-63 tok/s (2026-10-17). Holding the clocks costs power: ~165 W instead of
  30-35 W at idle on this card.
- **Draft head from a Q4_K copy** (on by default after 147054b; `MK_DRAFT_Q4=0` turns it off): greedy decode +2.1% on
  the short bench prompts, no measurable change at 38k and 101.6k, +90 MiB of VRAM (2026-10-17, Linux). Greedy output
  is unchanged; sampled replies keep the same distribution but differ from earlier builds for the same seed, because
  the drafts' probabilities changed (docs/ENGINE.md, "The GGUF").

**Q4_K_M:** unsloth's plain file runs on the same engine. Its last full run was on 2026-10-14 (commit b3be928, one
session with UD-Q4_K_XL, q8_0 KV, 7 drafts): short prompts 148.1 vs UD's 144.7 tok/s, 145.7 vs 128.6 at 11.6k, 109.4 vs
109.9 at 38k and 106.0 vs 102.0 at 101.6k context, 20.7 vs 21.1 GB of VRAM, and 231 vs 230 of the 282 questions right.
With upstream open-jet's own benchmark scripts in that session, the engine wrote code at 173.3 tok/s, decoded with 30k
tokens of code in context at 117.5, ran their reasoning test at 125.4 and read a 29.6k-token prompt at 1,811 tok/s.
Upstream's engine (open-jet 2993f47, 4 drafts) ran the same scripts on this card on 2026-10-13: 133.8, 87.4, 89.1 and
1,527 tok/s.

### Exactness

- Greedy token ids are identical at 0, 4 and 7 drafts. With sampling, the verification keeps the token distribution of
  sampling without drafts (chi-square and PIT tests).
- With exact reuse on (`--exact-reuse`, set by the launchers), a reused prefix (an edit or a follow-up turn) answers bit
  for bit like a fresh prefill of the whole conversation. A parked conversation restored from host memory, and a
  conversation kept through a side request, continue bit for bit as if nothing else had run in between.
- The output is not byte-identical to llama.cpp's, whose arithmetic is ordered and rounded differently. Scored against
  llama.cpp's own next-token distributions (13,299 positions of code and prose; reference: llama.cpp generating token by
  token with its q8_0 KV cache), the engine agrees on 98.737% of the top tokens with a mean KL divergence of 0.00119
  nats (q8_0 KV; 2026-10-17, the same report since 2026-10-15). For scale, llama.cpp's own batched run scores 98.489% /
  0.00166 against that reference (2026-10-11). At 32k context: 99.121% / 0.00219. Details:
  [docs/ENGINE.md](docs/ENGINE.md#correctness).

**Preset D'** (`MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048`) stores K at 6 bits and V at 5 bits, and keeps 8 bits for the
most recent 2,048 positions and for the MTP layer. Measured on 2026-10-15 at 7d67c5d:
- KL against llama.cpp: 0.00119 vs q8_0's 0.00119 on the short prompts, 0.00236 vs 0.00219 at 32k, and 0.00161 vs
  0.00163 at 64k. A plain 4-bit cache (`MK_KV='*=4'`) scores 0.00547 on the short prompts.
- The idle server holds 20.3 GB instead of 21.3 GB at 120k, and the model's full **262,144-token context** fits on 24 GB.
  At that size, reading a 260,896-token prompt took 334 s, decode at that depth ran at 63 tok/s, a follow-up turn took
  1.3 s, and the card held 23.4 GB.

**Images** (Ryzen 9 5900XT, 2026-10-16):
- A 196 / 300 / 1024-token image encodes in 0.51 / 0.78 / 3.56 s back to back (llama.cpp's libmtmd on the same CPU:
  0.80 / 1.29 / 6.07 s). A new image after the CPU has been idle takes ~1.2-1.5 s for 300 tokens while the cores ramp up.
- On six image prompts, the mean KL against llama.cpp fed the same image rows is 0.0006-0.0059 (text only: 0.0012).
- The encoder's rows are within 4e-6 to 4e-5 (relative) of an fp64 forward of the vision tower.

**Video** (same CPU, 2026-10-18; frames sampled at 4 fps as llama.cpp does, two consecutive frames per grid of tokens):
- A 640 x 360 clip costs ~450 prompt tokens per second of video (228 at 2 fps) and its frame pairs encode in 0.56 s
  each, about real time at 4 fps. 320 x 240: 167 tokens per second, 0.22 s per pair; 1280 x 720: 1861 tokens per
  second, ~3.2 s per pair.
- The prompt is llama-server's: the same prompt token counts on four clips, the same frames (ffmpeg), and frame-pair
  rows within 5e-6 to 1.1e-5 of an fp64 forward (libmtmd's: 5.5e-3 to 2.2e-2).

Earlier results with charts and raw data (one session, 2026-10-14) and the gates behind each change up to 2026-10-16
are in the fork's [perf docs](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf/perf). How the engine works
inside: [docs/ENGINE.md](docs/ENGINE.md).

## Requirements

- **GPU:** NVIDIA Ampere or newer (sm_86 tested; the kernels use sm_80+ features) with 24 GB of VRAM and at least 80
  SMs for more than 4 drafts (50 for up to 4; the engine checks at load).
- **CUDA:** 13.2 is what we build with; older 12.x versions are untested.
- **OS:** Linux, WSL2, or Windows 10 / 11 x64 ([Windows](#windows) has its own build steps).
- **RAM:** about 28 GB free to compile the served configuration (ptxas). The lean build needs ~19 GB.
- **llama.cpp b10246**, built at `../llama.cpp`. The engine uses its tokenizer (libllama), and the vision library and
  reference tools use libmtmd.
- **Python** 3.10+ with jinja2.
- **CPU for the fast image encoder:** x86-64 with AVX2 + FMA (Zen 2+, Haswell+). Other CPUs fall back to libmtmd.
- **ffmpeg** for video input only (optional): on `PATH`, named by `MK_FFMPEG` / `--ffmpeg`, or the one bundled with
  `pip install imageio-ffmpeg`. Without it the server answers video requests with a 400.

## Build

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && git -C ../llama.cpp checkout b10246
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build ../llama.cpp/build -j
make -j2        # qwen38_mk (CLI), libqwen38mk.so (server), tools/, libqwen38vision.so: ~15 min
pip install -r requirements.txt
```

- A plain `make` builds the configuration we serve: wide verify, up to 7 drafts, and K5
  (`MKFLAGS=-DMK_WV=1 -DMK_MAXT=8 -DMK_WV_Q6F=1 -DMK_K5=1`).
- `make MKFLAGS=` builds the lean engine. It compiles in ~10 minutes and takes at most 4 drafts (run it with
  `MK_DRAFT=4`).
- Most of the build is ptxas on the megakernel: one thread with ~25 GB of heap. The Makefile sets
  `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, which lets it use transparent huge pages and runs it ~20% faster (same
  machine code). It needs THP in `madvise` or `always` mode (`/sys/kernel/mm/transparent_hugepage/enabled`).
- `make` targets the installed GPU's architecture; override it with `ARCH=sm_89` or similar.
- `ptxas.log` lists registers and spills per function.
- The vision library is built when `../llama.cpp/build/bin/libmtmd.so` exists (llama.cpp's default). `make vision`
  builds it explicitly.

Models (unsloth's files, used unmodified):

```sh
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir ~/models
```

Use the BF16 mmproj. The encoder widens any file to fp32, so a Q8_0 mmproj is no faster and is less accurate.

Other Qwen3.8-27B GGUFs (fine-tunes such as abliterated or uncensored conversions) run when every tensor has a type
the engine reads ([docs/ENGINE.md](docs/ENGINE.md#the-gguf)): `token_embd` Q4_K, Q6_K or Q8_0, `output` Q6_K or Q8_0.
Some conversions drop the MTP layer (`blk.64`) the engine drafts with:

- Without one the engine loads with a notice and decodes without drafting (depth 0, about a third of the drafted
  speed); it allocates no MTP weights and no MTP KV slot. `--no-mtp` (`MK_NO_MTP=1`) does the same for a file that has
  one.
- `--mtp FILE` (`MK_MTP=FILE` for scripts/serve.sh) takes the MTP layer from another GGUF of the same model, e.g. a
  separate `mtp-*.gguf`, instead of the model file's own. The engine checks that it belongs to the model
  (architecture, sizes, vocabulary) and re-quantizes its matrices to Q4_K like the built-in layer's.

The MTP layer only proposes drafts, which the main model verifies: output is the same with any MTP layer or none
(greedy bit for bit), only the acceptance rate and so the speed can change.

Checked on 2026-10-18 (greedy output the same at depth 0, 4 and 7; top-token agreement with llama.cpp running the same
file, 98.7% for UD-Q4_K_XL):

| file | what it needs | agreement |
|---|---|---|
| OrcaSAQ-2-27B-Uncensored | Q6_K `token_embd` and `output` | 99.0% |
| Qwen3.8-27B-Uncensored-Cyber IQ4_XS (imatrix, plus-mtp) | Q8_0 `token_embd` and `output` | 99.1% |
| Qwen3.8-27B-Uncensored-Cyber Q4_K_M | no MTP layer: depth 0, or `--mtp` with its `mtp-*-Q4_0.gguf` | 99.1% |
| Huihui-Qwen3.8-27B-abliterated Q4_K_L | Q8_0 `token_embd` and `output`; 19.7 GB of weights, so a short context on 24 GB | 98.7% |

Huihui's abliterated UD-DW-Q4_K_M ran before these changes and still does. A fine-tune's MTP layer drafted as well as
the original's on these files (49-54% of drafts accepted at depth 7).

## Run

```sh
scripts/serve.sh                  # http://127.0.0.1:8080, 120k context, q8_0 KV, 7 drafts, exact reuse, images on
scripts/serve.sh 8080 64000       # port, context
MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048 scripts/serve.sh 8080 262144   # preset D', the full context
```

The settings (model path, host, drafts, prefill chunk, exact reuse, KV preset, keep-warm) are listed at the top of
[scripts/serve.sh](scripts/serve.sh). Under WSL2, keep the GGUF on the Linux filesystem: files on `/mnt/c` load several
times slower. With the model on ext4 the server was ready ~12 s after start (WSL2), and ~7 s natively on Windows from
an NVMe disk (2026-10-16).

After each request the engine keeps the GPU's clocks up for `MK_KEEP_WARM_S` seconds (default 120, `0` = off) with a
one-thread kernel, so a reply after a pause starts at full speed (numbers above); the driver's power settings are left
alone. It costs power while it runs: an RTX 3090 draws ~165 W instead of 30-35 W at idle.

Several conversations can share the engine (one request at a time, in arrival order): a request for another conversation
parks the cached one in host memory (`MK_PARK_MIB`, default 8192 MiB, allocated on first use; `0` = off), and it comes
back bit for bit with its next request instead of a full re-prefill (~65 ms for a 38k-token conversation instead of ~24
s; WSL2, 2026-10-19). A short first-turn request with a short reply, such as a client's title request, runs as a side
request (`MK_SIDE_MAX`, default 8192 tokens; `0` = off): the cached conversation stays and only the rows the side
request overwrites go to host memory and back (~10-20 ms). Details: [docs/ENGINE.md](docs/ENGINE.md#server).

The console shows three lines per request (what it asks for, its prompt: cached / prefilled tokens and prefill speed,
its reply: decode speed, thinking / answer / tool-call tokens, tokens per engine cycle, drafts accepted) and, in a
terminal, a status line updated in place with the prefill's progress and time left or the decode's speed (see
[docs/ENGINE.md](docs/ENGINE.md#server)).

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write quicksort in C."}], "max_tokens": 512}'

# an image (base64 data: URL)
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}]}]}'

# a video clip (base64, llama-server's input_video; needs ffmpeg)
VID=$(base64 -w0 clip.mp4)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What happens in this video?"},
    {"type": "input_video", "input_video": {"data": "'"$VID"'"}}]}]}'
```

What the API supports:
- **Sampling:** temperature, top_p, top_k (1..32; <= 0 means 32), min_p, seed, and presence / frequency penalty.
  Absent values default to Unsloth's thinking-mode preset for Qwen3.8: temperature 1.0, top_p 0.95, top_k 20 and
  min_p 0. There is no repetition penalty.
- **Chat template:** the one stored in the GGUF, with `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, ...)
  and OpenAI `tools`; `--chat-template-file` replaces it, as in llama-server (`MK_CHAT_TEMPLATE_FILE` in the launchers).
  Example: a fine-tune whose GGUF ships Qwen's original template answers a `developer` message, a second system
  message, a turn without a user query or `reasoning_effort` "high" with a 400; unsloth's template (the one in its
  Qwen3.8-27B GGUFs) takes all four.
- **Images:** base64 `data:` URLs only.
- **Video:** `input_video` parts (`{"data": <base64>}`, as llama-server) or `video_url` parts with a base64 `data:` URL;
  ffmpeg samples `--video-fps` frames per second (default 4) and the prompt is laid out as llama.cpp does
  ([docs/ENGINE.md](docs/ENGINE.md#video)). There is no audio.
- **Tool calls:** a call counts only when it is complete; one cut by `max_tokens` is returned as text in `content`, with
  `finish_reason` "length".
- **Limits:** request bodies up to 128 MB (`MK_MAX_BODY_MB`), 32 images per request (`MK_MAX_IMAGES`), 100 megapixels
  per image (`MK_IMAGE_MAX_PIXELS`, checked before the image is decoded), 256 video frames per request
  (`MK_MAX_VIDEO_FRAMES`: 64 s at 4 fps). A refused upload still gets its 4xx answer:
  the server drains the rest of the body (bounded) before it closes. A request whose client disconnects stops at the
  next prefill chunk (2048 tokens) or decode cycle, and a stopped prompt leaves nothing behind; an engine error in a
  stream ends it with an error event and `data: [DONE]`.
- **Requests:** one at a time.

[docs/ENGINE.md](docs/ENGINE.md#server) has the details, the server's options and the engine's environment variables.

## Windows

The same sources build natively with MSVC and nvcc. On an RTX 3090 we compared the Windows build with the Linux build
of the same source (fdad907), on the same model file, on 2026-10-16:

| check | Windows vs Linux |
|---|---|
| greedy token ids, `-d 7` and `-d 0` | identical |
| `--kld` report against llama.cpp (13,299 positions) | identical |
| server replies: text, thinking, an image, an image follow-up turn | identical |
| ptxas registers and spills, every function | identical |
| decode at 38k context | 173.0 vs 173.1 tok/s |
| prefill of a 38k-token prompt | 22.3 vs 22.5 s |
| decode, chat prompts 2-4 (512 tokens, `-d 7`) | within 1.5% |
| image encode, 196 / 300 / 1024 tokens | 0.59 / 0.81 / 3.56 s vs 0.50 / 0.80 / 3.95 s |

The first chat prompt after the model load decoded faster on Windows: 220 vs 132-145 tok/s, with the same tokens. The
likely cause is that the GPU's clocks ramp up faster there. The Windows build of 147054b (2026-10-17) passed the same
identity checks (ids, `--kld` report, server replies) against the Linux references.

The image encoder's rows used to differ between the two builds at fp32 rounding level (relative 2e-6 to 1.3e-5, below
either's error against an fp64 forward of the tower): its attention took e^x from the C library, and glibc's and
Microsoft's `expf` differ in the last bit for some inputs. Since 2026-10-18 the encoder computes it itself, with glibc's
algorithm, so the Linux rows stayed as they were and the Windows rows now equal them bit for bit (stills and video frame
pairs, at any thread count).

Requirements:
- **Driver:** an NVIDIA driver that supports CUDA 13.2.
- **Compiler:** Visual Studio 2022 Build Tools with the C++ x64 tools (MSVC v143). CUDA 13.2 does not accept VS 2026's
  compiler.
- **CUDA 13.2:** the installer, or NVIDIA's redistributable archives (`cuda_nvcc`, `cuda_cudart`, `cuda_cccl`,
  `cuda_crt`, `libnvvm`; `cuda_cuobjdump` for the checks) unpacked into one folder. The engine links the CUDA runtime
  statically, so at run time only the driver is needed.
- **Tools:** CMake and Ninja for llama.cpp, and Python 3.10+.
- **RAM:** about 27 GB free for the engine compile. Don't run WSL2 jobs at the same time: the Linux VM's memory comes
  out of the same pool.

Build, in PowerShell, from an "x64 Native Tools" prompt for VS 2022 (or after `vcvars64.bat`), with llama.cpp next to
this repository:

```powershell
git clone https://github.com/ggml-org/llama.cpp ..\llama.cpp; git -C ..\llama.cpp checkout b10246
cmake -S ..\llama.cpp -B ..\llama.cpp\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build ..\llama.cpp\build --target llama mtmd
.\build.ps1 -Cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"   # ~15 min (the default is $env:CUDA_PATH)
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
```

- **llama.cpp:** a CPU-only build is enough here. The engine uses only its tokenizer and, for images, libmtmd.
- **What `build.ps1` builds:** `qwen38mk.dll` (server engine), `qwen38_mk.exe` (CLI), `qwen38vision.dll` and
  `tools\*.exe`. It uses the Makefile's nvcc flags and `MKFLAGS` (`-MkFlags ""` for the lean engine), writes
  `ptxas.log`, and copies llama.cpp's DLLs next to the outputs.
- **Partial builds:** `-Targets engine|vision|tools` builds one part. An engine object that is newer than the sources
  and was built with the same flags is reused.

Run (cmd):

```bat
set MK_MODEL=D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
set MK_PYTHON=%CD%\.venv\Scripts\python.exe
scripts\serve.bat                 :: http://127.0.0.1:8080, 120k context, q8_0 KV, 7 drafts
```

`scripts\serve.bat` takes the same settings as `scripts/serve.sh`, for example `MK_KV` / `MK_KV_WIN` for preset D', or
`MK_HOST=0.0.0.0` for the LAN. Images are on when `mmproj-BF16.gguf` sits next to the model or `MK_MMPROJ` names it;
`MK_MMPROJ=off` turns them off. From an NVMe disk the server is ready in ~7 s.

## Checking a build

- **Server tests:** `python -m pytest -q tests` runs without a GPU, against fake engines.
- **GGUF parser:** `make tests/gguf_fuzz && tests/gguf_fuzz` feeds `gguf.h` malformed files under AddressSanitizer
  and UBSan (CPU only, Linux).
- **Depth independence:** the same prompt at `-d 0`, `-d 4` and `-d 7` must give identical ids:
  `./qwen38_mk -m <gguf> -p "..." -n 256 -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin --ids`
- **Accuracy against llama.cpp:** `tools/llama_ref` writes a reference file, and `./qwen38_mk -m <gguf> --kld <ref>
  --score-path mixed --kv q8` scores the engine against it ([docs/ENGINE.md](docs/ENGINE.md#correctness)).
- **Images:** `tools/mkv_cli` encodes images and writes their rows. `tools/vision_ref` writes llama.cpp's reference
  for an image prompt plus the rows it fed, and `qwen38_mk --kld <ref> --kld-images <rows>` scores it.
- **Speed:** `python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin --model <gguf>` (4 chat
  prompts through the CLI; it uses the default KV cache, so at most 4 drafts). `bench/*.cu` are kernel
  microbenchmarks.

## Files

- `qwen38_mk.cu`, `qwen38_mk.h`, `prefill*.cuh`, `wv*.cuh`, `loader.cuh`, `gguf.h`: the engine
- `capi.cpp`: the C API of `libqwen38mk.so` (and the keep-warm thread)
- `main.cu`: the `qwen38_mk` CLI
- `server/mk_server.py`: the HTTP server
- `scripts/serve.sh`, `scripts/serve.bat`: the launchers (Linux / WSL2, Windows)
- `build.ps1`, `compat_win32.h` / `.cpp`, `vision/win32.h` / `.cpp`: the Windows build and its POSIX / Win32 shims
  (compiled only on Windows)
- `vision/`: the CPU image encoder (`libqwen38vision.so`) and its tests
- `tools/`: llama.cpp reference tools and the draft-vocabulary builder
- `bench/`: benchmarks
- `tests/`: server tests
- `draft_vocab_32k.bin`: the MTP draft head's token subset

## Origin and license

This repository is the `megakernel/` directory of [open-jet](https://github.com/L-Forster/open-jet) by L-Forster, split
out with its history. It adds the engine work from our open-jet fork
([branch `qwen38-perf`](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf)). Parts of it went upstream as
open-jet PRs #19 and #20, and #23 is open. This repository started from that branch's `megakernel/` directory at 62c8d7a
(2026-10-16), and development of the engine continues here.

Licensed under the GNU AGPL v3, like open-jet ([LICENSE](LICENSE)).
