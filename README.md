# qwen38-megakernel

A standalone, OpenAI-compatible server for **Qwen3.8-27B on one 24 GB NVIDIA GPU**, with image input. It contains three
parts:

- **The engine** is a CUDA megakernel tuned on an RTX 3090. Each speculative decode cycle is one persistent kernel: the
  model's own MTP head drafts up to 7 tokens, and all 8 rows are verified at once with int8 tensor-core GEMMs. Prompts
  go through a separate batched tensor-core prefill. Weights stay in their GGUF k-quant encoding. It runs unsloth's
  `Q4_K_M` and `UD-Q4_K_XL` files unmodified.
- **The server** (`server/mk_server.py`) provides `/v1/chat/completions` (streaming, tools, `reasoning_content`),
  `/v1/models`, `/health` and `/props`. Follow-up turns and edits reuse the cached prefix through DeltaNet checkpoints.
- **The vision encoder** turns OpenAI `image_url` parts into model input. It runs **on the CPU** (fp32 AVX2 / FMA), so
  images cost no VRAM and the full context stays available.

It runs on Linux, WSL2 and native Windows. The Windows build gives the same output as the Linux build, bit for bit,
at the same speed ([Windows](#windows)).

## Numbers

These are from one RTX 3090 (24 GB) with unsloth's UD-Q4_K_XL, 120k context, one request at a time, greedy unless noted.
They were measured on 2026-10-15 with the text engine, before image input was added. Text output has been bit-identical
since then, but decode speed has not been re-measured.

| | llama.cpp (tuned, 4 MTP drafts, q8_0 KV) | this, q8_0 KV | this, preset D' |
|---|---|---|---|
| greedy decode, short prompts | 82.7 tok/s | **151.3** | 150.2 |
| sampled, thinking on | 65.8 tok/s | **134.9** | 134.1 |
| decode at 38k context | 62.3 tok/s | **114.3** | 101.7 |
| decode at 101.6k context | 53.4 tok/s | **105.4** | 96.8 |
| read a 101.6k-token prompt | 101.0 s | **77.7 s** | 82.2 s |
| first token, short prompt | 124 ms | **68 ms** | 70 ms |
| edit an earlier message in an 11.7k chat | 329 ms | **75 ms** | 79 ms |
| VRAM, whole card | 22.8 GB | 21.3 GB | **20.3 GB** |
| 282 benchmark questions (GSM8K, MMLU-Pro, ARC-C, MATH-500, HumanEval, NIAH, LongBench v2) | 229 right, 2,727 s | 230 right, **1,739 s** | 230 right, 1,775 s |

The llama.cpp column is the best of three earlier sessions on the same machine with the same client.

**Exactness:**
- Greedy token ids are identical at draft depths 0, 4 and 7, so speculation does not change the output.
- A reused prefix (an edit or a follow-up turn) answers bit for bit like a fresh prefill.
- Against llama.cpp's own next-token distributions (KL divergence, nats), the q8_0 KV cache scores 0.0012. For scale,
  llama.cpp batched scores 0.0017 against its own token-by-token run.

**Preset D'** stores K at 6 bits and V at 5 bits, and keeps 8 bits for the recent window and the MTP layer.
- KL against llama.cpp: 0.00119 vs q8's 0.00119 on short prompts, 0.00236 vs 0.00219 at 32k, and 0.00161 vs 0.00163 at
  64k. A plain 4-bit cache scores 0.00547 on the short prompts.
- It uses about 1.1 GB less VRAM at 120k, which is what lets the model's full **262,144-token context** fit on 24 GB.
  At that size, reading 261k tokens from scratch takes 334 s, decode at that depth runs at 63 tok/s, a follow-up turn
  takes 1.3 s, and the card holds 23.4 GB.

**Images** (Ryzen 9 5900XT):
- A 196 / 300 / 1024-token image encodes in 0.51 / 0.78 / 3.56 s back to back (llama.cpp's libmtmd: 0.80 / 1.29 /
  6.07 s). The first new image after the CPU has been idle takes ~1.2-1.5 s for 300 tokens while the cores ramp up.
- On six image prompts, the mean KL against llama.cpp is 0.0006-0.0061 (text-only: 0.0012).
- The encoder's rows are within 4e-6 to 4e-5 (relative) of an fp64 forward of the vision tower.

The full tables, charts, raw data and the gates behind each change are in the fork's
[perf docs](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf/perf). The engine's internals are covered
in [docs/ENGINE.md](docs/ENGINE.md).

## Requirements

- **GPU:** NVIDIA Ampere or newer (sm_86 tested; the kernels use sm_80+ features) with 24 GB of VRAM.
- **CUDA:** 13.2 is what we build with; older 12.x versions are untested.
- **OS:** Linux, WSL2, or Windows 10 / 11 x64 ([Windows](#windows) has its own build steps).
- **RAM:** about 27 GB free to compile the served configuration (ptxas). The lean build needs ~12 GB.
- **llama.cpp b10246**, built at `../llama.cpp`. The engine uses its tokenizer (libllama), and the vision library and
  reference tools use libmtmd.
- **Python** 3.10+ with jinja2.
- **CPU for the fast image encoder:** x86-64 with AVX2 + FMA (Zen 2+, Haswell+). Other CPUs fall back to libmtmd.

## Build

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && git -C ../llama.cpp checkout b10246
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build ../llama.cpp/build -j
make -j2        # qwen38_mk (CLI), libqwen38mk.so (server), tools/, libqwen38vision.so: ~20 min
pip install -r requirements.txt
```

- A plain `make` builds the configuration we serve: wide verify and up to 7 drafts
  (`MKFLAGS=-DMK_WV=1 -DMK_MAXT=8 -DMK_WV_Q6F=1 -DMK_K5=1`).
- `make MKFLAGS=` builds the lean engine. It compiles in a few minutes and takes at most 4 drafts (run it with
  `MK_DRAFT=4`).
- `make` targets the installed GPU's architecture; override it with `ARCH=sm_89` or similar.
- `ptxas.log` lists registers and spills per function.
- The vision library is built when `../llama.cpp/build/bin/libmtmd.so` exists (llama.cpp's default). `make vision`
  builds it explicitly.

Models (unsloth's files, used unmodified):

```sh
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir ~/models
```

Use the BF16 mmproj. The encoder widens any file to fp32, so a Q8_0 mmproj is no faster and is less accurate.

## Run

```sh
scripts/serve.sh                  # http://127.0.0.1:8080, 120k context, q8_0 KV, 7 drafts, images on
scripts/serve.sh 8080 64000       # port, context
MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048 scripts/serve.sh 8080 262144   # preset D', the full context
```

The settings (model path, host, drafts, prefill chunk, exact reuse, KV preset) are listed at the top of
[scripts/serve.sh](scripts/serve.sh). Model load takes ~9 s from an ext4 disk. Under WSL2, keep the GGUF on the Linux
filesystem: files on `/mnt/c` load several times slower.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write quicksort in C."}], "max_tokens": 512}'

# an image (base64 data: URL)
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}]}]}'
```

What the API supports:
- **Sampling:** temperature, top_p, top_k (<= 32), min_p, seed, and presence / frequency penalty. There is no
  repetition penalty.
- **Chat template:** the one stored in the GGUF, with `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, ...)
  and OpenAI `tools`.
- **Images:** base64 `data:` URLs only. There is no video or audio.
- **Requests:** one at a time.

[docs/ENGINE.md](docs/ENGINE.md#server) has the details.

## Windows

The same sources build natively with MSVC and nvcc. On an RTX 3090 we compared the Windows build with the Linux build
of the same source, on the same model file:

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
likely cause is that the GPU's clocks ramp up faster there. The Windows encoder's image rows differ from the Linux encoder's at fp32 rounding level (relative 2e-6 to 1.3e-5). That
is smaller than either's error against an fp64 forward of the tower, and the image replies above were identical.

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
.\build.ps1 -Cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"   # ~20 min (the default is $env:CUDA_PATH)
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
- **Depth independence:** the same prompt at `-d 0`, `-d 4` and `-d 7` must give identical ids:
  `./qwen38_mk -m <gguf> -p "..." -n 256 -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin --ids`
- **Accuracy against llama.cpp:** `tools/llama_ref` writes a reference file, and `./qwen38_mk -m <gguf> --kld <ref>
  --score-path mixed --kv q8` scores the engine against it ([docs/ENGINE.md](docs/ENGINE.md#results), "Correctness").
- **Images:** `tools/mkv_cli` encodes images and writes their rows. `tools/vision_ref` writes llama.cpp's reference
  for an image prompt plus the rows it fed, and `qwen38_mk --kld <ref> --kld-images <rows>` scores it.
- **Speed:** `python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin --model <gguf>` (4 chat
  prompts). `bench/*.cu` are kernel microbenchmarks.

## Files

- `qwen38_mk.cu`, `qwen38_mk.h`, `prefill*.cuh`, `wv*.cuh`, `loader.cuh`, `gguf.h`: the engine
- `capi.cpp`: the C API of `libqwen38mk.so`
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
