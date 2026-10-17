#!/usr/bin/env bash
# Serve Qwen3.8-27B through the megakernel's OpenAI-compatible server (Linux / WSL2). Usage: scripts/serve.sh [port] [ctx]
# Defaults: port 8080, 120,000 tokens of context, q8_0 KV cache, 7 MTP drafts, exact reuse on, images on when the mmproj
# file and libqwen38vision.so exist. Settings (environment):
#   MK_MODEL            the GGUF (default ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; Q4_K_M runs on the same engine). Keep it
#                       on a Linux filesystem under WSL2: /mnt/c or /mnt/g paths load several times slower.
#   MK_MMPROJ           the vision projector (default: mmproj-BF16.gguf, unsloth's file name, next to the model or in
#                       ~/models, else ~/models/mmproj-Qwen3.8-27B-BF16.gguf; MK_MMPROJ= turns images off)
#   MK_IMAGE_MAX_TOKENS tokens per image (default: the mmproj's, 4096); fewer tokens encode faster
#   MK_HOST             listen address (default 127.0.0.1; 0.0.0.0 for other machines on the LAN)
#   MK_DRAFT            MTP drafts per cycle (default 7; a `make MKFLAGS=` build takes at most 4)
#   MK_EXACT_REUSE=0    turn exact reuse off: a reused prompt then appends the previous reply instead of re-prefilling it
#                       (faster follow-up turns, but not bit-identical to the same request sent fresh)
#   MK_PF_CAP           prefill chunk tokens (default 2048; 1024 above 96k context: 0.34 GB less VRAM, prefill ~4.5% slower)
#   MK_KV, MK_KV_WIN    low-bit KV cache (opt-in), preset D': MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048 (6-bit K, 5-bit V,
#                       8-bit MTP layer and recent window: 1.1 GB less VRAM at 120k, ~162k context in the same VRAM,
#                       up to 262,144 tokens on 24 GB; near-lossless, see README)
#   MK_KEEP_WARM_S      seconds the engine keeps the GPU's clocks up after each request (default 300, 0 = off)
#   MK_PYTHON           Python with jinja2 (default python3)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8080}"
CTX="${2:-120000}"
MODEL="${MK_MODEL:-$HOME/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
PY="${MK_PYTHON:-python3}"
[ -f "$HERE/libqwen38mk.so" ] || { echo "missing $HERE/libqwen38mk.so: run make (see README)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "missing model $MODEL (set MK_MODEL)" >&2; exit 1; }
if [ "$CTX" -gt 96000 ]; then export MK_PF_CAP="${MK_PF_CAP:-1024}"; fi
XR=(--exact-reuse) XS=on
[ "${MK_EXACT_REUSE:-1}" = 0 ] && XR=() XS=off
if [ -n "${MK_MMPROJ+x}" ]; then MMPROJ="$MK_MMPROJ"
else MMPROJ=$(ls "$(dirname "$MODEL")/mmproj-BF16.gguf" "$HOME/models/mmproj-BF16.gguf" "$HOME/models/mmproj-Qwen3.8-27B-BF16.gguf" \
                    2>/dev/null | head -n1 || true)
fi
VA=() VS=off
if [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ] && [ -f "$HERE/libqwen38vision.so" ]; then
    VA=(--mmproj "$MMPROJ" ${MK_IMAGE_MAX_TOKENS:+--image-max-tokens "$MK_IMAGE_MAX_TOKENS"})
    VS="on the CPU ($(basename "$MMPROJ"))"
fi
echo "megakernel: $MODEL, ctx $CTX, KV q8_0${MK_KV:+ + MK_KV=$MK_KV (window ${MK_KV_WIN:-off})}, ${MK_DRAFT:-7} MTP drafts," \
     "prefill chunk ${MK_PF_CAP:-2048}, exact reuse $XS, images $VS, http://${MK_HOST:-127.0.0.1}:$PORT"
exec "$PY" "$HERE/server/mk_server.py" -m "$MODEL" --host "${MK_HOST:-127.0.0.1}" --port "$PORT" -c "$CTX" --kv q8 \
    -d "${MK_DRAFT:-7}" "${XR[@]}" "${VA[@]}"
