#!/usr/bin/env python3
"""OpenAI-compatible chat server for the Qwen3.8-27B megakernel (a drop-in for the parts of
llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF, or the one given by --chat-template-file / --chat-template (llama-server's options), with
enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs and tools via `tools`;
reasoning is returned as `reasoning_content`, Qwen's XML tool calls as structured `tool_calls`.
Sampling: temperature, top_p, top_k (1..32; <= 0: 32, the engine's maximum), min_p, seed; stop strings
(`stop`).  Successive requests of a conversation reuse the processed tokens: the new prompt is
matched against the sequence so far and only the rest is prefilled. When the new prompt diverges inside
what the engine already processed, it rewinds to the last DeltaNet checkpoint at or before the
divergence (saved during earlier prefills: at each generation prompt, each message start and every
--ck-every tokens; a pool of MK_CK_SLOTS, default 8).
POST /v1/messages (+ /v1/messages/count_tokens): the Anthropic Messages API over the same chat path.
GET /health (503 "Loading model" until the model is loaded), /v1/models, /props, /slots, /metrics; POST /tokenize,
/detokenize, /apply-template, /v1/chat/completions/input_tokens (llama-server's shapes).  One request at a time.
After each request the engine keeps the GPU's clocks up for MK_KEEP_WARM_S seconds (default 120, 0 = off), so a
reply after a pause starts at full speed.
Limits (environment): MK_MAX_BODY_MB (request body, default 128), MK_MAX_IMAGES (images per request, default 32),
MK_MAX_VIDEO_FRAMES (video frames per request, default 256);
the image library refuses images over MK_IMAGE_MAX_PIXELS (default 100,000,000) before decoding them. A request
whose client disconnects stops at the next prefill chunk (2048 tokens) or engine cycle. A connection the server closes
while the client is still sending (an early 4xx for a body it does not read) is half-closed and drained first (bounded),
so the client reads the answer instead of a connection reset.
"""
import argparse
import base64
import binascii
import codecs
import contextlib
import ctypes
import hashlib
import json
import os
import random
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK_MAX = 32
# sampling defaults (a request's absent / null fields): Unsloth's thinking-mode preset for Qwen3.8
DEF_TEMP, DEF_TOP_P, DEF_TOP_K, DEF_MIN_P = 1.0, 0.95, 20, 0.0
# request limits; main() reads them from the environment
MAX_BODY_MB = 128  # MK_MAX_BODY_MB: request body (several screenshots as base64 data: URLs fit)
MAX_IMAGES = 32  # MK_MAX_IMAGES: images per request
MAX_STOP, MAX_STOP_LEN = 64, 1024  # stop strings per request, characters per stop string
MAX_VIDEO_FRAMES = 256  # MK_MAX_VIDEO_FRAMES: video frames per request (all of its videos; 64 s at 4 fps)
VIDEO_DECODE_S = 300  # a video's ffmpeg decode is stopped after this many seconds (400)
VIDEO_TS_MS = 5000  # llama.cpp's video timestamp interval (mtmd_helper_video_init_params_default)
# lingering close (HTTPServer.shutdown_request): what a client still sends after the server's last answer is read and
# dropped for at most LINGER_S seconds / LINGER_BYTES bytes, LINGER_IDLE_S seconds without data (nginx's defaults)
LINGER_S, LINGER_IDLE_S, LINGER_BYTES = 30.0, 5.0, 1 << 30
# the engine and encoder libraries next to the repo root: qwen38mk.dll / qwen38vision.dll on Windows
LIB_MK, LIB_VISION = ("qwen38mk.dll", "qwen38vision.dll") if os.name == "nt" else ("libqwen38mk.so", "libqwen38vision.so")


def log(*a):
    print("[mk-server]", *a, file=sys.stderr, flush=True)


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1, mtp=None):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int]),
            "mk_last_error": (cp, []),
            "mk_chat_template": (ctypes.c_int, [vp, cp, ctypes.c_int]),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_piece": (ctypes.c_int, [vp, ctypes.c_int, cp, ctypes.c_int]),
            "mk_is_eog": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_set_sampling": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                               ctypes.c_ulonglong]),
            "mk_set_penalty": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float]),
            "mk_prefill": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_prefill_ck": (ctypes.c_int, [vp, ip, ctypes.c_int, ctypes.c_int]),
            "mk_step": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_reset": (None, [vp]),
            "mk_truncate": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_sequence": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_committed": (ctypes.c_int, [vp]),
            "mk_checkpoint_pos": (ctypes.c_int, [vp]),
            "mk_rewind_to_checkpoint": (ctypes.c_int, [vp]),
        }
        opt = {  # checkpoint pool (libraries before it: one checkpoint)
            "mk_prefill_cks": (ctypes.c_int, [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int]),
            "mk_checkpoints": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_rewind_to": (ctypes.c_int, [vp, ctypes.c_int]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        self.pool = all(hasattr(L, name) for name in opt)
        if self.pool:
            for name, (res, args) in opt.items():
                f = getattr(L, name)
                f.restype, f.argtypes = res, args
        # images (libraries before it: text only)
        self.images = hasattr(L, "mk_prefill_img")
        if self.images:
            L.mk_prefill_img.restype = ctypes.c_int
            L.mk_prefill_img.argtypes = [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ip, ip, ip,
                                         ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
            L.mk_n_embd.restype, L.mk_n_embd.argtypes = ctypes.c_int, []
        dv = draft_vocab.encode() if draft_vocab else None
        if mtp is not None:  # --mtp FILE / --no-mtp ("none"); otherwise the model file's MTP layer (or MK_MTP / MK_NO_MTP)
            if not hasattr(L, "mk_new_mtp"):
                raise RuntimeError("--mtp / --no-mtp need a newer libqwen38mk (mk_new_mtp)")
            L.mk_new_mtp.restype = vp
            L.mk_new_mtp.argtypes = [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int, cp]
            self.h = L.mk_new_mtp(model.encode(), ctx, n_draft, dv, kv_mode, mtp.encode())
        else:
            self.h = L.mk_new(model.encode(), ctx, n_draft, dv, kv_mode)
        if not self.h:
            raise RuntimeError(L.mk_last_error().decode())
        self.has_mtp = True  # False: no MTP layer, plain decoding at depth 0
        if hasattr(L, "mk_has_mtp"):
            L.mk_has_mtp.restype, L.mk_has_mtp.argtypes = ctypes.c_int, [vp]
            self.has_mtp = L.mk_has_mtp(self.h) == 1
        self.ctx = ctx
        self._pieces = {}
        # MK_EXACT engines (exact reuse): prefill rows do not depend on the chunking; see Server.load_prompt
        self.exact = False
        if hasattr(L, "mk_exact"):
            L.mk_exact.restype, L.mk_exact.argtypes = ctypes.c_int, [vp]
            self.exact = L.mk_exact(self.h) == 1
        # keep-warm (MK_KEEP_WARM_S; libraries before it: none)
        self._warm = hasattr(L, "mk_keep_warm")
        if self._warm:
            L.mk_keep_warm.restype, L.mk_keep_warm.argtypes = ctypes.c_double, [vp, ctypes.c_int]
        # prefill cancel (libraries before it: a disconnect during a prefill is noticed when it ends)
        self._cancel = hasattr(L, "mk_cancel")
        if self._cancel:
            L.mk_cancel.restype, L.mk_cancel.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        # decode statistics; the library's build (commit, engine flags; libraries before it: unknown)
        self._stats = hasattr(L, "mk_stats")
        if self._stats:
            lp = ctypes.POINTER(ctypes.c_long)
            L.mk_stats.restype, L.mk_stats.argtypes = None, [vp, lp, lp, lp]
        self.n_vocab = 0  # (libraries before mk_n_vocab: unknown)
        if hasattr(L, "mk_n_vocab"):
            L.mk_n_vocab.restype, L.mk_n_vocab.argtypes = ctypes.c_int, [vp]
            self.n_vocab = L.mk_n_vocab(self.h)
        self.build = None
        if hasattr(L, "mk_build_info"):
            L.mk_build_info.restype, L.mk_build_info.argtypes = ctypes.c_char_p, []
            raw = L.mk_build_info().decode(errors="replace")
            try:
                self.build = json.loads(raw)
            except ValueError:
                self.build = {"raw": raw}

    def stats(self):
        """(decode cycles, drafted tokens, accepted drafts) since the load, or None (a library without mk_stats)"""
        if not self._stats:
            return None
        c, d, a = ctypes.c_long(), ctypes.c_long(), ctypes.c_long()
        self.L.mk_stats(self.h, ctypes.byref(c), ctypes.byref(d), ctypes.byref(a))
        return c.value, d.value, a.value

    def _err(self, what):
        return RuntimeError(f"{what}: {self.L.mk_last_error().decode()}")

    def chat_template(self):
        n = self.L.mk_chat_template(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.L.mk_chat_template(self.h, buf, n)
        return buf.raw[:n].decode()

    def tokenize(self, text):
        b = text.encode()
        cap = len(b) + 16
        out = (ctypes.c_int * cap)()
        n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        if n < 0:
            cap = -n
            out = (ctypes.c_int * cap)()
            n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        return list(out[:n])

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is None:
            buf = ctypes.create_string_buffer(256)
            n = self.L.mk_piece(self.h, tok, buf, 256)
            p = self._pieces[tok] = buf.raw[:max(n, 0)]
        return p

    def is_eog(self, tok):
        return bool(self.L.mk_is_eog(self.h, tok))

    def keep_warm(self, on=True):
        """on: keep the GPU's clocks up for MK_KEEP_WARM_S seconds from now (the engine stops the load before its next
        GPU work); off: stop now. Returns the window in seconds (0: off)."""
        return self.L.mk_keep_warm(self.h, 1 if on else 0) if self._warm else 0.0

    @contextlib.contextmanager
    def cancel_when(self, gone, every=0.05):
        """runs the block (a prefill call) next to a watcher thread that polls gone() every `every` seconds: once it
        returns True, the engine stops the prefill at its next chunk boundary (mk_cancel: a host flag) and the call
        raises PrefillCancelled, with the engine back at its last checkpoint before the prefill"""
        if not self._cancel or gone is None:
            yield
            return
        self.L.mk_cancel(self.h, 0)
        done = threading.Event()

        def watch():
            while not done.wait(every):
                if gone():
                    self.L.mk_cancel(self.h, 1)
                    return
        th = threading.Thread(target=watch, name="mk-cancel-watch", daemon=True)
        th.start()
        try:
            yield
        finally:
            done.set()
            th.join()
            self.L.mk_cancel(self.h, 0)

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        if self.L.mk_set_sampling(self.h, temp, top_p, top_k, min_p, seed) < 0:
            raise self._err("sampling")

    def set_penalty(self, presence, frequency):
        if self.L.mk_set_penalty(self.h, presence, frequency) < 0:
            raise self._err("penalty")

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        """cks: more checkpoint positions (absolute; the engine moves them onto its scan grid), ck_at stays exact.
        images: (pos, nx, ny, rows address) of each image the fed positions overlap (absolute positions)"""
        arr = (ctypes.c_int * len(toks))(*toks)
        if images:
            k = len(images)
            ca = (ctypes.c_int * max(len(cks), 1))(*cks)
            pos, nx, ny = ((ctypes.c_int * k)(*[im[j] for im in images]) for j in range(3))
            rows = (ctypes.c_void_p * k)(*[im[3] for im in images])
            r = self.L.mk_prefill_img(self.h, arr, len(toks), ca, len(cks), ck_at, pos, nx, ny, rows, k)
        elif self.pool and cks:
            ca = (ctypes.c_int * len(cks))(*cks)
            r = self.L.mk_prefill_cks(self.h, arr, len(toks), ca, len(cks), ck_at)
        else:
            r = self.L.mk_prefill_ck(self.h, arr, len(toks), ck_at)
        if r == -2:  # (mk_cancel)
            raise PrefillCancelled(self.L.mk_last_error().decode())
        if r < 0:
            raise self._err("prefill")
        return r

    def step(self):
        out = (ctypes.c_int * 8)()
        n = self.L.mk_step(self.h, out, 8)
        if n < 0:
            raise self._err("decode")
        return list(out[:n])

    def reset(self):
        self.L.mk_reset(self.h)

    def truncate(self, n):
        if self.L.mk_truncate(self.h, n) < 0:
            raise self._err("truncate")

    def sequence(self):
        n = self.L.mk_sequence(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_sequence(self.h, out, n)
        return list(out[:n])

    def committed(self):
        return self.L.mk_committed(self.h)

    def checkpoint_pos(self):
        return self.L.mk_checkpoint_pos(self.h)

    def rewind_to_checkpoint(self):
        return self.L.mk_rewind_to_checkpoint(self.h) == 0

    def checkpoints(self):
        if not self.pool:
            ck = self.checkpoint_pos()
            return [ck] if ck >= 0 else []
        n = self.L.mk_checkpoints(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_checkpoints(self.h, out, n)
        return list(out[:n])

    def rewind_to(self, pos):
        if not self.pool:
            return pos == self.checkpoint_pos() and self.rewind_to_checkpoint()
        return self.L.mk_rewind_to(self.h, pos) == 0


class Vision:
    """ctypes binding of vision/mk_vision.h (libqwen38vision.so): the CPU image encoder (llama.cpp's libmtmd), plus a
    cache of decoded / encoded images by content hash, so a conversation re-sending its images encodes each once.
    A request pins its images (image() .. release()): the cache never frees an image a request still holds, so it can
    hold more than --image-cache images while a request runs."""

    class Image:
        def __init__(self, vision, handle, key):
            self.v, self.h, self.key = vision, handle, key
            nx, ny = ctypes.c_int(), ctypes.c_int()
            self.n = vision.L.mkv_image_grid(handle, ctypes.byref(nx), ctypes.byref(ny))
            self.nx, self.ny = nx.value, ny.value
            self.encode_s = None
            self.pins = 0  # requests holding it (Vision.image / Vision.release)

        def rows(self):
            """address of the [n][n_embd] fp32 rows, encoding on the first call"""
            L = self.v.L
            if not self.h:
                raise RuntimeError("image encode: the image was freed")
            if self.encode_s is None:
                if L.mkv_image_encode(self.v.ctx, self.h) != 0:
                    raise RuntimeError(f"image encode: {L.mkv_last_error().decode()}")
                self.encode_s = L.mkv_image_encode_ms(self.h) / 1e3
            return L.mkv_image_rows(self.h)

        def free(self):
            if self.h:
                self.v.L.mkv_image_free(self.h)
                self.h = None

    class Video:
        """a decoded video in llama.cpp's layout: parts [str | Image], each Image one frame or a frame pair (Qwen-VL's
        temporal patch of 2: one grid of tokens); owned by Vision.videos (its images are pinned like images)"""

        def __init__(self, key, parts, n_frames):
            self.key, self.parts, self.n_frames = key, parts, n_frames
            self.images = [p for p in parts if not isinstance(p, str)]

        def free(self):
            for im in self.images:
                im.free()

    class Params(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("min_tokens", ctypes.c_int), ("max_tokens", ctypes.c_int),
                    ("flash_attn", ctypes.c_int), ("verbose", ctypes.c_int), ("backend", ctypes.c_int)]

    BACKENDS = {"auto": 0, "mtmd": 1, "cpu": 2}
    frames_ok = False  # the library has video frames (mkv_frames_load)
    videos = None      # key -> Video, least recently used first (__init__)
    video_cache_max = 0

    def __init__(self, lib, mmproj, model, threads=0, min_tokens=0, max_tokens=0, cache=8, backend="auto", video_cache=2):
        L = self.L = ctypes.CDLL(lib)
        vp, cp = ctypes.c_void_p, ctypes.c_char_p
        for name, res, args in (
                ("mkv_params_default", Vision.Params, []), ("mkv_new", vp, [cp, cp, Vision.Params]),
                ("mkv_free", None, [vp]), ("mkv_last_error", cp, []), ("mkv_n_embd", ctypes.c_int, [vp]),
                ("mkv_n_threads", ctypes.c_int, [vp]), ("mkv_backend", ctypes.c_int, [vp]),
                ("mkv_image_load", vp, [vp, cp, ctypes.c_size_t]),
                ("mkv_image_grid", ctypes.c_int, [vp, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]),
                ("mkv_image_encode", ctypes.c_int, [vp, vp]), ("mkv_image_rows", vp, [vp]),
                ("mkv_image_encode_ms", ctypes.c_double, [vp]), ("mkv_image_free", None, [vp])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        # video frames (libraries before it: images only)
        self.frames_ok = hasattr(L, "mkv_frames_load")
        if self.frames_ok:
            L.mkv_frames_load.restype = vp
            L.mkv_frames_load.argtypes = [vp, cp, cp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        p = L.mkv_params_default()
        p.n_threads, p.min_tokens, p.max_tokens, p.backend = threads, min_tokens, max_tokens, Vision.BACKENDS[backend]
        self.ctx = L.mkv_new(mmproj.encode(), model.encode(), p)
        if not self.ctx:
            raise RuntimeError(f"vision: {L.mkv_last_error().decode()}")
        self.n_embd = L.mkv_n_embd(self.ctx)
        self.n_threads = L.mkv_n_threads(self.ctx)
        self.backend = {1: "libmtmd", 2: "vit_cpu"}.get(L.mkv_backend(self.ctx), "?")
        self.cache = {}  # key -> Image, least recently used first
        self.cache_max = max(1, cache)
        self.videos = {}  # key -> Video, least recently used first
        self.video_cache_max = max(0, video_cache)

    def image(self, data):
        """the Image of these file bytes (decoded and preprocessed; encoded lazily by rows()), pinned: the caller
        releases it (release) when its request is done"""
        key = hashlib.sha256(data).hexdigest()
        im = self.cache.pop(key, None)
        if im is None:
            h = self.L.mkv_image_load(self.ctx, data, len(data))
            if not h:
                raise BadRequest(f"image: {self.L.mkv_last_error().decode()}")
            im = Vision.Image(self, h, key)
        self.cache[key] = im
        im.pins += 1
        return im

    def video(self, data, decoder, fps, max_tokens, max_frames):
        """the Video of these file bytes (decoded by decoder (VideoDecoder) at fps, frames resized to at most max_tokens
        tokens (0: the image limit); its images are preprocessed, encoded lazily by rows()), pinned like image(). A
        video of more than max_frames frames is refused (BadRequest) while it decodes."""
        key = f"{hashlib.sha256(data).hexdigest()}/{fps!r}/{max_tokens}"
        v = self.videos.pop(key, None)
        if v is not None and v.n_frames > max_frames:
            self.videos[key] = v
            raise BadRequest(f"video of {v.n_frames} frames at {fps:g} fps: over the {max_frames} frames left for this "
                             f"request (MK_MAX_VIDEO_FRAMES {MAX_VIDEO_FRAMES})")
        if v is None:
            parts, n = [], 0
            try:
                for kind, x in video_parts(decoder.frames(data, fps, max_frames), fps):
                    if kind == "text":
                        parts.append(x)
                        continue
                    (w, h, a), b = x[0], x[1]
                    n += 2 if b else 1
                    hd = self.L.mkv_frames_load(self.ctx, a, b[2] if b else None, w, h, max_tokens)
                    if not hd:
                        raise BadRequest(f"video frame: {self.L.mkv_last_error().decode()}")
                    parts.append(Vision.Image(self, hd, f"{key}/{len(parts)}"))
            except BaseException:
                for im in parts:
                    if not isinstance(im, str):
                        im.free()
                raise
            v = Vision.Video(key, parts, n)
        self.videos[key] = v
        for im in v.images:
            im.pins += 1
        return v

    def release(self, images):
        """unpins the images of a request (image() / video() once per pin), then frees the least recently used unpinned
        images beyond --image-cache and videos beyond --video-cache"""
        for im in images:
            im.pins -= 1
        extra = len(self.cache) - self.cache_max
        for key in [k for k, im in self.cache.items() if im.pins <= 0][:max(extra, 0)]:
            self.cache.pop(key).free()
        extra = len(self.videos or ()) - self.video_cache_max
        for key in [k for k, v in (self.videos or {}).items() if all(im.pins <= 0 for im in v.images)][:max(extra, 0)]:
            self.videos.pop(key).free()

    def warm_up(self):
        """one encode of a 64 x 64 grey PNG (not cached): the first encode after the engine's load pays ~0.5 s once
        (every weight streamed in again, the workers' scratch), here at startup instead of in the first image request"""
        w = h = 64
        raw = b"".join(bytes([0]) + bytes([128, 128, 128]) * w for _ in range(h))

        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
        png = (bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        hd = self.L.mkv_image_load(self.ctx, png, len(png))
        if not hd or self.L.mkv_image_encode(self.ctx, hd) != 0:
            raise RuntimeError(f"vision warm-up: {self.L.mkv_last_error().decode()}")
        self.L.mkv_image_free(hd)


def image_bytes(url):
    """the file bytes of an OpenAI image_url (a data: URL, as llama-server takes them by default)"""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise BadRequest("image_url: only data: URLs (base64) are supported")
    head, sep, payload = url.partition(",")
    if not sep or ";base64" not in head:
        raise BadRequest("image_url: expected a base64 data: URL")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise BadRequest(f"image_url: bad base64 ({e})")


# ---------------------------------------------------------------------------------------------
# video: frames from ffmpeg, laid out as llama.cpp's mtmd does (mtmd-helper.cpp video helpers + mtmd_tokenize)


class VideoBytes(bytes):
    """the file bytes of a video content part (normalize_messages' media list holds images as plain bytes)"""


VIDEO_PARTS = ("input_video", "video_url")  # llama-server's / vLLM's content part types


def video_bytes(part):
    """the file bytes of a video content part: llama-server's {"type": "input_video", "input_video": {"data": base64}}
    (or {"url": ...}), or {"type": "video_url", "video_url": {"url": ...}}; raw base64 or a base64 data: URL (no remote
    or file URLs)"""
    t = part.get("type")
    v = part.get(t)
    s = (v.get("data") or v.get("url")) if isinstance(v, dict) else v
    if not isinstance(s, str) or not s:
        raise BadRequest(f"{t}: expected base64 data")
    if s.startswith("data:"):
        head, sep, s = s.partition(",")
        if not sep or ";base64" not in head:
            raise BadRequest(f"{t}: expected a base64 data: URL")
    elif re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        raise BadRequest(f"{t}: only base64 data (or a base64 data: URL) is supported, not {s.split(':')[0]}:// URLs")
    try:
        data = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as e:
        raise BadRequest(f"{t}: bad base64 ({e})")
    if not data:
        raise BadRequest(f"{t}: empty video")
    return VideoBytes(data)


def f32(x):
    """x rounded to float32 (llama.cpp computes the video timestamps in float)"""
    return struct.unpack("f", struct.pack("f", x))[0]


def video_timestamp(k, fps, nxt):
    """mtmd-helper's timestamp text after frame k (0-based) at fps (float32) given the next threshold nxt (ms):
    (text or None, the next threshold); float32 arithmetic as there"""
    elapsed = f32(f32(f32(k) / fps) * 1000.0)
    if elapsed < nxt:
        return None, nxt
    s = f32(elapsed / 1000.0)
    minutes = int(f32(s / 60.0))
    sec = f32(s - f32(minutes * 60.0))
    return f"[{minutes}m{sec:.2f}s]", f32(nxt + VIDEO_TS_MS)


def video_parts(frames, fps):
    """a video's parts as llama.cpp lays them out: ("text", "Video:"), then each frame with a timestamp text
    ("[0m5.00s]") after every frame that reaches the next 5 s mark (frame 0 included); consecutive frames of one size
    (no text between) merge in pairs ("frames", (a, b)), a run's odd last frame stays alone ("frames", (a, None)).
    frames: (width, height, rgb bytes) in order; fps: the sampling rate (float32 in the timestamps)."""
    fps = f32(fps)
    yield "text", "Video:"
    nxt, pend = 0.0, None
    for k, fr in enumerate(frames):
        if pend is not None and pend[:2] != fr[:2]:  # (frames of different sizes do not merge)
            yield "frames", (pend, None)
            pend = None
        if pend is None:
            pend = fr
        else:
            yield "frames", (pend, fr)
            pend = None
        ts, nxt = video_timestamp(k, fps, nxt)
        if ts:
            if pend is not None:
                yield "frames", (pend, None)
                pend = None
            yield "text", ts
    if pend is not None:
        yield "frames", (pend, None)


def read_ppm(f):
    """the next binary PPM (P6, maxval 255) of a stream: (width, height, rgb bytes), None at its end"""
    def token():
        t = b""
        while True:
            c = f.read(1)
            if not c:
                return t or None
            if c == b"#" and not t:  # (a comment line)
                while c and c not in b"\r\n":
                    c = f.read(1)
                continue
            if c.isspace():
                if t:
                    return t
                continue
            t += c
            if len(t) > 16:
                raise ValueError("bad PPM header")
    magic = token()
    if magic is None:
        return None
    w, h, mx = token(), token(), token()
    if magic != b"P6" or not (w and h and mx) or not (w.isdigit() and h.isdigit()) or mx != b"255":
        raise ValueError("bad PPM header")
    w, h = int(w), int(h)
    if not (0 < w <= 65535 and 0 < h <= 65535):
        raise ValueError(f"bad PPM size {w} x {h}")
    data = f.read(w * h * 3)
    if len(data) != w * h * 3:
        raise ValueError("truncated frame")
    return w, h, data


def find_ffmpeg(explicit=None):
    """the ffmpeg executable for video input: explicit (--ffmpeg) or MK_FFMPEG (a file, or a directory with it), else
    ffmpeg on PATH, else the imageio-ffmpeg package's bundled binary (pip install imageio-ffmpeg); None: none found"""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    p = explicit or os.environ.get("MK_FFMPEG")
    if p:
        p = os.path.join(p, exe) if os.path.isdir(p) else p
        return p if os.path.isfile(p) else None
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # optional
        p = imageio_ffmpeg.get_ffmpeg_exe()
        return p if p and os.path.isfile(p) else None
    except Exception:  # (not installed, or no binary for this platform)
        return None


class VideoDecoder:
    """video file bytes -> RGB24 frames through an ffmpeg executable, sampled by ffmpeg's fps filter as llama.cpp's
    mtmd-helper does (`-vf fps=F`, rgb24), but read as a PPM stream (each frame carries its size: no ffprobe needed)"""

    def __init__(self, exe):
        self.exe = exe
        self.version = ""

    def check(self, timeout=20):
        """runs `ffmpeg -version`; the first line of its output (RuntimeError if it does not run)"""
        try:
            r = subprocess.run([self.exe, "-hide_banner", "-version"], stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"{self.exe}: {e}")
        if r.returncode != 0:
            raise RuntimeError(f"{self.exe} -version: exit {r.returncode}")
        self.version = (r.stdout.decode(errors="replace").splitlines() or ["?"])[0]
        return self.version

    def frames(self, data, fps, max_frames, timeout=None):
        """yields (width, height, rgb bytes) per frame; BadRequest for a video ffmpeg cannot decode, one of more than
        max_frames frames (when the next frame arrives) or a decode over `timeout` seconds"""
        timeout = VIDEO_DECODE_S if timeout is None else timeout
        with tempfile.TemporaryDirectory(prefix="mkvideo") as d:
            # (a name without an extension and only the file protocol: a "video" that is a playlist or a concat list
            # cannot make ffmpeg open URLs or other files; ffmpeg 6 refuses an HLS playlist without .m3u8 by itself)
            path = os.path.join(d, "input")
            with open(path, "wb") as f:
                f.write(data)
            cmd = [self.exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file", "-i", path,
                   "-an", "-sn", "-vf", f"fps={fps:.6f}", "-f", "image2pipe", "-c:v", "ppm", "-pix_fmt", "rgb24", "pipe:1"]
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as e:
                raise BadRequest(f"video: cannot run ffmpeg ({e})")
            err = bytearray()

            def drain():  # (stderr in a thread: a chatty ffmpeg must not block on a full pipe)
                for line in p.stderr:
                    err.extend(line)
                    del err[:-4096]
            th = threading.Thread(target=drain, name="mk-ffmpeg-stderr", daemon=True)
            th.start()
            killed = threading.Event()

            def kill():
                killed.set()
                p.kill()
            timer = threading.Timer(timeout, kill)
            timer.start()
            n = 0
            try:
                while True:
                    try:
                        fr = read_ppm(p.stdout)
                    except ValueError as e:
                        if killed.is_set():
                            break
                        raise BadRequest(f"video: bad frame stream from ffmpeg ({e})")
                    if fr is None:
                        break
                    n += 1
                    if n > max_frames:
                        raise BadRequest(f"video of more than {max_frames} frames at {fps:g} fps (MK_MAX_VIDEO_FRAMES "
                                         f"{MAX_VIDEO_FRAMES} per request)")
                    yield fr
            finally:
                timer.cancel()
                if p.poll() is None:
                    p.kill()
                p.stdout.close()
                p.wait()
                th.join(5)
            if killed.is_set():
                raise BadRequest(f"video: decoding took over {timeout} s")
            if n == 0:
                msg = err.decode(errors="replace").strip().splitlines()
                raise BadRequest("video: no frames decoded" + (f" ({msg[-1][:300]})" if msg else ""))


# ---------------------------------------------------------------------------------------------
# chat template and output parsing


def make_template(src):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    return env.from_string(src)


# llama.cpp's CHATML_TEMPLATE_SRC: `--chat-template chatml`, and the template of a GGUF that has none
CHATML_TEMPLATE = ("{%- for message in messages -%}\n"
                   "  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n"
                   "{%- endfor -%}\n"
                   "{%- if add_generation_prompt -%}\n"
                   "  {{- '<|im_start|>assistant\\n' -}}\n"
                   "{%- endif -%}")
# the conversation a template given at startup must render (llama-server renders it at startup too)
TEMPLATE_EXAMPLE = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}, {"role": "user", "content": "How are you?"}]


def chat_template_source(kind, value):
    """the chat template of --chat-template-file PATH (kind "file": the file's text, UTF-8; CRLF line ends are read as
    LF, as llama-server on Windows reads it, and a byte-order mark is dropped) or --chat-template TEMPLATE (kind
    "inline": the Jinja source itself; "chatml" = llama.cpp's ChatML template), checked at startup: it must compile
    and render a short conversation with its messages in it. ValueError names the problem."""
    opt = "--chat-template-file" if kind == "file" else "--chat-template"
    if kind == "file":
        try:
            with open(value, "rb") as f:
                raw = f.read()
        except OSError as e:
            raise ValueError(f"{opt}: cannot read {value}: {e.strerror or e}") from None
        try:
            src = raw.decode("utf-8-sig").replace("\r\n", "\n")
        except UnicodeDecodeError as e:
            raise ValueError(f"{opt}: {value} is not UTF-8 text ({e.reason} at byte {e.start})") from None
        what = value
    else:
        src, what = (CHATML_TEMPLATE if value == "chatml" else value), "the template"
    try:
        tmpl = make_template(src)
    except jinja2.exceptions.TemplateSyntaxError as e:
        raise ValueError(f"{opt}: {what} does not compile: {e.message} (line {e.lineno})") from None
    try:
        text = tmpl.render(messages=TEMPLATE_EXAMPLE, add_generation_prompt=True)
    except Exception as e:  # (raise_exception, an undefined name, or a Python error inside an expression)
        raise ValueError(f"{opt}: {what} fails on a short conversation (system, user, assistant, user): "
                         f"{type(e).__name__}: {e}") from None
    if TEMPLATE_EXAMPLE[-1]["content"] not in text:
        raise ValueError(f"{opt}: {what} does not render the conversation's messages: not a chat template"
                         + (" (the only built-in name is chatml)" if kind == "inline" else ""))
    return src


class ChatTemplateArg(argparse.Action):
    """--chat-template-file / --chat-template: one setting, the last one given wins (as in llama-server)"""

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, ("file" if option_string == "--chat-template-file" else "inline", value))


class BadRequest(Exception):
    pass


class ClientGone(Exception):
    """the client closed its connection: the request stops, nothing is sent"""


class PrefillCancelled(Exception):
    """the engine stopped a prefill at a chunk boundary (Engine.cancel_when); nothing of it was kept"""


MEDIA_MARKER = "<__media__>"  # llama-server's (mtmd_default_marker)


def normalize_messages(messages, media=None):
    """media (a list: the server has a vision encoder): image_url parts and video parts (input_video / video_url)
    become MEDIA_MARKER and their file bytes are appended to media in order (videos as VideoBytes). A message with
    media is flattened to one string as llama-server does (text parts joined by newlines, no newline next to an image
    or video), so the prompt around each is llama-server's."""
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text", "image_url", *VIDEO_PARTS):
                    raise BadRequest(f"unsupported content part type {part.get('type')!r} (text, image_url, "
                                     f"input_video and video_url only)")
            kinds = {p.get("type") for p in content if isinstance(p, dict)}
            if kinds & {"image_url", *VIDEO_PARTS}:
                if media is None:
                    what = "image" if "image_url" in kinds else "video"
                    raise BadRequest(f"{what} input is not supported: start the server with --mmproj")
                text, after_media = "", False
                for part in content:
                    if part.get("type") in VIDEO_PARTS:
                        media.append(video_bytes(part))
                        text += MEDIA_MARKER
                        after_media = True
                    elif part.get("type") == "image_url":
                        iu = part.get("image_url")
                        media.append(image_bytes(iu.get("url") if isinstance(iu, dict) else iu))
                        text += MEDIA_MARKER
                        after_media = True
                    else:
                        if not after_media and text:
                            text += "\n"
                        text += str(part.get("text") or "")
                        after_media = False
                m["content"] = text
        if content is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


# a tool call counts only with both closing tags: one cut by max_tokens (or left open) is not a call
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>((?:(?!<tool_call>).)*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def unparsed_tool_call(text):
    """the tool-call markup after the last complete call that is not a call itself (cut by max_tokens, left open or
    malformed), from its <tool_call> to the end; empty if there is none"""
    end = 0
    for m in TOOL_CALL_RE.finditer(text):
        end = m.end()
    i = text.find("<tool_call>", end)
    return text[i:] if i >= 0 else ""


def parse_tool_calls(text, tools):
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        props = schemas.get(name, {})
        args = {}
        for p in PARAM_RE.finditer(m.group(2)):
            key, val = p.group(1).strip(), p.group(2)
            typ = (props.get(key) or {}).get("type")
            if typ == "string" or (typ is None and not val.strip()[:1] in "[{0123456789-tfn"):
                args[key] = val
            else:
                try:
                    args[key] = json.loads(val)
                except json.JSONDecodeError:
                    args[key] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def _off(limit, above=False):
    """a check that a sampler field has its 'off' value: None, or <= limit (above: >= limit)"""
    def ok(v):
        if v is None:
            return True
        try:
            return float(v) >= limit if above else float(v) <= limit
        except (TypeError, ValueError):
            return False
    return ok


# llama.cpp sampler fields the engine does not have (it samples with temperature, top_k, top_p, min_p and the presence /
# frequency penalties): accepted only with the value that turns them off (llama.cpp's web UI sends them all)
SAMPLER_OFF = {"mirostat": _off(0), "typical_p": _off(1, True), "typ_p": _off(1, True), "tfs_z": _off(1, True),
               "xtc_probability": _off(0), "dry_multiplier": _off(0), "top_n_sigma": _off(0), "dynatemp_range": _off(0),
               "min_tokens": _off(0), "n_probs": _off(0)}


def check_params(body):
    """400 for the request fields the server cannot honour (they would change the output and need engine work, e.g.
    logprobs, n > 1, constrained sampling): never ignored silently. Harmless ones (user, metadata, store, samplers,
    cache_prompt, ...) are accepted."""
    def bad(field, why):
        raise BadRequest(f"{field}: {why}")
    if not isinstance(body.get("messages"), list):
        bad("messages", "an array of messages is required")
    if body.get("n") not in (None, 1):
        bad("n", "one choice per request (the engine runs one sequence): send several requests instead")
    if body.get("logprobs") or body.get("top_logprobs") not in (None, 0):
        bad("logprobs", "token log-probabilities are not available from the megakernel (they need engine support)")
    if body.get("logit_bias"):
        bad("logit_bias", "not supported by the megakernel's sampler")
    rf = body.get("response_format")
    if rf is not None and not (isinstance(rf, dict) and rf.get("type", "text") == "text"):
        bad("response_format", "only {\"type\": \"text\"}: JSON modes need constrained sampling (a grammar), which the "
                               "megakernel does not have")
    for k in ("json_schema", "grammar"):
        if body.get(k):
            bad(k, "constrained sampling (a grammar) is not supported by the megakernel")
    tc = body.get("tool_choice")
    if tc not in (None, "auto", "none"):
        bad("tool_choice", "\"auto\" or \"none\": forcing a tool call (required / a named function) needs constrained "
                           "sampling, which the megakernel does not have")
    if body.get("tools") is not None and not isinstance(body["tools"], list):
        bad("tools", "an array of tools")
    if body.get("functions") or body.get("function_call"):
        bad("functions", "the legacy function-calling fields are not supported: use tools")
    if body.get("ignore_eos"):
        bad("ignore_eos", "not supported by the megakernel")
    if body.get("audio") or "audio" in (body.get("modalities") or []):
        bad("audio", "audio output is not supported")
    if body.get("add_generation_prompt") is False or body.get("continue_final_message"):
        bad("add_generation_prompt", "continuing the final message is not supported")
    for k, ok in SAMPLER_OFF.items():
        if not ok(body.get(k)):
            bad(k, f"{body.get(k)!r} is not supported: the megakernel samples with temperature, top_k, top_p, min_p "
                   "and the presence / frequency penalties only")


def stop_strings(v, field="stop"):
    """the stop strings of a request: a string or a list of strings (empty ones dropped, as llama-server does)"""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise BadRequest(f"{field} must be a string or a list of strings")
    v = [s for s in v if s]
    if len(v) > MAX_STOP or any(len(s) > MAX_STOP_LEN for s in v):
        raise BadRequest(f"{field}: at most {MAX_STOP} strings of at most {MAX_STOP_LEN} characters")
    return v


class StopMatcher:
    """Cuts the generated text at the first occurrence of a stop string, as llama-server does: on the raw text (reasoning,
    content and tool-call markup alike), the stop string and everything after it dropped. Text that could still become
    the start of a stop string is held back until the next piece decides it (or the end releases it)."""

    def __init__(self, stops):
        self.stops = stops
        self.held = ""
        self.word = None  # the stop string found

    def _partial(self, buf):
        """length of the longest suffix of buf that is a proper prefix of a stop string"""
        best = 0
        for s in self.stops:
            j = buf.find(s[0], max(0, len(buf) - len(s) + 1))  # (scanned from the longest candidate)
            while 0 <= j and len(buf) - j > best:
                if s.startswith(buf[j:]):
                    best = len(buf) - j
                    break
                j = buf.find(s[0], j + 1)
        return best

    def feed(self, text, final=False):
        """the text that can go out now; after a match (self.word set) nothing more"""
        if not self.stops or self.word is not None:
            return text if self.word is None else ""
        buf = self.held + text
        # (a match cannot start in text released before: its start would have been held)
        at = -1
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (at < 0 or i < at):
                at, self.word = i, s
        if at >= 0:
            self.held = ""
            return buf[:at]
        keep = 0 if final else self._partial(buf)
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool-call markup."""

    def __init__(self, thinking, tools):
        self.mode = "reasoning" if thinking else "content_start"
        self.tools = tools
        self.buf = ""
        self.tool_text = ""
        self.tool_sep = ""  # the whitespace between the content and the first <tool_call>
        self.content_seen = False

    def feed(self, text, final=False):
        """returns [(kind, text)] with kind in {reasoning, content}"""
        out = []
        self.buf += text
        while True:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if i >= 0:
                    out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len("</think>"):]
                    self.mode = "content_start"
                    continue
                keep = 0 if final else len("</think>") - 1
                if len(self.buf) > keep:
                    out.append(("reasoning", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.mode == "content_start":
                self.buf = self.buf.lstrip()
                if not self.buf:
                    break
                self.mode = "content"
                continue
            if self.mode == "content":
                if self.tools:
                    i = self.buf.find("<tool_call>")
                    if i >= 0:
                        head = self.buf[:i].rstrip()
                        out.append(("content", head))
                        self.tool_sep = self.buf[len(head):i]
                        self.tool_text = self.buf[i:]
                        self.buf = ""
                        self.mode = "tools"
                        break
                    keep = 0 if final else len("<tool_call>") - 1
                else:
                    keep = 0
                if final:
                    out.append(("content", self.buf.rstrip()))
                    self.buf = ""
                elif len(self.buf) > keep:
                    # trailing whitespace waits for more text: it is dropped before a tool call or the end
                    seg = self.buf[:len(self.buf) - keep]
                    head = seg.rstrip()
                    out.append(("content", head))
                    self.buf = seg[len(head):] + self.buf[len(self.buf) - keep:]
                break
            # tools: collect everything
            self.tool_text += self.buf
            self.buf = ""
            break
        out = [(k, t) for k, t in out if t]
        self.content_seen = self.content_seen or any(k == "content" for k, _ in out)
        return out

    def unparsed(self):
        """the tool-call markup that is not a complete call (unparsed_tool_call), as the rest of the content: the
        client gets the text, as from llama-server, instead of a call it would execute half-written"""
        rest = unparsed_tool_call(self.tool_text)
        return (self.tool_sep if self.content_seen else "") + rest if rest else ""


# ---------------------------------------------------------------------------------------------
# Anthropic Messages API (POST /v1/messages, /v1/messages/count_tokens): translated to the chat request above and back,
# so one code path generates for both (as llama-server's server_chat_convert_anthropic_to_oai)

# Claude Code's system prompt starts with a billing header whose cch= value changes on every request: normalized (as
# llama-server does), or the prompt would differ from its first tokens on and no prefix could be reused
BILLING_RE = re.compile(r"^(x-anthropic-billing-header:.*?cch=)[^;\s]{5};")


def _anthropic_image(src):
    if not isinstance(src, dict):
        raise BadRequest("image: a source object is required")
    if src.get("type") == "base64":
        return f"data:{src.get('media_type') or 'image/jpeg'};base64,{src.get('data') or ''}"
    if src.get("type") == "url":
        return str(src.get("url") or "")  # (refused by image_bytes: only data: URLs)
    raise BadRequest(f"image source type {src.get('type')!r} is not supported (base64)")


def _tool_result(content):
    """a tool_result block's content as an OpenAI tool message's: a string, or parts when it has images"""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise BadRequest("tool_result content: a string or an array of blocks")
    parts = []
    for c in content:
        t = c.get("type") if isinstance(c, dict) else None
        if t == "text":
            parts.append({"type": "text", "text": str(c.get("text") or "")})
        elif t == "image":
            parts.append({"type": "image_url", "image_url": {"url": _anthropic_image(c.get("source"))}})
        else:
            raise BadRequest(f"tool_result content block type {t!r} is not supported (text, image)")
    if any(p["type"] == "image_url" for p in parts):
        return parts
    return "\n".join(p["text"] for p in parts)


def anthropic_to_chat(body):
    """an Anthropic Messages request as the server's chat request. system: a string or text blocks (joined by blank
    lines); content blocks text / image / thinking (-> reasoning_content) / tool_use (-> tool_calls) / tool_result (->
    tool messages, before the rest of their message: they answer the previous turn's calls); tools input_schema ->
    parameters; tool_choice auto / none / any / tool; stop_sequences -> stop; thinking enabled / adaptive -> thinking on,
    disabled or absent -> off (Anthropic's default; chat_template_kwargs win)"""
    if not isinstance(body.get("messages"), list):
        raise BadRequest("messages: an array of messages is required")
    msgs = []
    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            texts = [system]
        elif isinstance(system, list) and all(isinstance(b, dict) and b.get("type") == "text" for b in system):
            texts = [str(b.get("text") or "") for b in system]
        else:
            raise BadRequest("system: a string or an array of text blocks")
        msgs.append({"role": "system", "content": "\n\n".join(BILLING_RE.sub(r"\1fffff;", t) for t in texts)})
    for m in body["messages"]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise BadRequest("messages: each message needs the role user or assistant")
        role, content = m["role"], m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise BadRequest("message content: a string or an array of content blocks")
        parts, calls, results, thinking = [], [], [], ""
        for b in content:
            t = b.get("type") if isinstance(b, dict) else None
            if t == "text":
                parts.append({"type": "text", "text": str(b.get("text") or "")})
            elif t == "image":
                parts.append({"type": "image_url", "image_url": {"url": _anthropic_image(b.get("source"))}})
            elif t == "thinking":
                thinking += str(b.get("thinking") or "")
            elif t == "redacted_thinking":
                pass
            elif t == "tool_use":
                calls.append({"id": str(b.get("id") or ""), "type": "function", "function": {
                    "name": str(b.get("name") or ""), "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
            elif t == "tool_result":
                results.append({"role": "tool", "tool_call_id": str(b.get("tool_use_id") or ""),
                                "content": _tool_result(b.get("content"))})
            else:
                raise BadRequest(f"content block type {t!r} is not supported (text, image, thinking, tool_use, tool_result)")
        msgs += results
        if parts or calls or thinking:
            nm = {"role": role, "content": parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text"
                  else (parts or "")}
            if calls:
                nm["tool_calls"] = calls
            if thinking:
                nm["reasoning_content"] = thinking
            msgs.append(nm)
    out = {"messages": msgs, "stream": bool(body.get("stream"))}
    out["max_tokens"] = body.get("max_tokens") if body.get("max_tokens") is not None else 4096  # (llama-server's)
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stop_sequences") is not None:
        out["stop"] = stop_strings(body["stop_sequences"], "stop_sequences")
    tools = body.get("tools")
    if tools:
        if not isinstance(tools, list):
            raise BadRequest("tools: an array of tools")
        out["tools"] = []
        for t in tools:
            if not isinstance(t, dict) or t.get("type") not in (None, "custom") or not t.get("name"):
                raise BadRequest(f"tools: only custom tools (name, input_schema), not {str(t)[:80]!r}")
            out["tools"].append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description") or "", "parameters": t.get("input_schema") or {}}})
    tc = body.get("tool_choice")
    if tc is not None:
        kind = tc.get("type") if isinstance(tc, dict) else None
        if kind in ("any", "tool"):
            raise BadRequest(f"tool_choice {kind}: forcing a tool call needs constrained sampling, which the megakernel "
                             "does not have (auto, none)")
        if kind not in ("auto", "none"):
            raise BadRequest("tool_choice: {\"type\": \"auto\"} or {\"type\": \"none\"}")
        out["tool_choice"] = kind
        if tc.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False
    if body.get("mcp_servers"):
        raise BadRequest("mcp_servers: not supported (the client runs its tools)")
    think = body.get("thinking")
    kw = dict(body.get("chat_template_kwargs") or {})
    if think is not None and not (isinstance(think, dict) and think.get("type") in ("enabled", "adaptive", "disabled")):
        raise BadRequest("thinking: {\"type\": \"enabled\" | \"adaptive\" | \"disabled\"}")
    kw.setdefault("enable_thinking", bool(think) and think["type"] != "disabled")
    out["chat_template_kwargs"] = kw
    return out


def anthropic_stop_reason(finish, stop_word):
    if finish == "length":
        return "max_tokens"
    if finish == "tool_calls":
        return "tool_use"
    return "stop_sequence" if stop_word is not None else "end_turn"


def anthropic_usage(usage):
    reused = usage["prompt_tokens_details"]["cached_tokens"]
    return {"input_tokens": usage["prompt_tokens"] - reused, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": reused, "output_tokens": usage["completion_tokens"]}


def anthropic_tool_use(call):
    try:
        args = json.loads(call["function"]["arguments"])
    except ValueError:
        args = {}
    return {"type": "tool_use", "id": "toolu_" + call["id"].removeprefix("call_"), "name": call["function"]["name"],
            "input": args if isinstance(args, dict) else {}}


ANTHROPIC_ERRORS = {400: "invalid_request_error", 404: "not_found_error", 413: "request_too_large",
                    500: "api_error", 501: "api_error", 503: "overloaded_error"}


# ---------------------------------------------------------------------------------------------


class Server:
    vision = None   # Vision (--mmproj), else text only
    decoder = None  # VideoDecoder (an ffmpeg was found): video input with the vision encoder
    video_ok = False  # video input works in this process (mmproj loaded, frame-pair encoder, ffmpeg): /props
    img_spans = ()  # the images of the engine's sequence: (pos, n, key); replaced, never mutated
    stop_word = None  # the stop string that ended the last reply (None: it ended otherwise)
    last_prompt = (0, 0)  # the running request's prompt: (tokens, reused tokens), once it is processed

    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.eng = Engine(args.lib, args.model, args.ctx, args.draft, args.draft_vocab,
                          {"f16": 0, "q8": 1, "auto": -1}[args.kv], "none" if args.no_mtp else args.mtp)
        if not self.eng.has_mtp:
            log("no MTP layer: plain decoding (depth 0)")
        # --chat-template-file / --chat-template (read and checked in main) replace the GGUF's template everywhere it is
        # used: chat and Anthropic requests, /apply-template, token counts, /props; a GGUF without one gets ChatML
        src = getattr(args, "chat_template_src", None)
        self.template_src = src if src is not None else self.eng.chat_template()
        if not self.template_src:
            log("the GGUF has no chat template: using ChatML (as llama-server; --chat-template-file to give one)")
            self.template_src = CHATML_TEMPLATE
        self.template = make_template(self.template_src)
        im = self.eng.tokenize("<|im_start|>")
        self.im_start = im[0] if len(im) == 1 else None  # checkpoint at the start of the generation prompt
        if args.mmproj:
            if not self.eng.images:
                raise RuntimeError(f"{args.lib} has no image input (mk_prefill_img): rebuild it")
            self.vision = Vision(args.vision_lib, args.mmproj, args.model, args.vision_threads, args.image_min_tokens,
                                 args.image_max_tokens, args.image_cache, args.vision_backend, args.video_cache)
            if self.vision.n_embd != self.eng.L.mk_n_embd():
                raise RuntimeError(f"mmproj rows of {self.vision.n_embd} for a model of {self.eng.L.mk_n_embd()}")
            ids = [self.eng.tokenize(t) for t in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")]
            if any(len(i) != 1 for i in ids):
                raise RuntimeError("the vocabulary has no <|vision_start|> / <|vision_end|> / <|image_pad|> tokens")
            self.vis_start, self.vis_end, self.img_pad = (i[0] for i in ids)
            self.setup_video(args)
        self.warm_up()
        self.model_id = os.path.basename(args.model)
        self.lock = threading.Lock()
        self.warned = set()
        vis = (f", vision on the CPU ({os.path.basename(args.mmproj)}, {self.vision.backend}, {self.vision.n_threads} threads)"
               if self.vision else "")
        self.stats0 = self.eng.stats()  # (the warm-up's cycle: /metrics counts from here)
        self.t_start = time.time()
        log(f"ready in {time.time() - t0:.1f} s: {self.model_id}, ctx {args.ctx}, {args.draft} drafts{vis}")
        self.warm_s = self.eng.keep_warm()  # (MK_KEEP_WARM_S: the first reply starts at full clocks too)

    def warm_up(self):
        """Runs one prefill and one decode cycle before the server listens.

        Loading allocates the weights and the KV cache, but the first kernel launches still need
        device memory of their own. When the context is too large for that, loading succeeds, /health
        answers, and the first request ends the process with a CUDA out-of-memory (CK exits), which
        the client sees as a dropped connection. Warming up here moves that failure to startup.
        The prompt must be at least two tokens: only then does prefill take the batched path, whose
        buffers are allocated on first use."""
        toks = self.eng.tokenize("Hello, how are you today?")
        assert len(toks) >= 2
        self.eng.prefill(toks)
        self.eng.step()
        self.eng.reset()
        if self.vision:
            self.vision.warm_up()

    def setup_video(self, args):
        """video input (input_video / video_url parts): needs the vision library's frame pairs (mkv_frames_load) and an
        ffmpeg (find_ffmpeg); without them video requests get a 400 and video_ok stays False"""
        if args.video_max_tokens and self.vision.backend != "vit_cpu":
            raise RuntimeError("--video-max-tokens needs the cpu vision backend (libmtmd resizes frames as images)")
        if not self.vision.frames_ok:
            log(f"video input off: {args.vision_lib} has no video frames (mkv_frames_load): rebuild it")
            return
        exe = find_ffmpeg(args.ffmpeg)
        if not exe:
            log("video input off: no ffmpeg (--ffmpeg / MK_FFMPEG, ffmpeg on PATH, or pip install imageio-ffmpeg)"
                if not (args.ffmpeg or os.environ.get("MK_FFMPEG")) else
                f"video input off: ffmpeg not found at {args.ffmpeg or os.environ.get('MK_FFMPEG')}")
            return
        dec = VideoDecoder(exe)
        try:
            dec.check()
        except RuntimeError as e:
            log(f"video input off: {e}")
            return
        self.decoder, self.video_ok = dec, True
        log(f"video input: {exe} ({dec.version}), {args.video_fps:g} fps, frames of at most "
            f"{args.video_max_tokens or args.image_max_tokens or 'the default'} tokens, {MAX_VIDEO_FRAMES} frames per request")

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log("warning:", msg)

    # ---- reporting: /props, /v1/models, /slots, /metrics ----
    METRICS = ("requests", "requests_bad", "requests_failed", "requests_cancelled", "deferred", "prompt_tokens",
               "prompt_tokens_reused", "prompt_seconds", "tokens_predicted", "predicted_seconds")
    _mlock = threading.Lock()
    t_start = 0.0
    warm_s = 0.0
    stats0 = stats1 = None  # engine (cycles, drafted, accepted) after the warm-up / after the last request

    def count(self, **kw):
        """adds to the counters of /metrics"""
        with self._mlock:
            m = self.__dict__.setdefault("metrics", dict.fromkeys(self.METRICS, 0))
            for k, v in kw.items():
                m[k] += v

    def counters(self):
        with self._mlock:
            return dict(self.__dict__.get("metrics") or dict.fromkeys(self.METRICS, 0))

    def busy(self):
        return self.lock.locked()

    def n_vocab(self):
        return getattr(self.eng, "n_vocab", 0)

    def prompt_tokens(self, body):
        """the token count of the request's prompt (rendered with the chat template, images as their tokens), without
        generating: /v1/messages/count_tokens, /v1/chat/completions/input_tokens"""
        text, _, media = self.render(body)
        if not media:
            return len(self.eng.tokenize(text))
        with self.lock:  # (the image cache belongs to the running request too)
            toks, images = self.tokenize_prompt(text, media)
            self.vision.release([im for _, im in images])
        return len(toks)

    def engine_info(self):
        """what the server knows for sure about its engine: arguments, environment, the library's own answers"""
        a, e, v = self.args, self.eng, self.vision
        g = lambda o, k, d=None: getattr(o, k, d)  # noqa: E731 (fake engines in the tests have fewer attributes)
        return {
            "build": g(e, "build"), "lib": os.path.basename(str(g(a, "lib", ""))),
            "n_ctx": a.ctx, "kv": g(a, "kv"),  # (the KV format asked for; "auto" is resolved inside the engine)
            "mtp": g(e, "has_mtp", True), "mtp_file": g(a, "mtp"),
            "draft_depth": g(a, "draft") if g(e, "has_mtp", True) else 0,
            "draft_vocab": os.path.basename(a.draft_vocab) if isinstance(g(a, "draft_vocab"), str) else None,
            "exact_reuse": bool(g(e, "exact", False)),
            "checkpoints": {"pool": bool(g(e, "pool", False)), "every": g(a, "ck_every"), "messages": bool(g(a, "ck_msgs")),
                            "slots": int(os.environ.get("MK_CK_SLOTS") or 8)},
            "keep_warm_s": self.warm_s, "prefill_cancel": bool(g(e, "_cancel", False)),
            "vision": None if v is None else {
                "mmproj": os.path.basename(str(g(a, "mmproj", ""))), "backend": g(v, "backend"),
                "threads": g(v, "n_threads"), "image_min_tokens": g(a, "image_min_tokens"),
                "image_max_tokens": g(a, "image_max_tokens"), "image_cache": g(v, "cache_max")},
            "limits": {"max_body_mb": MAX_BODY_MB, "max_images": MAX_IMAGES, "max_stop": MAX_STOP,
                       "image_max_pixels": int(os.environ.get("MK_IMAGE_MAX_PIXELS") or 100_000_000)},
            "env": {k: val for k, val in sorted(os.environ.items()) if k.startswith("MK_")},
            "busy": self.busy(), "uptime_s": round(time.time() - self.t_start, 1) if self.t_start else 0.0,
        }

    def props(self):
        """llama-server's GET /props fields we have, plus `engine` (engine_info)"""
        b = getattr(self.eng, "build", None) or {}
        params = {"temperature": DEF_TEMP, "top_k": DEF_TOP_K, "top_p": DEF_TOP_P, "min_p": DEF_MIN_P, "seed": -1,
                  "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": -1, "n_predict": -1, "stop": [],
                  "stream": False, "n_probs": 0}
        return {
            "default_generation_settings": {"n_ctx": self.args.ctx, "params": params},
            "total_slots": 1, "model_alias": self.model_id, "model_path": getattr(self.args, "model", None),
            "modalities": {"vision": self.vision is not None, "video": bool(getattr(self, "video_ok", False)),
                           "audio": False},
            "media_marker": MEDIA_MARKER, "endpoint_slots": True, "endpoint_props": False, "endpoint_metrics": True,
            "chat_template": getattr(self, "template_src", None),
            "build_info": f"{b.get('commit', 'unknown')} {b.get('flags', '')}".strip() if b else "unknown",
            "is_sleeping": False, "engine": self.engine_info(),
        }

    def models(self):
        """GET /v1/models: the one model, with llama-server's meta where we have it"""
        meta = {"n_ctx": self.args.ctx}
        try:
            meta["size"] = os.path.getsize(self.args.model)
        except (OSError, TypeError):
            pass
        if getattr(self.eng, "images", False):
            meta["n_embd"] = self.eng.L.mk_n_embd()
        if self.n_vocab():
            meta["n_vocab"] = self.n_vocab()
        caps = ["completion"] + (["multimodal"] if self.vision is not None else [])
        return {"object": "list",
                "data": [{"id": self.model_id, "object": "model", "created": int(self.t_start), "owned_by": "openjet",
                          "meta": meta}],
                "models": [{"name": self.model_id, "model": self.model_id, "type": "model", "capabilities": caps,
                            "details": {"format": "gguf"}}]}

    def slots(self):
        """GET /slots: the one slot (llama-server's fields we have)"""
        return [{"id": 0, "n_ctx": self.args.ctx, "is_processing": self.busy(),
                 "speculative": bool(getattr(self.eng, "has_mtp", True)) and bool(getattr(self.args, "draft", 0))}]

    def metrics_text(self):
        """GET /metrics: Prometheus text; llama-server's names (llamacpp:) where they mean the same, megakernel: else"""
        m = self.counters()
        s0, s1 = self.stats0, self.stats1
        cyc, drafted, accepted = ((b - a for a, b in zip(s0, s1)) if s0 and s1 else (0, 0, 0))
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed (prefilled; reused ones not).",
             m["prompt_tokens"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt process time", m["prompt_seconds"]),
            ("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.", m["tokens_predicted"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Predict process time", m["predicted_seconds"]),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.",
             m["prompt_tokens"] / m["prompt_seconds"] if m["prompt_seconds"] else 0.0),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.",
             m["tokens_predicted"] / m["predicted_seconds"] if m["predicted_seconds"] else 0.0),
            ("llamacpp:requests_processing", "gauge", "Number of requests processing.", int(self.busy())),
            ("llamacpp:requests_deferred", "gauge", "Number of requests deferred.", m["deferred"]),
            ("megakernel:prompt_tokens_reused_total", "counter", "Prompt tokens reused from the engine's sequence.",
             m["prompt_tokens_reused"]),
            ("megakernel:decode_cycles_total", "counter", "Engine decode cycles (one verify + MTP draft each).", cyc),
            ("megakernel:draft_tokens_total", "counter", "MTP draft tokens produced.", drafted),
            ("megakernel:draft_tokens_accepted_total", "counter", "MTP draft tokens accepted.", accepted),
            ("megakernel:requests_total", "counter", "Generation requests received.", m["requests"]),
            ("megakernel:requests_bad_total", "counter", "Generation requests answered with a 4xx.", m["requests_bad"]),
            ("megakernel:requests_failed_total", "counter", "Generation requests that failed (5xx).",
             m["requests_failed"]),
            ("megakernel:requests_cancelled_total", "counter", "Generation requests whose client disconnected.",
             m["requests_cancelled"]),
        ]
        return "".join(f"# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v:g}\n" if isinstance(v, float) else
                       f"# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v}\n" for n, t, h, v in rows)

    def render(self, body):
        """(prompt text, thinking, image file bytes in prompt order: one MEDIA_MARKER in the text each)"""
        kw = dict(body.get("chat_template_kwargs") or {})
        eff = body.get("reasoning_effort")  # (OpenAI's top-level field; chat_template_kwargs win)
        if isinstance(eff, str) and eff:
            if eff == "none":  # (as llama-server)
                kw.setdefault("enable_thinking", False)
            else:  # (the template knows low / medium / high / xhigh)
                kw.setdefault("reasoning_effort", {"minimal": "low"}.get(eff, eff))
        media = [] if self.vision else None
        ctx = dict(messages=normalize_messages(body.get("messages") or [], media), add_generation_prompt=True)
        if body.get("tools"):
            ctx["tools"] = body["tools"]
        for k in ("enable_thinking", "preserve_thinking", "reasoning_effort"):
            if k in kw:
                ctx[k] = kw[k]
        try:
            text = self.template.render(**ctx)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}")
        thinking = text.endswith("<think>\n")
        n_img = sum(1 for d in media or [] if not isinstance(d, VideoBytes))
        if n_img > MAX_IMAGES:
            raise BadRequest(f"{n_img} images in one request: at most {MAX_IMAGES} (MK_MAX_IMAGES)")
        if n_img < len(media or []) and not self.video_ok:
            raise BadRequest("video input is not available: " + (
                "the server found no ffmpeg (--ffmpeg / MK_FFMPEG, ffmpeg on PATH, or pip install imageio-ffmpeg)"
                if self.vision is not None and self.vision.frames_ok else "the vision library has no video frames"))
        if media and text.count(MEDIA_MARKER) != len(media):
            raise BadRequest(f"the rendered prompt has {text.count(MEDIA_MARKER)} media markers for {len(media)} images / videos")
        return text, thinking, media or []

    def tokenize_prompt(self, text, media):
        """the prompt's token ids and its images [(pos, Vision.Image)]: each MEDIA_MARKER becomes <|vision_start|>, the
        image's rows (<|image_pad|> placeholders, nx * ny of them), <|vision_end|>; the text between is tokenized piece
        by piece, as mtmd_tokenize does. A video's marker becomes its parts (Vision.video: texts, and frames / frame
        pairs each like an image). The images are pinned (Vision.image / video): the caller releases them."""
        if not media:
            return self.eng.tokenize(text), []
        toks, images = [], []
        segs = text.split(MEDIA_MARKER)
        frames_left = MAX_VIDEO_FRAMES

        def add_image(im, j=None):  # (j: its slot in images, else appended)
            toks.append(self.vis_start)
            if j is None:
                images.append((len(toks), im))
            else:
                images[j] = (len(toks), im)
            toks.extend([self.img_pad] * im.n)
            toks.append(self.vis_end)
        try:
            for i, seg in enumerate(segs):
                if seg:
                    toks += self.eng.tokenize(seg)
                if i < len(media):
                    if isinstance(media[i], VideoBytes):
                        # llama.cpp's layout: "Video:", frames / frame pairs as images, timestamp texts between (each
                        # text tokenized on its own, as mtmd_tokenize does)
                        v = self.vision.video(media[i], self.decoder, self.args.video_fps, self.args.video_max_tokens,
                                              frames_left)
                        j = len(images)
                        images.extend((None, im) for im in v.images)  # (all pinned now: released below on an error)
                        frames_left -= v.n_frames
                        for part in v.parts:
                            if isinstance(part, str):
                                toks += self.eng.tokenize(part)
                            else:
                                add_image(part, j)
                                j += 1
                    else:
                        add_image(self.vision.image(media[i]))
                    if len(toks) + 16 >= self.args.ctx:  # (no more images decoded for a prompt that cannot run)
                        raise BadRequest(f"prompt of at least {len(toks)} tokens does not fit the context of {self.args.ctx}")
        except BaseException:
            self.vision.release([im for _, im in images])
            raise
        return toks, images

    def ck_positions(self, toks, start):
        """checkpoints for a prefill of toks[start:] besides the generation prompt: each message start and every
        --ck-every tokens (absolute positions; the engine keeps the most useful ones in its pool)"""
        cks = set()
        every = self.args.ck_every
        if every > 0:
            cks.update(range((start // every + 1) * every, len(toks) - 1, every))
        if self.args.ck_msgs and self.im_start is not None:
            cks.update(k for k in range(start + 1, len(toks) - 1) if toks[k] == self.im_start)
        return sorted(cks)

    def load_prompt(self, toks, images=(), gone=None):
        """brings the engine to `toks` processed, reusing what it has; returns (first token, reused count).
        images: [(pos, Vision.Image)] of the prompt. gone (the client left: () -> bool) is watched during the prefill,
        which then stops at a chunk boundary with PrefillCancelled (Engine.cancel_when)"""
        eng = self.eng
        seq = eng.sequence()
        lcp = 0
        for a, b in zip(seq, toks):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(toks) - 1)  # at least one token is fed (it produces the next one)
        # rewind checkpoint at the last <|im_start|> (the generation prompt): a follow-up turn re-renders this turn
        # from there (thinking mode: <think>\n\n</think> instead of <think>\n), so it must not sit at the prompt's end
        gen = -1
        if self.im_start is not None:
            for k in range(len(toks) - 1, -1, -1):
                if toks[k] == self.im_start:
                    gen = k
                    break
        # images: their rows' placeholder ids are alike, so the processed sequence matches only up to the first image
        # that is not the same image at the same position in both
        spans = [(p, im.n, im.key) for p, im in images]
        for p, _, _ in set(spans) ^ set(self.img_spans):
            lcp = min(lcp, p)

        def feed(start):  # prefill toks[start:] with the images it overlaps (encoded now, each once: Vision's cache)
            fed = [(p, im) for p, im in images if p + im.n > start]
            if any(im.encode_s is None for _, im in fed):
                # the CPU encodes leave the GPU idle (up to seconds): keep its clocks up until the prefill's first launch
                eng.keep_warm()
            imgs, armed = [], time.monotonic()
            for p, im in fed:  # (a video's frame pairs encode for seconds to minutes: a client that left stops them)
                if imgs and im.encode_s is None:
                    if gone is not None and gone():
                        raise PrefillCancelled(f"the client left during the image encodes ({len(imgs)} of {len(fed)} done)")
                    if time.monotonic() - armed > 30:  # (keep-warm re-armed through long encodes)
                        eng.keep_warm()
                        armed = time.monotonic()
                imgs.append((p, im.nx, im.ny, im.rows()))
            with eng.cancel_when(gone):
                if imgs:
                    return eng.prefill(toks[start:], gen, self.ck_positions(toks, start), imgs), start
                return eng.prefill(toks[start:], gen, self.ck_positions(toks, start)), start

        self.img_spans = []  # (until the prefill succeeds)
        # exact reuse (an MK_EXACT engine, --exact-reuse): never append to the sequence. Its last prompt row and the
        # reply were processed by the decode kernels, whose arithmetic differs from the batched prefill's; restoring
        # the last checkpoint at or before the divergence (all of them sit inside earlier batched prefills, on the
        # absolute 32-token scan grid) and prefilling from there computes what a prefill from 0 computes, bit for bit.
        if seq and lcp >= eng.committed() and not eng.exact:
            eng.truncate(lcp)
            r = feed(lcp)
        else:
            # the last checkpoint at or before the divergence (one inside a kept image: the image is fed again)
            ck = max((p for p in eng.checkpoints() if 0 < p <= lcp), default=0)
            if ck and eng.rewind_to(ck):
                r = feed(ck)
            else:
                eng.reset()
                r = feed(0)
        self.img_spans = spans
        return r

    def sampling(self, body):
        """top_k: absent -> DEF_TOP_K; <= 0 ("off" for llama.cpp / OpenAI-style clients) -> TOPK_MAX, the engine's
        maximum; above TOPK_MAX -> TOPK_MAX with a one-time warning"""
        try:
            temp = float(body["temperature"] if body.get("temperature") is not None else DEF_TEMP)
            top_p = float(body["top_p"] if body.get("top_p") is not None else DEF_TOP_P)
            top_k = int(body["top_k"]) if body.get("top_k") is not None else DEF_TOP_K
            min_p = float(body["min_p"] if body.get("min_p") is not None else DEF_MIN_P)
            seed = body.get("seed")
            seed = random.getrandbits(63) if seed is None or int(seed) < 0 else int(seed)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("temperature, top_p, top_k, min_p and seed must be numbers")
        if top_k <= 0:
            top_k = TOPK_MAX
        elif top_k > TOPK_MAX:
            self.warn_once("top_k", f"top_k {top_k} -> {TOPK_MAX} (the megakernel samples from at most {TOPK_MAX} candidates)")
            top_k = TOPK_MAX
        # presence / frequency penalty, vLLM's semantics: tokens generated in this response so far (not the prompt)
        pen = []
        for k in ("presence_penalty", "frequency_penalty"):
            try:
                v = float(body.get(k) or 0)
            except (TypeError, ValueError):
                raise BadRequest(f"{k} must be a number")
            if not -2.0 <= v <= 2.0:
                raise BadRequest(f"{k} must be in [-2, 2]")
            pen.append(v)
        self.eng.set_penalty(*pen)
        if float(body.get("repetition_penalty") or body.get("repeat_penalty") or 1.0) != 1.0:
            self.warn_once("rep", "repetition_penalty is not supported by the megakernel and is ignored")
        if temp <= 0:
            self.eng.set_sampling(0.0, 1.0, TOPK_MAX, 0.0, seed)
        else:
            self.eng.set_sampling(temp, top_p, top_k, min_p, seed)

    def generate(self, body, emit, gone=lambda: False):
        """runs one chat completion; emit(kind, payload) is called with ('reasoning'|'content', text) deltas.
        Returns (finish_reason, tool_calls, usage, timings). emit returning False, or gone() returning True (the client
        closed its connection; asked before the prefill, during it by a watcher thread (the prefill stops at its next
        chunk) and before each engine cycle) stops it with ClientGone."""
        check_params(body)
        text, thinking, media = self.render(body)
        toks, images = self.tokenize_prompt(text, media)
        try:
            return self._generate(body, emit, gone, thinking, toks, images)
        finally:
            if images:  # (the request's images stay pinned until here)
                self.vision.release([im for _, im in images])
            if getattr(self.eng, "stats", None) is not None:  # (/metrics: the engine's counters, read while it is idle)
                self.stats1 = self.eng.stats()

    def _generate(self, body, emit, gone, thinking, toks, images):
        max_new = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or -1
        try:
            max_new = int(max_new)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("max_tokens must be an integer")
        limit = self.args.ctx - len(toks) - 16
        if limit <= 0:
            raise BadRequest(f"prompt of {len(toks)} tokens does not fit the context of {self.args.ctx}")
        max_new = limit if max_new <= 0 else min(max_new, limit)
        # tool_choice "none": the tools stay in the prompt, calls are not parsed (returned as text), as llama-server does
        tools = (body.get("tools") or []) if body.get("tool_choice") != "none" else []
        stopper = StopMatcher(stop_strings(body.get("stop")))
        self.stop_word = None
        self.sampling(body)
        if gone():  # (it left while the request waited for the engine)
            log("the client disconnected before its prompt was processed")
            raise ClientGone()
        t0 = time.time()
        enc0 = [im.encode_s for _, im in images]
        try:
            tok, reused = self.load_prompt(toks, images, gone)
        except PrefillCancelled as e:  # (the engine kept nothing of it)
            log(f"prompt {len(toks)} tok: the client disconnected during its prefill, stopped after {time.time() - t0:.2f} s ({e})")
            raise ClientGone()
        t1 = time.time()
        self.last_prompt = (len(toks), reused)  # (an Anthropic stream's message_start reports it)
        enc = [im.encode_s for (_, im), e in zip(images, enc0) if e is None and im.encode_s is not None]
        if images:
            log(f"images: {len(images)} ({', '.join(f'{im.nx}x{im.ny}' for _, im in images)} tokens), "
                f"{len(enc)} encoded on the CPU in {sum(enc):.2f} s")
        parser = OutputParser(thinking, tools)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        n_gen, finish, pending, aborted, dropped = 0, "length", [tok], False, 0
        while n_gen < max_new and not aborted:
            if not pending:
                if gone():  # once per engine cycle: a closed connection stops the GPU work here
                    aborted = True
                    break
                pending = self.eng.step()
            t = pending.pop(0)
            n_gen += 1
            if self.eng.is_eog(t):
                finish = "stop"
                break
            for kind, delta in parser.feed(stopper.feed(dec.decode(self.eng.piece(t)))):
                if emit(kind, delta) is False:
                    aborted = True
                    break
            if stopper.word is not None:
                # a stop string: the reply ends at it, and the tokens after this one (the rest of the cycle's accepted
                # drafts) are dropped. The engine's sequence keeps them, as after max_tokens: the next request finds
                # the divergence (load_prompt) and continues from the reply that was sent.
                finish, dropped = "stop", len(pending)
                break
        if aborted:  # (the engine's state is that of a reply cut by max_tokens: the next request reuses it as usual)
            log(f"prompt {len(toks)} tok ({reused} reused), {n_gen} tok in {time.time() - t1:.2f} s: the client "
                f"disconnected, stopped")
            raise ClientGone()
        for kind, delta in parser.feed(stopper.feed(dec.decode(b"", final=True), final=True), final=True):
            emit(kind, delta)
        self.stop_word = stopper.word
        calls = parse_tool_calls(parser.tool_text, tools) if parser.tool_text else []
        if body.get("parallel_tool_calls") is False:
            calls = calls[:1]  # (the first call only)
        rest = parser.unparsed()
        if rest:
            emit("content", rest)
        # a call cut by max_tokens is text, and the reply ends with "length" (complete calls before it are still sent)
        if calls and not (rest and finish == "length"):
            finish = "tool_calls"
        t2 = time.time()
        log(f"prompt {len(toks)} tok ({reused} reused) in {t1 - t0:.2f} s, {n_gen} tok in {t2 - t1:.2f} s "
            f"({n_gen / max(t2 - t1, 1e-9):.1f} tok/s), {finish}"
            + (f" (stop string; {dropped} tokens of the cycle dropped)" if stopper.word is not None else ""))
        self.count(prompt_tokens=len(toks) - reused, prompt_tokens_reused=reused, prompt_seconds=t1 - t0,
                   tokens_predicted=n_gen, predicted_seconds=t2 - t1)
        usage = {"prompt_tokens": len(toks), "completion_tokens": n_gen, "total_tokens": len(toks) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": reused}}
        timings = {"prompt_n": len(toks) - reused, "prompt_ms": (t1 - t0) * 1e3, "predicted_n": n_gen,
                   "predicted_ms": (t2 - t1) * 1e3, "predicted_per_second": n_gen / max(t2 - t1, 1e-9)}
        return finish, calls, usage, timings


POST_ROUTES = {"/v1/chat/completions": "chat", "/chat/completions": "chat", "/v1/messages": "messages",
               "/v1/messages/count_tokens": "count_tokens", "/tokenize": "tokenize", "/detokenize": "detokenize",
               "/apply-template": "apply_template", "/v1/chat/completions/input_tokens": "input_tokens",
               "/chat/completions/input_tokens": "input_tokens"}
# llama-server's other POST routes: 501 (not supported here) instead of 404
UNSUPPORTED = {"/completion", "/completions", "/v1/completions", "/v1/responses", "/responses", "/v1/responses/input_tokens",
               "/responses/input_tokens", "/infill", "/embedding", "/embeddings", "/v1/embeddings", "/rerank", "/reranking",
               "/v1/rerank", "/v1/reranking", "/v1/audio/transcriptions", "/audio/transcriptions", "/lora-adapters",
               "/v1/chat/completions/control", "/models", "/v1/streams/lookup", "/tools", "/cors-proxy"}


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def client_gone(self):
            """True once the client has closed its connection (end of file or reset). The request was read in full, so
            anything readable is the next request of a keep-alive client (still connected) or the close."""
            try:
                if not select.select([self.connection], [], [], 0)[0]:
                    return False
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

        def _json(self, code, obj, close=False):
            """close: then close the connection (a request body that was not read must not be taken for the next request)"""
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def _body_length(self):
            """(the request body's length, None), or (None, (status, message)) when it is not acceptable: chunked or
            no Content-Length (411), a malformed one (400), over MK_MAX_BODY_MB (413, refused before reading)"""
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                return None, (411, "chunked request bodies are not supported: send a Content-Length")
            vals = [v.strip() for v in self.headers.get_all("Content-Length") or []]
            if not vals:
                return None, (411, "a Content-Length header is required")
            if len(set(vals)) > 1 or not re.fullmatch(r"[0-9]+", vals[0]):
                return None, (400, f"bad Content-Length {vals[0][:40]!r}")
            n = int(vals[0]) if len(vals[0]) <= 18 else 1 << 62
            if n > MAX_BODY_MB << 20:
                return None, (413, f"request body of {n} bytes: over the limit of {MAX_BODY_MB} MB (MK_MAX_BODY_MB)")
            return n, None

        def _refuse_body(self):
            """answers an unacceptable request body (_body_length) without reading it; True if it did"""
            refused = self._body_length()[1]
            if refused:
                self._json(refused[0], {"error": {"message": refused[1], "type": "invalid_request_error"}}, close=True)
            return bool(refused)

        def handle_expect_100(self):
            # (curl sends large bodies after a 100 Continue: a refused one is answered before it is sent)
            if self.command == "POST" and self._refuse_body():
                return False
            return super().handle_expect_100()

        def _text(self, code, text, ctype="text/plain; charset=utf-8", headers=()):
            data = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, srv.models())
            elif path == "/props":
                self._json(200, srv.props())
            elif path == "/slots":
                if parse_qs(query).get("fail_on_no_slot", [""])[0] and srv.busy():
                    self._json(503, {"error": {"code": 503, "message": "no slot available", "type": "unavailable_error"}})
                else:
                    self._json(200, srv.slots())
            elif path == "/metrics":
                self._text(200, srv.metrics_text(), "text/plain; version=0.0.4",
                           [("Process-Start-Time-Unix", str(int(srv.t_start)))])
            else:
                self._json(404, {"error": {"message": "not found", "type": "not_found_error"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            if self._refuse_body():
                return
            n = self._body_length()[0]
            raw = self.rfile.read(n) if n else b""
            if len(raw) < n:  # (the client closed the connection mid-body)
                self.close_connection = True
                return
            route = POST_ROUTES.get(path)
            anth = path.startswith("/v1/messages")  # (errors in Anthropic's shape)
            if route is None:
                known = path in UNSUPPORTED or path.startswith(("/slots", "/props", "/models/"))
                self._error(501 if known else 404, f"{path} is not supported by the megakernel server", anth)
                return
            try:
                body = json.loads(raw or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None
            if not isinstance(body, dict):
                self._error(400, "the request body must be a JSON object", anth)
                return
            if route in ("tokenize", "detokenize", "apply_template", "input_tokens", "count_tokens"):
                try:  # (no generation: the tokenizer, from this thread; images take the engine's lock)
                    self._json(200, getattr(self, "_" + route)(body))
                except BadRequest as e:
                    self._error(400, str(e), anth)
                return
            srv.count(requests=1, deferred=1)  # (deferred: waiting for the engine)
            with srv.lock:
                srv.count(deferred=-1)
                try:
                    if route == "messages":
                        self._anthropic(body)
                    elif body.get("stream"):
                        self._stream(body)
                    else:
                        self._complete(body)
                except BadRequest as e:
                    srv.count(requests_bad=1)
                    self._error(400, str(e), anth)
                except (BrokenPipeError, ConnectionResetError, ClientGone):
                    srv.count(requests_cancelled=1)
                    self.close_connection = True
                except Exception as e:  # engine failure: report it, keep serving
                    srv.count(requests_failed=1)
                    log("error:", repr(e))
                    try:
                        self._error(500, str(e), anth)
                    except Exception:
                        pass
                finally:
                    srv.eng.keep_warm()  # (MK_KEEP_WARM_S: the window restarts after every request)

        def _error(self, code, message, anthropic=False):
            if anthropic:
                self._json(code, {"type": "error", "error": {"type": ANTHROPIC_ERRORS.get(code, "api_error"), "message": message}})
            else:
                kind = {400: "invalid_request_error", 404: "not_found_error", 501: "not_supported"}.get(code, "server_error")
                self._json(code, {"error": {"message": message, "type": kind}})

        # ---- routes without generation (llama-server's) ----
        def _tokenize(self, body):
            content = body.get("content")
            if body.get("parse_special") is False:
                raise BadRequest("parse_special: false is not supported (special tokens are always parsed)")
            if isinstance(content, str):
                content = [content]
            if not isinstance(content, list) or not all(isinstance(c, (str, int)) and not isinstance(c, bool) for c in content):
                raise BadRequest("content: a string, or an array of strings and token ids")
            # (ids of the tokenizer's own, or ids given in the content: those only within the vocabulary get a piece)
            toks = [(t, isinstance(c, str)) for c in content for t in (srv.eng.tokenize(c) if isinstance(c, str) else [c])]
            if not body.get("with_pieces"):
                return {"tokens": [t for t, _ in toks]}
            out = []
            for t, own in toks:
                p = srv.eng.piece(t) if own or 0 <= t < srv.n_vocab() else b""
                try:
                    out.append({"id": t, "piece": p.decode()})
                except UnicodeDecodeError:
                    out.append({"id": t, "piece": list(p)})
            return {"tokens": out}

        def _detokenize(self, body):
            toks = body.get("tokens") or []
            nv = srv.n_vocab()
            if not nv:
                raise BadRequest("/detokenize needs a newer libqwen38mk (mk_n_vocab)")
            if not isinstance(toks, list) or not all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < nv for t in toks):
                raise BadRequest(f"tokens: an array of token ids in [0, {nv})")
            return {"content": b"".join(srv.eng.piece(t) for t in toks).decode(errors="replace")}

        def _apply_template(self, body):
            check_params(body)
            return {"prompt": srv.render(body)[0]}

        def _input_tokens(self, body):
            check_params(body)
            return {"input_tokens": srv.prompt_tokens(body), "object": "response.input_tokens"}

        def _count_tokens(self, body):
            return {"input_tokens": srv.prompt_tokens(anthropic_to_chat(body))}

        # ---- Anthropic Messages ----
        def _anthropic(self, body):
            chat = anthropic_to_chat(body)
            model = body.get("model") if isinstance(body.get("model"), str) and body.get("model") else srv.model_id
            if chat["stream"]:
                return self._anthropic_stream(chat, model)
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, _ = srv.generate(chat, lambda k, t: parts[k].append(t), self.client_gone)
            blocks = []
            reasoning = "".join(parts["reasoning"]).strip()
            if reasoning:
                blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
            if parts["content"]:
                blocks.append({"type": "text", "text": "".join(parts["content"])})
            blocks += [anthropic_tool_use(c) for c in calls]
            self._json(200, {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": model,
                             "content": blocks, "stop_reason": anthropic_stop_reason(finish, srv.stop_word),
                             "stop_sequence": srv.stop_word, "usage": anthropic_usage(usage)})

        def _anthropic_stream(self, chat, model):
            """Anthropic's SSE: message_start, ping, content blocks (thinking / text deltas as they come, tool_use blocks
            with their whole input as one input_json_delta at the end), message_delta, message_stop"""
            mid = "msg_" + uuid.uuid4().hex
            st = {"started": False, "index": -1, "open": None, "ws": ""}  # ws: the thinking's held trailing whitespace

            def send(data, last=False):
                self.wfile.write(b"%x\r\n" % len(data) + data + (b"\r\n0\r\n\r\n" if last else b"\r\n"))
                self.wfile.flush()

            def event(name, obj, last=False):
                send(f"event: {name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode(), last)

            def start():
                if st["started"]:
                    return
                st["started"] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                n, reused = srv.last_prompt
                event("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None, "usage": {"input_tokens": n - reused, "cache_creation_input_tokens": 0,
                                                     "cache_read_input_tokens": reused, "output_tokens": 0}}})
                event("ping", {"type": "ping"})

            def close():
                if st["open"]:
                    event("content_block_stop", {"type": "content_block_stop", "index": st["index"]})
                    st["open"] = None

            def block(cb):
                close()
                st["index"] += 1
                st["open"] = cb["type"]
                event("content_block_start", {"type": "content_block_start", "index": st["index"], "content_block": cb})

            def delta(d):
                event("content_block_delta", {"type": "content_block_delta", "index": st["index"], "delta": d})

            def emit(kind, text):
                try:
                    start()
                    if kind == "reasoning":  # (trimmed at both ends, as the non-streamed reply's thinking block)
                        t = st["ws"] + text
                        if st["open"] != "thinking":
                            t = t.lstrip()
                            if not t:
                                return True
                            block({"type": "thinking", "thinking": "", "signature": ""})
                        body = t.rstrip()
                        st["ws"] = t[len(body):]
                        if body:
                            delta({"type": "thinking_delta", "thinking": body})
                    else:
                        if st["open"] != "text":
                            block({"type": "text", "text": ""})
                        delta({"type": "text_delta", "text": text})
                except (BrokenPipeError, ConnectionResetError):
                    return False
                return True

            check_params(chat)  # (parameter / render errors: a 400 before the stream starts)
            srv.render(chat)
            try:
                finish, calls, usage, _ = srv.generate(chat, emit, self.client_gone)
            except Exception as e:
                if not st["started"] or isinstance(e, (BrokenPipeError, ConnectionResetError, ClientGone)):
                    raise
                bad = isinstance(e, BadRequest)
                if not bad:
                    log("error:", repr(e))
                event("error", {"type": "error", "error": {"type": "invalid_request_error" if bad else "api_error",
                                                         "message": str(e)}}, last=True)
                return
            start()
            close()
            for c in calls:
                tu = anthropic_tool_use(c)
                block({**tu, "input": {}})
                delta({"type": "input_json_delta", "partial_json": json.dumps(tu["input"], ensure_ascii=False)})
            close()
            event("message_delta", {"type": "message_delta", "delta": {
                "stop_reason": anthropic_stop_reason(finish, srv.stop_word), "stop_sequence": srv.stop_word},
                "usage": anthropic_usage(usage)})
            event("message_stop", {"type": "message_stop"}, last=True)

        def _complete(self, body):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = srv.generate(body, lambda k, t: parts[k].append(t), self.client_gone)
            msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if calls else "")}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": srv.model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage, "timings": timings})

        def _stream(self, body):
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            started = [False]

            def send(data, last=False):
                """one event as a chunk of the body; last: then the end of the body"""
                self.wfile.write(b"%x\r\n" % len(data) + data + (b"\r\n0\r\n\r\n" if last else b"\r\n"))
                self.wfile.flush()

            def chunk(delta, finish=None, usage=None, timings=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_id,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                if timings is not None:
                    obj["timings"] = timings
                send(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

            def head():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    chunk({"role": "assistant", "content": None})

            def emit(kind, text):
                head()
                try:
                    chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})
                except (BrokenPipeError, ConnectionResetError):
                    return False
                return True

            # parameter / render errors are reported as a 400 before the stream starts
            check_params(body)
            srv.render(body)
            try:
                finish, calls, usage, timings = srv.generate(body, emit, self.client_gone)
            except Exception as e:
                if not started[0] or isinstance(e, (BrokenPipeError, ConnectionResetError, ClientGone)):
                    raise  # nothing sent yet: do_POST answers with a status (400 / 500)
                # the status (200) and part of the reply are out: the error is the stream's last event, OpenAI style
                bad = isinstance(e, BadRequest)
                if not bad:
                    log("error:", repr(e))
                err = {"message": str(e), "type": "invalid_request_error" if bad else "server_error", "code": 400 if bad else 500}
                send(b"data: " + json.dumps({"error": err}, ensure_ascii=False).encode() + b"\n\n")
                send(b"data: [DONE]\n\n", last=True)
                return
            head()
            for i, c in enumerate(calls):
                chunk({"tool_calls": [{"index": i, **c}]})
            chunk({}, finish=finish, timings=timings)
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk(None, usage=usage)
            send(b"data: [DONE]\n\n", last=True)

    return H


def make_loading_handler():
    """the handler while the model loads (the port opens first, as llama-server's does): every request gets llama-server's
    503 "Loading model" and the connection closes (a request body is not read: the lingering close drains it)"""
    class Loading(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _loading(self):
            data = json.dumps({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def handle_expect_100(self):
            self._loading()
            return False

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _loading

    return Loading


def linger(sock, seconds=None, idle=None, cap=None):
    """after shutdown(SHUT_WR): reads and drops what the client still sends until it closes, at most `cap` bytes and
    `seconds` (`idle` seconds without data). Closing a socket with unread data makes the kernel send a reset, which can
    reach the client before it has read the answer."""
    end = time.monotonic() + (LINGER_S if seconds is None else seconds)
    left = LINGER_BYTES if cap is None else cap
    idle = LINGER_IDLE_S if idle is None else idle
    buf = bytearray(1 << 16)
    try:
        while left > 0:
            t = end - time.monotonic()
            if t <= 0:
                break
            sock.settimeout(min(t, idle))
            n = sock.recv_into(buf, min(left, len(buf)))
            if not n:
                break
            left -= n
    except OSError:  # (timeout, reset)
        pass


class HTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a lingering close: when the server ends a connection (an early 400 / 411 / 413 for a
    request body it does not read, the base class's errors, Connection: close), it half-closes it (the client reads the
    whole answer and the end of the stream), then drains what the client still sends (linger) before closing, so the
    client is not reset in the middle of its upload and gets the answer (the reason) instead of a connection error."""
    daemon_threads = True

    def shutdown_request(self, request):
        try:
            request.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        else:
            linger(request)
        self.close_request(request)


def env_int(name, default):
    """a positive integer setting from the environment (default when unset or empty)"""
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    if not v.isdecimal() or int(v) < 1:
        sys.exit(f"{name}: expected a positive integer, not {v!r}")
    return int(v)


def main():
    global MAX_BODY_MB, MAX_IMAGES, MAX_VIDEO_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("-c", "--ctx", type=int, default=32768)
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--mtp", default=None,
                    help="the MTP draft layer from another GGUF of the same model (default: the model file's; env MK_MTP)")
    ap.add_argument("--no-mtp", action="store_true", help="no MTP layer: plain decoding, depth 0 (env MK_NO_MTP=1)")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), LIB_MK))
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 cache does not fit)")
    ap.add_argument("--ck-every", type=int, default=4096,
                    help="DeltaNet checkpoint every N prompt tokens (0: off); see MK_CK_SLOTS for the pool size")
    ap.add_argument("--ck-msgs", type=int, default=1, help="DeltaNet checkpoint at each message start (1/0)")
    ap.add_argument("--exact-reuse", action="store_true",
                    help="reused requests answer exactly as fresh ones (MK_EXACT=1): restore a checkpoint and prefill "
                         "instead of appending to decoded rows (costs prompt time when a long reply is reused)")
    ap.add_argument("--chat-template-file", dest="chat_template", action=ChatTemplateArg, metavar="PATH",
                    help="Jinja chat template file to use instead of the GGUF's (as llama-server), e.g. another "
                         "template for a fine-tune of the same model")
    ap.add_argument("--chat-template", dest="chat_template", action=ChatTemplateArg, metavar="TEMPLATE",
                    help="the chat template itself (Jinja source; 'chatml': llama.cpp's ChatML) instead of the GGUF's; "
                         "the last of --chat-template / --chat-template-file wins")
    ap.add_argument("--mmproj", default=None,
                    help="vision projector GGUF (e.g. mmproj-*.gguf): image_url inputs, encoded on the CPU (no VRAM)")
    ap.add_argument("--vision-lib", default=os.path.join(os.path.dirname(HERE), LIB_VISION))
    ap.add_argument("--vision-threads", type=int, default=0,
                    help="CPU threads of the image encoder (0: every logical CPU but two)")
    ap.add_argument("--vision-backend", choices=["auto", "cpu", "mtmd"], default="auto",
                    help="image encoder: cpu (AVX2 / FMA fp32 kernels), mtmd (llama.cpp's), auto (cpu if the CPU has AVX2 + FMA)")
    ap.add_argument("--image-min-tokens", type=int, default=0, help="minimum tokens per image (0: the mmproj's default)")
    ap.add_argument("--image-max-tokens", type=int, default=0,
                    help="maximum tokens per image (0: the mmproj's default, 4096 for Qwen3.x; fewer encode faster)")
    ap.add_argument("--image-cache", type=int, default=8,
                    help="images kept decoded / encoded for reuse (a request holds all of its own while it runs)")
    ap.add_argument("--video-fps", type=float, default=4.0,
                    help="video input (input_video / video_url parts; --mmproj and ffmpeg): frames sampled per second "
                         "(llama.cpp's default 4); every two frames are one grid of tokens")
    ap.add_argument("--video-max-tokens", type=int, default=0,
                    help="maximum tokens per video frame (pair) (0: --image-max-tokens, as llama.cpp; fewer encode faster "
                         "and fit longer clips; cpu vision backend)")
    ap.add_argument("--video-cache", type=int, default=2,
                    help="videos kept decoded / encoded for reuse by the next requests of a conversation")
    ap.add_argument("--ffmpeg", default=None,
                    help="ffmpeg executable (or its directory) for video input (env MK_FFMPEG; default: ffmpeg on PATH, "
                         "else the imageio-ffmpeg package's)")
    args = ap.parse_args()
    if not 0 < args.video_fps <= 30:
        ap.error("--video-fps: expected a rate in (0, 30]")
    if args.video_max_tokens and args.video_max_tokens < max(8, args.image_min_tokens):
        ap.error("--video-max-tokens: expected 0 or at least 8 (and --image-min-tokens)")
    args.chat_template_src = None  # (before the port opens: a bad template fails at once, not after the model load)
    if args.chat_template:
        try:
            args.chat_template_src = chat_template_source(*args.chat_template)
        except ValueError as e:
            ap.error(str(e))
        kind, value = args.chat_template
        log(f"chat template: {value if kind == 'file' else 'from --chat-template'} instead of the GGUF's")
    MAX_BODY_MB = env_int("MK_MAX_BODY_MB", MAX_BODY_MB)
    MAX_IMAGES = env_int("MK_MAX_IMAGES", MAX_IMAGES)
    MAX_VIDEO_FRAMES = env_int("MK_MAX_VIDEO_FRAMES", MAX_VIDEO_FRAMES)
    if args.exact_reuse:
        os.environ["MK_EXACT"] = "1"  # read by the engine at construction
    # the port opens first (a busy port fails before the model loads); until the load is done every request gets a 503
    # (llama-server's "Loading model"), then the handler is swapped for the real one (new connections take it)
    httpd = HTTPServer((args.host, args.port), make_loading_handler())
    log(f"listening on http://{args.host}:{args.port} (503 until the model is loaded)")
    box = {}

    def load():
        try:
            srv = Server(args)
            if srv.eng.exact:
                log("exact reuse: on (MK_EXACT)")
            box["srv"] = srv
            httpd.RequestHandlerClass = make_handler(srv)
        except BaseException as e:  # (the server exits: serve_forever returns)
            box["error"] = e
            traceback.print_exc()
            httpd.shutdown()

    threading.Thread(target=load, name="mk-load", daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if "srv" in box:
            box["srv"].eng.keep_warm(False)
        httpd.server_close()
    if "error" in box:
        sys.exit(f"[mk-server] startup failed: {box['error']}")


if __name__ == "__main__":
    main()
