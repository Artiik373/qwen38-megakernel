"""Tests of server/mk_server.py without a GPU: request parsing, prompt reuse and image handling against fake engines.
(Moved from open-jet's tests/test_megakernel_backend.py with the server itself.)"""
import base64
import http.client
import importlib.util
import json
import socket
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def _load_mk_server():
    path = ROOT / "server" / "mk_server.py"
    spec = importlib.util.spec_from_file_location("mk_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MegakernelServerParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _feed_all(self, parser, chunks):
        out = []
        for c in chunks:
            out += parser.feed(c)
        out += parser.feed("", final=True)
        merged = {"reasoning": "", "content": ""}
        for kind, text in out:
            merged[kind] += text
        return merged

    def test_reasoning_and_content_are_split_across_chunk_boundaries(self) -> None:
        parser = self.mk.OutputParser(thinking=True, tools=[])
        text = "Think about it.\n</think>\n\nThe answer is 4."
        merged = self._feed_all(parser, [text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual(merged["reasoning"], "Think about it.\n")
        self.assertEqual(merged["content"], "The answer is 4.")

    def test_tool_call_markup_is_held_back_and_parsed(self) -> None:
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}}}}}]
        parser = self.mk.OutputParser(thinking=False, tools=tools)
        text = ("Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n"
                "<parameter=max_lines>\n10\n</parameter>\n</function>\n</tool_call>")
        merged = self._feed_all(parser, [text[i:i + 5] for i in range(0, len(text), 5)])
        self.assertEqual(merged["content"], "Let me look.")
        calls = self.mk.parse_tool_calls(parser.tool_text, tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/etc/hostname", "max_lines": 10})

    def test_string_parameters_are_not_json_decoded(self) -> None:
        tools = [{"type": "function", "function": {"name": "run", "parameters": {
            "type": "object", "properties": {"cmd": {"type": "string"}}}}}]
        calls = self.mk.parse_tool_calls("<tool_call>\n<function=run>\n<parameter=cmd>\n123\n</parameter>\n</function>\n</tool_call>", tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"cmd": "123"})

    def test_assistant_tool_call_arguments_are_decoded_for_the_template(self) -> None:
        msgs = self.mk.normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}])
        self.assertEqual(msgs[0]["content"], "")
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})

    def test_images_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self.mk.normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])



class _FakeEngine:
    """MKEngine's sequence / checkpoint semantics, without a GPU"""

    exact = False  # (an MK_EXACT engine never appends; see Server.load_prompt)

    def __init__(self):
        self.seq, self.dn_c, self.cks, self.calls = [], 0, [], []

    def sequence(self):
        return list(self.seq)

    def committed(self):
        return self.dn_c

    def checkpoints(self):
        return sorted(self.cks)

    def truncate(self, n):
        self.seq = self.seq[:n]

    def reset(self):
        self.seq, self.dn_c, self.cks = [], 0, []
        self.calls.append(("reset",))

    def rewind_to(self, pos):
        if pos not in self.cks:
            return False
        self.seq, self.dn_c = self.seq[:pos], pos
        self.cks = [c for c in self.cks if c <= pos]
        self.calls.append(("rewind", pos))
        return True

    def prefill(self, toks, ck_at=-1, cks=()):
        n0 = len(self.seq)
        self.seq += toks
        self.cks += [c for c in [*cks, ck_at] if n0 < c < len(self.seq)]
        self.dn_c = len(self.seq)
        self.calls.append(("prefill", n0, len(toks), ck_at, tuple(cks)))
        self.seq.append(7)  # the first generated token
        return 7


class MegakernelPromptReuseTests(unittest.TestCase):
    IM = 1  # <|im_start|>

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _server(self, every=8):
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.im_start = _FakeEngine(), self.IM
        srv.args = mock.Mock(ck_every=every, ck_msgs=1)
        return srv

    def test_checkpoints_at_message_starts_the_generation_prompt_and_every_n_tokens(self) -> None:
        srv = self._server()
        toks = [self.IM, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, self.IM, 5, 5, self.IM, 9, 9]
        self.assertEqual(srv.load_prompt(toks), (7, 0))
        self.assertEqual(srv.eng.calls, [("reset",), ("prefill", 0, len(toks), 15, (8, 12, 15, 16))])

    def test_a_divergence_rewinds_to_the_last_checkpoint_before_it(self) -> None:
        srv = self._server()
        first = [self.IM, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, self.IM, 5, 5, self.IM, 9, 9]
        srv.load_prompt(first)
        srv.eng.dn_c = len(srv.eng.seq) + 5  # decoding went on
        second = first[:14] + [6, 6, self.IM, 9, 9]  # diverges at 14: checkpoints 8, 12 are before it
        self.assertEqual(srv.load_prompt(second), (7, 12))
        self.assertEqual(srv.eng.calls[-2:], [("rewind", 12), ("prefill", 12, len(second) - 12, 16, (16,))])

    def test_an_extension_of_the_sequence_is_appended_without_a_rewind(self) -> None:
        srv = self._server(every=0)
        first = [self.IM, 5, 5, self.IM, 9]
        srv.load_prompt(first)
        second = srv.eng.sequence() + [self.IM, 6, self.IM, 9]
        self.assertEqual(srv.load_prompt(second), (7, len(first) + 1))
        self.assertEqual(srv.eng.calls[-1], ("prefill", len(first) + 1, 4, len(second) - 2, (len(second) - 2,)))

    def test_no_checkpoint_before_the_divergence_means_a_full_prefill(self) -> None:
        srv = self._server(every=0)
        srv.load_prompt([self.IM, 5, 5, self.IM, 9])
        srv.eng.dn_c += 3
        self.assertEqual(srv.load_prompt([self.IM, 6, 6, self.IM, 9]), (7, 0))
        self.assertEqual(srv.eng.calls[-2][0], "reset")


class _FakeSamplingEngine:
    """records what Server.sampling passes to the engine"""

    def __init__(self):
        self.sampling = self.penalty = None

    def set_penalty(self, presence, frequency):
        self.penalty = (presence, frequency)

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        self.sampling = (temp, top_p, top_k, min_p, seed)


class MegakernelSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _top_k(self, body):
        """(the top_k the engine gets, the warnings given) for a sampled request with these fields"""
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.warned = _FakeSamplingEngine(), set()
        srv.sampling({"temperature": 0.7, "seed": 1, **body})
        return srv.eng.sampling[2], srv.warned

    def test_top_k_absent_or_off_is_the_engine_maximum_without_a_warning(self) -> None:
        for body in ({}, {"top_k": None}, {"top_k": 0}, {"top_k": -1}):
            self.assertEqual(self._top_k(body), (self.mk.TOPK_MAX, set()), body)

    def test_top_k_in_range_is_kept(self) -> None:
        self.assertEqual(self._top_k({"top_k": 20}), (20, set()))
        self.assertEqual(self._top_k({"top_k": self.mk.TOPK_MAX}), (self.mk.TOPK_MAX, set()))

    def test_top_k_above_the_maximum_is_clamped_with_a_warning(self) -> None:
        self.assertEqual(self._top_k({"top_k": 64}), (self.mk.TOPK_MAX, {"top_k"}))


class _FakeVisionEngine(_FakeEngine):
    """_FakeEngine with mk_prefill_img's images argument and a character tokenizer"""

    def tokenize(self, text):
        return [ord(c) for c in text]

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        self.calls.append(("images", tuple((p, nx, ny) for p, nx, ny, _ in images)))
        return super().prefill(toks, ck_at, cks)


class _FakeImage:
    def __init__(self, key, nx, ny):
        self.key, self.nx, self.ny, self.n = key, nx, ny, nx * ny
        self.encoded = 0

    def rows(self):
        self.encoded += 1
        return 0  # (an address)


class MegakernelVisionTests(unittest.TestCase):
    VS, VE, PAD, IM = 1001, 1002, 1003, 1

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _media(self, content):
        media = []
        msgs = self.mk.normalize_messages([{"role": "user", "content": content}], media)
        return msgs[0]["content"], media

    def test_image_parts_are_flattened_as_llama_server_does(self) -> None:
        url = {"url": "data:image/png;base64,iVBORw0KGgo="}
        img = {"type": "image_url", "image_url": url}
        m = self.mk.MEDIA_MARKER
        self.assertEqual(self._media([{"type": "text", "text": "a"}, img, {"type": "text", "text": "b"}])[0], f"a{m}b")
        self.assertEqual(self._media([img, {"type": "text", "text": "b"}])[0], f"{m}b")
        self.assertEqual(self._media([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, img])[0], f"a\nb{m}")
        text, media = self._media([img, img])
        self.assertEqual((text, len(media)), (m + m, 2))
        self.assertEqual(media[0][:4], bytes([0x89, 0x50, 0x4E, 0x47]))  # (the decoded data: URL)
        # a message without images keeps its parts, as before
        self.assertIsInstance(self._media([{"type": "text", "text": "a"}])[0], list)

    def test_remote_image_urls_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self._media([{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}])

    def _server(self, images):
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.im_start = _FakeVisionEngine(), self.IM
        srv.args = mock.Mock(ck_every=0, ck_msgs=1, ctx=4096)
        srv.vis_start, srv.vis_end, srv.img_pad = self.VS, self.VE, self.PAD
        srv.vision = mock.Mock()
        srv.vision.image.side_effect = lambda data: images[data]
        return srv

    def test_each_image_becomes_its_rows_between_vision_start_and_end(self) -> None:
        im = _FakeImage("k", 3, 2)
        srv = self._server({b"png": im})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"ab{m}c", [b"png"])
        self.assertEqual(toks, [ord("a"), ord("b"), self.VS] + [self.PAD] * 6 + [self.VE, ord("c")])
        self.assertEqual(images, [(3, im)])

    def test_a_different_image_at_the_same_place_is_not_reused(self) -> None:
        a, b = _FakeImage("a", 2, 2), _FakeImage("b", 2, 2)
        srv = self._server({b"A": a, b"B": b})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"x{m}yz", [b"A"])
        srv.load_prompt(toks, images)
        self.assertEqual(srv.eng.calls[-2], ("images", ((2, 2, 2),)))
        toks2, images2 = srv.tokenize_prompt(f"x{m}yz", [b"B"])
        self.assertEqual(toks2, toks)  # (the placeholders are alike)
        _, reused = srv.load_prompt(toks2, images2)
        self.assertEqual(reused, 0)  # (no checkpoint before the image: a full prefill)
        self.assertEqual(b.encoded, 1)

    def test_the_same_image_is_reused_and_a_rewind_into_it_feeds_it_again(self) -> None:
        a = _FakeImage("a", 4, 4)
        srv = self._server({b"A": a})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"x{m}yz", [b"A"])
        srv.load_prompt(toks, images)
        srv.eng.cks.append(10)  # a checkpoint inside the image (positions 2 .. 17)
        srv.eng.dn_c = len(srv.eng.seq) + 3
        toks2, images2 = srv.tokenize_prompt(f"x{m}yw", [b"A"])
        _, reused = srv.load_prompt(toks2, images2)
        self.assertEqual(reused, 10)
        self.assertEqual(srv.eng.calls[-3:-1], [("rewind", 10), ("images", ((2, 4, 4),))])


# === sec:harness ===
# The HTTP handler end to end (make_handler on a local port) over a scripted engine: no GPU, no model.

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


class _ScriptEngine(_FakeEngine):
    """generates a fixed text: one token per piece (the first from the prefill, two per decode cycle), then an
    end-of-generation token (eog=False: filler pieces forever); delay: seconds per decode cycle; fail_at: the decode
    cycle that raises, as the engine does on an error"""

    EOG, FILL = 900000, 900001

    def __init__(self, pieces, eog=True, delay=0.0, fail_at=None):
        super().__init__()
        self.pieces, self.eog, self.delay, self.fail_at = [p.encode() for p in pieces], eog, delay, fail_at
        self.i = self.cycles = 0

    def tokenize(self, text):
        return [ord(c) for c in text]

    def set_penalty(self, *a):
        pass

    def set_sampling(self, *a):
        pass

    def keep_warm(self, on=True):
        return 0.0

    def _next(self):
        self.i += 1
        if self.i <= len(self.pieces):
            return 1000 + self.i - 1
        return self.EOG if self.eog else self.FILL

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        super().prefill(toks, ck_at, cks)
        self.i = 0  # (each request replays the script)
        return self._next()

    def step(self):
        self.cycles += 1
        if self.cycles == self.fail_at:
            raise RuntimeError("decode: no finite logit (test)")
        time.sleep(self.delay)
        return [self._next(), self._next()]

    def is_eog(self, tok):
        return tok == self.EOG

    def piece(self, tok):
        return b" x" if tok == self.FILL else self.pieces[tok - 1000]


def _chat_server(mk, eng, ctx=4096):
    srv = object.__new__(mk.Server)
    srv.eng, srv.im_start, srv.vision = eng, None, None
    srv.template = mk.make_template(TEMPLATE)
    srv.args = mock.Mock(ctx=ctx, ck_every=0, ck_msgs=0)
    srv.lock, srv.warned, srv.model_id = threading.Lock(), set(), "test"
    return srv


class _Http:
    """make_handler(srv) on 127.0.0.1:<free port>"""

    def __init__(self, mk, srv):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), mk.make_handler(srv))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, timeout=20):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def raw(self, data, timeout=3.0, until=None):
        """sends raw bytes; returns what comes back until the server closes, `until` ends the data, or the timeout"""
        with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as s:
            s.sendall(data)
            out = b""
            try:
                while not (until and out.endswith(until)):
                    b = s.recv(65536)
                    if not b:
                        break
                    out += b
            except (socket.timeout, ConnectionResetError):  # (a reset after the answer: an unread request body)
                pass
            return out


def _request(body, headers=None):
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Content-Length": str(len(data)), **(headers or {})}
    return (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n" + b"".join(f"{k}: {v}\r\n".encode() for k, v in h.items())
            + b"\r\n" + data)


def _sse(raw):
    """the data: payloads of a chunked SSE response, decoded strictly (a status line inside the body fails here)"""
    head, _, body = raw.partition(b"\r\n\r\n")
    events, pos = [], 0
    while True:
        eol = body.index(b"\r\n", pos)
        n = int(body[pos:eol], 16)
        if n == 0:
            return head, events
        chunk = body[eol + 2:eol + 2 + n]
        assert body[eol + 2 + n:eol + 4 + n] == b"\r\n"
        events += [ln[6:] for ln in chunk.split(b"\n") if ln.startswith(b"data: ")]
        pos = eol + 4 + n


class _HttpCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def serve(self, eng, ctx=4096):
        srv = _chat_server(self.mk, eng, ctx)
        http_ = _Http(self.mk, srv)
        self.addCleanup(http_.close)
        return srv, http_


# === sec:b3 ===
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}}]
CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n</function>\n</tool_call>"


class ToolCallCutTests(_HttpCase):
    """a tool call counts only with its closing tags: one cut by max_tokens is text and the reply ends with 'length'"""

    def test_only_a_closed_call_is_parsed(self) -> None:
        self.assertEqual(len(self.mk.parse_tool_calls(CALL, TOOLS)), 1)
        for n in range(len(CALL)):
            self.assertEqual(self.mk.parse_tool_calls(CALL[:n], TOOLS), [], CALL[:n])
            self.assertEqual(self.mk.unparsed_tool_call(CALL[:n]), CALL[CALL.find("<tool_call>"):n] if n >= 11 else "")
        self.assertEqual(self.mk.unparsed_tool_call(CALL + "\n" + CALL[:30]), CALL[:30])

    PIECES = (lambda t: [t[i:i + 3] for i in range(0, len(t), 3)])("Let me look.\n\n" + CALL)

    def _reply(self, max_tokens, stream):
        pieces = self.PIECES
        _, http_ = self.serve(_ScriptEngine(pieces))
        body = {"messages": [{"role": "user", "content": "hostname?"}], "tools": TOOLS, "max_tokens": max_tokens,
                "stream": stream}
        if not stream:
            status, data = http_.post(body)
            self.assertEqual(status, 200)
            o = json.loads(data)
            c = o["choices"][0]
            return pieces, c["finish_reason"], c["message"].get("content"), c["message"].get("tool_calls") or []
        _, events = _sse(http_.raw(_request(body), until=b"0\r\n\r\n"))
        self.assertEqual(events[-1], b"[DONE]")
        deltas = [c for e in events[:-1] for c in json.loads(e)["choices"]]
        content = "".join(d["delta"].get("content") or "" for d in deltas)
        calls = [tc for d in deltas for tc in d["delta"].get("tool_calls") or []]
        return pieces, [d["finish_reason"] for d in deltas if d["finish_reason"]][-1], content, calls

    def test_a_complete_call_is_a_tool_call(self) -> None:
        for stream in (False, True):
            _, finish, content, calls = self._reply(500, stream)
            self.assertEqual((finish, content, len(calls)), ("tool_calls", "Let me look.", 1), stream)
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/etc/hostname"})

    def test_a_call_cut_by_max_tokens_is_text_and_length(self) -> None:
        n = len(self.PIECES)
        for cut in (1, 2, 4, 12):  # inside </tool_call>, </function>, the parameter
            for stream in (False, True):
                pieces, finish, content, calls = self._reply(n - cut, stream)
                self.assertEqual((finish, calls), ("length", []), (cut, stream))
                self.assertEqual(content, "".join(pieces[:n - cut]), (cut, stream))  # (all of the generated text)


# === sec:b1 ===
class _FakeVisionLib:
    """libqwen38vision's C API over Python objects; using a freed image is recorded (the real library crashes)"""

    def __init__(self):
        self.live, self.freed, self.misuse, self.next = {}, [], [], 1

    def mkv_image_load(self, ctx, data, n):
        if data.startswith(b"bad"):
            return None
        h, self.next = self.next, self.next + 1
        self.live[h] = data
        return h

    def mkv_last_error(self):
        return b"cannot decode the image"

    def mkv_image_grid(self, h, nx, ny):
        nx._obj.value, ny._obj.value = 2, 2
        return 4

    def mkv_image_encode(self, ctx, h):
        if h not in self.live:
            self.misuse.append(("encode", h))
            return -1
        return 0

    def mkv_image_encode_ms(self, h):
        return 1.0

    def mkv_image_rows(self, h):
        if h not in self.live:
            self.misuse.append(("rows", h))
        return 4096 + 16 * (h or 0)

    def mkv_image_free(self, h):
        self.freed.append(self.live.pop(h))


class ImagePinTests(_HttpCase):
    """a request owns every image it references until it is done; at most MK_MAX_IMAGES images per request"""

    def serve_vision(self, cache=8):
        eng = _ScriptEngine(["ok"])
        srv, http_ = self.serve(eng)
        lib = _FakeVisionLib()
        v = object.__new__(self.mk.Vision)
        v.L, v.ctx, v.cache, v.cache_max = lib, 1, {}, cache
        srv.vision, srv.vis_start, srv.vis_end, srv.img_pad = v, 1001, 1002, 1003
        return v, lib, http_

    def _images(self, datas):
        parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(d).decode()}}
                 for d in datas]
        return {"messages": [{"role": "user", "content": [{"type": "text", "text": "what?"}] + parts}], "max_tokens": 4}

    def test_more_images_than_the_cache_in_one_request(self) -> None:
        v, lib, http_ = self.serve_vision(cache=8)
        datas = [b"img%d" % i for i in range(9)]
        status, data = http_.post(self._images(datas))
        self.assertEqual((status, lib.misuse, lib.freed), (200, [], [b"img0"]), data[:300])  # (freed after the request)
        self.assertEqual(len(v.cache), 8)
        self.assertTrue(all(im.pins == 0 for im in v.cache.values()))
        status, _ = http_.post(self._images(datas))  # again: 8 cached, the first decoded again
        self.assertEqual((status, lib.misuse, len(v.cache)), (200, [], 8))

    def test_the_same_image_twice_in_a_request(self) -> None:
        v, lib, http_ = self.serve_vision(cache=1)
        status, _ = http_.post(self._images([b"a", b"b", b"a"]))
        self.assertEqual((status, lib.misuse, lib.freed, list(v.cache)), (200, [], [b"b"], list(v.cache)))
        self.assertEqual(len(v.cache), 1)

    def test_a_failed_request_releases_its_images(self) -> None:
        v, lib, http_ = self.serve_vision(cache=2)
        status, data = http_.post(self._images([b"a", b"b", b"c", b"bad"]))
        self.assertEqual(status, 400)
        self.assertIn(b"cannot decode the image", data)
        self.assertEqual((len(v.cache), lib.freed, lib.misuse), (2, [b"a"], []))
        self.assertTrue(all(im.pins == 0 for im in v.cache.values()))

    def test_too_many_images_is_a_400(self) -> None:
        v, lib, http_ = self.serve_vision()
        with mock.patch.object(self.mk, "MAX_IMAGES", 4):
            status, data = http_.post(self._images([b"i%d" % i for i in range(5)]))
            self.assertEqual(status, 400)
            self.assertIn(b"MK_MAX_IMAGES", data)
            self.assertEqual(lib.live, {})  # (refused before decoding any)
            self.assertEqual(http_.post(self._images([b"i%d" % i for i in range(4)]))[0], 200)


if __name__ == "__main__":
    unittest.main()
