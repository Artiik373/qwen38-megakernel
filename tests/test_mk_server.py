"""Tests of server/mk_server.py without a GPU: request parsing, prompt reuse and image handling against fake engines.
(Moved from open-jet's tests/test_megakernel_backend.py with the server itself.)"""
import importlib.util
import json
import sys
import unittest
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
        srv.args = mock.Mock(ck_every=0, ck_msgs=1)
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


if __name__ == "__main__":
    unittest.main()
