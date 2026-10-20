"""API completeness of server/mk_server.py without a GPU (round 1019b, review chunk 3's C3-7 and gaps G1, G2, G4-G7):
tool calls whose values hold tool-call markup, tool-call arguments streamed as they are generated (OpenAI and Anthropic
shapes, the official SDKs' stream accumulators when installed), Anthropic typed tools and unknown blocks, the context
overflow error, reasoning format and template defaults, llama-server's extra request fields, chunked request bodies."""
import http.client
import json
import unittest

from test_mk_server import CALL, TEMPLATE, TOOLS, _HttpCase, _ScriptEngine, _load_mk_server, _request, _sse, _sse_events

WRITE = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "integer"}, "opts": {"type": "object"}}}}}]


def _call(name, **params):
    """Qwen's markup of a call (values as given: strings raw, others as their JSON text)"""
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _pieces(text, n=3):
    return [text[i:i + n] for i in range(0, len(text), n)]


def _accumulate(deltas):
    """an OpenAI stream's tool_calls deltas by index, as the SDKs accumulate them: (calls, the deltas of each index)"""
    calls, per = {}, {}
    for tc in deltas:
        per.setdefault(tc["index"], []).append(tc)
        c = calls.setdefault(tc["index"], {"id": tc.get("id"), "type": tc.get("type"), "function": {"name": "", "arguments": ""}})
        for k in ("name", "arguments"):
            c["function"][k] += tc["function"].get(k) or ""
    return [calls[i] for i in sorted(calls)], per


class ToolCallParserTests(unittest.TestCase):
    """C3-7: a value runs to its </parameter>; the incremental parser's decisions do not depend on the pieces"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def test_a_value_may_hold_tool_call_markup(self) -> None:
        mk = self.mk
        # (review chunk 3's probe input: was no call, the markup returned as text)
        probe = "<tool_call>\n<function=f>\n<parameter=x>\n<tool_call>\n</parameter>\n</function>\n</tool_call>"
        calls = mk.parse_tool_calls(probe, [])
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls], [("f", {"x": "<tool_call>"})])
        self.assertEqual(mk.unparsed_tool_call(probe), "")
        doc = "Calls look like <tool_call>\n<function=read>\n<parameter=path>\n/x\n</parameter>\n</function>\n</tool_call>, see."
        text = _call("write", path="notes.md", content=doc) + "\n" + _call("write", path="b", content="2")
        calls = mk.parse_tool_calls(text, WRITE)
        self.assertEqual([json.loads(c["function"]["arguments"])["content"] for c in calls], [doc.split("\n</parameter>")[0], "2"])
        # an example call inside a value that never closes (cut by max_tokens) is not run
        cut = "<tool_call>\n<function=write>\n<parameter=content>\nlike <tool_call>\n<function=rm>\n</function>\n</tool_call>\n"
        self.assertEqual((mk.parse_tool_calls(cut, WRITE), mk.unparsed_tool_call(cut)), ([], cut))
        # a new <tool_call> between the parameters (not in a value): the first was left open, the second is a call
        text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n" + _call("b", y="2")
        self.assertEqual([c["function"]["name"] for c in mk.parse_tool_calls(text, [])], ["b"])

    def test_values_and_types(self) -> None:
        args = lambda text, tools=WRITE: [json.loads(c["function"]["arguments"]) for c in self.mk.parse_tool_calls(text, tools)]
        self.assertEqual(args(_call("write", path="123", mode=7, opts={"a": [1]})), [{"path": "123", "mode": 7, "opts": {"a": [1]}}])
        self.assertEqual(args(_call("write", mode="seven")), [{"mode": "seven"}])  # (not JSON: the text)
        self.assertEqual(args(_call("other", n="12", s="text", b="true", e="")), [{"n": 12, "s": "text", "b": True, "e": ""}])
        self.assertEqual(args("<tool_call>\n<function=write>\n<parameter=content>\n\na\n\n</parameter>\n</function>\n</tool_call>"),
                         [{"content": "\na\n"}])  # (one newline at each end belongs to the markup)
        self.assertEqual(args("<tool_call><function=write><parameter=path>p</parameter></function></tool_call>"), [{"path": "p"}])
        self.assertEqual(args("<tool_call>\n<function=free>\n</function>\n</tool_call>", []), [{}])

    def _events(self, text, tools, n):
        p = self.mk.ToolCallParser(tools, events=True)
        evs = []
        for piece in _pieces(text, n):
            evs += p.feed(piece)
        evs += p.feed("", final=True)
        return p, evs

    def test_pieces_do_not_matter_and_fragments_add_up(self) -> None:
        texts = [_call("write", path="a\\b \"q\" é", content="x\ny\n" * 5, mode=3) + "\n\n" + _call("free"),
                 "<tool_call>\n<function=write>\n<parameter=content>\n<tool_call></parameter>\n</function>\n</tool_call>",
                 _call("write", path="p") + "\n<tool_call>\n<function=write>\n<parameter=path>\ncut",
                 "<tool_call>\n<function=write>\n<parameter=opts>\n{\"k\": 1}\n</parameter>\n</function>\nx</function>\n</tool_call>"]
        for text in texts:
            whole = self.mk.ToolCallParser(WRITE)
            whole.feed(text, final=True)
            for n in range(1, 9):
                p, evs = self._events(text, WRITE, n)
                self.assertEqual([c["function"] for c in p.calls], [c["function"] for c in whole.calls], (text, n))
                self.assertEqual(p.unparsed(), whole.unparsed(), (text, n))
                frags = {}
                for ev in evs:
                    frags[ev[1]] = frags.get(ev[1], "") + (ev[2] if ev[0] == "args" else "")
                ids = [ev[2] for ev in evs if ev[0] == "start"]
                for i, cid in enumerate(ids):
                    done = [c for c in p.calls if c["id"] == cid]
                    if done:
                        self.assertEqual(frags[i], done[0]["function"]["arguments"], (text, n))
                    else:
                        with self.assertRaises(ValueError):  # (a call left open never has valid JSON)
                            json.loads(frags[i])

    def test_strings_stream_typed_values_come_whole(self) -> None:
        text = _call("write", content="z" * 60, mode=12345)
        _, evs = self._events(text, WRITE, 2)
        frags = [ev[2] for ev in evs if ev[0] == "args"]
        self.assertGreater(sum("z" in f for f in frags), 20)  # (the string as it is generated)
        self.assertIn(', "mode": 12345', frags)  # (the integer whole, at its end)
        self.assertEqual(frags[-1], "}")


class ToolCallStreamTests(_HttpCase):
    """G2: the stream's tool calls as they are generated, == the non-streamed reply's"""

    def _openai(self, pieces, body):
        _, http_ = self.serve(_ScriptEngine(pieces))
        base = {"messages": [{"role": "user", "content": "go"}], "tools": WRITE, "max_tokens": 500, **body}
        status, data = http_.post({**base, "stream": False})
        self.assertEqual(status, 200, data)
        whole = json.loads(data)["choices"][0]
        _, events = _sse(http_.raw(_request({**base, "stream": True}), until=b"0\r\n\r\n"))
        self.assertEqual(events[-1], b"[DONE]")
        deltas = [c for e in events[:-1] for c in json.loads(e)["choices"]]
        tcs = [tc for d in deltas for tc in d["delta"].get("tool_calls") or []]
        content = "".join(d["delta"].get("content") or "" for d in deltas)
        return whole, deltas, tcs, content, [d["finish_reason"] for d in deltas if d["finish_reason"]][-1]

    def test_openai_stream_shape_and_identity(self) -> None:
        text = "Writing.\n\n" + _call("write", path="/tmp/a.txt", content="line \"1\"\nline 2\né中", mode=420) + "\n" + \
            _call("write", path="/tmp/b", opts={"x": [1, 2]})
        whole, deltas, tcs, content, finish = self._openai(_pieces(text), {})
        calls, per = _accumulate(tcs)
        self.assertEqual((finish, whole["finish_reason"], content, whole["message"]["content"]),
                         ("tool_calls", "tool_calls", "Writing.", "Writing."))
        self.assertEqual(len(calls), 2)
        for i, (c, w) in enumerate(zip(calls, whole["message"]["tool_calls"])):
            first = per[i][0]
            self.assertEqual(set(first), {"index", "id", "type", "function"})
            self.assertEqual((first["type"], first["function"]), ("function", {"name": "write", "arguments": ""}))
            self.assertTrue(all(set(d) == {"index", "function"} and set(d["function"]) == {"arguments"} for d in per[i][1:]))
            self.assertGreater(len(per[i]), 4)  # (pieces, not one delta at the end)
            self.assertEqual(c["function"], w["function"])  # (byte for byte, the ids aside)
            self.assertTrue(c["id"].startswith("call_"))

    def test_parallel_tool_calls_false_and_stop_strings(self) -> None:
        text = _call("write", path="a") + "\n" + _call("write", path="b")
        whole, _, tcs, _, finish = self._openai(_pieces(text), {"parallel_tool_calls": False})
        calls, _ = _accumulate(tcs)
        self.assertEqual(([c["function"] for c in calls], finish), ([whole["message"]["tool_calls"][0]["function"]], "tool_calls"))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "a"})
        # a stop string inside a call: the call is left open (finish stop; streamed: arguments never valid JSON)
        whole, _, tcs, content, finish = self._openai(_pieces(_call("write", path="abcSTOPdef")), {"stop": ["STOP"]})
        self.assertEqual((whole["finish_reason"], whole["message"].get("tool_calls")), ("stop", None))
        self.assertTrue(whole["message"]["content"].startswith("<tool_call>"))
        calls, _ = _accumulate(tcs)
        self.assertEqual((finish, content, len(calls)), ("stop", "", 1))
        self.assertEqual(calls[0]["function"]["arguments"], '{"path": "abc')

    def test_markup_inside_a_value_streams_too(self) -> None:
        doc = "a <tool_call> b </function> c"
        whole, _, tcs, content, finish = self._openai(_pieces(_call("write", content=doc)), {})
        calls, _ = _accumulate(tcs)
        self.assertEqual((finish, content), ("tool_calls", ""))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"content": doc})
        self.assertEqual(calls[0]["function"], whole["message"]["tool_calls"][0]["function"])

    def test_anthropic_stream(self) -> None:
        text = "Let me write.\n\n" + _call("write", path="/x", content="q" * 40, mode=1) + "\n" + _call("write", path="/y")
        tools = [{"name": "write", "input_schema": WRITE[0]["function"]["parameters"]}]
        body = {"model": "m", "max_tokens": 500, "messages": [{"role": "user", "content": "go"}], "tools": tools}
        _, http_ = self.serve(_ScriptEngine(_pieces(text)))
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=20)
        c.request("POST", "/v1/messages", json.dumps(body))
        whole = json.loads(c.getresponse().read())
        c.close()
        raw = http_.raw(_request({**body, "stream": True}).replace(b"/v1/chat/completions", b"/v1/messages"), until=b"0\r\n\r\n")
        _, ev = _sse_events(raw)
        starts = [(d["index"], d["content_block"]) for n, d in ev if n == "content_block_start"]
        self.assertEqual([b["type"] for _, b in starts], ["text", "tool_use", "tool_use"])
        inputs = {}
        for n, d in ev:
            if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta":
                inputs.setdefault(d["index"], []).append(d["delta"]["partial_json"])
        self.assertGreater(len(inputs[1]), 4)
        uses = [b for b in whole["content"] if b["type"] == "tool_use"]
        for (i, b), u in zip(starts[1:], uses):
            self.assertEqual((b["name"], b["input"], b["id"][:6]), ("write", {}, "toolu_"))
            self.assertEqual(json.loads("".join(inputs[i])), u["input"])
        self.assertEqual(ev[-2][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual([n for n, _ in ev].count("content_block_stop"), 3)


class AnthropicBlockTests(_HttpCase):
    """G1: Anthropic typed tools become functions, unknown content blocks are skipped (as llama-server), documents with
    a text source become text; each kind is a one-time console warning"""

    def test_typed_tools(self) -> None:
        notes = []
        tools = [{"name": "read", "description": "reads", "input_schema": {"type": "object", "properties": {}}},
                 {"type": "custom", "name": "c", "input_schema": {"type": "object"}},
                 {"type": "bash_20250124", "name": "bash"}, {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                 {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        out = self.mk.anthropic_to_chat({"messages": [], "tools": tools}, notes)
        self.assertEqual([t["function"] for t in out["tools"]], [
            {"name": "read", "description": "reads", "parameters": {"type": "object", "properties": {}}},
            {"name": "c", "description": "", "parameters": {"type": "object"}},
            {"name": "bash", "description": "", "parameters": {}},
            {"name": "str_replace_based_edit_tool", "description": "", "parameters": {}},
            {"name": "web_search", "description": "", "parameters": {}}])
        self.assertEqual([k for k, _ in notes], ["anthropic-tool-bash_20250124", "anthropic-tool-text_editor_20250728",
                                                 "anthropic-tool-web_search_20250305"])
        for bad in ([{"type": "bash_20250124"}], ["bash"], [{"name": ""}]):
            with self.assertRaises(self.mk.BadRequest):
                self.mk.anthropic_to_chat({"messages": [], "tools": bad})

    def test_blocks(self) -> None:
        notes = []
        content = [{"type": "text", "text": "Summarize."},
                   {"type": "document", "title": "notes.txt", "source": {"type": "text", "media_type": "text/plain", "data": "A B"}},
                   {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "one"},
                                                                              {"type": "text", "text": "two"}]},
                    "context": "ctx"},
                   {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
                   {"type": "search_result", "source": "https://x", "title": "t", "content": [{"type": "text", "text": "s"}]},
                   {"type": "container_upload", "file_id": "f"}]
        msgs = [{"role": "user", "content": content},
                {"role": "assistant", "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}},
                    {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
                    {"type": "text", "text": "Found it."},
                    {"type": "tool_use", "id": "toolu_9", "name": "read", "input": {"p": 1}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": [
                    {"type": "text", "text": "r1"}, {"type": "document", "source": {"type": "text", "data": "d1"}},
                    {"type": "search_result", "source": "u", "title": "t", "content": []}]}]}]
        m = self.mk.anthropic_to_chat({"messages": msgs}, notes)["messages"]
        self.assertEqual(m[0], {"role": "user", "content": [{"type": "text", "text": "Summarize."},
                                                             {"type": "text", "text": "notes.txt\n\nA B"},
                                                             {"type": "text", "text": "ctx\n\none\ntwo"}]})
        self.assertEqual((m[1]["content"], [c["function"]["name"] for c in m[1]["tool_calls"]]), ("Found it.", ["read"]))
        self.assertEqual(m[2], {"role": "tool", "tool_call_id": "toolu_9", "content": "r1\nd1"})
        self.assertEqual(sorted({k for k, _ in notes}), [
            "anthropic-block-container_upload", "anthropic-block-search_result", "anthropic-block-server_tool_use",
            "anthropic-block-web_search_tool_result", "anthropic-doc-base64"])

    def test_system_messages_inside_the_conversation(self) -> None:
        # (Claude Code 2.1 sends its reminders as a system message after the first user message: was a 400)
        body = {"system": [{"type": "text", "text": "S"}], "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "system", "content": [{"type": "text", "text": "reminder", "cache_control": {"type": "ephemeral"}}]},
            {"role": "system", "content": "plain"}]}
        m = self.mk.anthropic_to_chat(body)["messages"]
        self.assertEqual([x["role"] for x in m], ["system", "user", "system", "system"])
        self.assertEqual((m[2]["content"], m[3]["content"]), ("reminder", "plain"))
        notes = []
        m = self.mk.anthropic_to_chat(body, notes, mid_system=False)["messages"]  # (Qwen's templates)
        self.assertEqual([x["role"] for x in m], ["system", "user", "user", "user"])
        self.assertEqual({k for k, _ in notes}, {"anthropic-mid-system"})
        # the server asks its template once
        strict = ("{% for m in messages %}{% if m.role == 'system' and not loop.first %}"
                  "{{ raise_exception('System message must be at the beginning.') }}{% endif %}"
                  "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                  "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.assertFalse(self.mk.takes_mid_system(self.mk.make_template(strict)))
        self.assertTrue(self.mk.takes_mid_system(self.mk.make_template(TEMPLATE)))
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        srv.template = self.mk.make_template(strict)
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("POST", "/v1/messages", json.dumps({**body, "max_tokens": 5}))
        r = c.getresponse()
        self.assertEqual((r.status, json.loads(r.read())["stop_reason"]), (200, "end_turn"))
        c.close()
        self.assertEqual((srv.mid_system, "anthropic-mid-system" in srv.warned), (False, True))

    def test_http(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["Hello", "."]))
        body = {"model": "m", "max_tokens": 20, "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                                          {"type": "redacted_thinking", "data": "x"},
                                                          {"type": "mcp_tool_use", "id": "m", "name": "n", "input": {}}]}]}
        for path in ("/v1/messages", "/v1/messages", "/v1/messages/count_tokens"):
            c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
            c.request("POST", path, json.dumps(body))
            r = c.getresponse()
            o = json.loads(r.read())
            c.close()
            self.assertEqual(r.status, 200, o)
        self.assertEqual(o, {"input_tokens": len(srv.render(self.mk.anthropic_to_chat(body))[0])})
        self.assertTrue({"anthropic-tool-web_fetch_20250910", "anthropic-block-mcp_tool_use"} <= srv.warned)


class SdkStreamTests(_HttpCase):
    """the official SDKs' stream accumulators rebuild the streamed calls (skipped when the SDK is not installed)"""

    TEXT = "Writing.\n\n" + _call("write", path="/tmp/a", content="x \"y\"\n" * 8, mode=7) + "\n" + _call("write", path="/b")

    def test_openai_sdk(self) -> None:
        try:
            import openai
        except ImportError:
            self.skipTest("openai is not installed")
        _, http_ = self.serve(_ScriptEngine(_pieces(self.TEXT)))
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{http_.port}/v1", api_key="x", max_retries=0)
        self.addCleanup(client.close)
        kw = dict(model="m", messages=[{"role": "user", "content": "go"}], tools=WRITE)
        whole = client.chat.completions.create(**kw)
        stream = getattr(client.chat.completions, "stream", None) or client.beta.chat.completions.stream
        with stream(**kw) as s:
            final = s.get_final_completion()
        a, b = whole.choices[0], final.choices[0]
        self.assertEqual((b.finish_reason, b.message.content), (a.finish_reason, a.message.content))
        self.assertEqual([(t.function.name, t.function.arguments) for t in b.message.tool_calls],
                         [(t.function.name, t.function.arguments) for t in a.message.tool_calls])
        self.assertEqual(json.loads(b.message.tool_calls[0].function.arguments)["mode"], 7)

    def test_anthropic_sdk(self) -> None:
        try:
            import anthropic
        except ImportError:
            self.skipTest("anthropic is not installed")
        _, http_ = self.serve(_ScriptEngine(_pieces(self.TEXT)))
        client = anthropic.Anthropic(base_url=f"http://127.0.0.1:{http_.port}", api_key="x", max_retries=0)
        self.addCleanup(client.close)
        kw = dict(model="m", max_tokens=500, messages=[{"role": "user", "content": "go"}],
                  tools=[{"name": "write", "input_schema": WRITE[0]["function"]["parameters"]}])
        whole = client.messages.create(**kw)
        with client.messages.stream(**kw) as s:
            final = s.get_final_message()
        self.assertEqual(final.stop_reason, whole.stop_reason)
        self.assertEqual([b.type for b in final.content], [b.type for b in whole.content])
        self.assertEqual([b.input for b in final.content if b.type == "tool_use"],
                         [b.input for b in whole.content if b.type == "tool_use"])


if __name__ == "__main__":
    unittest.main()
