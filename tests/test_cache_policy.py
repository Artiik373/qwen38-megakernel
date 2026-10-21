"""Tests of server/mk_server.py's host cache policy without a GPU: which requests run as side requests (a one-shot
without tools, whatever its max_tokens), which edits of a conversation's history park nothing, and which parked
conversations have low priority (compaction debris), with request shapes the agent harnesses send. Fake engines and the
test server come from test_mk_server.py and test_engine_side.py."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_engine_side as side  # noqa: E402  (its helpers; its test cases are not collected from here)
import test_mk_server as base  # noqa: E402

TOOLS = base.TOOLS


class SideRuleTests(base._HttpCase):
    """S1: with parking on, a request without tools and without an earlier assistant reply is a side request whatever
    its max_tokens; one with tools still needs 0 < max_tokens <= SIDE_REPLY_MAX"""

    LONG = base.HostCacheTests.LONG
    chat, metrics, quiet = base.HostCacheTests.chat, base.HostCacheTests.metrics, base.HostCacheTests.quiet
    serve_store = side.SideConversationTests.serve_store
    follow_up = staticmethod(side.SideConversationTests.follow_up)
    side_ops = staticmethod(side.SideConversationTests.side_ops)

    def run_after_long(self, eng, *requests, **store):
        """a long conversation first, then each request (messages, extra body fields): the log"""
        srv, http_ = self.serve_store(eng, park_mib=4, **store)

        def run():
            self.chat(http_, [{"role": "user", "content": self.LONG}], tools=TOOLS)
            for msgs, kw in requests:
                self.chat(http_, msgs, **kw)
        return srv, self.quiet(run)

    # request shapes of the 1021a harness captures (notes/hpol/survey.md), shortened to the test tokenizer (streamed or
    # not makes no difference to the rule: sent unstreamed here)
    ONE_SHOTS = {
        "goose session name (no max_tokens)": ([{"role": "system", "content": "Generate a short title."},
                                                {"role": "user", "content": "---BEGIN USER MESSAGES--- fix it"}], {}),
        "aider commit message (no max_tokens, non-stream)": ([{"role": "system", "content": "Write a commit message."},
                                                              {"role": "user", "content": "diff: -a +b"}],
                                                             {"temperature": 0}),
        "dsh title (max_completion_tokens 4096)": ([{"role": "developer", "content": "Session title."},
                                                    {"role": "user", "content": "[{\"seq\":8,\"text\":\"hi\"}]"}],
                                                   {"max_completion_tokens": 4096}),
        "crush title (max_tokens 32768)": ([{"role": "system", "content": "Title it."},
                                            {"role": "user", "content": "Generate a title for: hi"}],
                                           {"max_tokens": 32768}),
        "openwebui tags (no max_tokens, one user message)": ([{"role": "user", "content": "### Task: tags for the chat"}],
                                                             {"stream": False}),
        "tool_choice none (tools rendered, no calls)": ([{"role": "user", "content": "a title please"}],
                                                        {"tools": TOOLS, "tool_choice": "none"}),
    }

    def test_one_shots_of_the_harnesses_are_side_requests_whatever_their_max_tokens(self) -> None:
        for name, (msgs, kw) in self.ONE_SHOTS.items():
            eng = side._SideEngine(["a", "b"])
            srv, log = self.run_after_long(eng, (msgs, kw))
            self.assertEqual([o[0] for o in eng.ops], ["side_begin", "park", "side_end_park"], (name, log))
            self.assertIn("side request done: the cached", log, name)
            self.assertNotIn("parked the cached conversation", log, name)

    def test_an_agent_session_first_turn_parks_the_live_conversation(self) -> None:
        for kw in ({"max_tokens": 32000}, {}, {"max_tokens": 4096}):  # (Claude Code, goose / openwebui, a capped agent)
            eng = side._SideEngine(["a", "b"])
            srv, log = self.run_after_long(eng, ([{"role": "user", "content": "new task"}], {"tools": TOOLS, **kw}))
            self.assertEqual([o[0] for o in eng.ops], ["park"], (kw, log))
            self.assertIn("parked the cached conversation (a new conversation)", log)
        eng = side._SideEngine(["a", "b"])  # (with tools and a bounded reply: a side request, as before)
        srv, log = self.run_after_long(eng, ([{"role": "user", "content": "new task"}],
                                             {"tools": TOOLS, "max_tokens": 8}))
        self.assertEqual([o[0] for o in eng.ops], ["side_begin", "park", "side_end_park"], log)

    def test_a_chat_without_tools_continues_beside_the_live_conversation(self) -> None:
        eng = side._SideEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, park_mib=4)
        c1 = [{"role": "user", "content": "hello there"}]
        out = {}

        def run():
            self.chat(http_, [{"role": "user", "content": self.LONG}], tools=TOOLS)
            kept = list(eng.seq)
            r1 = self.chat(http_, c1)  # (turn 1: a side request, its conversation parked as a side entry)
            out["c2"] = self.chat(http_, self.follow_up(c1, r1))  # (turn 2, no max_tokens: continues as a side request)
            self.assertEqual(eng.seq, kept)  # (the long conversation never left the engine)
        log = self.quiet(run)
        self.assertEqual([o[0] for o in eng.ops], ["side_begin", "park", "side_end_park", "side_begin", "side_unpark",
                                                   "park", "side_end_park"], log)
        self.assertGreater(out["c2"]["usage"]["prompt_tokens_details"]["cached_tokens"], 0, log)
        self.assertIn("its parked conversation (", log)

    def test_a_request_without_tools_runs_beside_an_agent_conversation_after_its_first_turn_too(self) -> None:
        c1 = [{"role": "user", "content": "hello there"}]
        c2 = c1 + [{"role": "assistant", "content": "ab"}, {"role": "user", "content": "and more"}]
        # (an engine without side conversations: turn 1 is not kept, so turn 2 has nothing to resume)
        for first, tools, want in (((self.LONG, TOOLS), None, ["side_begin", "side_end"] * 2),  # (beside an agent)
                                   ((self.LONG, None), None, ["side_begin", "side_end", "park"])):  # (beside a chat)
            eng = base._StoreEngine(["a", "b"])
            srv, http_ = self.serve_store(eng, park_mib=4)
            log = self.quiet(lambda: (self.chat(http_, [{"role": "user", "content": first[0]}], tools=first[1]),
                                      self.chat(http_, c1, tools=tools), self.chat(http_, c2, tools=tools)))
            self.assertEqual([o[0] for o in eng.ops], want, (first[1] is not None, log))
        eng = base._StoreEngine(["a", "b"])  # (parking off: as before, a later turn is no side request)
        srv, http_ = self.serve_store(eng, park_mib=0)
        self.quiet(lambda: (self.chat(http_, [{"role": "user", "content": self.LONG}], tools=TOOLS),
                            self.chat(http_, c2)))
        self.assertEqual(eng.ops, [])

    def test_side_requests_off_or_parking_off_give_the_previous_behaviour(self) -> None:
        msgs, kw = self.ONE_SHOTS["goose session name (no max_tokens)"]
        eng = side._SideEngine(["a", "b"])
        srv, log = self.run_after_long(eng, (msgs, kw), side_max=0)  # (--side-max 0: no side requests, a park)
        self.assertEqual([o[0] for o in eng.ops], ["park"], log)
        eng = side._SideEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, park_mib=0)  # (--park-mib 0: every short first turn, as before)
        self.quiet(lambda: (self.chat(http_, [{"role": "user", "content": self.LONG}], tools=TOOLS),
                            self.chat(http_, [{"role": "user", "content": "a new task"}], tools=TOOLS)))
        self.assertEqual(self.side_ops(eng), ["side_begin", "side_end"])

    def test_the_store_room_a_side_request_takes_follows_its_max_tokens(self) -> None:
        room = 100  # (SIDE_REPLY_ROOM for the test: the test context holds 4096 tokens)
        for kw, reply in (({"max_tokens": 40}, 40), ({}, room), ({"max_tokens": 32768}, room)):
            eng = side._SideEngine(["a", "b"])
            srv, http_ = self.serve_store(eng, park_mib=4)
            needs = []
            make_room = srv.make_room
            srv.make_room = lambda need, budget, keep=None, side_only=False: (  # noqa: E731
                needs.append(need), make_room(need, budget, keep, side_only))[1]
            with mock.patch.object(self.mk, "SIDE_REPLY_ROOM", room):
                self.quiet(lambda: (self.chat(http_, [{"role": "user", "content": self.LONG}], tools=TOOLS),
                                    self.chat(http_, [{"role": "user", "content": "a title please"}], **kw)))
            pt, rings, state, ck = eng.kv_sizes()
            prompt = len("<|im_start|>user\na title please<|im_end|>\n<|im_start|>assistant\n")
            self.assertLess(prompt + reply, len(eng.sequence()))  # (the cached sequence is longer: the reply decides)
            self.assertEqual(needs[0], state + rings + pt * (prompt + reply) + 2 * ck, kw)


class EditedHistoryTests(base._HttpCase):
    """S5: a request that repeats the cached conversation up to its first reply and diverges later continues that
    conversation with edited history: nothing is parked; a new conversation (another first message) is parked"""

    chat, quiet = base.HostCacheTests.chat, base.HostCacheTests.quiet
    SYS = {"role": "system", "content": "You are an agent. " * 3}
    CATALOG = " <reminder>memory catalog: " + "entry " * 12 + "</reminder>"  # (Qwen Code's moving block)

    def serve(self, eng, ctx=4096):
        srv, http_ = base._HttpCase.serve(self, eng, ctx)
        srv.im_start = ord("<")  # (the test tokenizer: one token per character)
        srv.asst_marker = tuple(ord(c) for c in "<|im_start|>assistant")
        srv.args.ck_msgs = 1  # (a checkpoint at each message start, as the server's default)
        return srv, http_

    def run_pair(self, first, second, tools=TOOLS):
        """first, then second (each a list of messages; a reply "ab" in between): (ops, log, second's cached tokens)"""
        eng = base._StoreEngine(["a", "b"])
        srv, http_ = base.HostCacheTests.serve_store(self, eng, side_max=0, park_mib=4)
        out = {}
        log = self.quiet(lambda: (self.chat(http_, first, tools=tools),
                                  out.setdefault("r", self.chat(http_, second, tools=tools))))
        return [o[0] for o in eng.ops], log, out["r"]["usage"]["prompt_tokens_details"]["cached_tokens"]

    def turn(self, *texts):
        """messages: the system prompt, then user / assistant texts alternately (tool results as ("tool", text))"""
        msgs = [self.SYS]
        for i, t in enumerate(texts):
            if isinstance(t, tuple):
                msgs.append({"role": t[0], "content": t[1]})
            else:
                msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
        return msgs

    def test_a_block_moved_out_of_a_later_message_parks_nothing(self) -> None:
        first = self.turn("fix the bug", "done, run tests?", "yes please" + self.CATALOG)
        second = self.turn("fix the bug", "done, run tests?", "yes please", "ab", "thanks" + self.CATALOG)
        ops, log, cached = self.run_pair(first, second)
        self.assertEqual(ops, [], log)
        self.assertIn("not parked: the request continues the cached conversation with its history edited", log)
        self.assertNotIn("parked the cached conversation", log)
        n = len("".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in second[:3]))
        self.assertGreaterEqual(cached, n, log)  # (it resumed at the edited message, from the engine's state)

    def test_cleared_tool_results_park_nothing(self) -> None:
        result = "file contents " * 8
        first = self.turn("read a and b", "<tool_call>read a</tool_call>", ("tool", result), "<tool_call>read b",
                          ("tool", result), "both read", "now summarize")
        second = self.turn("read a and b", "<tool_call>read a</tool_call>", ("tool", "[cleared]"),
                           "<tool_call>read b", ("tool", result), "both read", "now summarize", "ab", "and more")
        ops, log, _ = self.run_pair(first, second)
        self.assertEqual(ops, [], log)
        self.assertIn("with its history edited", log)

    def test_a_chat_edit_after_the_first_reply_parks_nothing_an_edit_of_the_first_message_parks(self) -> None:
        first = self.turn("tell me about cats", "cats are mammals " * 4, "and dogs? " * 6)
        ops, log, _ = self.run_pair(first, self.turn("tell me about cats", "cats are mammals " * 4, "and birds?"),
                                    tools=None)
        self.assertEqual(ops, [], log)
        ops, log, _ = self.run_pair(first, self.turn("tell me about whales", "cats are mammals " * 4, "and dogs? " * 6),
                                    tools=None)
        self.assertEqual(ops, ["park"], log)
        self.assertIn("parked the cached conversation (a new conversation)", log)

    def test_a_new_session_of_the_same_harness_still_parks_the_cached_one(self) -> None:
        head = "Working directory context: " + "x" * 60 + "\n"  # (shared by both sessions' first messages)
        first = self.turn(head + "task A", "<tool_call>ls</tool_call>", ("tool", "a.py b.py"), "ok", "next step")
        for second in (self.turn(head + "task B"),  # (its first turn)
                       self.turn(head + "task B", "<tool_call>ls</tool_call>", ("tool", "c.py"), "ok", "go on")):
            ops, log, _ = self.run_pair(first, second)
            self.assertEqual(ops, ["park"], (second, log))
            self.assertNotIn("history edited", log)

    def test_without_the_role_tokens_every_divergence_parks_as_before(self) -> None:
        first = self.turn("fix the bug", "done, run tests?", "yes please" + self.CATALOG)
        second = self.turn("fix the bug", "done, run tests?", "yes please", "ab", "thanks" + self.CATALOG)
        eng = base._StoreEngine(["a", "b"])
        srv, http_ = base.HostCacheTests.serve_store(self, eng, side_max=0, park_mib=4)
        srv.asst_marker = ()  # (a vocabulary without a single <|im_start|> token)
        self.quiet(lambda: (self.chat(http_, first, tools=TOOLS), self.chat(http_, second, tools=TOOLS)))
        self.assertEqual([o[0] for o in eng.ops], ["park"])


class DebrisTests(base._HttpCase):
    """S3: a summary request (no tools) that replaces a conversation with tools parks it with low priority, and the
    summary's own conversation is low too when the compacted conversation replaces it; low entries go before real
    parked conversations"""

    chat, quiet = base.HostCacheTests.chat, base.HostCacheTests.quiet
    LONG = base.HostCacheTests.LONG
    serve = EditedHistoryTests.serve

    def setup(self, side_max=200, park_mib=4, pieces=("a", "b")):
        eng = side._SideEngine(list(pieces))
        srv, http_ = base.HostCacheTests.serve_store(self, eng, side_max=side_max, park_mib=park_mib)
        return eng, srv, http_

    @staticmethod
    def entries(srv):
        """the parked conversations, least recently parked first: (tokens, side, low)"""
        return [(len(e.seq), e.side, e.low) for e in srv.parked.values()]

    def compaction(self, http_, task, summary_msgs, after=""):
        """an agent conversation (tools), its summary request (no tools), the compacted conversation (tools)"""
        self.chat(http_, [{"role": "user", "content": task + self.LONG}], tools=TOOLS)
        r = self.chat(http_, summary_msgs)
        summary = "Summary: " + r["choices"][0]["message"]["content"] + " go on" + after
        self.chat(http_, [{"role": "user", "content": summary}], tools=TOOLS)

    def test_a_summary_and_the_conversation_it_replaced_are_parked_with_low_priority(self) -> None:
        eng, srv, http_ = self.setup()
        summary = "Summarize this conversation for a continuation: " + "the user asked to read files. " * 6
        log = self.quiet(lambda: self.compaction(http_, "A: ", [{"role": "system", "content": "You summarize."},
                                                               {"role": "user", "content": summary}]))
        self.assertIn("(a new conversation, low priority: replaced by a request without tools)", log)
        self.assertIn("(a new conversation, low priority: the conversation of a request without tools that replaced "
                      "one with tools)", log)
        self.assertEqual([low for _, _, low in self.entries(srv)], [True, True], log)
        self.assertEqual(srv.engine_info()["host_cache"]["parked_low"], 2)

    def test_a_summary_with_the_whole_conversation_as_messages_is_debris_too(self) -> None:
        eng, srv, http_ = self.setup()
        msgs = [{"role": "system", "content": "You summarize conversations."},  # (Crush's shape: its own system prompt)
                {"role": "user", "content": "B: " + self.LONG}, {"role": "assistant", "content": "ab"},
                {"role": "user", "content": "Provide a detailed summary of our conversation above."}]
        log = self.quiet(lambda: self.compaction(http_, "B: ", msgs))
        self.assertEqual([low for _, _, low in self.entries(srv)], [True, True], log)

    def test_low_priority_entries_go_before_a_real_parked_conversation(self) -> None:
        eng, srv, http_ = self.setup(park_mib=2)  # (2 MiB: two long conversations, or one and the small debris)
        summary = "Summarize this conversation for a continuation: " + "the user asked to read files. " * 6

        def run():
            self.chat(http_, [{"role": "user", "content": "Z: " + self.LONG}], tools=TOOLS)  # (a real conversation)
            # (parks Z, then A with low priority, then the summary's conversation with low priority)
            self.compaction(http_, "A: ", [{"role": "user", "content": summary}], after=self.LONG)
            self.chat(http_, [{"role": "user", "content": "C: " + self.LONG}], tools=TOOLS)  # (parks the compacted)
        log = self.quiet(run)
        self.assertIn(", low priority) to make room", log)
        z = [o for o in eng.ops if o[0] == "park"][0]  # (Z's park: never dropped)
        self.assertNotIn(("drop", z[1]), eng.ops, log)
        self.assertEqual(self.entries(srv)[0][:2], (z[2], False), log)

    def test_after_a_summary_beside_it_a_replaced_conversation_is_low_a_continued_one_is_not(self) -> None:
        summary = [{"role": "user", "content": "Summarize: " + "the user asked to read files. " * 3}]  # (side request)
        task = [{"role": "user", "content": "A: " + self.LONG}]
        new = [{"role": "user", "content": "Summary: done. go on"}]  # (the compacted conversation)
        long, short = ["w"] * (self.mk.SUMMARY_ANSWER_MIN + 8), ["a", "b"]  # (a summary's answer, a title's)
        for pieces, then, low in ((long, new, True),  # (a summary, then the compacted conversation: low)
                                  (long, None, False),  # (the conversation continues uncompacted, then a new one)
                                  (short, new, False)):  # (a long one-shot with a short answer: not a summary)
            eng, srv, http_ = self.setup(pieces=pieces)

            def run():
                r = self.chat(http_, task, tools=TOOLS)
                self.chat(http_, summary)
                if then is None:
                    self.chat(http_, task + [{"role": "assistant", "content": r["choices"][0]["message"]["content"]},
                                             {"role": "user", "content": "more"}], tools=TOOLS)
                self.chat(http_, then or [{"role": "user", "content": "another task"}], tools=TOOLS)
            log = self.quiet(run)
            self.assertIn("side request done", log)
            real = [e for e in srv.parked.values() if not e.side]
            self.assertEqual([e.low for e in real], [low], (len(pieces), then is None, log))
            self.assertEqual("low priority: replaced right after a summary request beside it" in log, low)

    def test_a_low_entry_that_comes_back_is_parked_normally_again(self) -> None:
        eng, srv, http_ = self.setup()
        summary = "Summarize this conversation for a continuation: " + "the user asked to read files. " * 6

        def run():
            r = self.chat(http_, [{"role": "user", "content": "A: " + self.LONG}], tools=TOOLS)
            self.chat(http_, [{"role": "user", "content": summary}])  # (A parked, low)
            a2 = [{"role": "user", "content": "A: " + self.LONG}, {"role": "assistant", "content": "ab"},
                  {"role": "user", "content": "more"}]
            self.assertEqual(r["choices"][0]["message"]["content"], "ab")
            self.chat(http_, a2, tools=TOOLS)  # (A comes back: it was real after all)
            self.chat(http_, [{"role": "user", "content": "B: " + self.LONG}], tools=TOOLS)  # (A parked again: normal)
        log = self.quiet(run)
        real = [(len(e.seq), e.low) for e in srv.parked.values() if not e.side]
        self.assertEqual(real[-1][1], False, (real, log))
        self.assertIn("restored the parked conversation", log)

    def test_make_room_takes_low_entries_with_the_side_ones(self) -> None:
        eng, srv, _ = self.setup(park_mib=1)
        mk = self.mk
        for pid, kind in ((1, ""), (2, "low"), (3, "side"), (4, "")):
            eng.entries[pid] = ([1] * 300, 0, [], 300000)
            srv.parked[pid] = mk.Parked(pid, [1] * 300, [], 0, [], -1, 300000, side=kind == "side", low=kind == "low")
        self.quiet(lambda: self.assertTrue(srv.make_room(100000, 1 << 20)))
        self.assertEqual(list(srv.parked), [1, 3, 4])  # (low entry 2 first: the oldest of the low tier)
        self.quiet(lambda: self.assertFalse(srv.make_room(700000, 1 << 20, side_only=True)))
        self.assertEqual(list(srv.parked), [1, 4])  # (side entry 3 went; side_only takes nothing else)


if __name__ == "__main__":
    unittest.main()
