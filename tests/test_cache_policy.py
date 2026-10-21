"""Tests of server/mk_server.py's host cache policy without a GPU (TODO #19): which requests run as side requests (a
one-shot without tools, whatever its max_tokens), with request shapes the agent harnesses send. Fake engines and the test
server come from test_mk_server.py and test_engine_side.py."""
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


if __name__ == "__main__":
    unittest.main()
