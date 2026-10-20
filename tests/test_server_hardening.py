"""Tests of server/mk_server.py's hardening without a GPU: the request gate (Host / Origin checks, CORS), special-token
text in message content, video frame sizes. Fake engines and the test server come from test_mk_server.py."""
import email.message
import http.client
import json
import os
import socket
import sys
import threading
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mk_server as base  # noqa: E402  (its helpers; its test cases are not collected from here)


def _headers(**kw):
    """an http.client-style header block: keyword names with _ as -, a list value = repeated header"""
    h = email.message.Message()
    for k, v in kw.items():
        for one in v if isinstance(v, list) else [v]:
            h[k.replace("_", "-")] = one
    return h


class _GateCase(base._HttpCase):
    def gated(self, eng=None, **gate):
        """the chat test server behind a Gate(**gate)"""
        srv, http_ = self.serve(eng or base._ScriptEngine(["ok"]))
        http_.httpd.gate = self.mk.Gate(**gate)
        return srv, http_

    @staticmethod
    def ask(http_, method, path, headers=(), body=None):
        """one request with these headers (and Connection: close; no Host unless given) to the server http_ (.port);
        (status, headers, body)"""
        lines = [f"{method} {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in headers] + ["Connection: close"]
        if body is not None:
            lines.append(f"Content-Length: {len(body)}")
        out = b""
        with socket.create_connection(("127.0.0.1", http_.port), timeout=10) as s:
            s.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + (body or b""))
            try:
                while True:
                    b = s.recv(65536)
                    if not b:
                        break
                    out += b
            except (socket.timeout, ConnectionResetError):
                pass
        head, _, rest = out.partition(b"\r\n\r\n")
        status_line, *hl = head.decode("latin-1").split("\r\n")
        hdrs = _headers()
        for ln in hl:
            k, _, v = ln.partition(":")
            hdrs[k.strip()] = v.strip()
        return int(status_line.split()[1]), hdrs, rest


class GateUnitTests(unittest.TestCase):
    """Gate.check per branch: (None, cors origin) to answer, (message, None) to refuse"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = base._load_mk_server()

    def ok(self, gate, **h):
        why, cors = gate.check(_headers(**h))
        self.assertIsNone(why, h)
        return cors

    def refused(self, gate, **h):
        why, cors = gate.check(_headers(**h))
        self.assertIsNotNone(why, h)
        self.assertIsNone(cors)
        return why

    def test_hosts(self) -> None:
        g = self.mk.Gate("127.0.0.1")
        for h in ("localhost", "localhost:18151", "LOCALHOST:1", "localhost.", "127.0.0.1", "127.0.0.1:18151",
                  "192.168.1.5:18151", "[::1]:18151", "[::1]", "[fe80::1]:80", "10.0.0.2:"):
            self.assertIsNone(self.ok(g, Host=h))
        self.assertIsNone(self.ok(g))  # (HTTP/1.0: no Host)
        why = self.refused(g, Host="attacker.example:18151")
        self.assertIn("--allow-host attacker.example", why)
        self.assertIn("MK_ALLOW_HOSTS", why)
        for h in ("evil.localhost", "127.0.0.1.nip.io", "a b", "x/y", "[::1", "localhost:99999x", "user@localhost"):
            self.refused(g, Host=h)
        self.assertIn("more than one Host", self.refused(g, Host=["localhost", "evil.example"]))
        # --host's own name, --allow-host names, '*'
        self.ok(self.mk.Gate("MyBox.lan"), Host="mybox.lan:18151")
        g = self.mk.Gate("0.0.0.0", allow_hosts=["Box.Example", "other"])
        self.ok(g, Host="box.example")
        self.ok(g, Host="other:1")
        self.refused(g, Host="box.example.attacker")
        self.ok(self.mk.Gate("127.0.0.1", allow_hosts=["*"]), Host="anything.example")

    def test_origins(self) -> None:
        g = self.mk.Gate("127.0.0.1", cors_origins=["http://LocalHost:5173/", "https://ui.example", "http://x:80"])
        self.assertEqual(g.origins, {"http://localhost:5173", "https://ui.example", "http://x"})
        h = "127.0.0.1:18151"
        self.assertIsNone(self.ok(g, Host=h))  # (no Origin: not a web page)
        # the server's own origin (scheme free, port 80 when Host has none)
        self.assertEqual(self.ok(g, Host=h, Origin="http://127.0.0.1:18151"), "http://127.0.0.1:18151")
        self.ok(g, Host="localhost", Origin="http://localhost")
        self.ok(g, Host="[::1]:18151", Origin="http://[::1]:18151")
        # listed origins (as browsers send them), whatever the Host
        self.assertEqual(self.ok(g, Host=h, Origin="http://localhost:5173"), "http://localhost:5173")
        self.ok(g, Host=h, Origin="https://ui.example")
        self.ok(g, Host=h, Origin="https://ui.example:443")
        self.ok(g, Host=h, Origin="http://x")
        # everything else: other sites, other local ports, another name of this machine, file: pages, garbage
        for o in ("http://attacker.example", "http://127.0.0.1:3000", "http://localhost:18151", "https://ui.example:8443",
                  "null", "http://localhost:5174", "file://", "http://127.0.0.1:18151/path", "javascript:alert(1)"):
            why = self.refused(g, Host=h, Origin=o)
            self.assertIn("is not allowed", why)
        self.assertIn("--cors-origin http://127.0.0.1:3000", self.refused(g, Host=h, Origin="http://127.0.0.1:3000"))
        self.assertIn("more than one Origin", self.refused(g, Host=h, Origin=["http://127.0.0.1:18151"] * 2))
        self.refused(g, Origin="http://127.0.0.1:18151")  # (no Host: not the server's own)
        # the Host check comes first
        self.refused(g, Host="attacker.example", Origin="http://localhost:5173")
        # a listed null (file: pages) matches only null
        g = self.mk.Gate(cors_origins=["null"])
        self.ok(g, Host=h, Origin="null")

    def test_bad_lists(self) -> None:
        for bad in ("*", "localhost:5173", "http://", "http://x/y", "http://x:70000"):
            with self.assertRaises(ValueError, msg=bad):
                self.mk.Gate(cors_origins=[bad])


class GateHttpTests(_GateCase):
    """the gate in the HTTP handlers: 403 before anything is read or run, CORS headers on every admitted response"""

    CHAT = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()

    def test_refused_requests_are_403_and_nothing_runs(self) -> None:
        srv, http_ = self.gated()
        st, h, body = self.ask(http_, "POST", "/v1/chat/completions",
                               [("Host", "rebound.example:18151"), ("Content-Type", "application/json")], self.CHAT)
        err = json.loads(body)["error"]
        self.assertEqual((st, err["code"], err["type"], h["Connection"]), (403, 403, "permission_error", "close"))
        self.assertIn("--allow-host rebound.example", err["message"])
        self.assertEqual(srv.eng.calls, [])  # (nothing prefilled)
        # a page of another origin: the simple request a page can send without a preflight
        st, h, body = self.ask(http_, "POST", "/v1/chat/completions",
                               [("Host", f"127.0.0.1:{http_.port}"), ("Origin", "http://attacker.example"),
                                ("Content-Type", "text/plain")], self.CHAT)
        self.assertEqual(st, 403)
        self.assertIsNone(h["Access-Control-Allow-Origin"])
        self.assertEqual(srv.eng.calls, [])
        # Anthropic's error shape on /v1/messages; GETs too
        st, _, body = self.ask(http_, "POST", "/v1/messages", [("Host", "rebound.example")], b"{}")
        self.assertEqual((st, json.loads(body)["error"]["type"]), (403, "permission_error"))
        st, _, body = self.ask(http_, "GET", "/props", [("Host", "rebound.example")])
        self.assertEqual(st, 403)
        self.assertNotIn(b"engine", body)

    def test_clients_without_origin_are_unaffected(self) -> None:
        srv, http_ = self.gated()
        # curl -d without -H (form content type), the SDKs (application/json), no Host at all (HTTP/1.0 style)
        for ctype in ("application/x-www-form-urlencoded", "application/json", "text/plain"):
            st, h, body = self.ask(http_, "POST", "/v1/chat/completions",
                                   [("Host", f"localhost:{http_.port}"), ("Content-Type", ctype)], self.CHAT)
            self.assertEqual(st, 200, body)
            self.assertIsNone(h["Access-Control-Allow-Origin"])
        st, _, _ = self.ask(http_, "GET", "/health", [])
        self.assertEqual(st, 200)

    def test_an_expect_100_request_is_refused_before_its_body(self) -> None:
        _, http_ = self.gated()
        with socket.create_connection(("127.0.0.1", http_.port), timeout=10) as s:
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: rebound.example\r\nContent-Length: 1000000\r\n"
                      b"Expect: 100-continue\r\n\r\n")
            got = s.recv(65536)
        self.assertTrue(got.startswith(b"HTTP/1.1 403"), got[:80])
        self.assertNotIn(b"100 Continue", got)

    def test_cors_headers_on_every_response_to_an_admitted_origin(self) -> None:
        srv, http_ = self.gated(base._ScriptEngine(["a", "b"]), cors_origins=["http://localhost:5173"])
        o = [("Host", f"127.0.0.1:{http_.port}"), ("Origin", "http://localhost:5173")]
        for method, path, body in (("GET", "/health", None), ("GET", "/nope", None),
                                   ("POST", "/v1/chat/completions", b"not json"),
                                   ("POST", "/v1/chat/completions", self.CHAT)):
            st, h, _ = self.ask(http_, method, path, o + [("Content-Type", "application/json")], body)
            self.assertEqual((h["Access-Control-Allow-Origin"], h["Vary"]), ("http://localhost:5173", "Origin"),
                             (method, path, st))
        stream = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": True}).encode()
        st, h, body = self.ask(http_, "POST", "/v1/chat/completions", o, stream)
        self.assertEqual((st, h["Content-Type"], h["Access-Control-Allow-Origin"]),
                         (200, "text/event-stream", "http://localhost:5173"))
        self.assertIn(b"[DONE]", body)
        # the server's own origin is admitted (same origin) with the same headers
        own = f"http://127.0.0.1:{http_.port}"
        st, h, _ = self.ask(http_, "GET", "/health", [("Host", f"127.0.0.1:{http_.port}"), ("Origin", own)])
        self.assertEqual((st, h["Access-Control-Allow-Origin"]), (200, own))

    def test_preflight(self) -> None:
        _, http_ = self.gated(cors_origins=["http://localhost:5173"])
        host = ("Host", f"127.0.0.1:{http_.port}")
        pre = [host, ("Origin", "http://localhost:5173"), ("Access-Control-Request-Method", "POST"),
               ("Access-Control-Request-Headers", "content-type, authorization, x-stainless-os")]
        st, h, body = self.ask(http_, "OPTIONS", "/v1/chat/completions", pre)
        self.assertEqual((st, body), (200, b""))
        self.assertEqual(h["Access-Control-Allow-Origin"], "http://localhost:5173")
        self.assertEqual(h["Access-Control-Allow-Methods"], "GET, POST, OPTIONS")
        self.assertEqual(h["Access-Control-Allow-Headers"], "content-type, authorization, x-stainless-os")
        self.assertEqual((h["Access-Control-Max-Age"], h["Vary"]), ("86400", "Origin"))
        self.assertIsNone(h["Access-Control-Allow-Private-Network"])
        st, h, _ = self.ask(http_, "OPTIONS", "/v1/messages", pre + [("Access-Control-Request-Private-Network", "true")])
        self.assertEqual((st, h["Access-Control-Allow-Private-Network"]), (200, "true"))
        # a header list that is not one is not echoed
        st, h, _ = self.ask(http_, "OPTIONS", "/v1/messages", pre[:3] + [("Access-Control-Request-Headers", "a\tb;")])
        self.assertEqual((st, h["Access-Control-Allow-Headers"]), (200, None))
        # other origins: 403
        st, h, _ = self.ask(http_, "OPTIONS", "/v1/chat/completions", [host, ("Origin", "http://attacker.example"),
                                                                     ("Access-Control-Request-Method", "POST")])
        self.assertEqual((st, h["Access-Control-Allow-Origin"]), (403, None))
        # a plain OPTIONS (no Origin): the methods
        st, h, _ = self.ask(http_, "OPTIONS", "*", [host])
        self.assertEqual((st, h["Allow"], h["Access-Control-Allow-Methods"]), (200, "GET, POST, OPTIONS", None))
        # keep-alive after a preflight: the next request on the connection is answered
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("OPTIONS", "/v1/chat/completions", headers=dict(pre[1:]))
        r = c.getresponse()
        r.read()
        c.request("GET", "/health")
        self.assertEqual((r.status, c.getresponse().status), (200, 200))
        c.close()

    def test_the_loading_handler(self) -> None:
        httpd = self.mk.HTTPServer(("127.0.0.1", 0), self.mk.make_loading_handler())
        httpd.gate = self.mk.Gate(cors_origins=["http://localhost:5173"])
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        http_ = types.SimpleNamespace(port=httpd.server_address[1])
        st, h, _ = self.ask(http_, "GET", "/health", [("Host", "localhost"), ("Origin", "http://localhost:5173")])
        self.assertEqual((st, h["Access-Control-Allow-Origin"]), (503, "http://localhost:5173"))
        st, _, _ = self.ask(http_, "GET", "/health", [("Host", "rebound.example")])
        self.assertEqual(st, 403)
        st, h, _ = self.ask(http_, "OPTIONS", "/v1/chat/completions", [
            ("Host", "localhost"), ("Origin", "http://localhost:5173"), ("Access-Control-Request-Method", "POST")])
        self.assertEqual((st, h["Access-Control-Allow-Methods"]), (200, "GET, POST, OPTIONS"))

    def test_no_gate_answers_everything(self) -> None:
        _, http_ = self.serve(base._ScriptEngine(["ok"]))  # (HTTPServer.gate None: as before)
        st, _, _ = self.ask(http_, "GET", "/health", [("Host", "anything.example"), ("Origin", "http://x.example")])
        self.assertEqual(st, 200)


if __name__ == "__main__":
    unittest.main()
