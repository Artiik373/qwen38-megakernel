#!/usr/bin/env python3
"""CPU reference of the QSA block selection (kuraneko1/Qwen3.8-27B-QSA-llama.cpp, src/models/qwen35.cpp build_qsa_mask).

Text-only positions (all three M-RoPE sections equal => plain NEOX rope over the first n_rot dims, base 1e7).
Used to validate the GPU selection: select(...) returns, per query, the sorted attended token indices.
Run `python3 bench/qsa_ref.py` for the self-check.
"""
import numpy as np

R, D, HEADS, TOPK, NROT, BASE = 4, 128, 4, 2048, 64, 1e7


def rms(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos):
    """x [..., D], pos broadcastable to x[..., 0]; NEOX pairs (i, i + NROT/2) over the first NROT dims."""
    h = NROT // 2
    f = BASE ** (-np.arange(h) * 2.0 / NROT)
    a = np.asarray(pos, dtype=np.float64)[..., None] * f
    c, s = np.cos(a), np.sin(a)
    y = x.copy()
    x0, x1 = x[..., :h], x[..., h:NROT]
    y[..., :h] = x0 * c - x1 * s
    y[..., h:NROT] = x0 * s + x1 * c
    return y


def pooled_keys(raw_k, k_pre_w, k_w, pos0=0):
    """raw_k [n, D] (k_proj(hidden)); returns [nb, D] pooled+normed+roped keys of the n // R complete blocks."""
    n = raw_k.shape[0]
    nb = n // R
    k = rms(raw_k, k_pre_w)[: nb * R].reshape(nb, R, D).mean(1)
    return rope(rms(k, k_w), pos0 + np.arange(nb) * R)


def select(q_raw, qn_w, pooled, qpos, n_kv, topk=TOPK):
    """q_raw [nq, HEADS*D] (q_proj(hidden)), qpos [nq] absolute positions (cache holds positions 0..n_kv-1).
    Returns list of sorted int arrays: attended token indices (before the causal mask is applied) per query,
    or None where the query is dense (n_kv <= topk or no complete block)."""
    nq = q_raw.shape[0]
    nb = pooled.shape[0]
    out = []
    if n_kv <= topk or nb == 0:
        return [None] * nq
    q = rope(rms(q_raw.reshape(nq, HEADS, D), qn_w), np.asarray(qpos)[:, None])
    kb = min(nb, -(-topk // R))
    for i in range(nq):
        s = np.maximum(pooled @ q[i].T, 0).sum(1) / np.sqrt(D)  # [nb]
        s = np.where((np.arange(nb) + 1) * R - 1 <= qpos[i], s, -np.inf)
        # deterministic tie-break: higher score first, lower block index first
        top = np.lexsort((np.arange(nb), -s))[:kb]
        top = top[np.isfinite(s[top])]
        blocks = set(top.tolist()) | {int(qpos[i]) // R}
        toks = np.concatenate([np.arange(b * R, b * R + R) for b in sorted(blocks)])
        out.append(toks[toks <= qpos[i]])
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 4000
    raw = rng.standard_normal((n, D))
    pk = pooled_keys(raw, np.ones(D), np.ones(D))
    assert pk.shape == (n // R, D)
    qs = rng.standard_normal((3, HEADS * D))
    sel = select(qs, np.ones(D), pk, np.array([n - 3, n - 2, n - 1]), n)
    for s_, p in zip(sel, (n - 3, n - 2, n - 1)):
        assert s_ is not None and s_.max() == p and p // R * R in s_ and len(s_) <= TOPK + R
        assert np.all(np.diff(s_) > 0)
    assert select(qs, np.ones(D), pk[:100], np.array([5, 6, 7]), 400) == [None] * 3  # dense below top_k
    print("qsa_ref ok", [len(s_) for s_ in sel])
