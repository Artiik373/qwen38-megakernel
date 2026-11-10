// MKEngine::Impl: the DSpine drafter's host side (dspine.h: kernels, dspine.cuh: their definitions). Loads the drafter GGUF,
// keeps its context K / V ring (the last WIN positions of the target's features, injected after every committed row) and
// drafts one block per decode cycle: the block pass, the target's lm_head on its mask rows, the 16 x 16 lattice walk.
// Everything runs on stream 0 behind the megakernel; the drafts go to the control block like an n-gram continuation.
// MK_DSPINE=FILE (the CLI's --dspine) turns it on; the MTP layer is then off. Not built or run on a GPU when written.

#include "engine_impl.h"
#include "dspine.cuh" // (the drafter's kernels: this TU, not the megakernel's, so tuning them rebuilds in seconds)
#include <chrono>

struct MKEngine::Impl::Dsp {
    dsp::Weights w{};
    dsp::Scratch s{};
    __half *kr = nullptr, *vr = nullptr;      // [NLY][WIN][NKV][HD] fp16
    int * tags = nullptr;                     // [WIN]: the position held by ring slot p & (WIN - 1), -1: none
    __nv_bfloat16 *cap = nullptr, *stg = nullptr; // the megakernel's feature rows [row][NF][D]; the prefill's (stg_rows rows)
    int stg_rows = 0;
    gguf::File mf;                            // the model file again: raw lm_head rows of the candidates
    const gguf::Tensor * ht = nullptr;
    size_t hrb = 0;
    int hty = 0;
    int lay[dsp::NF] = {};                    // target layers whose output feeds the drafter (0-based)
    int mask_tok = 248070;
    int next = -1;                            // the ring holds the contiguous positions up to next - 1 (-1: empty)
    std::vector<int> hc, hd;                  // host: candidates, drafts
    std::vector<uint8_t> hrows;
    explicit Dsp(const std::string & model) : mf(model) {}
};

void MKEngine::Impl::dsp_init(const std::string & path) {
    auto d = std::make_shared<Dsp>(model_path);
    gguf::File df(path);
    if (df.strs["general.architecture"] != "dflash") throw std::runtime_error("--dspine " + path + ": expected architecture dflash");
    auto bad = [&](const std::string & what) { throw std::runtime_error("--dspine " + path + ": " + what); };
    auto upload = [&](const gguf::Tensor & t) {
        uint8_t * p = dalloc<uint8_t>(t.nbytes, 0);
        CK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return p;
    };
    auto mat = [&](const std::string & name, int K, int N) {
        const gguf::Tensor & t = df.get(name, {(uint64_t) K, (uint64_t) N});
        const int ty = (int) t.type;
        if (ty != dsp::TY_F32 && ty != dsp::TY_Q8_0 && ty != dsp::TY_Q4K && ty != dsp::TY_Q6K && ty != dsp::TY_BF16)
            bad(name + " has ggml type " + std::to_string(ty) + " (supported: F32, BF16, Q8_0, Q4_K, Q6_K)");
        dsp::Mat m;
        m.type = ty;
        m.K = K;
        m.N = N;
        m.rb = (size_t) gguf::type_row_bytes(t.type, (uint64_t) K);
        if (t.nbytes != m.rb * (size_t) N) bad(name + ": size does not match its shape");
        m.w = upload(t);
        return m;
    };
    auto vec = [&](const std::string & name, int n) {
        const gguf::Tensor & t = df.get(name, {(uint64_t) n});
        if (t.type != dsp::TY_F32) bad(name + " is not F32");
        return reinterpret_cast<const float *>(upload(t));
    };
    constexpr int D = dsp::D, FF = dsp::FF, QD = dsp::QD, KD = dsp::KD, CODE = dsp::CODE, RANK = dsp::RANK, HD = dsp::HD;
    dsp::Weights & w = d->w;
    w.fc = mat("fc.weight", dsp::FD, D);
    w.code = mat("dspine_codebook.weight", CODE, VOCAB);
    w.gate = mat("dspine_gate.weight", CODE, RANK);
    w.pgate = mat("dspine_pred_gate.weight", CODE, RANK);
    w.prev = mat("dspine_prev.weight", CODE, RANK);
    w.encn = vec("enc.output_norm.weight", D);
    w.outn = vec("output_norm.weight", D);
    w.gbias = vec("dspine_gate.bias", RANK);
    for (int l = 0; l < dsp::NLY; l++) {
        const std::string b = "blk." + std::to_string(l) + ".";
        dsp::Layer & L = w.L[l];
        L.q = mat(b + "attn_q.weight", D, QD);
        L.k = mat(b + "attn_k.weight", D, KD);
        L.v = mat(b + "attn_v.weight", D, KD);
        L.o = mat(b + "attn_output.weight", QD, D);
        L.gate = mat(b + "ffn_gate.weight", D, FF);
        L.up = mat(b + "ffn_up.weight", D, FF);
        L.down = mat(b + "ffn_down.weight", FF, D);
        L.rd = mat(b + "dspine_read.weight", D, CODE);
        L.wr = mat(b + "dspine_write.weight", RANK, D);
        L.an = vec(b + "attn_norm.weight", D);
        L.fn = vec(b + "ffn_norm.weight", D);
        L.qn = vec(b + "attn_q_norm.weight", HD);
        L.kn = vec(b + "attn_k_norm.weight", HD);
    }
    if (auto it = df.floats.find("dflash.attention.layer_norm_rms_epsilon"); it != df.floats.end()) w.eps = it->second;
    if (auto it = df.floats.find("dflash.rope.freq_base"); it != df.floats.end()) w.theta = it->second;
    if (auto it = df.ints.find("tokenizer.ggml.mask_token_id"); it != df.ints.end()) d->mask_tok = (int) it->second;
    const auto tl = df.iarrs.find("dflash.target_layers"); // 1-based: the output of layer v - 1
    if (tl == df.iarrs.end() || (int) tl->second.size() != dsp::NF) bad("dflash.target_layers is missing or not 5 layers");
    for (int j = 0; j < dsp::NF; j++) {
        d->lay[j] = tl->second[j] - 1;
        if (d->lay[j] < 0 || d->lay[j] >= NL) bad("dflash.target_layers outside the target model");
    }
    // the target's lm_head rows, raw (the device copy is repacked): the lattice scores 16 candidates per position
    const char * hn = d->mf.tensors.count("output.weight") ? "output.weight" : "token_embd.weight";
    d->ht = &d->mf.get(hn);
    d->hty = (int) d->ht->type;
    if (d->hty != dsp::TY_F32 && d->hty != dsp::TY_Q8_0 && d->hty != dsp::TY_Q4K && d->hty != dsp::TY_Q6K && d->hty != dsp::TY_BF16)
        bad(std::string("the target's ") + hn + " has ggml type " + std::to_string(d->hty) + " (the lattice reads F32, BF16, Q8_0, Q4_K, Q6_K)");
    if (d->ht->ne.size() != 2 || d->ht->ne[0] != (uint64_t) D || d->ht->ne[1] != (uint64_t) VOCAB) bad(std::string("the target's ") + hn + " has an unexpected shape");
    d->hrb = (size_t) gguf::type_row_bytes(d->ht->type, D);

    dsp::Scratch & s = d->s;
    constexpr int B = dsp::BMAX, LR = dsp::LROWS, TK = dsp::TOPK;
    s.feat = dalloc<float>((size_t) B * dsp::FD, 0);
    s.g = dalloc<float>((size_t) B * D, 0);
    s.kb = dalloc<float>((size_t) B * KD, 0);
    s.vb = dalloc<float>((size_t) B * KD, 0);
    s.x = dalloc<float>((size_t) B * D, 0);
    s.xn = dalloc<float>((size_t) B * D, 0);
    s.q = dalloc<float>((size_t) B * QD, 0);
    s.att = dalloc<float>((size_t) B * QD, 0);
    s.ffg = dalloc<float>((size_t) B * FF, 0);
    s.ffu = dalloc<float>((size_t) B * FF, 0);
    s.codes = dalloc<float>((size_t) B * CODE, 0);
    s.bq = dalloc<float>((size_t) B * RANK, 0);
    s.bs = dalloc<float>((size_t) B * RANK, 0);
    s.bv = dalloc<float>((size_t) B * RANK, 0);
    s.bmsg = dalloc<float>((size_t) B * RANK, 0);
    s.bdelta = dalloc<float>((size_t) B * D, 0);
    s.raw = dalloc<float>((size_t) B * D, 0);
    s.rms = dalloc<float>(B, 0);
    s.toks = dalloc<int>(B, 0);
    s.cand = dalloc<int>((size_t) B * TK, 0);
    s.drafts = dalloc<int>(B, 0);
    s.pc = dalloc<float>((size_t) LR * CODE, 0);
    s.gp = dalloc<float>((size_t) LR * RANK, 0);
    s.pv = dalloc<float>((size_t) LR * RANK, 0);
    s.lmsg = dalloc<float>((size_t) LR * RANK, 0);
    s.ldelta = dalloc<float>((size_t) LR * D, 0);
    s.refined = dalloc<float>((size_t) LR * D, 0);
    s.table = dalloc<float>((size_t) (B - 1) * TK * TK, 0);
    s.hrows = dalloc<uint8_t>((size_t) LR * d->hrb, 0);
    d->kr = dalloc<__half>((size_t) dsp::NLY * dsp::WIN * KD, 0);
    d->vr = dalloc<__half>((size_t) dsp::NLY * dsp::WIN * KD, 0);
    d->tags = dalloc<int>(dsp::WIN, 0);
    CK(cudaMemset(d->tags, 0xff, dsp::WIN * sizeof(int)));
    d->cap = dalloc<__nv_bfloat16>((size_t) (std::max(MAXT, MAXD) + 1) * dsp::NF * D, 0);
    d->hc.resize((size_t) B * TK);
    d->hd.resize(B);
    d->hrows.resize((size_t) LR * d->hrb);
    CK(dsp::set_capture(d->cap, d->lay, dsp::NF));
    dspine = d;
    fprintf(stderr, "[mk] DSpine drafter %s: layers %d %d %d %d %d, mask token %d, up to %d drafts per cycle\n", path.c_str(), d->lay[0], d->lay[1],
            d->lay[2], d->lay[3], d->lay[4], d->mask_tok, std::min(n_draft_max, dsp::BMAX - 1));
}

void MKEngine::Impl::dsp_invalidate() {
    if (!dspine) return;
    CK(cudaMemsetAsync(dspine->tags, 0xff, dsp::WIN * sizeof(int), 0));
    dspine->next = -1;
}

// feat: M <= BMAX rows [M][NF][D] of bf16 (device) for the positions pos0 .. pos0 + M - 1 -> the ring
void MKEngine::Impl::dsp_inject(const __nv_bfloat16 * feat, int pos0, int M) {
    Dsp & d = *dspine;
    if (M < 1 || M > dsp::BMAX) throw std::runtime_error("dsp_inject: 1 .. 16 rows");
    if (d.next >= 0 && pos0 > d.next) dsp_invalidate(); // a gap: what the ring holds does not lead up to these rows
    constexpr int D = dsp::D, KD = dsp::KD;
    const dsp::Weights & w = d.w;
    const dsp::Scratch & s = d.s;
    dsp::bf2f(feat, s.feat, (size_t) M * dsp::FD, 0);
    dsp::gemm(w.fc, s.feat, dsp::FD, s.g, D, M, false, 0);
    dsp::rms(s.g, w.encn, s.g, nullptr, M, D, D, w.eps, 0);
    for (int l = 0; l < dsp::NLY; l++) {
        dsp::gemm(w.L[l].k, s.g, D, s.kb, KD, M, false, 0);
        dsp::gemm(w.L[l].v, s.g, D, s.vb, KD, M, false, 0);
        const size_t o = (size_t) l * dsp::WIN * KD;
        dsp::inject_kv(s.kb, s.vb, w.L[l].kn, d.kr + o, d.vr + o, pos0, M, w.theta, w.eps, 0);
    }
    dsp::set_tags(d.tags, pos0, M, 0);
    d.next = pos0 + M;
}

// nd drafts for the anchor hctl.tok[0] at position hctl.n (the ring holds the positions before it): written to the host and
// device control block as fixed drafts; returns their count
int MKEngine::Impl::dsp_draft(int nd) {
    Dsp & d = *dspine;
    Ctl & c = hctl;
    nd = std::min(nd, dsp::BMAX - 1);
    if (nd < 1) return 0;
    pf_init(); // (the target's embedding and lm_head GEMM run on the prefill scratch)
    static const bool prof = getenv("MK_DSPINE_TIME") != nullptr; // per-stage ms of a draft cycle (each stage ends in a stream sync)
    static double acc[4];
    static int cnt;
    auto tm = std::chrono::steady_clock::now();
    auto mark = [&](int i) {
        if (!prof) return;
        CK(cudaStreamSynchronize(0));
        const auto t = std::chrono::steady_clock::now();
        acc[i] += std::chrono::duration<double, std::milli>(t - tm).count();
        tm = t;
    };
    constexpr int D = dsp::D, FF = dsp::FF, QD = dsp::QD, KD = dsp::KD, CODE = dsp::CODE, RANK = dsp::RANK, TK = dsp::TOPK;
    const int M = nd + 1, n = c.n;
    const dsp::Weights & w = d.w;
    const dsp::Scratch & s = d.s;
    int toks[dsp::BMAX];
    toks[0] = c.tok[0];
    for (int t = 1; t < M; t++) toks[t] = d.mask_tok;
    CK(cudaMemcpyAsync(s.toks, toks, M * sizeof(int), cudaMemcpyHostToDevice, 0));
    CK(cudaMemcpyAsync(hp.tok, toks, M * sizeof(int), cudaMemcpyHostToDevice, 0));
    if (hw.embd_t == T_Q5K) mkl::pk_embed<T_Q5K>(M);
    else if (hw.embd_t == T_Q6K) mkl::pk_embed<T_Q6K>(M);
    else if (hw.embd_t == T_Q8_0) mkl::pk_embed<T_Q8_0>(M);
    else if (hw.embd_t == T_IQ4X) mkl::pk_embed<T_IQ4X>(M);
    else if (hw.embd_t == T_IQ2S) mkl::pk_embed<T_IQ2S>(M);
    else mkl::pk_embed<T_Q4K>(M);
    CK(cudaMemcpyAsync(s.x, hp.x, (size_t) M * D * sizeof(float), cudaMemcpyDeviceToDevice, 0));
    for (int l = 0; l < dsp::NLY; l++) {
        const dsp::Layer & L = w.L[l];
        const size_t o = (size_t) l * dsp::WIN * KD;
        // attention (the context ring + this block, causal inside it)
        dsp::rms(s.x, L.an, s.xn, nullptr, M, D, D, w.eps, 0);
        dsp::gemm(L.q, s.xn, D, s.q, QD, M, false, 0);
        dsp::gemm(L.k, s.xn, D, s.kb, KD, M, false, 0);
        dsp::gemm(L.v, s.xn, D, s.vb, KD, M, false, 0);
        dsp::qk_prep(s.q, s.kb, L.qn, L.kn, n, M, w.theta, w.eps, 0);
        dsp::attn(s.q, s.kb, s.vb, d.kr + o, d.vr + o, d.tags, n, M, s.att, 0);
        dsp::gemm(L.o, s.att, QD, s.x, D, M, true, 0);
        // FFN
        dsp::rms(s.x, L.fn, s.xn, nullptr, M, D, D, w.eps, 0);
        dsp::gemm(L.gate, s.xn, D, s.ffg, FF, M, false, 0);
        dsp::gemm(L.up, s.xn, D, s.ffu, FF, M, false, 0);
        dsp::swiglu(s.ffg, s.ffu, (size_t) M * FF, 0);
        dsp::gemm(L.down, s.ffg, FF, s.x, D, M, true, 0);
        // code bridge: row t reads the code of row t - 1's state (row 0: the anchor token's codebook row)
        dsp::rms(s.x, nullptr, s.xn, s.rms, M, D, D, w.eps, 0);
        dsp::gemm(L.rd, s.xn, D, s.pc, CODE, M, false, 0); // (s.pc: scratch here, the lattice's input later)
        dsp::codes(s.pc, w.code, s.toks, M, s.codes, w.eps, 0);
        dsp::gemm(w.gate, s.codes, CODE, s.bq, RANK, M, false, 0);
        dsp::gemm(w.pgate, s.codes, CODE, s.bs, RANK, M, false, 0);
        dsp::gemm(w.prev, s.codes, CODE, s.bv, RANK, M, false, 0);
        dsp::msg(s.bq, s.bs, s.bv, w.gbias, M, s.bmsg, 0);
        dsp::gemm(L.wr, s.bmsg, RANK, s.bdelta, D, M - 1, false, 0);
        dsp::apply(s.x, s.bdelta, s.rms, M, s.raw, 0);
    }
    mark(0); // embed + 5 layers with the bridge
    // the target's output norm + lm_head on the mask rows (rows 1 .. M - 1) -> the 16 best candidates of each
    CK(cudaMemcpyAsync(hp.x, s.x + D, (size_t) (M - 1) * D * sizeof(float), cudaMemcpyDeviceToDevice, 0));
    pf_norm_quant(hp.x, w.outn, M - 1);
    const int tile = std::min(128, hp.cap * FF / VOCAB); // logits rows that fit in y0 (as pf_run's)
    for (int r0 = 0; r0 < M - 1; r0 += tile) {
        const int nt = std::min(tile, M - 1 - r0);
        gemm(hw.head, nt, hp.y0, VOCAB, pf::OUT_STORE, r0);
        dsp::topk(hp.y0, VOCAB, nt, s.cand + (size_t) (1 + r0) * TK, 0);
    }
    CK(cudaMemcpy(d.hc.data(), s.cand, (size_t) M * TK * sizeof(int), cudaMemcpyDeviceToHost));
    mark(1); // output norm + lm_head + top-16 + candidate copy
    // lattice: the head's raw rows of every candidate; the table of scores [position][predecessor][candidate]
    for (int t = 1; t < M; t++)
        for (int k = 0; k < TK; k++) {
            const int id = d.hc[(size_t) t * TK + k];
            if (id < 0 || id >= VOCAB) throw std::runtime_error("DSpine: candidate token out of range");
            memcpy(d.hrows.data() + ((size_t) (t - 1) * TK + k) * d.hrb, d.ht->data + (size_t) id * d.hrb, d.hrb);
        }
    CK(cudaMemcpyAsync(s.hrows, d.hrows.data(), (size_t) (M - 1) * TK * d.hrb, cudaMemcpyHostToDevice, 0));
    const int LR = (M - 1) * TK;
    dsp::pred_codes(s.cand, s.toks, w.code, M, s.pc, 0);
    dsp::gemm(w.pgate, s.pc, CODE, s.gp, RANK, LR, false, 0);
    dsp::gemm(w.prev, s.pc, CODE, s.pv, RANK, LR, false, 0);
    dsp::lat_msg(s.gp, s.pv, s.bq, w.gbias, M, s.lmsg, 0);
    dsp::gemm(w.L[dsp::NLY - 1].wr, s.lmsg, RANK, s.ldelta, D, LR, false, 0);
    dsp::refine(s.raw, s.rms, s.ldelta, w.outn, M, w.eps, s.refined, 0);
    dsp::table(s.hrows, d.hrb, d.hty, s.refined, M, s.table, 0);
    dsp::walk(s.cand, s.table, M, s.drafts, 0);
    CK(cudaMemcpy(d.hd.data(), s.drafts, (size_t) nd * sizeof(int), cudaMemcpyDeviceToHost));
    mark(2); // head-row gather + lattice kernels + draft copy
    if (prof && ++cnt % 50 == 0)
        fprintf(stderr, "[dspine] avg ms/draft over %d: block %.2f, head %.2f, lattice %.2f\n", cnt, acc[0] / cnt, acc[1] / cnt, acc[2] / cnt);
    CK(cudaGetLastError());
    // fixed drafts for the next verify pass (as the n-gram continuation: Ctl::dsmp 0)
    for (int i = 0; i < nd; i++) c.tok[1 + i] = d.hd[i];
    char * dc = reinterpret_cast<char *>(hb.ctl);
    CK(cudaMemcpyAsync(dc + offsetof(Ctl, tok) + sizeof(int), d.hd.data(), (size_t) nd * sizeof(int), cudaMemcpyHostToDevice, 0));
    if (c.dsmp) {
        c.dsmp = 0;
        CK(cudaMemcpyAsync(dc + offsetof(Ctl, dsmp), &c.dsmp, sizeof(int), cudaMemcpyHostToDevice, 0));
    }
    return nd;
}

// pf_run: after layer L of a chunk (hp.x = its output, T rows): the rows the drafter reads go to the staging buffer
void MKEngine::Impl::dsp_pf_stage(int L, int T) {
    Dsp & d = *dspine;
    if (d.stg_rows < T) { // (lazily: ~0.1 GB at the default chunk; the buffer only grows)
        d.stg = dalloc<__nv_bfloat16>((size_t) hp.cap * dsp::NF * dsp::D, 0);
        d.stg_rows = hp.cap;
    }
    for (int j = 0; j < dsp::NF; j++)
        if (d.lay[j] == L) dsp::stage(hp.x, d.stg, j, T, 0);
}

// pf_run: after the chunk's last layer: the positions n .. n + T - 1 from `lo` on into the ring
void MKEngine::Impl::dsp_pf_inject(int n, int T, int lo) {
    Dsp & d = *dspine;
    for (int p = std::max(n, lo); p < n + T; p += dsp::BMAX)
        dsp_inject(d.stg + (size_t) (p - n) * dsp::NF * dsp::D, p, std::min<int>(dsp::BMAX, n + T - p));
}

// the megakernel's feature rows of the last launch (rows 0 .. M - 1 at positions p0 ..) into the ring
void MKEngine::Impl::dsp_cap_inject(int p0, int M) { dsp_inject(dspine->cap, p0, M); }
