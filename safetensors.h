// Direct safetensors source for gguf::File (included at the end of gguf.h): a Hugging Face checkpoint directory (or its
// model.safetensors) holding a Qwen3_5ForCausalLM in compressed-tensors "nvfp4-pack-quantized" form. The engine reads the
// same tensor names and shapes it reads from a qwen35 GGUF, so this file presents the checkpoint as one:
//   * NVFP4 linears (`X.weight_packed` U8 [N, K/2], `X.weight_scale` F8_E4M3 [N, K/16], `X.weight_global_scale` F32 [1]) stay in
//     the mapping (Tensor::data / nv_scale, type NVFP4); the loader repacks them straight from there (repack_nvf4). The
//     linear-attention V-head reorder (llama.cpp's _reorder_v_heads: HF grouped by K head -> the GGUF's tiled order) is a
//     row map / 16-column-group map applied by that repack, not a copy of the weights.
//   * small tensors are converted into owned F32 buffers as llama.cpp's converter writes them: RMS norm weights + 1 (not
//     linear_attn.norm), A_log -> -exp(A_log), dt_bias, conv1d squeezed, V heads reordered; ssm_alpha / ssm_beta (NVFP4 here,
//     but F32 in the engine) dequantised.
//   * the BF16 embedding and lm_head become owned Q8_0 (ggml's quantize_row_q8_0_ref), the types the engine reads.
//   * the metadata the loader reads (architecture, block count, widths, RoPE base, RMS epsilon, context length) comes from
//     config.json. input_global_scale and the calibrated k_scale / v_scale are not read: the engine quantises its
//     activations and KV cache itself.
// The vocabulary is not here: the tokenizer needs a vocab-only GGUF next to the checkpoint (tokenizer.gguf, see
// gguf::tokenizer_path).
#pragma once

#include <cmath>
#include <filesystem>
#include <memory>
#include <thread>

namespace gguf {

namespace stjson {
struct V {
    enum T { Null, Num, Str, Arr, Obj, Bool } t = Null;
    double n = 0;
    std::string s;
    std::vector<V> a;
    std::vector<std::pair<std::string, V>> o;
    const V * get(const std::string & k) const {
        for (const auto & p : o)
            if (p.first == k) return &p.second;
        return nullptr;
    }
};
struct Parser {
    const char * p;
    const char * e;
    [[noreturn]] void fail(const char * w) const { throw std::runtime_error(std::string("bad json: ") + w); }
    void ws() {
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
    }
    void lit(const char * s) {
        const size_t n = strlen(s);
        if ((size_t) (e - p) < n || memcmp(p, s, n) != 0) fail("literal");
        p += n;
    }
    std::string str() {
        if (p >= e || *p != '"') fail("string expected");
        p++;
        std::string r;
        while (p < e && *p != '"') {
            if (*p != '\\') {
                r += *p++;
                continue;
            }
            if (++p >= e) fail("escape");
            const char c = *p++;
            switch (c) {
                case 'n': r += '\n'; break;
                case 't': r += '\t'; break;
                case 'r': r += '\r'; break;
                case 'b': r += '\b'; break;
                case 'f': r += '\f'; break;
                case 'u': {
                    if (e - p < 4) fail("\\u");
                    unsigned cp = (unsigned) strtoul(std::string(p, 4).c_str(), nullptr, 16);
                    p += 4;
                    if (cp < 0x80) r += (char) cp;
                    else if (cp < 0x800) r += (char) (0xC0 | (cp >> 6)), r += (char) (0x80 | (cp & 63));
                    else r += (char) (0xE0 | (cp >> 12)), r += (char) (0x80 | ((cp >> 6) & 63)), r += (char) (0x80 | (cp & 63));
                    break;
                }
                default: r += c;
            }
        }
        if (p >= e) fail("unterminated string");
        p++;
        return r;
    }
    V val(int depth = 0) {
        if (depth > 64) fail("too deep");
        ws();
        if (p >= e) fail("unexpected end");
        V v;
        if (*p == '{') {
            v.t = V::Obj;
            p++;
            ws();
            if (p < e && *p == '}') return p++, v;
            for (;;) {
                ws();
                std::string k = str();
                ws();
                if (p >= e || *p++ != ':') fail("':' expected");
                v.o.emplace_back(std::move(k), val(depth + 1));
                ws();
                if (p < e && *p == ',') { p++; continue; }
                if (p < e && *p == '}') { p++; break; }
                fail("',' or '}' expected");
            }
        } else if (*p == '[') {
            v.t = V::Arr;
            p++;
            ws();
            if (p < e && *p == ']') return p++, v;
            for (;;) {
                v.a.push_back(val(depth + 1));
                ws();
                if (p < e && *p == ',') { p++; continue; }
                if (p < e && *p == ']') { p++; break; }
                fail("',' or ']' expected");
            }
        } else if (*p == '"') {
            v.t = V::Str;
            v.s = str();
        } else if (*p == 't') {
            lit("true");
            v.t = V::Bool;
            v.n = 1;
        } else if (*p == 'f') {
            lit("false");
            v.t = V::Bool;
        } else if (*p == 'n') {
            lit("null");
        } else {
            char * end = nullptr;
            v.n = strtod(p, &end);
            if (end == p) fail("value");
            v.t = V::Num;
            p = end;
        }
        return v;
    }
};
inline V parse(const char * b, size_t n) {
    Parser ps{b, b + n};
    V v = ps.val();
    return v;
}
} // namespace stjson

// e4m3 (also UE4M3: the sign bit is 0 in a scale) -> float
inline float nv_e4m3(uint8_t b) {
    const int ex = (b >> 3) & 15, m = b & 7;
    const float v = ex ? std::ldexp(1.0f + (float) m / 8.0f, ex - 7) : (float) m * 0.001953125f;
    return (b & 0x80) ? -v : v;
}
inline float bf16_f(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
inline uint16_t f32_f16(float f) { // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int32_t ex = (int32_t) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t) (s | 0x7c00u | (m ? 0x200u : 0u));
    if (ex >= 31) return (uint16_t) (s | 0x7c00u);
    if (ex <= 0) {
        if (ex < -10) return (uint16_t) s;
        m |= 0x800000u;
        const int sh = 14 - ex;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) ex << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) r++;
    return (uint16_t) (s | r);
}

template <class F> inline void st_par(int64_t n, F && f) {
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    std::vector<std::exception_ptr> err((size_t) nt);
    for (int t = 0; t < nt; t++)
        th.emplace_back([&, t] {
            try {
                for (int64_t i = t; i < n; i += nt) f(i);
            } catch (...) {
                err[(size_t) t] = std::current_exception();
            }
        });
    for (auto & x : th) x.join();
    for (auto & x : err)
        if (x) std::rethrow_exception(x);
}

// one NVFP4 row (dst row `row`, through the row / column-group maps) as floats: code * ue4m3 scale / weight_global_scale
inline void nv_dequant_row(const Tensor & t, uint64_t row, float * out) {
    static const float E2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const uint64_t K = t.ne[0], G = K / 16;
    const uint64_t sr = t.nv_rows ? (*t.nv_rows)[row] : row;
    for (uint64_t g = 0; g < G; g++) {
        const uint64_t sg = t.nv_cols ? (*t.nv_cols)[g] : g;
        const float sc = nv_e4m3(t.nv_scale[sr * G + sg]) / t.nv_gs;
        const uint8_t * q = t.data + sr * (K / 2) + sg * 8;
        for (int i = 0; i < 8; i++) {
            out[16 * g + 2 * i] = E2M1[q[i] & 15] * sc;
            out[16 * g + 2 * i + 1] = E2M1[q[i] >> 4] * sc;
        }
    }
}

inline bool File::is_safetensors_path(const std::string & p) {
    std::error_code ec;
    if (std::filesystem::is_directory(p, ec)) return true;
    return p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0;
}

// the vocab-only GGUF the tokenizer is read from: the model file itself, or tokenizer.gguf next to a safetensors checkpoint
inline std::string tokenizer_path(const std::string & model) {
    if (!File::is_safetensors_path(model)) return model;
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::is_directory(model, ec) ? std::filesystem::path(model) : std::filesystem::path(model).parent_path();
    const std::filesystem::path tk = dir / "tokenizer.gguf";
    if (!std::filesystem::exists(tk, ec))
        throw std::runtime_error(tk.string() + " is missing: a safetensors model needs a vocab-only GGUF for its tokenizer (python llama.cpp/convert_hf_to_gguf.py " +
                                 dir.string() + " --vocab-only --outfile " + tk.string() + ")");
    return tk.string();
}

inline void File::open_safetensors(const std::string & path0) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path stp(path0), dir;
    if (fs::is_directory(stp, ec)) {
        dir = stp;
        stp = dir / "model.safetensors";
    } else {
        dir = stp.parent_path();
    }
    const std::string path = stp.string();
    int fd = mk_open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path + " (a safetensors model directory holds model.safetensors; sharded checkpoints are not supported)");
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        close(fd);
        throw std::runtime_error("not a safetensors file: " + path);
    }
    size = (size_t) st.st_size;
    void * mp = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mp == MAP_FAILED) throw std::runtime_error("mmap failed");
    base = (const uint8_t *) mp;
    try {
        uint64_t hl;
        memcpy(&hl, base, 8);
        if (hl > size - 8 || hl > (1u << 28)) throw std::runtime_error("bad safetensors header length");
        const stjson::V hdr = stjson::parse((const char *) base + 8, (size_t) hl);
        const uint8_t * data0 = base + 8 + hl;
        const uint64_t dsize = size - 8 - hl;
        struct Ent {
            std::string dtype;
            std::vector<uint64_t> shape;
            uint64_t a = 0, b = 0;
        };
        std::map<std::string, Ent> ent;
        for (const auto & kv : hdr.o) {
            if (kv.first == "__metadata__") continue;
            Ent e;
            const stjson::V * dt = kv.second.get("dtype"), * sh = kv.second.get("shape"), * of = kv.second.get("data_offsets");
            if (!dt || !sh || !of || of->a.size() != 2) throw std::runtime_error("bad safetensors entry " + kv.first);
            e.dtype = dt->s;
            for (const auto & d : sh->a) e.shape.push_back((uint64_t) d.n);
            e.a = (uint64_t) of->a[0].n;
            e.b = (uint64_t) of->a[1].n;
            if (e.a > e.b || e.b > dsize) throw std::runtime_error("safetensors tensor " + kv.first + " runs past the end of the file");
            ent.emplace(kv.first, std::move(e));
        }
        auto cfgj = [&]() {
            std::string txt;
            const std::string cp = (dir / "config.json").string();
            FILE * f = fopen(cp.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open " + cp + " (the model dimensions come from it)");
            char buf[65536];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
            fclose(f);
            return stjson::parse(txt.data(), txt.size());
        };
        const stjson::V cfg0 = cfgj();
        const stjson::V & cfg = cfg0.get("text_config") ? *cfg0.get("text_config") : cfg0;
        auto cnum = [&](const char * k) -> double {
            const stjson::V * v = cfg.get(k);
            if (!v || v->t != stjson::V::Num) throw std::runtime_error(std::string("config.json: missing ") + k);
            return v->n;
        };
        const stjson::V * ra = cfg.get("architectures");
        if (!ra || ra->a.empty() || (ra->a[0].s != "Qwen3_5ForCausalLM" && ra->a[0].s != "Qwen3_5ForConditionalGeneration"))
            throw std::runtime_error("config.json: architecture " + (ra && !ra->a.empty() ? ra->a[0].s : std::string("missing")) + ", expected Qwen3_5ForCausalLM");
        const uint64_t nl = (uint64_t) cnum("num_hidden_layers");
        const uint64_t NKH = (uint64_t) cnum("linear_num_key_heads"), NVH = (uint64_t) cnum("linear_num_value_heads"),
                       HKD = (uint64_t) cnum("linear_key_head_dim"), HVD = (uint64_t) cnum("linear_value_head_dim");
        if (!NKH || NVH % NKH) throw std::runtime_error("config.json: linear_num_value_heads is not a multiple of linear_num_key_heads");
        const uint64_t VPK = NVH / NKH;
        ints["qwen35.block_count"] = nl;
        ints["qwen35.embedding_length"] = (uint64_t) cnum("hidden_size");
        ints["qwen35.feed_forward_length"] = (uint64_t) cnum("intermediate_size");
        ints["qwen35.attention.head_count"] = (uint64_t) cnum("num_attention_heads");
        ints["qwen35.attention.head_count_kv"] = (uint64_t) cnum("num_key_value_heads");
        ints["qwen35.attention.key_length"] = (uint64_t) cnum("head_dim");
        ints["qwen35.attention.value_length"] = (uint64_t) cnum("head_dim");
        ints["qwen35.context_length"] = (uint64_t) cnum("max_position_embeddings");
        strs["general.architecture"] = "qwen35";
        floats["qwen35.attention.layer_norm_rms_epsilon"] = (float) cnum("rms_norm_eps");
        {
            const stjson::V * rp = cfg.get("rope_parameters");
            const stjson::V * th = rp ? rp->get("rope_theta") : cfg.get("rope_theta");
            if (th && th->t == stjson::V::Num) floats["qwen35.rope.freq_base"] = (float) th->n;
        }

        // V head h (tiled order, the GGUF's) is head (h % NKH) * VPK + h / NKH of the checkpoint (llama.cpp _reorder_v_heads)
        auto head_map = [&](uint64_t per_head, uint64_t unit) { // map over `unit`-sized groups; per_head elements per head
            auto v = std::make_shared<std::vector<uint32_t>>();
            const uint64_t gph = per_head / unit;
            for (uint64_t h = 0; h < NVH; h++) {
                const uint64_t sh = (h % NKH) * VPK + h / NKH;
                for (uint64_t g = 0; g < gph; g++) v->push_back((uint32_t) (sh * gph + g));
            }
            return v;
        };
        const uint64_t QK = 2 * HKD * NKH; // q and k channels of in_proj_qkv / conv1d, in front of the V channels
        auto rows_v = [&](uint64_t per_head, uint64_t front) { // identity over `front` rows, then the V heads reordered
            auto v = std::make_shared<std::vector<uint32_t>>();
            for (uint64_t i = 0; i < front; i++) v->push_back((uint32_t) i);
            const auto hm = head_map(per_head, 1);
            for (uint32_t s : *hm) v->push_back((uint32_t) (front + s));
            return v;
        };

        auto own = [&](size_t n) -> uint8_t * {
            owned.emplace_back(new uint8_t[n]);
            return owned.back().get();
        };
        auto raw = [&](const std::string & n, const char * dt) -> const Ent & {
            const auto it = ent.find(n);
            if (it == ent.end()) throw std::runtime_error("missing tensor " + n);
            if (it->second.dtype != dt) throw std::runtime_error("tensor " + n + " is " + it->second.dtype + ", expected " + dt);
            return it->second;
        };
        auto add = [&](const std::string & gname, Tensor && t) {
            t.name = gname;
            if (!tensors.emplace(gname, std::move(t)).second) throw std::runtime_error("duplicate tensor " + gname);
        };
        auto bf16 = [&](const std::string & n, uint64_t want) -> const uint16_t * {
            const Ent & e = raw(n, "BF16");
            if ((e.b - e.a) / 2 != want) throw std::runtime_error("tensor " + n + " has an unexpected size");
            return (const uint16_t *) (data0 + e.a);
        };
        // F32 vector from a BF16 tensor: src(i) -> out[i], optionally + 1
        auto f32_from_bf16 = [&](const std::string & hf, const std::string & gname, std::vector<uint64_t> ne, bool plus1,
                                 const std::vector<uint32_t> * map, const std::function<float(float)> & fn = nullptr) {
            uint64_t n = 1;
            for (uint64_t d : ne) n *= d;
            const uint16_t * s = bf16(hf, n);
            float * o = (float *) own((size_t) n * 4);
            for (uint64_t i = 0; i < n; i++) {
                float v = bf16_f(s[map ? (*map)[i] : i]);
                if (plus1) v += 1.0f;
                o[i] = fn ? fn(v) : v;
            }
            Tensor t;
            t.type = F32;
            t.ne = std::move(ne);
            t.data = (const uint8_t *) o;
            t.nbytes = n * 4;
            add(gname, std::move(t));
        };
        auto q8_from_bf16 = [&](const std::string & hf, const std::string & gname, uint64_t K, uint64_t N) {
            const uint16_t * s = bf16(hf, K * N);
            if (K % 32) throw std::runtime_error(hf + ": row length is not a multiple of 32");
            const uint64_t rb = K / 32 * 34;
            uint8_t * o = own((size_t) (rb * N));
            st_par((int64_t) N, [&](int64_t r) {
                std::vector<float> x(32);
                const uint16_t * sr = s + (uint64_t) r * K;
                uint8_t * orow = o + (uint64_t) r * rb;
                for (uint64_t b = 0; b < K / 32; b++) {
                    float amax = 0.f;
                    for (int j = 0; j < 32; j++) {
                        x[j] = bf16_f(sr[b * 32 + j]);
                        amax = std::max(amax, std::fabs(x[j]));
                    }
                    const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
                    const uint16_t dh = f32_f16(d);
                    memcpy(orow + b * 34, &dh, 2);
                    for (int j = 0; j < 32; j++) orow[b * 34 + 2 + j] = (uint8_t) (int8_t) std::roundf(x[j] * id);
                }
            });
            Tensor t;
            t.type = Q8_0;
            t.ne = {K, N};
            t.data = o;
            t.nbytes = rb * N;
            add(gname, std::move(t));
        };
        // the NVFP4 linear `hf` (.weight_packed / .weight_scale / .weight_global_scale) as tensor gname
        auto nvfp4 = [&](const std::string & hf, const std::string & gname, std::shared_ptr<const std::vector<uint32_t>> rows,
                         std::shared_ptr<const std::vector<uint32_t>> cols) {
            const Ent & p = raw(hf + ".weight_packed", "U8"), & s = raw(hf + ".weight_scale", "F8_E4M3"),
                      & g = raw(hf + ".weight_global_scale", "F32");
            if (p.shape.size() != 2 || s.shape.size() != 2 || g.b - g.a != 4) throw std::runtime_error(hf + ": bad NVFP4 shapes");
            const uint64_t N = p.shape[0], K = 2 * p.shape[1];
            if (K % 256 || s.shape[0] != N || s.shape[1] != K / 16 || p.b - p.a != N * K / 2 || s.b - s.a != N * K / 16)
                throw std::runtime_error(hf + ": NVFP4 tensor of " + shape_str({N, K}) + " with scales " + shape_str(s.shape) + " (rows must be whole 256-blocks)");
            float gs;
            memcpy(&gs, data0 + g.a, 4);
            if (!(gs > 0) || !std::isfinite(gs)) throw std::runtime_error(hf + ": weight_global_scale " + std::to_string(gs));
            if (rows && rows->size() != N) throw std::runtime_error(hf + ": row map of " + std::to_string(rows->size()) + " rows, " + std::to_string(N) + " expected");
            if (cols && cols->size() != K / 16) throw std::runtime_error(hf + ": column map of " + std::to_string(cols->size()) + " groups, " + std::to_string(K / 16) + " expected");
            Tensor t;
            t.type = NVFP4;
            t.ne = {K, N};
            t.data = data0 + p.a;
            t.nbytes = N * K / 2;
            t.nv_scale = data0 + s.a;
            t.nv_scale_bytes = N * K / 16;
            t.nv_gs = gs;
            t.nv_rows = std::move(rows);
            t.nv_cols = std::move(cols);
            add(gname, std::move(t));
        };
        // ssm_alpha / ssm_beta: the engine reads them as F32 rows
        auto nv_to_f32 = [&](const std::string & hf, const std::string & gname, std::shared_ptr<const std::vector<uint32_t>> rows) {
            nvfp4(hf, "tmp." + gname, rows, nullptr);
            Tensor src = std::move(tensors["tmp." + gname]);
            tensors.erase("tmp." + gname);
            const uint64_t K = src.ne[0], N = src.ne[1];
            float * o = (float *) own((size_t) (K * N * 4));
            st_par((int64_t) N, [&](int64_t r) { nv_dequant_row(src, (uint64_t) r, o + (uint64_t) r * K); });
            Tensor t;
            t.type = F32;
            t.ne = {K, N};
            t.data = (const uint8_t *) o;
            t.nbytes = K * N * 4;
            add(gname, std::move(t));
        };

        const uint64_t D = ints["qwen35.embedding_length"], VOCAB = (uint64_t) cnum("vocab_size");
        q8_from_bf16("model.embed_tokens.weight", "token_embd.weight", D, VOCAB);
        q8_from_bf16("lm_head.weight", "output.weight", D, VOCAB);
        f32_from_bf16("model.norm.weight", "output_norm.weight", {D}, true, nullptr);
        for (uint64_t i = 0; i < nl; i++) {
            const std::string hl = "model.layers." + std::to_string(i) + ".", gb = "blk." + std::to_string(i) + ".";
            f32_from_bf16(hl + "input_layernorm.weight", gb + "attn_norm.weight", {D}, true, nullptr);
            f32_from_bf16(hl + "post_attention_layernorm.weight", gb + "post_attention_norm.weight", {D}, true, nullptr);
            nvfp4(hl + "mlp.gate_proj", gb + "ffn_gate.weight", nullptr, nullptr);
            nvfp4(hl + "mlp.up_proj", gb + "ffn_up.weight", nullptr, nullptr);
            nvfp4(hl + "mlp.down_proj", gb + "ffn_down.weight", nullptr, nullptr);
            if (ent.count(hl + "linear_attn.A_log")) {
                const auto hm = head_map(1, 1);
                const std::string la = hl + "linear_attn.";
                nvfp4(la + "in_proj_qkv", gb + "attn_qkv.weight", rows_v(HVD, QK), nullptr);
                nvfp4(la + "in_proj_z", gb + "attn_gate.weight", rows_v(HVD, 0), nullptr);
                nvfp4(la + "out_proj", gb + "ssm_out.weight", nullptr, head_map(HVD, 16));
                nv_to_f32(la + "in_proj_a", gb + "ssm_alpha.weight", hm);
                nv_to_f32(la + "in_proj_b", gb + "ssm_beta.weight", hm);
                f32_from_bf16(la + "A_log", gb + "ssm_a", {NVH}, false, hm.get(), [](float v) { return -std::exp(v); });
                f32_from_bf16(la + "dt_bias", gb + "ssm_dt.bias", {NVH}, false, hm.get());
                f32_from_bf16(la + "norm.weight", gb + "ssm_norm.weight", {HVD}, false, nullptr);
                // conv1d [channels, 1, taps] -> [channels, taps]; the V channels reordered with the V heads
                const Ent & ce = raw(la + "conv1d.weight", "BF16");
                if (ce.shape.size() != 3 || ce.shape[1] != 1 || ce.shape[0] != QK + NVH * HVD)
                    throw std::runtime_error(la + "conv1d.weight has shape " + shape_str(ce.shape));
                const uint64_t taps = ce.shape[2], ch = ce.shape[0];
                const auto cm = rows_v(HVD, QK);
                const uint16_t * cs = (const uint16_t *) (data0 + ce.a);
                float * co = (float *) own((size_t) (ch * taps * 4));
                for (uint64_t c = 0; c < ch; c++)
                    for (uint64_t k = 0; k < taps; k++) co[c * taps + k] = bf16_f(cs[(*cm)[c] * taps + k]);
                Tensor t;
                t.type = F32;
                t.ne = {taps, ch};
                t.data = (const uint8_t *) co;
                t.nbytes = ch * taps * 4;
                add(gb + "ssm_conv1d.weight", std::move(t));
            } else {
                nvfp4(hl + "self_attn.q_proj", gb + "attn_q.weight", nullptr, nullptr);
                nvfp4(hl + "self_attn.k_proj", gb + "attn_k.weight", nullptr, nullptr);
                nvfp4(hl + "self_attn.v_proj", gb + "attn_v.weight", nullptr, nullptr);
                nvfp4(hl + "self_attn.o_proj", gb + "attn_output.weight", nullptr, nullptr);
                const uint64_t HD = ints["qwen35.attention.key_length"];
                f32_from_bf16(hl + "self_attn.q_norm.weight", gb + "attn_q_norm.weight", {HD}, true, nullptr);
                f32_from_bf16(hl + "self_attn.k_norm.weight", gb + "attn_k_norm.weight", {HD}, true, nullptr);
            }
        }
        // anything else in the file the engine does not read must be a known-unused tensor
        for (const auto & kv : ent) {
            const std::string & n = kv.first;
            auto ends = [&](const char * s) { return n.size() >= strlen(s) && n.compare(n.size() - strlen(s), strlen(s), s) == 0; };
            if (ends(".weight_packed") || ends(".weight_scale") || ends(".weight_global_scale") || ends(".input_global_scale") ||
                ends(".k_scale") || ends(".v_scale"))
                continue;
            if (n == "lm_head.weight" || n == "model.embed_tokens.weight" || n == "model.norm.weight") continue;
            if (n.compare(0, 13, "model.layers.") == 0 && atoll(n.c_str() + 13) < (long long) nl) continue;
            throw std::runtime_error("unexpected tensor " + n + " (an MTP layer or other extra is not read from safetensors)");
        }
        owned_st = true;
    } catch (const std::exception & e) {
        owned.clear();
        munmap((void *) base, size);
        base = nullptr;
        throw std::runtime_error(path0 + ": " + e.what());
    }
}

} // namespace gguf
