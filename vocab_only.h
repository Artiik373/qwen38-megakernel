#pragma once
// llama.cpp reads the tokenizer from the model file, and it rejects ggml types it does not know (llamAmpere's AI+ types
// 111 / 112 / 114: "invalid ggml type"). vocab_path() returns the model itself when every tensor type is known to
// llama.cpp; otherwise a vocab-only copy (the header and all key-values, zero tensors) in the temp directory.
// Not part of the device build: only capi.cpp and main.cu include it.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace gguf {

inline std::string vocab_path(const std::string & model) {
    namespace fs = std::filesystem;
    std::ifstream in(model, std::ios::binary);
    if (!in) return model; // (the engine reports an unreadable file itself)
    auto rd = [&](void * p, size_t n) {
        in.read(static_cast<char *>(p), (std::streamsize) n);
        if (!in) throw std::runtime_error("truncated gguf header");
    };
    auto rd32 = [&]() { uint32_t v; rd(&v, 4); return v; };
    auto rd64 = [&]() { uint64_t v; rd(&v, 8); return v; };
    auto skip = [&](uint64_t n) { in.seekg((std::streamoff) n, std::ios::cur); if (!in) throw std::runtime_error("truncated gguf header"); };
    auto skip_str = [&]() { skip(rd64()); };
    static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    std::function<void(uint32_t)> skip_val = [&](uint32_t t) {
        if (t == 8) skip_str();
        else if (t == 9) {
            const uint32_t et = rd32();
            const uint64_t n = rd64();
            if (et < 13 && et != 8 && et != 9) skip(n * sz[et]);
            else for (uint64_t i = 0; i < n; i++) skip_val(et);
        } else if (t < 13) skip(sz[t]);
        else throw std::runtime_error("bad gguf value type");
    };
    try {
        char magic[4];
        rd(magic, 4);
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') return model;
        rd32();
        const uint64_t nt = rd64(), nkv = rd64();
        for (uint64_t i = 0; i < nkv; i++) { skip_str(); skip_val(rd32()); }
        const std::streamoff kv_end = in.tellg();
        bool unknown = false;
        for (uint64_t i = 0; i < nt; i++) {
            skip_str();
            const uint32_t nd = rd32();
            skip(8ull * nd);
            if (rd32() >= 43) unknown = true; // GGML_TYPE_COUNT of llama.cpp
            skip(8);
        }
        if (!unknown) return model;
        std::error_code ec;
        const uint64_t sig = std::hash<std::string>{}(model) ^ (uint64_t) fs::file_size(model, ec) * 1099511628211ull;
        const fs::path out = fs::temp_directory_path() / ("mk_vocab_" + std::to_string(sig) + ".gguf");
        std::vector<char> buf((size_t) kv_end);
        in.clear();
        in.seekg(0);
        rd(buf.data(), buf.size());
        for (int i = 8; i < 16; i++) buf[i] = 0; // n_tensors = 0
        std::ofstream o(out, std::ios::binary | std::ios::trunc);
        o.write(buf.data(), (std::streamsize) buf.size());
        if (!o) throw std::runtime_error("cannot write " + out.string());
        fprintf(stderr, "[mk] the model has ggml types llama.cpp does not know: tokenizer read from %s\n", out.string().c_str());
        return out.string();
    } catch (const std::exception & e) {
        fprintf(stderr, "[mk] vocab_path: %s\n", e.what());
        return model;
    }
}

} // namespace gguf
