// Minimal read-only GGUF parser (mmap based). Only what the megakernel loader needs.
// Every read is checked against the file size and every size is computed without overflow: a truncated or malformed
// file throws std::runtime_error (never a read outside the mapping or a division by zero). Every length and count is
// checked against the bytes left before anything is allocated for it: memory grows with the file's size (a string up to
// the rest of the file), never with a number the file declares.
// A tensor's byte size comes from its type and dimensions (never from anything else the file says); its type must be one
// the engine reads, every dimension at least 1, and offset + size inside the data section. Readers that expect a shape
// ask for it (File::get(name, dims), shape_is): a tensor with the right name but another shape (fewer rows, an extra
// dimension) is refused with its name, so no reader trusts a size the file declares. tests/gguf_fuzz.cpp feeds it
// malformed files.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <cmath>

#ifdef _WIN32
#include "compat_win32.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
inline int mk_open(const char * path, int flags) { return open(path, flags); }
#endif

namespace gguf {

enum GType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, IQ1_M = 29, BF16 = 30, NVFP4 = 40,
    AIT3 = 111, AIT4 = 112, AIT5_U = 114 // llamAmpere AI+ (ait_dec.cuh): AIT3 / AIT4 tile layout, AIT5_U unit layout
};

struct Tensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    uint64_t offset = 0;
    const uint8_t * data = nullptr;
    uint64_t nbytes = 0;
    // NVFP4 (safetensors source): data = packed e2m1 [rows, K/2]; nv_scale = UE4M3 group scales [rows, K/16];
    // value = e2m1 * ue4m3 / nv_gs. Row r / 16-column group g of this tensor are source row (*nv_rows)[r] / group (*nv_cols)[g].
    const uint8_t * nv_scale = nullptr;
    uint64_t nv_scale_bytes = 0;
    float nv_gs = 1;
    std::shared_ptr<const std::vector<uint32_t>> nv_rows, nv_cols;
};

inline uint64_t type_row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
        case F32: return n * 4;
        case F16: case BF16: return n * 2;
        case Q4_0: return n / 32 * 18;
        case Q4_1: return n / 32 * 20;
        case Q5_0: return n / 32 * 22;
        case Q5_1: return n / 32 * 24;
        case Q8_0: return n / 32 * 34;
        case Q4_K: return n / 256 * 144;
        case Q5_K: return n / 256 * 176;
        case Q6_K: return n / 256 * 210;
        case Q3_K: return n / 256 * 110;
        case IQ3_S: return n / 256 * 110;
        case IQ4_XS: return n / 256 * 136;
        case Q2_K: return n / 256 * 84;
        case IQ2_XXS: return n / 256 * 66;
        case IQ2_XS: return n / 256 * 74;
        case IQ3_XXS: return n / 256 * 98;
        case IQ2_S: return n / 256 * 82;
        case IQ1_M: return n / 256 * 56;
        case IQ4_NL: return n / 32 * 18;
        case AIT3: return n / 512 * 194;   // 32 rows x 512 K chunks of 32 (64 k + 2) bytes: the total is exact, the rows interleave
        case AIT4: return n / 512 * 258;
        case AIT5_U: return n / 128 * 82;
        case NVFP4: return n / 2; // values only: the UE4M3 group scales are a second array (Tensor::nv_scale)
        default: throw std::runtime_error("unsupported ggml type " + std::to_string(type));
    }
}

// weights per block of a type (a row holds whole blocks); 0: a type the engine does not read
inline uint64_t type_block(uint32_t type) {
    switch (type) {
        case F32: case F16: case BF16: return 1;
        case Q4_0: case Q4_1: case Q5_0: case Q5_1: case Q8_0: case IQ4_NL: return 32;
        case Q4_K: case Q5_K: case Q6_K: case Q3_K: case IQ3_S: case IQ4_XS: case Q2_K: case IQ2_XXS: case IQ2_XS: case IQ3_XXS:
        case IQ2_S: case IQ1_M: case NVFP4: return 256;
        case AIT3: case AIT4: return 512;
        case AIT5_U: return 128;
        default: return 0;
    }
}

inline std::string shape_str(const std::vector<uint64_t> & ne) {
    std::string s = "[";
    for (size_t d = 0; d < ne.size(); d++) s += (d ? ", " : "") + std::to_string(ne[d]);
    return s + "]";
}

// ne is exactly `dims` (GGUF order: ne[0] = the row length), missing or further dimensions being 1
inline bool shape_is(const std::vector<uint64_t> & ne, const std::vector<uint64_t> & dims) {
    for (size_t d = 0; d < std::max(ne.size(), dims.size()); d++)
        if ((d < ne.size() ? ne[d] : 1) != (d < dims.size() ? dims[d] : 1)) return false;
    return true;
}

struct File {
    const uint8_t * base = nullptr;
    size_t size = 0;
    std::map<std::string, Tensor> tensors;
    std::map<std::string, uint64_t> ints;
    std::map<std::string, std::string> strs;
    std::map<std::string, float> floats; // float32 kvs (rope.freq_base, layer_norm_rms_epsilon)
    std::map<std::string, uint64_t> arrs; // element count of each array kv (e.g. tokenizer.ggml.tokens: the vocabulary)
    std::map<std::string, std::vector<int64_t>> iarrs; // integer arrays of up to 16 elements (rope.dimension_sections), and in full: *.attention.compress_ratios (QSA)

    static constexpr uint64_t MAX_ALIGNMENT = 1u << 20; // general.alignment: a power of two up to this

    std::vector<std::unique_ptr<uint8_t[]>> owned; // buffers of tensors built at load (safetensors source)
    bool owned_st = false;
    static bool is_safetensors_path(const std::string & p);
    void open_safetensors(const std::string & path);

    explicit File(const std::string & path) {
        if (is_safetensors_path(path)) {
            open_safetensors(path);
            return;
        }
        int fd = mk_open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            throw std::runtime_error("cannot stat " + path);
        }
        if (st.st_size < 24) { // magic, version, tensor and kv counts
            close(fd);
            throw std::runtime_error("not a gguf file (" + std::to_string((long long) st.st_size) + " bytes): " + path);
        }
        size = (size_t) st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base = (const uint8_t *) p;
        try {
            parse();
        } catch (const std::exception & e) {
            munmap((void *) base, size);
            base = nullptr;
            throw std::runtime_error(path + ": " + e.what());
        }
    }
    ~File() { release(); }

    // unmaps the file once its tensors are copied out. The metadata stays (each tensor's type, shape and size; its
    // data pointer is cleared): a mapping held for the engine's life keeps the whole model in the page cache, which
    // under WSL stays counted against Windows.
    void release() {
        if (base) munmap((void *) base, size);
        base = nullptr;
        for (auto & t : tensors) t.second.data = nullptr, t.second.nv_scale = nullptr;
        owned.clear();
    }

    const Tensor & get(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }
    // the tensor, which must have the shape `dims` (shape_is): its nbytes then cover exactly what a reader of that shape
    // reads (the type's row bytes x the rows)
    const Tensor & get(const std::string & name, const std::vector<uint64_t> & dims) const {
        const Tensor & t = get(name);
        if (!shape_is(t.ne, dims))
            throw std::runtime_error("tensor " + name + " has shape " + shape_str(t.ne) + ", expected " + shape_str(dims));
        return t;
    }

  private:
    size_t pos = 0; // invariant: pos <= size

    [[noreturn]] void fail(const std::string & what) const {
        throw std::runtime_error("bad gguf: " + what + " (at byte " + std::to_string(pos) + " of " + std::to_string(size) + ")");
    }
    void need(uint64_t n) const {
        if (n > size - pos) fail("truncated file");
    }
    template <class T> T rd() { need(sizeof(T)); T v; memcpy(&v, base + pos, sizeof(T)); pos += sizeof(T); return v; }
    void skip(uint64_t n) { need(n); pos += (size_t) n; }
    std::string rds() {
        const uint64_t n = rd<uint64_t>();
        need(n);
        std::string s((const char *) base + pos, (size_t) n);
        pos += (size_t) n;
        return s;
    }
    // bytes of a fixed-size value type (0: string, array or invalid)
    static uint64_t scalar_bytes(uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: return 1;
            case 2: case 3: return 2;
            case 4: case 5: case 6: return 4;
            case 10: case 11: case 12: return 8;
            default: return 0;
        }
    }
    // an integer element of type t at p (signed types sign-extended, a u64 as its bits)
    static int64_t int_at(uint32_t t, const uint8_t * p) {
        switch (t) {
            case 0: case 7: return *p;
            case 1: return (int8_t) *p;
            case 2: { uint16_t v; memcpy(&v, p, 2); return v; }
            case 3: { int16_t v; memcpy(&v, p, 2); return v; }
            case 4: { uint32_t v; memcpy(&v, p, 4); return v; }
            case 5: { int32_t v; memcpy(&v, p, 4); return v; }
            default: { int64_t v; memcpy(&v, p, 8); return v; } // 10, 11
        }
    }

    void skip_val(uint32_t t, const std::string & key) {
        switch (t) {
            case 0: case 1: case 7: ints[key] = rd<uint8_t>(); break;
            case 2: case 3: skip(2); break;
            case 4: ints[key] = rd<uint32_t>(); break;
            case 5: ints[key] = (uint64_t) rd<int32_t>(); break;
            case 6: floats[key] = rd<float>(); break;
            case 8: strs[key] = rds(); break;
            case 9: { // arrays are skipped but short integer ones; their elements are scalars or strings (ggml rejects nested arrays too)
                const uint32_t et = rd<uint32_t>();
                const uint64_t n = rd<uint64_t>();
                arrs[key] = n;
                if (const uint64_t eb = scalar_bytes(et)) {
                    if (n > (size - pos) / eb) fail("array " + key + " of " + std::to_string(n) + " elements runs past the end");
                    const bool cr = key.size() >= 15 && key.compare(key.size() - 15, 15, "compress_ratios") == 0; // QSA: per-layer block size (0: dense), kept in full
                    if ((n <= 16 || cr) && et != 6 && et != 12) { // (not f32 / f64)
                        std::vector<int64_t> & v = iarrs[key];
                        for (uint64_t i = 0; i < n; i++) v.push_back(int_at(et, base + pos + i * eb));
                    }
                    pos += (size_t) (n * eb);
                } else if (et == 8) {
                    if (n > (size - pos) / 8) fail("array " + key + " of " + std::to_string(n) + " strings runs past the end");
                    for (uint64_t i = 0; i < n; i++) skip(rd<uint64_t>());
                } else {
                    fail("array " + key + " of element type " + std::to_string(et));
                }
                break;
            }
            case 10: ints[key] = rd<uint64_t>(); break;
            case 11: ints[key] = (uint64_t) rd<int64_t>(); break;
            case 12: skip(8); break;
            default: fail("kv " + key + " of type " + std::to_string(t));
        }
    }

    void parse() {
        if (memcmp(base, "GGUF", 4) != 0) throw std::runtime_error("not a gguf file");
        pos = 4;
        const uint32_t version = rd<uint32_t>();
        if (version != 2 && version != 3) fail("unsupported version " + std::to_string(version));
        const uint64_t n_tensors = rd<uint64_t>();
        const uint64_t n_kv = rd<uint64_t>();
        // a kv takes at least 13 bytes (key length, type, a 1-byte value), a tensor info at least 32
        if (n_kv > (size - pos) / 13) fail(std::to_string(n_kv) + " kvs do not fit the file");
        if (n_tensors > (size - pos) / 32) fail(std::to_string(n_tensors) + " tensors do not fit the file");
        std::set<std::string> keys;
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rds();
            if (!keys.insert(key).second) fail("duplicate key " + key);
            const uint32_t t = rd<uint32_t>();
            skip_val(t, key);
        }
        std::vector<Tensor> infos;
        for (uint64_t i = 0; i < n_tensors; i++) {
            Tensor t;
            t.name = rds();
            const uint32_t nd = rd<uint32_t>();
            if (nd < 1 || nd > 4) fail("tensor " + t.name + " has " + std::to_string(nd) + " dimensions");
            uint64_t nel = 1;
            for (uint32_t d = 0; d < nd; d++) {
                const uint64_t ne = rd<uint64_t>();
                if (ne == 0) fail("tensor " + t.name + " has a dimension of 0"); // (an empty tensor: nothing to read)
                if (ne > (uint64_t) INT64_MAX || nel > (uint64_t) INT64_MAX / ne)
                    fail("tensor " + t.name + " has too many elements");
                nel *= ne;
                t.ne.push_back(ne);
            }
            t.type = rd<uint32_t>();
            t.offset = rd<uint64_t>();
            infos.push_back(std::move(t));
        }
        const uint64_t align = ints.count("general.alignment") ? ints["general.alignment"] : 32;
        if (align == 0 || (align & (align - 1)) != 0 || align > MAX_ALIGNMENT)
            fail("general.alignment " + std::to_string(align) + " is not a power of two up to " + std::to_string(MAX_ALIGNMENT));
        const uint64_t data_start = (pos + align - 1) / align * align;
        if (data_start > size) fail("the data section starts past the end of the file");
        const uint64_t data_size = size - data_start;
        for (auto & t : infos) {
            const uint64_t bl = type_block(t.type);
            if (!bl || t.type == NVFP4) fail("tensor " + t.name + " has unsupported ggml type " + std::to_string(t.type));
            if (t.ne[0] % bl != 0) fail("tensor " + t.name + ": a row of " + std::to_string(t.ne[0]) + " is not whole blocks");
            uint64_t rows = 1;
            for (size_t d = 1; d < t.ne.size(); d++) rows *= t.ne[d]; // (< 2^63: checked above)
            const uint64_t bb = type_row_bytes(t.type, bl), nb = t.ne[0] / bl; // bytes per block, blocks per row
            if (nb > data_size / bb) fail("tensor " + t.name + " is larger than the file");
            const uint64_t rb = nb * bb; // = type_row_bytes(t.type, t.ne[0])
            if (rows && rb > data_size / rows) fail("tensor " + t.name + " is larger than the file");
            t.nbytes = rb * rows;
            if (t.offset % align != 0) fail("tensor " + t.name + " offset " + std::to_string(t.offset) + " is not aligned");
            if (t.offset > data_size || t.nbytes > data_size - t.offset) fail("tensor " + t.name + " runs past the end of the file");
            t.data = base + data_start + t.offset;
            const std::string name = t.name;
            if (!tensors.emplace(name, std::move(t)).second) fail("duplicate tensor " + name);
        }
    }
};

} // namespace gguf

// safetensors source (File::open_safetensors); needs the complete File
#include "safetensors.h"

