// aarch64 (MLIR dialect) -> Mach-O ARM64 binary translator.
//
// First-light scope: lower a module whose top-level ops are
// aarch64.func (with a flat sequence of aarch64.movz / movk / mov_x /
// bl / svc / ret instruction ops inside) into a runnable Mach-O ARM64
// binary that does not require any libSystem stubs or GOT. The
// special function name `_start` is the program entry; it is placed
// first in the `__text` section so LC_MAIN.entryoff lands on it.
//
// The envelope mirrors the macho_exit reference layout used by
// mlir_wasm_to_macho.c (the n_stubs == 0 / no __DATA shape), so we
// know the load-command sequence and ad-hoc signature format are
// accepted by macOS / AMFI. The only difference from the existing
// backend is that the contents of __text are produced by walking the
// aarch64 dialect rather than the WASM bytecode.
//
// Everything outside the first-light op set returns `false` with a
// diagnostic; coverage grows as new aarch64.* ops are added.

#include "mlir_aarch64_to_macho.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

// =============================================================================
// Growable byte buffer + endian helpers + SHA-256. These mirror the
// helpers in mlir_wasm_to_macho.c (which keeps them file-local), so we
// duplicate them here rather than tying the two backends together.
// =============================================================================
typedef struct { uint8_t *data; size_t len, cap; } Buf;

static void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 1024;
    while (b->len + add > nc) nc *= 2;
    b->data = (uint8_t *)realloc(b->data, nc);
    b->cap = nc;
}
static void buf_u8(Buf *b, uint8_t v) { buf_grow(b, 1); b->data[b->len++] = v; }
static void buf_append(Buf *b, const void *p, size_t n) {
    buf_grow(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static void buf_pad_to(Buf *b, size_t target) {
    if (target <= b->len) return;
    buf_grow(b, target - b->len);
    memset(b->data + b->len, 0, target - b->len);
    b->len = target;
}
static void buf_le16(Buf *b, uint16_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
}
static void buf_le32(Buf *b, uint32_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
    buf_u8(b, (uint8_t)((v >> 16) & 0xff));
    buf_u8(b, (uint8_t)((v >> 24) & 0xff));
}
static void buf_le64(Buf *b, uint64_t v) {
    for (int i = 0; i < 8; i++) buf_u8(b, (uint8_t)((v >> (8 * i)) & 0xff));
}
static void buf_be32(Buf *b, uint32_t v) {
    buf_u8(b, (uint8_t)((v >> 24) & 0xff));
    buf_u8(b, (uint8_t)((v >> 16) & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
    buf_u8(b, (uint8_t)(v & 0xff));
}
static void buf_uleb(Buf *b, uint64_t v) {
    do {
        uint8_t byte = (uint8_t)(v & 0x7f);
        v >>= 7;
        if (v) byte |= 0x80;
        buf_u8(b, byte);
    } while (v);
}
static void buf_cstr(Buf *b, const char *s) {
    while (*s) buf_u8(b, (uint8_t)*s++);
    buf_u8(b, 0);
}
static void buf_patch_le32(Buf *b, size_t pos, uint32_t v) {
    b->data[pos + 0] = (uint8_t)(v & 0xff);
    b->data[pos + 1] = (uint8_t)((v >> 8) & 0xff);
    b->data[pos + 2] = (uint8_t)((v >> 16) & 0xff);
    b->data[pos + 3] = (uint8_t)((v >> 24) & 0xff);
}
static void buf_patch_le64(Buf *b, size_t pos, uint64_t v) {
    for (int k = 0; k < 8; k++)
        b->data[pos + (size_t)k] = (uint8_t)(v >> (8 * k));
}

#define SHA256_DIGEST_LEN 32
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint32_t datalen;
    uint8_t  data[64];
} Sha256;
static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
static inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
static void sha256_xform(Sha256 *s, const uint8_t *d) {
    uint32_t a,b,c,e,f,g,h,t1,t2,m[64];
    uint32_t dd;
    for (int i = 0, j = 0; i < 16; i++, j += 4) {
        m[i] = ((uint32_t)(d[j]   & 0xFFu) << 24) | ((uint32_t)(d[j+1] & 0xFFu) << 16) |
               ((uint32_t)(d[j+2] & 0xFFu) << 8)  |  (uint32_t)(d[j+3] & 0xFFu);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(m[i-15], 7) ^ rotr32(m[i-15], 18) ^ (m[i-15] >> 3);
        uint32_t s1 = rotr32(m[i-2], 17) ^ rotr32(m[i-2], 19)  ^ (m[i-2] >> 10);
        m[i] = m[i-16] + s0 + m[i-7] + s1;
    }
    a = s->state[0]; b = s->state[1]; c = s->state[2]; dd = s->state[3];
    e = s->state[4]; f = s->state[5]; g = s->state[6]; h = s->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        t1 = h + S1 + ch + SHA256_K[i] + m[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        t2 = S0 + mj;
        h = g; g = f; f = e; e = dd + t1;
        dd = c; c = b; b = a; a = t1 + t2;
    }
    s->state[0] += a; s->state[1] += b; s->state[2] += c; s->state[3] += dd;
    s->state[4] += e; s->state[5] += f; s->state[6] += g; s->state[7] += h;
}
static void sha256(const uint8_t *data, size_t n, uint8_t out[32]) {
    Sha256 s;
    s.state[0] = 0x6a09e667; s.state[1] = 0xbb67ae85;
    s.state[2] = 0x3c6ef372; s.state[3] = 0xa54ff53a;
    s.state[4] = 0x510e527f; s.state[5] = 0x9b05688c;
    s.state[6] = 0x1f83d9ab; s.state[7] = 0x5be0cd19;
    s.bitlen = 0; s.datalen = 0;
    while (n) {
        size_t take = 64 - s.datalen;
        if (take > n) take = n;
        memcpy(s.data + s.datalen, data, take);
        s.datalen += (uint32_t)take; data += take; n -= take;
        if (s.datalen == 64) { sha256_xform(&s, s.data); s.bitlen += 512; s.datalen = 0; }
    }
    uint32_t i = s.datalen;
    if (s.datalen < 56) {
        s.data[i++] = 0x80;
        while (i < 56) s.data[i++] = 0;
    } else {
        s.data[i++] = 0x80;
        while (i < 64) s.data[i++] = 0;
        sha256_xform(&s, s.data);
        memset(s.data, 0, 56);
    }
    s.bitlen += (uint64_t)s.datalen * 8;
    for (int k = 7; k >= 0; k--) s.data[56 + (7 - k)] = (uint8_t)(s.bitlen >> (8 * k));
    sha256_xform(&s, s.data);
    for (int k = 0; k < 8; k++) {
        out[4*k + 0] = (uint8_t)(s.state[k] >> 24);
        out[4*k + 1] = (uint8_t)(s.state[k] >> 16);
        out[4*k + 2] = (uint8_t)(s.state[k] >> 8);
        out[4*k + 3] = (uint8_t)(s.state[k]);
    }
}

// =============================================================================
// AArch64 instruction encoders. Each helper returns the 32-bit little-
// endian instruction word for the given operands. These are the only
// ops the first-light slice supports; the table grows op-by-op.
// =============================================================================
static uint32_t arm64_movz(uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    // 5283/d283 0000 base. sf=1 sets bit 31.
    uint32_t base = sf ? 0xd2800000u : 0x52800000u;
    return base | ((uint32_t)(hw & 0x3) << 21) | ((uint32_t)imm16 << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_movk(uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    uint32_t base = sf ? 0xf2800000u : 0x72800000u;
    return base | ((uint32_t)(hw & 0x3) << 21) | ((uint32_t)imm16 << 5)
                | (uint32_t)(rd & 0x1f);
}
// `mov Xd, Xn` (alias for ORR Xd, XZR, Xn): aa0003e0 | Rn<<16 | Rd.
static uint32_t arm64_mov_x(uint8_t rd, uint8_t rn) {
    return 0xaa0003e0u | ((uint32_t)(rn & 0x1f) << 16) | (uint32_t)(rd & 0x1f);
}
// `bl <pc-rel>` (the imm26 carries `byte_offset / 4`, sign-extended).
static uint32_t arm64_bl(int32_t imm26) {
    return 0x94000000u | ((uint32_t)imm26 & 0x03ffffffu);
}
static uint32_t arm64_svc(uint16_t imm16) {
    return 0xd4000001u | ((uint32_t)imm16 << 5);
}
static uint32_t arm64_ret(void) { return 0xd65f03c0u; }

static void emit_word(Buf *b, uint32_t w) { buf_le32(b, w); }

// =============================================================================
// Per-function emission. A pass over an aarch64.func produces a byte
// buffer + a list of `bl` call sites that need PC-relative patching
// once all function offsets are known.
// =============================================================================
typedef struct {
    string   callee;        // referenced symbol name
    uint32_t fn_off;        // byte offset within the function's code
} BlReloc;

typedef struct {
    string    name;
    bool      exported;
    Buf       code;
    BlReloc  *relocs;
    size_t    n_relocs, c_relocs;
    uint32_t  text_off;     // assigned after layout
} EmittedFunc;

static void ef_add_reloc(EmittedFunc *e, string callee, uint32_t off) {
    if (e->n_relocs == e->c_relocs) {
        e->c_relocs = e->c_relocs ? e->c_relocs * 2 : 4;
        e->relocs = (BlReloc *)realloc(e->relocs, e->c_relocs * sizeof(BlReloc));
    }
    e->relocs[e->n_relocs].callee = callee;
    e->relocs[e->n_relocs].fn_off = off;
    e->n_relocs++;
}

static int64_t attr_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static bool attr_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}
static string attr_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}

static bool emit_aarch64_func(MLIR_OpHandle fn, EmittedFunc *out) {
    out->name     = attr_s(fn, "sym_name");
    out->exported = attr_b(fn, "exported");

    if (MLIR_GetOpNumRegions(fn) < 1) {
        fprintf(stderr, "aarch64->macho: aarch64.func has no region\n");
        return false;
    }
    MLIR_BlockHandle blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(fn, 0), 0);
    size_t n = MLIR_GetBlockNumOps(blk);

    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        MLIR_OpType  t  = MLIR_GetOpType(op);
        switch (t) {
        case OP_TYPE_AARCH64_MOVZ: {
            uint8_t rd  = (uint8_t)attr_i(op, "rd");
            uint16_t im = (uint16_t)attr_i(op, "imm16");
            uint8_t hw  = (uint8_t)attr_i(op, "hw");
            bool   sf   = attr_b(op, "sf");
            emit_word(&out->code, arm64_movz(rd, im, hw, sf));
            break;
        }
        case OP_TYPE_AARCH64_MOVK: {
            uint8_t rd  = (uint8_t)attr_i(op, "rd");
            uint16_t im = (uint16_t)attr_i(op, "imm16");
            uint8_t hw  = (uint8_t)attr_i(op, "hw");
            bool   sf   = attr_b(op, "sf");
            emit_word(&out->code, arm64_movk(rd, im, hw, sf));
            break;
        }
        case OP_TYPE_AARCH64_MOV_X: {
            uint8_t rd = (uint8_t)attr_i(op, "rd");
            uint8_t rn = (uint8_t)attr_i(op, "rn");
            emit_word(&out->code, arm64_mov_x(rd, rn));
            break;
        }
        case OP_TYPE_AARCH64_BL: {
            string callee = attr_s(op, "callee");
            uint32_t off = (uint32_t)out->code.len;
            // Placeholder displacement; patched in post-pass.
            emit_word(&out->code, arm64_bl(0));
            ef_add_reloc(out, callee, off);
            break;
        }
        case OP_TYPE_AARCH64_SVC: {
            uint16_t im = (uint16_t)attr_i(op, "imm16");
            emit_word(&out->code, arm64_svc(im));
            break;
        }
        case OP_TYPE_AARCH64_RET:
            emit_word(&out->code, arm64_ret());
            break;
        default: {
            string nm = MLIR_GetOpName(op);
            fprintf(stderr,
                "aarch64->macho: unsupported aarch64 op '%.*s' "
                "(kind=%d)\n", (int)nm.size, nm.str, (int)t);
            return false;
        }
        }
    }
    return true;
}

// =============================================================================
// Mach-O envelope constants (mirror mlir_wasm_to_macho.c).
// =============================================================================
#define MH_MAGIC_64            0xfeedfacfu
#define CPU_ARCH_ABI64         0x01000000u
#define CPU_TYPE_ARM           12u
#define CPU_TYPE_ARM64         (CPU_TYPE_ARM | CPU_ARCH_ABI64)

#define LC_REQ_DYLD            0x80000000u
#define LC_SEGMENT_64          0x19u
#define LC_SYMTAB              0x02u
#define LC_DYSYMTAB            0x0bu
#define LC_LOAD_DYLIB          0x0cu
#define LC_LOAD_DYLINKER       0x0eu
#define LC_UUID                0x1bu
#define LC_CODE_SIGNATURE      0x1du
#define LC_FUNCTION_STARTS     0x26u
#define LC_DATA_IN_CODE        0x29u
#define LC_SOURCE_VERSION      0x2au
#define LC_BUILD_VERSION       0x32u
#define LC_MAIN                (0x28u | LC_REQ_DYLD)
#define LC_DYLD_EXPORTS_TRIE   (0x33u | LC_REQ_DYLD)
#define LC_DYLD_CHAINED_FIXUPS (0x34u | LC_REQ_DYLD)

#define MH_EXECUTE             2u
#define MH_FLAGS_EXEC          2097285u

#define VM_PROT_READ           1u
#define VM_PROT_WRITE          2u
#define VM_PROT_EXECUTE        4u

#define TEXT_VM_BASE           0x100000000ULL
#define TEXT_FILE_BASE         0u
#define VMSEG_SIZE             0x4000u   // 16 KiB minimum mach-o page

// =============================================================================
// Top-level translator.
// =============================================================================
bool mlir_aarch64_to_macho(MLIR_Context *ctx, MLIR_OpHandle module,
                           uint8_t **out_data, size_t *out_size) {
    (void)ctx;
    *out_data = NULL; *out_size = 0;
    if (!module) return false;
    if (MLIR_GetOpNumRegions(module) < 1) return false;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(MLIR_GetOpRegion(module, 0), 0);
    size_t n_top = MLIR_GetBlockNumOps(mb);

    // -----------------------------------------------------------------
    // Collect functions, find `_start`. `_start` must be placed first
    // in __text so LC_MAIN.entryoff equals text_section_off.
    // -----------------------------------------------------------------
    EmittedFunc *efs = (EmittedFunc *)calloc(n_top, sizeof(EmittedFunc));
    size_t n_funcs = 0;
    size_t start_idx = (size_t)-1;
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(op) != OP_TYPE_AARCH64_FUNC) {
            string nm = MLIR_GetOpName(op);
            fprintf(stderr,
                "aarch64->macho: unexpected top-level op '%.*s'\n",
                (int)nm.size, nm.str);
            free(efs); return false;
        }
        if (!emit_aarch64_func(op, &efs[n_funcs])) {
            for (size_t k = 0; k <= n_funcs; k++) {
                free(efs[k].code.data); free(efs[k].relocs);
            }
            free(efs); return false;
        }
        if (efs[n_funcs].name.size == 6
            && memcmp(efs[n_funcs].name.str, "_start", 6) == 0) {
            start_idx = n_funcs;
        }
        n_funcs++;
    }
    if (start_idx == (size_t)-1) {
        fprintf(stderr, "aarch64->macho: no `_start` function in module\n");
        for (size_t k = 0; k < n_funcs; k++) {
            free(efs[k].code.data); free(efs[k].relocs);
        }
        free(efs); return false;
    }

    // -----------------------------------------------------------------
    // Layout: _start first, then everything else in source order.
    // -----------------------------------------------------------------
    uint32_t cursor = 0;
    efs[start_idx].text_off = cursor; cursor += (uint32_t)efs[start_idx].code.len;
    for (size_t i = 0; i < n_funcs; i++) {
        if (i == start_idx) continue;
        efs[i].text_off = cursor;
        cursor += (uint32_t)efs[i].code.len;
    }
    uint32_t text_size = cursor;
    while (text_size % 4) text_size++;

    // -----------------------------------------------------------------
    // Patch `bl` PC-relative displacements. Each reloc identifies its
    // callee by symbol name; resolve against the local table only —
    // first-light has no inter-module calls.
    // -----------------------------------------------------------------
    for (size_t i = 0; i < n_funcs; i++) {
        for (size_t k = 0; k < efs[i].n_relocs; k++) {
            BlReloc *r = &efs[i].relocs[k];
            size_t   tgt = (size_t)-1;
            for (size_t j = 0; j < n_funcs; j++) {
                if (efs[j].name.size == r->callee.size
                    && memcmp(efs[j].name.str, r->callee.str,
                              r->callee.size) == 0) {
                    tgt = j; break;
                }
            }
            if (tgt == (size_t)-1) {
                fprintf(stderr,
                    "aarch64->macho: bl to unknown symbol '%.*s'\n",
                    (int)r->callee.size, r->callee.str);
                for (size_t k2 = 0; k2 < n_funcs; k2++) {
                    free(efs[k2].code.data); free(efs[k2].relocs);
                }
                free(efs); return false;
            }
            uint32_t src_pc = efs[i].text_off + r->fn_off;
            uint32_t dst_pc = efs[tgt].text_off;
            int32_t  rel    = (int32_t)dst_pc - (int32_t)src_pc;
            int32_t  imm26  = rel >> 2;
            uint32_t insn   = arm64_bl(imm26);
            efs[i].code.data[r->fn_off + 0] = (uint8_t)(insn      );
            efs[i].code.data[r->fn_off + 1] = (uint8_t)(insn >>  8);
            efs[i].code.data[r->fn_off + 2] = (uint8_t)(insn >> 16);
            efs[i].code.data[r->fn_off + 3] = (uint8_t)(insn >> 24);
        }
    }

    // -----------------------------------------------------------------
    // Layout constants. This is the macho_exit reference layout
    // (n_stubs == 0, no __DATA segment). See mlir_wasm_to_macho.c for
    // the full annotated walk-through; the load-command list here
    // matches that file's `macho_exit` shape byte-for-byte.
    // -----------------------------------------------------------------
    const uint32_t n_stubs = 0;
    const uint32_t stub_size = 12;
    const uint32_t got_size  = n_stubs * 8;

    const uint32_t n_cmds      = 17;
    const uint32_t sizeofcmds  = 976;
    uint32_t text_section_off  = (32u + sizeofcmds + 15u) & ~15u;
    if (text_section_off < 1040u) text_section_off = 1040u;

    const uint32_t stubs_off    = text_section_off + text_size;
    const uint32_t stubs_size   = n_stubs * stub_size;
    const uint32_t cstring_off  = stubs_off + stubs_size;
    const uint32_t cstring_size = 0;

    uint32_t text_seg_end  = cstring_off + cstring_size;
    uint32_t text_seg_size = (text_seg_end + (VMSEG_SIZE - 1u)) & ~(VMSEG_SIZE - 1u);
    if (text_seg_size < VMSEG_SIZE) text_seg_size = VMSEG_SIZE;
    const uint64_t data_const_vm_base   = TEXT_VM_BASE + text_seg_size;
    const uint32_t data_const_file_base = text_seg_size;
    const uint64_t linkedit_vm_base     = data_const_vm_base + VMSEG_SIZE;
    const uint32_t linkedit_file_base   = data_const_file_base + VMSEG_SIZE;

    // -----------------------------------------------------------------
    // Build the image.
    // -----------------------------------------------------------------
    Buf img = {0};
    img.cap = 1 << 15; img.data = (uint8_t *)malloc(img.cap);

    // mach_header_64
    buf_le32(&img, MH_MAGIC_64);
    buf_le32(&img, CPU_TYPE_ARM64);
    buf_le32(&img, 0);
    buf_le32(&img, MH_EXECUTE);
    buf_le32(&img, n_cmds);
    buf_le32(&img, sizeofcmds);
    buf_le32(&img, MH_FLAGS_EXEC);
    buf_le32(&img, 0);

    // LC_SEGMENT_64 __PAGEZERO
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 72);
    { static const char SEG[16] = "__PAGEZERO"; buf_append(&img, SEG, 16); }
    buf_le64(&img, 0);
    buf_le64(&img, TEXT_VM_BASE);
    buf_le64(&img, 0); buf_le64(&img, 0);
    buf_le32(&img, 0); buf_le32(&img, 0);
    buf_le32(&img, 0); buf_le32(&img, 0);

    // LC_SEGMENT_64 __TEXT (3 sections: __text, __stubs, __cstring)
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 312);
    { static const char SEG[16] = "__TEXT"; buf_append(&img, SEG, 16); }
    buf_le64(&img, TEXT_VM_BASE);
    buf_le64(&img, (uint64_t)text_seg_size);
    buf_le64(&img, TEXT_FILE_BASE);
    buf_le64(&img, (uint64_t)text_seg_size);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, 3);
    buf_le32(&img, 0);
    {
        static const char SN[16] = "__text";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + text_section_off);
        buf_le64(&img, (uint64_t)text_size);
        buf_le32(&img, text_section_off);
        buf_le32(&img, 4);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0x80000400u);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0);
    }
    {
        static const char SN[16] = "__stubs";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + stubs_off);
        buf_le64(&img, (uint64_t)stubs_size);
        buf_le32(&img, stubs_off);
        buf_le32(&img, 2);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0x80000408u);
        buf_le32(&img, n_stubs);
        buf_le32(&img, stub_size);
        buf_le32(&img, 0);
    }
    {
        static const char SN[16] = "__cstring";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + cstring_off);
        buf_le64(&img, (uint64_t)cstring_size);
        buf_le32(&img, cstring_off);
        buf_le32(&img, 0);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 2);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0);
    }

    // LC_SEGMENT_64 __DATA_CONST (1 section: __got, empty)
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 152);
    { static const char SEG[16] = "__DATA_CONST"; buf_append(&img, SEG, 16); }
    buf_le64(&img, data_const_vm_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, (uint64_t)data_const_file_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, 1);
    buf_le32(&img, 16);                  // SG_READ_ONLY
    {
        static const char SN[16] = "__got";
        static const char SG[16] = "__DATA_CONST";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, data_const_vm_base);
        buf_le64(&img, (uint64_t)got_size);
        buf_le32(&img, data_const_file_base);
        buf_le32(&img, 3);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 6);
        buf_le32(&img, 0);
        buf_le32(&img, 0);
        buf_le32(&img, 0);
    }

    // LC_SEGMENT_64 __LINKEDIT (filesize patched at end)
    size_t pos_linkedit_seg = img.len;
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 72);
    { static const char SEG[16] = "__LINKEDIT"; buf_append(&img, SEG, 16); }
    buf_le64(&img, linkedit_vm_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, linkedit_file_base);
    buf_le64(&img, 0);                   // PLACEHOLDER filesize
    buf_le32(&img, VM_PROT_READ);
    buf_le32(&img, VM_PROT_READ);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_linkedit_filesize = pos_linkedit_seg + 8 + 16 + 8 + 8 + 8;

    size_t pos_lc_chained_fixups   = img.len;
    buf_le32(&img, LC_DYLD_CHAINED_FIXUPS); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_exports_trie     = img.len;
    buf_le32(&img, LC_DYLD_EXPORTS_TRIE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_symtab           = img.len;
    buf_le32(&img, LC_SYMTAB); buf_le32(&img, 24);
    buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_dysymtab         = img.len;
    buf_le32(&img, LC_DYSYMTAB); buf_le32(&img, 80);
    for (int k = 0; k < 18; k++) buf_le32(&img, 0);

    // LC_LOAD_DYLINKER
    buf_le32(&img, LC_LOAD_DYLINKER);
    buf_le32(&img, 32);
    buf_le32(&img, 12);
    { static const char nm[20] = "/usr/lib/dyld"; buf_append(&img, nm, 20); }

    // LC_UUID
    buf_le32(&img, LC_UUID); buf_le32(&img, 24);
    {
        static const uint8_t U[16] = {
            0x27,0x07,0xdd,0x62,0x09,0x67,0x3c,0xc0,
            0xb2,0xac,0xef,0xc3,0x2b,0x1c,0xf6,0x3a};
        buf_append(&img, U, 16);
    }

    // LC_BUILD_VERSION
    buf_le32(&img, LC_BUILD_VERSION); buf_le32(&img, 32);
    buf_le32(&img, 1);
    buf_le32(&img, 0x000f0700);
    buf_le32(&img, 0);
    buf_le32(&img, 1);
    buf_le32(&img, 3);
    buf_le32(&img, 0x04ce0100);

    // LC_SOURCE_VERSION
    buf_le32(&img, LC_SOURCE_VERSION); buf_le32(&img, 16);
    buf_le64(&img, 0);

    // LC_MAIN
    buf_le32(&img, LC_MAIN); buf_le32(&img, 24);
    buf_le64(&img, (uint64_t)text_section_off);
    buf_le64(&img, 32ULL * 1024 * 1024);

    // LC_LOAD_DYLIB libSystem (kept so the load-command layout matches
    // the macho_exit reference even though we don't import anything).
    buf_le32(&img, LC_LOAD_DYLIB); buf_le32(&img, 56);
    buf_le32(&img, 24);
    buf_le32(&img, 2);
    buf_le32(&img, 0x054c0000);
    buf_le32(&img, 0x00010000);
    { static const char nm[32] = "/usr/lib/libSystem.B.dylib"; buf_append(&img, nm, 32); }

    size_t pos_lc_function_starts  = img.len;
    buf_le32(&img, LC_FUNCTION_STARTS); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_data_in_code     = img.len;
    buf_le32(&img, LC_DATA_IN_CODE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_code_sig         = img.len;
    buf_le32(&img, LC_CODE_SIGNATURE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);

    buf_pad_to(&img, text_section_off);

    // __text — _start first, then everything else.
    buf_append(&img, efs[start_idx].code.data, efs[start_idx].code.len);
    for (size_t i = 0; i < n_funcs; i++) {
        if (i == start_idx) continue;
        buf_append(&img, efs[i].code.data, efs[i].code.len);
    }
    buf_pad_to(&img, stubs_off);
    // __stubs empty.
    buf_pad_to(&img, cstring_off);
    // __cstring empty.

    // Pad to __DATA_CONST.
    buf_pad_to(&img, data_const_file_base);
    // __got empty (zero bytes written; the segment file size is
    // VMSEG_SIZE, padded below when we pad to linkedit_file_base).
    buf_pad_to(&img, linkedit_file_base);

    // =====================================================================
    // __LINKEDIT
    // =====================================================================
    size_t linkedit_start = img.len;

    // ---- chained fixups blob (no imports) ----
    size_t chained_start = img.len;
    uint32_t fx_imports_off = 0x50u;
    uint32_t fx_symbols_off = fx_imports_off + n_stubs * 4u;
    buf_le32(&img, 0);
    buf_le32(&img, 0x20);
    buf_le32(&img, fx_imports_off);
    buf_le32(&img, fx_symbols_off);
    buf_le32(&img, n_stubs);
    buf_le32(&img, 1);
    buf_le32(&img, 0);
    buf_le32(&img, 0);

    uint32_t fx_seg_count = 4u;
    buf_le32(&img, fx_seg_count);
    buf_le32(&img, 0); buf_le32(&img, 0);
    buf_le32(&img, 0x18);
    buf_le32(&img, 0);
    buf_le32(&img, 0);

    buf_le32(&img, 0x18);
    buf_le16(&img, 0x4000);
    buf_le16(&img, 6);
    buf_le64(&img, (uint64_t)data_const_file_base);
    buf_le32(&img, 0);
    buf_le16(&img, 1);
    buf_le16(&img, 0xffff);                // page_start[0] = DYLD_CHAINED_PTR_START_NONE

    // imports table: empty (n_stubs = 0).
    // symbols table: leading 0 + trailing pad.
    buf_u8(&img, 0);
    buf_u8(&img, 0);
    buf_u8(&img, 0);
    while ((img.len - chained_start) % 8) buf_u8(&img, 0);
    uint32_t chained_size = (uint32_t)(img.len - chained_start);

    // ---- exports trie ----
    size_t exports_start = img.len;
    buf_u8(&img, 0x00); buf_u8(&img, 0x01);
    buf_cstr(&img, "_");
    buf_uleb(&img, 0x12);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x00); buf_u8(&img, 0x02);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x03); buf_u8(&img, 0x00);
    buf_uleb(&img, (uint64_t)text_section_off);
    buf_u8(&img, 0x00);

    buf_u8(&img, 0x00); buf_u8(&img, 0x02);
    buf_cstr(&img, "_mh_execute_header"); buf_uleb(&img, 0x09);
    buf_cstr(&img, "main"); buf_uleb(&img, 0x0d);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00);
    while ((img.len - exports_start) % 8) buf_u8(&img, 0);
    uint32_t exports_size = (uint32_t)(img.len - exports_start);

    // ---- function starts ----
    size_t fs_start = img.len;
    buf_uleb(&img, (uint64_t)text_section_off);
    buf_u8(&img, 0);
    while ((img.len - fs_start) % 8) buf_u8(&img, 0);
    uint32_t fs_size = (uint32_t)(img.len - fs_start);

    // ---- symtab + strtab ----
    Buf strtab = {0};
    buf_u8(&strtab, 0x20);
    buf_u8(&strtab, 0x00);
    uint32_t str_mh   = (uint32_t)strtab.len; buf_cstr(&strtab, "__mh_execute_header");
    uint32_t str_main = (uint32_t)strtab.len; buf_cstr(&strtab, "_main");
    while (strtab.len % 8) buf_u8(&strtab, 0);

    Buf symtab = {0};
    // __mh_execute_header
    buf_le32(&symtab, str_mh);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0010);
    buf_le64(&symtab, TEXT_VM_BASE);
    // _main
    buf_le32(&symtab, str_main);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0000);
    buf_le64(&symtab, TEXT_VM_BASE + text_section_off);

    uint32_t n_syms       = 2;
    uint32_t n_undefs     = 0;
    uint32_t iundefsym    = 2;

    Buf indsyms = {0};

    size_t symtab_off  = img.len;
    buf_append(&img, symtab.data, symtab.len);
    size_t indsyms_off = img.len;
    buf_append(&img, indsyms.data, indsyms.len);
    size_t strtab_off  = img.len;
    buf_append(&img, strtab.data, strtab.len);

    while ((img.len - linkedit_start) % 16) buf_u8(&img, 0);

    // ---- code signature ----
    uint32_t code_limit = (uint32_t)img.len;
    size_t code_sig_off = img.len;

    const uint32_t page_size  = 4096;
    const uint32_t page_shift = 12;
    const char *ident = "tinyc.out";
    uint32_t ident_len = (uint32_t)strlen(ident);
    const uint32_t n_slots = (code_limit + page_size - 1) / page_size;
    const uint32_t ident_offset = 88;
    const uint32_t n_special_slots = 2;
    const uint32_t hash_offset = ident_offset + ident_len + 1
                               + n_special_slots * SHA256_DIGEST_LEN;
    const uint32_t cd_len  = hash_offset + n_slots * SHA256_DIGEST_LEN;
    const uint32_t req_len = 12;
    const uint32_t cms_len = 8;
    const uint32_t sb_header = 12 + 3 * 8;
    const uint32_t sb_len_unpadded = sb_header + cd_len + req_len + cms_len;
    const uint32_t code_sig_size = (sb_len_unpadded + 15u) & ~15u;

    buf_patch_le32(&img, pos_lc_chained_fixups   + 8,  (uint32_t)chained_start);
    buf_patch_le32(&img, pos_lc_chained_fixups   + 12, chained_size);
    buf_patch_le32(&img, pos_lc_exports_trie     + 8,  (uint32_t)exports_start);
    buf_patch_le32(&img, pos_lc_exports_trie     + 12, exports_size);
    buf_patch_le32(&img, pos_lc_symtab           + 8,  (uint32_t)symtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 12, n_syms);
    buf_patch_le32(&img, pos_lc_symtab           + 16, (uint32_t)strtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 20, (uint32_t)strtab.len);
    buf_patch_le32(&img, pos_lc_dysymtab         + 8,  0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 12, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 16, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 20, 2);
    buf_patch_le32(&img, pos_lc_dysymtab         + 24, iundefsym);
    buf_patch_le32(&img, pos_lc_dysymtab         + 28, n_undefs);
    buf_patch_le32(&img, pos_lc_dysymtab         + 32, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 36, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 40, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 44, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 48, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 52, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 56, (uint32_t)indsyms_off);
    buf_patch_le32(&img, pos_lc_dysymtab         + 60, 2u * n_stubs);
    buf_patch_le32(&img, pos_lc_dysymtab         + 64, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 68, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 72, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 76, 0);
    buf_patch_le32(&img, pos_lc_function_starts  + 8,  (uint32_t)fs_start);
    buf_patch_le32(&img, pos_lc_function_starts  + 12, fs_size);
    buf_patch_le32(&img, pos_lc_data_in_code     + 8,  (uint32_t)fs_start + fs_size);
    buf_patch_le32(&img, pos_lc_data_in_code     + 12, 0);
    buf_patch_le32(&img, pos_lc_code_sig         + 8,  (uint32_t)code_sig_off);
    buf_patch_le32(&img, pos_lc_code_sig         + 12, code_sig_size);

    uint64_t linkedit_filesize_v =
        (uint64_t)(code_limit + code_sig_size) - linkedit_file_base;
    uint64_t linkedit_vmsize_v =
        (linkedit_filesize_v + (VMSEG_SIZE - 1u)) & ~(uint64_t)(VMSEG_SIZE - 1u);
    if (linkedit_vmsize_v < VMSEG_SIZE) linkedit_vmsize_v = VMSEG_SIZE;
    buf_patch_le64(&img, pos_linkedit_filesize,        linkedit_filesize_v);
    buf_patch_le64(&img, pos_linkedit_filesize - 16,   linkedit_vmsize_v);

    free(symtab.data); free(indsyms.data); free(strtab.data);

    Buf cs = {0};
    {
        Buf req = {0};
        buf_be32(&req, 0xfade0c01);
        buf_be32(&req, req_len);
        buf_be32(&req, 0);

        uint8_t req_hash[SHA256_DIGEST_LEN];
        sha256(req.data, req.len, req_hash);

        Buf cd = {0};
        buf_be32(&cd, 0xfade0c02);
        buf_be32(&cd, cd_len);
        buf_be32(&cd, 0x00020400);
        buf_be32(&cd, 0x00000002);          // CS_ADHOC
        buf_be32(&cd, hash_offset);
        buf_be32(&cd, ident_offset);
        buf_be32(&cd, n_special_slots);
        buf_be32(&cd, n_slots);
        buf_be32(&cd, code_limit);
        buf_u8(&cd, SHA256_DIGEST_LEN);
        buf_u8(&cd, 2);
        buf_u8(&cd, 0);
        buf_u8(&cd, page_shift);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        while (cd.len < 64) buf_u8(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, text_seg_size);
        buf_be32(&cd, 0);
        buf_be32(&cd, 1);                   // CS_EXECSEG_MAIN_BINARY
        while (cd.len < ident_offset) buf_u8(&cd, 0);
        buf_append(&cd, ident, ident_len);
        buf_u8(&cd, 0);
        buf_append(&cd, req_hash, SHA256_DIGEST_LEN);
        for (int z = 0; z < SHA256_DIGEST_LEN; z++) buf_u8(&cd, 0);
        for (uint32_t i = 0; i < n_slots; i++) {
            uint32_t start  = i * page_size;
            uint32_t remain = code_limit - start;
            uint32_t len    = remain < page_size ? remain : page_size;
            uint8_t  d[SHA256_DIGEST_LEN];
            sha256(img.data + start, len, d);
            buf_append(&cd, d, sizeof(d));
        }

        const uint32_t cd_off  = sb_header;
        const uint32_t req_off = cd_off + cd_len;
        const uint32_t cms_off = req_off + req_len;
        buf_be32(&cs, 0xfade0cc0);
        buf_be32(&cs, sb_len_unpadded);
        buf_be32(&cs, 3);
        buf_be32(&cs, 0x00000000);
        buf_be32(&cs, cd_off);
        buf_be32(&cs, 0x00000002);
        buf_be32(&cs, req_off);
        buf_be32(&cs, 0x00010000);
        buf_be32(&cs, cms_off);
        buf_append(&cs, cd.data, cd.len);
        buf_append(&cs, req.data, req.len);
        buf_be32(&cs, 0xfade0b01);
        buf_be32(&cs, cms_len);
        free(cd.data);
        free(req.data);
    }
    while (cs.len < code_sig_size) buf_u8(&cs, 0);
    buf_append(&img, cs.data, cs.len);
    free(cs.data);

    for (size_t i = 0; i < n_funcs; i++) {
        free(efs[i].code.data);
        free(efs[i].relocs);
    }
    free(efs);

    *out_data = img.data;
    *out_size = img.len;
    return true;
}
