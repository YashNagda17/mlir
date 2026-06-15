// llvm dialect (flat CFG, LP64) -> static x86_64 ELF executable.
//
// The Linux-native sibling of mlir_llvm_to_aarch64.c + mlir_aarch64_to_macho.c.
// It consumes the SAME shared front half (C -> llvm dialect, then mem2reg /
// load_cse / arith_gvn / dce), but expects the standard (scf->cf) lowering so
// the body is a flat basic-block CFG with block arguments (MLIR's phi form):
// llvm.br / llvm.cond_br / llvm.icmp / llvm.add / llvm.call / llvm.return / ...
//
// Milestone-1 code generator: a deliberately simple, correct, slot-based
// allocator. Every SSA value lives in a stack slot; operands are loaded into
// scratch registers, the operation is performed, and the result is stored back.
// No register allocation yet (that converges onto the shared, target-
// parameterized allocator in a follow-up). System access is via direct Linux
// syscalls emitted as tiny local thunks (no libc / PLT / GOT / dynamic linker).
//
// Native-only; not part of the tinyC self-host source set.

#include "mlir_llvm_to_x64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir_api.h"
#include "mlir_obj.h"
#include "mlir_elf.h"

// ---------------------------------------------------------------------------
// x86-64 register numbers.
// ---------------------------------------------------------------------------
enum { RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7,
       R8=8, R9=9, R10=10, R11=11 };
// SysV integer argument registers, in order.
static const uint8_t ARG_REGS[6] = { RDI, RSI, RDX, RCX, R8, R9 };
// Linux x86_64 syscall numbers.
#define NR_READ  0
#define NR_WRITE 1
#define NR_EXIT  60
#define NR_MMAP  9

// condition codes (jcc/setcc low nibble)
enum { CC_E=0x4, CC_NE=0x5, CC_B=0x2, CC_AE=0x3, CC_BE=0x6, CC_A=0x7,
       CC_L=0xC, CC_GE=0xD, CC_LE=0xE, CC_G=0xF };

// ---------------------------------------------------------------------------
// Value -> stack-slot map (open-addressing on the value handle).
// ---------------------------------------------------------------------------
typedef struct { uintptr_t *k; int32_t *v; size_t cap, n; } VMap;
static void vmap_init(VMap *m, size_t want) {
    size_t cap = 16; while (cap < want * 2) cap *= 2;
    m->k = (uintptr_t *)calloc(cap, sizeof(uintptr_t));
    m->v = (int32_t *)malloc(cap * sizeof(int32_t));
    m->cap = cap; m->n = 0;
}
static void vmap_put(VMap *m, MLIR_ValueHandle key, int32_t val) {
    uintptr_t k = (uintptr_t)key;
    size_t i = (k * 1099511628211ull) & (m->cap - 1);
    while (m->k[i]) { if (m->k[i] == k) { m->v[i] = val; return; } i = (i + 1) & (m->cap - 1); }
    m->k[i] = k; m->v[i] = val; m->n++;
}
static bool vmap_get(VMap *m, MLIR_ValueHandle key, int32_t *out) {
    uintptr_t k = (uintptr_t)key;
    size_t i = (k * 1099511628211ull) & (m->cap - 1);
    while (m->k[i]) { if (m->k[i] == k) { *out = m->v[i]; return true; } i = (i + 1) & (m->cap - 1); }
    return false;
}
static void vmap_free(VMap *m) { free(m->k); free(m->v); m->k = NULL; m->v = NULL; }

// Global symbol -> byte offset in the module data blob.
typedef struct { char **name; uint32_t *off; size_t n, cap; } GMap;
static bool gmap_get(GMap *g, string nm, uint32_t *out) {
    for (size_t i = 0; i < g->n; i++)
        if (strlen(g->name[i]) == nm.size && memcmp(g->name[i], nm.str, nm.size) == 0) {
            *out = g->off[i]; return true;
        }
    return false;
}

// ---------------------------------------------------------------------------
// Per-function lowering state.
// ---------------------------------------------------------------------------
typedef struct {
    int    target_block;  // index of the target block
    uint32_t field_off;   // offset of the rel32 field within the function code
} BrFix;

typedef struct {
    MLIR_Context *ctx;
    ObjFunc      *f;
    VMap          slots;     // value -> rbp-relative byte offset (positive = below rbp)
    GMap         *gm;
    uint32_t      frame;     // total frame size (16-aligned)
    uint32_t      edge_tmp;  // rbp offset base of the parallel-move temp area
    // block layout
    uint32_t     *block_off; // code offset of each block (filled as emitted)
    size_t        n_blocks;
    BrFix        *fix; size_t n_fix, c_fix;
    bool          ok;
} FnCtx;

static void fail(FnCtx *F, const char *msg, string extra) {
    if (F->ok) {
        if (extra.size) fprintf(stderr, "x64: %s '%.*s'\n", msg, (int)extra.size, extra.str);
        else            fprintf(stderr, "x64: %s\n", msg);
    }
    F->ok = false;
}

// --- byte emission ---------------------------------------------------------
static void eb(FnCtx *F, uint8_t b) { buf_u8(&F->f->code, b); }
static void e32(FnCtx *F, uint32_t v) { buf_le32(&F->f->code, v); }
static uint32_t here(FnCtx *F) { return (uint32_t)F->f->code.len; }

static void rex(FnCtx *F, int w, int r, int x, int b) {
    uint8_t v = 0x40 | (w << 3) | ((r >= 8) << 2) | ((x >= 8) << 1) | (b >= 8);
    if (v != 0x40 || w) eb(F, v);
}
static void modrm(FnCtx *F, int mod, int reg, int rm) {
    eb(F, (uint8_t)((mod << 6) | ((reg & 7) << 3) | (rm & 7)));
}

// mov reg, imm  (w=1 -> full 64-bit immediate via movabs)
static void mov_ri(FnCtx *F, uint8_t reg, int64_t imm, int w) {
    if (w) {
        rex(F, 1, 0, 0, reg);
        eb(F, 0xB8 + (reg & 7));
        buf_le64(&F->f->code, (uint64_t)imm);
    } else {
        if (reg >= 8) eb(F, 0x41);
        eb(F, 0xB8 + (reg & 7));
        e32(F, (uint32_t)imm);
    }
}
// generic ALU r/m64-or-32, r : opcode form (dst = dst OP src)
static void alu_rr(FnCtx *F, uint8_t op, uint8_t dst, uint8_t src, int w) {
    rex(F, w, src, 0, dst);
    eb(F, op);
    modrm(F, 3, src, dst);
}
static void mov_rr(FnCtx *F, uint8_t dst, uint8_t src, int w) {
    if (dst == src && !w) return;
    alu_rr(F, 0x89, dst, src, w);
}
static void imul_rr(FnCtx *F, uint8_t dst, uint8_t src, int w) {
    rex(F, w, dst, 0, src);
    eb(F, 0x0F); eb(F, 0xAF);
    modrm(F, 3, dst, src);
}
// load reg, [rbp - off]
static void load_rbp(FnCtx *F, uint8_t reg, int32_t off, int w) {
    rex(F, w, reg, 0, 0);
    eb(F, 0x8B);
    modrm(F, 2, reg, RBP);
    e32(F, (uint32_t)(-off));
}
// store [rbp - off], reg
static void store_rbp(FnCtx *F, int32_t off, uint8_t reg, int w) {
    rex(F, w, reg, 0, 0);
    eb(F, 0x89);
    modrm(F, 2, reg, RBP);
    e32(F, (uint32_t)(-off));
}
static void lea_rbp(FnCtx *F, uint8_t reg, int32_t off) {
    rex(F, 1, reg, 0, 0);
    eb(F, 0x8D);
    modrm(F, 2, reg, RBP);
    e32(F, (uint32_t)(-off));
}
// lea reg, [rip + disp32]  (records a data reloc onto the global at `data_off`)
static void lea_rip_data(FnCtx *F, uint8_t reg, int64_t data_off) {
    rex(F, 1, reg, 0, 0);
    eb(F, 0x8D);
    modrm(F, 0, reg, RBP); /* rm=101 + mod=00 => RIP-relative */
    uint32_t field = here(F);
    e32(F, 0);
    obj_add_reloc(F->f, field, field + 4, true, NULL, data_off);
}
// width-aware memory load/store through an address already in `addr`
static void load_mem(FnCtx *F, uint8_t dst, uint8_t addr, int bytes, bool sgn) {
    if (bytes == 8) { rex(F,1,dst,0,addr); eb(F,0x8B); modrm(F,0,dst,addr); }
    else if (bytes == 4) { rex(F,0,dst,0,addr); eb(F,0x8B); modrm(F,0,dst,addr);
                           if (sgn) { /* movsxd done by caller width */ } }
    else if (bytes == 2) { rex(F,1,dst,0,addr); eb(F,0x0F); eb(F, sgn?0xBF:0xB7); modrm(F,0,dst,addr); }
    else { rex(F,1,dst,0,addr); eb(F,0x0F); eb(F, sgn?0xBE:0xB6); modrm(F,0,dst,addr); }
}
static void store_mem(FnCtx *F, uint8_t addr, uint8_t src, int bytes) {
    if (bytes == 1) { rex(F,0,src,0,addr); eb(F,0x88); modrm(F,0,src,addr); }
    else if (bytes == 2) { eb(F,0x66); rex(F,0,src,0,addr); eb(F,0x89); modrm(F,0,src,addr); }
    else if (bytes == 4) { rex(F,0,src,0,addr); eb(F,0x89); modrm(F,0,src,addr); }
    else { rex(F,1,src,0,addr); eb(F,0x89); modrm(F,0,src,addr); }
}

// ---------------------------------------------------------------------------
// type / constant helpers
// ---------------------------------------------------------------------------
static int int_bits(MLIR_Context *ctx, MLIR_ValueHandle v) {
    string s = MLIR_GetTypeString(ctx, MLIR_GetValueType(v));
    if (s.size >= 1 && s.str[0] == 'i') {
        int w = 0; bool any = false;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] < '0' || s.str[i] > '9') return 0;
            w = w * 10 + (s.str[i] - '0'); any = true;
        }
        return any ? w : 0;
    }
    return 0;  // ptr / other
}
static int val_w(MLIR_Context *ctx, MLIR_ValueHandle v) {
    int b = int_bits(ctx, v);
    return (b == 0 || b > 32) ? 1 : 0;  // ptr and i64 -> 64-bit; else 32-bit
}
static bool is_const(MLIR_OpHandle def, int64_t *val) {
    if (def == MLIR_INVALID_HANDLE) return false;
    string nm = MLIR_GetOpName(def);
    if (nm.size == sizeof("llvm.mlir.constant")-1 &&
        memcmp(nm.str, "llvm.mlir.constant", nm.size) == 0) {
        MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(def, "value");
        if (a == MLIR_INVALID_HANDLE) return false;
        *val = MLIR_GetAttributeInteger(a);
        return true;
    }
    if ((nm.size == sizeof("llvm.mlir.zero")-1 &&
         memcmp(nm.str, "llvm.mlir.zero", nm.size) == 0) ||
        (nm.size == sizeof("llvm.mlir.null")-1 &&
         memcmp(nm.str, "llvm.mlir.null", nm.size) == 0)) {
        *val = 0; return true;
    }
    return false;
}
static bool op_is(MLIR_OpHandle op, const char *n) {
    string nm = MLIR_GetOpName(op);
    size_t l = strlen(n);
    return nm.size == l && memcmp(nm.str, n, l) == 0;
}

// SymbolRef attributes print with a leading '@' (e.g. "@printI64"); function
// sym_name attributes do not. Strip it so the two match.
static string strip_at(string s) {
    if (s.size && s.str[0] == '@') { s.str++; s.size--; }
    return s;
}

// Load an operand value into `reg`. Handles constants, globals, slots.
static void load_val(FnCtx *F, MLIR_ValueHandle v, uint8_t reg) {
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(v);
    int64_t c;
    if (is_const(def, &c)) { mov_ri(F, reg, c, val_w(F->ctx, v)); return; }
    if (def != MLIR_INVALID_HANDLE && op_is(def, "llvm.mlir.undef")) {
        mov_ri(F, reg, 0, val_w(F->ctx, v)); return;
    }
    if (def != MLIR_INVALID_HANDLE && op_is(def, "llvm.mlir.addressof")) {
        MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(def, "global_name");
        if (a == MLIR_INVALID_HANDLE) a = MLIR_GetOpAttributeByName(def, "value");
        string gn = strip_at(a ? MLIR_GetAttributeString(a) : (string){0});
        uint32_t off;
        if (!gn.size || !gmap_get(F->gm, gn, &off)) { fail(F, "unknown global", gn); return; }
        lea_rip_data(F, reg, off); return;
    }
    int32_t slot;
    if (!vmap_get(&F->slots, v, &slot)) { fail(F, "value with no slot", (string){0}); return; }
    load_rbp(F, reg, slot, val_w(F->ctx, v));
}
static void store_val(FnCtx *F, MLIR_ValueHandle v, uint8_t reg) {
    int32_t slot;
    if (!vmap_get(&F->slots, v, &slot)) { fail(F, "result with no slot", (string){0}); return; }
    store_rbp(F, slot, reg, val_w(F->ctx, v));
}

#include "mlir_llvm_to_x64_emit.inc"
