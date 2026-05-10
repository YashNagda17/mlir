// Native implementation of MLIR_TranslateModuleToWasm.
//
// Walks an `llvm`-dialect builtin.module via the public mlir_api.h
// surface and emits a wasm32 *relocatable* object byte stream that is
// link-compatible with the existing tinyc wasm build pipeline:
//
//     wasm-ld <module>.wasm.o runtime_wasm.o start_wasm.o -o <name>.wasm
//     wasmtime <name>.wasm
//
// In other words: this file is a minimal hand-written replacement for
// the LLVM WebAssembly backend on the NATIVE arm of
// MLIR_TranslateModuleToWasm.
//
// SCOPE (initial scaffold):
//   - All WASM binary writer infrastructure: LEB128, sections,
//     symbol/reloc/signature tables, custom `linking` and
//     `reloc.CODE` sections per the WebAssembly tool conventions.
//   - Module-level walker that collects llvm.func defs/decls, builds
//     function signatures, and emits the type/import/function/code
//     sections.
//   - Per-function emitter that handles the small subset of LLVM
//     dialect ops needed for the `simple.tc` end-to-end test:
//
//         llvm.func (def & decl)
//         llvm.return                 (with optional i32 value)
//         llvm.mlir.constant          (i32 / i64)
//         llvm.alloca                 (shadow stack)
//         llvm.store / llvm.load      (i32 / i64)
//         llvm.sext                   (i32 -> i64)
//         llvm.call                   (direct, fixed arity)
//
//     Anything else returns an empty string from
//     mlir_translate_to_wasm_native; the test runner's
//     `wasm_native_skip` field marks unsupported tests.
//
// Future PRs grow op coverage and unskip more tests.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"

// Forward-declared entry point — invoked from MLIR_TranslateModuleToWasm
// in mlir_api_impl_upstream.cpp on the NATIVE branch.
string mlir_translate_to_wasm_native(MLIR_Context *ctx, MLIR_OpHandle module);

// =============================================================================
// Growable byte buffer.
// =============================================================================
typedef struct { uint8_t *data; size_t len, cap; } Buf;

static void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 256;
    while (b->len + add > nc) nc *= 2;
    b->data = (uint8_t *)realloc(b->data, nc);
    b->cap = nc;
}
static void buf_putc(Buf *b, uint8_t c) { buf_grow(b, 1); b->data[b->len++] = c; }
static void buf_append(Buf *b, const void *p, size_t n) {
    buf_grow(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static void buf_cstr(Buf *b, const char *s) { buf_append(b, s, strlen(s)); }
static void buf_free(Buf *b) { free(b->data); b->data = NULL; b->len = b->cap = 0; }

// =============================================================================
// LEB128 encoders.
// =============================================================================
static void leb_u(Buf *b, uint64_t v) {
    do {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        if (v) byte |= 0x80;
        buf_putc(b, byte);
    } while (v);
}
static void leb_s(Buf *b, int64_t v) {
    bool more = true;
    while (more) {
        uint8_t byte = v & 0x7f;
        v >>= 7;  // arithmetic shift on signed
        bool sign = (byte & 0x40) != 0;
        if ((v == 0 && !sign) || (v == -1 && sign)) more = false;
        else byte |= 0x80;
        buf_putc(b, byte);
    }
}
// Five-byte zero-padded uleb128 — used inside the CODE section for
// values that get patched by relocations (the linker rewrites the LEB
// in place and the reloc target offset must always remain valid).
static void leb_u_pad5(Buf *b, uint32_t v) {
    for (int i = 0; i < 5; i++) {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        if (i < 4) byte |= 0x80;
        buf_putc(b, byte);
    }
}

// =============================================================================
// WASM constants.
// =============================================================================
enum {
    SEC_CUSTOM   = 0,
    SEC_TYPE     = 1,
    SEC_IMPORT   = 2,
    SEC_FUNCTION = 3,
    SEC_CODE     = 10,

    IMP_FUNC   = 0,
    IMP_TABLE  = 1,
    IMP_MEMORY = 2,
    IMP_GLOBAL = 3,

    VT_I32 = 0x7f,
    VT_I64 = 0x7e,
    VT_F32 = 0x7d,
    VT_F64 = 0x7c,

    LINK_VERSION = 2,
    LINK_SUB_SYMBOL_TABLE = 8,

    SYM_FUNCTION = 0,
    SYM_DATA     = 1,
    SYM_GLOBAL   = 2,

    SYMF_BINDING_LOCAL = 0x02,
    SYMF_UNDEFINED     = 0x10,
    SYMF_EXPORTED      = 0x20,
    SYMF_EXPLICIT_NAME = 0x40,

    R_WASM_FUNCTION_INDEX_LEB = 0,
    R_WASM_MEMORY_ADDR_LEB    = 3,
    R_WASM_GLOBAL_INDEX_LEB   = 7,
};

// =============================================================================
// Simple string utilities + name comparison helpers (mirroring
// mlir_translate_to_llvm_ir.c).
// =============================================================================
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}
static char *xstrdupn(const char *s, size_t n) {
    char *r = (char *)malloc(n + 1);
    memcpy(r, s, n);
    r[n] = 0;
    return r;
}
static char *xstrdup_str(string s) { return xstrdupn(s.str, s.size); }

static MLIR_AttributeHandle find_attr(MLIR_OpHandle op, const char *name) {
    size_t n = MLIR_GetOpNumAttributes(op);
    for (size_t i = 0; i < n; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        string an = MLIR_GetAttributeName(a);
        if (name_eq(an, name)) return a;
    }
    return MLIR_INVALID_HANDLE;
}

// Convert MLIR LLVM-dialect type -> WASM value type byte. Returns 0 if
// the type is not a primitive scalar.
static uint8_t wasm_vt(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return VT_I32;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return VT_I32;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return VT_F32;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return VT_F64;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8 || w == 16 || w == 32) return VT_I32;
        if (w == 64) return VT_I64;
    }
    return 0;
}

// Size in bytes of a primitive scalar/pointer type. Returns 0 on
// unsupported aggregates (caller signals error).
static unsigned type_size_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return 8;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8) return 1;
        if (w == 16) return 2;
        if (w == 32) return 4;
        if (w == 64) return 8;
    }
    return 0;
}

// =============================================================================
// Function-signature interner.
// =============================================================================
typedef struct {
    uint8_t *params;  // value-type bytes
    size_t   nparams;
    uint8_t *results;
    size_t   nresults;
} Sig;

typedef struct {
    Sig    *e;
    size_t  n, cap;
} SigTab;

static bool sig_eq(const Sig *a, const Sig *b) {
    if (a->nparams != b->nparams || a->nresults != b->nresults) return false;
    if (memcmp(a->params, b->params, a->nparams) != 0) return false;
    if (memcmp(a->results, b->results, a->nresults) != 0) return false;
    return true;
}
static uint32_t sig_intern(SigTab *t, const uint8_t *params, size_t np,
                           const uint8_t *results, size_t nr) {
    Sig probe;
    probe.params = (uint8_t *)params; probe.nparams = np;
    probe.results = (uint8_t *)results; probe.nresults = nr;
    for (size_t i = 0; i < t->n; i++) if (sig_eq(&t->e[i], &probe)) return (uint32_t)i;
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 8;
        t->e = (Sig *)realloc(t->e, t->cap * sizeof(Sig));
    }
    Sig *s = &t->e[t->n];
    s->nparams = np; s->params = (uint8_t *)malloc(np ? np : 1); memcpy(s->params, params, np);
    s->nresults = nr; s->results = (uint8_t *)malloc(nr ? nr : 1); memcpy(s->results, results, nr);
    return (uint32_t)t->n++;
}
static void sigtab_free(SigTab *t) {
    for (size_t i = 0; i < t->n; i++) { free(t->e[i].params); free(t->e[i].results); }
    free(t->e); t->e = NULL; t->n = t->cap = 0;
}

// =============================================================================
// Module-level state.
// =============================================================================

// Function record. Indices laid out in the wasm function-index space:
//   [0 .. n_imports)        : imported funcs
//   [n_imports .. n_imports + n_defs) : defined funcs
typedef struct {
    char    *name;
    uint32_t sig;
    bool     imported;
    bool     exported;       // true for `__original_main`
    Buf      body;           // raw body (locals + instructions + 0x0b end), valid if !imported
    uint32_t func_index;     // index in function-index space
    uint32_t sym_index;      // index in symbol table

    // Reloc records emitted into this function's body (offsets are
    // relative to the start of `body`). We accumulate them per-func and
    // adjust to CODE-section-relative offsets at serialization time.
    struct {
        uint8_t  type;
        uint32_t body_offset;
        uint32_t sym_idx;
    } *relocs;
    size_t n_relocs, c_relocs;
} Func;

typedef struct {
    Func   *e;
    size_t  n, cap;
} FuncTab;

static uint32_t func_add(FuncTab *t, const char *name, uint32_t sig, bool imported) {
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 16;
        t->e = (Func *)realloc(t->e, t->cap * sizeof(Func));
    }
    memset(&t->e[t->n], 0, sizeof(Func));
    t->e[t->n].name = xstrdupn(name, strlen(name));
    t->e[t->n].sig = sig;
    t->e[t->n].imported = imported;
    return (uint32_t)t->n++;
}
static int func_find(FuncTab *t, const char *name) {
    for (size_t i = 0; i < t->n; i++) if (strcmp(t->e[i].name, name) == 0) return (int)i;
    return -1;
}
static void func_add_reloc(Func *f, uint8_t type, uint32_t body_off, uint32_t sym_idx) {
    if (f->n_relocs == f->c_relocs) {
        f->c_relocs = f->c_relocs ? f->c_relocs * 2 : 4;
        f->relocs = realloc(f->relocs, f->c_relocs * sizeof(*f->relocs));
    }
    f->relocs[f->n_relocs].type = type;
    f->relocs[f->n_relocs].body_offset = body_off;
    f->relocs[f->n_relocs].sym_idx = sym_idx;
    f->n_relocs++;
}
static void functab_free(FuncTab *t) {
    for (size_t i = 0; i < t->n; i++) {
        free(t->e[i].name);
        buf_free(&t->e[i].body);
        free(t->e[i].relocs);
    }
    free(t->e); t->e = NULL; t->n = t->cap = 0;
}

typedef struct {
    SigTab   sigs;
    FuncTab  funcs;
    uint32_t stack_pointer_global; // global index for env.__stack_pointer (always 0)
    uint32_t stack_pointer_sym;    // its symbol-table index (assigned in translate_module)
} ModCtx;

// Sym indices are assigned eagerly so reloc records emitted during op
// translation can reference the correct table slot:
//   sym 0       : __stack_pointer (imported global)
//   sym 1..n    : functions, in `M->funcs` order (imports first, then defs)
static void assign_sym_indices(ModCtx *M) {
    M->stack_pointer_sym = 0;
    for (size_t i = 0; i < M->funcs.n; i++) {
        M->funcs.e[i].sym_index = (uint32_t)(i + 1);
    }
}

// =============================================================================
// Per-function emission state.
// =============================================================================
typedef struct {
    uintptr_t key;
    uint32_t  local_index;
} VLocal;

typedef struct {
    MLIR_Context *ctx;
    ModCtx       *M;
    Func         *F;
    Buf          *body;       // alias of F->body for convenience

    // Locals: parameters first (fixed by signature), then a flat list of
    // synthetic locals appended in declaration order. We emit grouped
    // (count, type) prelude pairs at finalize time, but local indices
    // are simply param_count + alloc_order so we can name them eagerly.
    uint8_t  *param_types;
    size_t    n_params;
    uint8_t  *local_types;     // flat, in alloc order
    size_t    n_locals, c_locals;

    // Map from MLIR_ValueHandle -> local index.
    VLocal   *vmap;
    size_t    n_vmap, c_vmap;

    // Shadow-stack frame: total bytes (16-aligned) and per-alloca byte
    // offset within the frame. The frame is laid out at SP. Local index
    // for the SP register is `sp_local`.
    uint32_t  frame_size;
    uint32_t  sp_local;          // 0xffffffff if no frame

    typeof(struct {
        uintptr_t key;
        uint32_t  offset;
    }) *amap;                    // alloca-result -> frame offset
    size_t    n_amap, c_amap;
} FnCtx;

static uint32_t local_alloc(FnCtx *F, uint8_t vt) {
    if (F->n_locals == F->c_locals) {
        F->c_locals = F->c_locals ? F->c_locals * 2 : 8;
        F->local_types = realloc(F->local_types, F->c_locals);
    }
    F->local_types[F->n_locals] = vt;
    return (uint32_t)(F->n_params + F->n_locals++);
}
static void vmap_set(FnCtx *F, MLIR_ValueHandle v, uint32_t li) {
    if (F->n_vmap == F->c_vmap) {
        F->c_vmap = F->c_vmap ? F->c_vmap * 2 : 16;
        F->vmap = realloc(F->vmap, F->c_vmap * sizeof(VLocal));
    }
    F->vmap[F->n_vmap].key = (uintptr_t)v;
    F->vmap[F->n_vmap].local_index = li;
    F->n_vmap++;
}
static int vmap_get(FnCtx *F, MLIR_ValueHandle v, uint32_t *out) {
    for (size_t i = 0; i < F->n_vmap; i++) {
        if (F->vmap[i].key == (uintptr_t)v) { *out = F->vmap[i].local_index; return 1; }
    }
    return 0;
}
static void amap_set(FnCtx *F, MLIR_ValueHandle v, uint32_t off) {
    if (F->n_amap == F->c_amap) {
        F->c_amap = F->c_amap ? F->c_amap * 2 : 8;
        F->amap = realloc(F->amap, F->c_amap * sizeof(*F->amap));
    }
    F->amap[F->n_amap].key = (uintptr_t)v;
    F->amap[F->n_amap].offset = off;
    F->n_amap++;
}
static int amap_get(FnCtx *F, MLIR_ValueHandle v, uint32_t *out) {
    for (size_t i = 0; i < F->n_amap; i++) {
        if (F->amap[i].key == (uintptr_t)v) { *out = F->amap[i].offset; return 1; }
    }
    return 0;
}

// Helpers that emit the SP global access opcode along with a
// R_WASM_GLOBAL_INDEX_LEB reloc record so the linker can patch the
// referenced global index when sections get merged.
static void emit_sp_global_get(FnCtx *F) {
    buf_putc(F->body, 0x23);                        // global.get
    uint32_t off = (uint32_t)F->body->len;
    leb_u_pad5(F->body, F->M->stack_pointer_global);
    func_add_reloc(F->F, R_WASM_GLOBAL_INDEX_LEB, off, F->M->stack_pointer_sym);
}
static void emit_sp_global_set(FnCtx *F) {
    buf_putc(F->body, 0x24);                        // global.set
    uint32_t off = (uint32_t)F->body->len;
    leb_u_pad5(F->body, F->M->stack_pointer_global);
    func_add_reloc(F->F, R_WASM_GLOBAL_INDEX_LEB, off, F->M->stack_pointer_sym);
}

// =============================================================================
// Per-op emitters.
// =============================================================================

// Push the value of `v` onto the wasm stack.
static bool emit_use(FnCtx *F, MLIR_ValueHandle v) {
    uint32_t li;
    if (!vmap_get(F, v, &li)) {
        fprintf(stderr, "wasm-emit: use of unbound value\n");
        return false;
    }
    buf_putc(F->body, 0x20);  // local.get
    leb_u(F->body, li);
    return true;
}

// Returns true on success, false on unsupported op.
static bool emit_op(FnCtx *F, MLIR_OpHandle op);

static bool emit_op(FnCtx *F, MLIR_OpHandle op) {
    string name = MLIR_GetOpName(op);

    // ---- llvm.mlir.constant ------------------------------------------------
    if (name_eq(name, "llvm.mlir.constant")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_TypeHandle  rt = MLIR_GetValueType(r);
        uint8_t vt = wasm_vt(F->ctx, rt);
        if (vt == 0) return false;
        MLIR_AttributeHandle va = find_attr(op, "value");
        if (va == MLIR_INVALID_HANDLE) return false;
        if (MLIR_GetAttributeKind(va) != MLIR_ATTR_KIND_INTEGER) return false;
        int64_t iv = MLIR_GetAttributeInteger(va);
        uint32_t li = local_alloc(F, vt);
        vmap_set(F, r, li);
        if (vt == VT_I32) {
            buf_putc(F->body, 0x41);  // i32.const
            leb_s(F->body, (int32_t)iv);
        } else if (vt == VT_I64) {
            buf_putc(F->body, 0x42);  // i64.const
            leb_s(F->body, iv);
        } else {
            return false;
        }
        buf_putc(F->body, 0x21);      // local.set
        leb_u(F->body, li);
        return true;
    }

    // ---- llvm.alloca -------------------------------------------------------
    // Result is a pointer (i32) into the shadow-stack frame.
    if (name_eq(name, "llvm.alloca")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        // Frame offset was assigned in the pre-walk.
        uint32_t off;
        if (!amap_get(F, r, &off)) return false;
        uint32_t li = local_alloc(F, VT_I32);
        vmap_set(F, r, li);
        // local.set li := sp_local + off
        buf_putc(F->body, 0x20); leb_u(F->body, F->sp_local);
        buf_putc(F->body, 0x41); leb_s(F->body, (int32_t)off);
        buf_putc(F->body, 0x6a);  // i32.add
        buf_putc(F->body, 0x21); leb_u(F->body, li);
        return true;
    }

    // ---- llvm.store --------------------------------------------------------
    if (name_eq(name, "llvm.store")) {
        if (MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 1);
        MLIR_TypeHandle  vt_ty = MLIR_GetValueType(val);
        uint8_t vt = wasm_vt(F->ctx, vt_ty);
        unsigned sz = type_size_bytes(F->ctx, vt_ty);
        if (vt == 0 || sz == 0) return false;
        if (!emit_use(F, ptr)) return false;
        if (!emit_use(F, val)) return false;
        // align (log2) + offset(=0)
        uint32_t align_log2;
        uint8_t opc;
        if (vt == VT_I32) {
            if (sz == 4) { opc = 0x36; align_log2 = 2; }
            else if (sz == 2) { opc = 0x3b; align_log2 = 1; }       // i32.store16
            else /* sz==1 */  { opc = 0x3a; align_log2 = 0; }       // i32.store8
        } else if (vt == VT_I64) {
            opc = 0x37; align_log2 = 3;                              // i64.store
        } else if (vt == VT_F32) {
            opc = 0x38; align_log2 = 2;
        } else if (vt == VT_F64) {
            opc = 0x39; align_log2 = 3;
        } else {
            return false;
        }
        buf_putc(F->body, opc);
        leb_u(F->body, align_log2);
        leb_u(F->body, 0);
        return true;
    }

    // ---- llvm.load ---------------------------------------------------------
    if (name_eq(name, "llvm.load")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 0);
        MLIR_TypeHandle  rt = MLIR_GetValueType(r);
        uint8_t vt = wasm_vt(F->ctx, rt);
        unsigned sz = type_size_bytes(F->ctx, rt);
        if (vt == 0 || sz == 0) return false;
        if (!emit_use(F, ptr)) return false;
        uint32_t align_log2;
        uint8_t opc;
        if (vt == VT_I32) {
            if (sz == 4)      { opc = 0x28; align_log2 = 2; }     // i32.load
            else if (sz == 2) { opc = 0x2e; align_log2 = 1; }     // i32.load16_s
            else /* sz==1 */  { opc = 0x2c; align_log2 = 0; }     // i32.load8_s
        } else if (vt == VT_I64) {
            opc = 0x29; align_log2 = 3;
        } else if (vt == VT_F32) {
            opc = 0x2a; align_log2 = 2;
        } else if (vt == VT_F64) {
            opc = 0x2b; align_log2 = 3;
        } else {
            return false;
        }
        buf_putc(F->body, opc);
        leb_u(F->body, align_log2);
        leb_u(F->body, 0);
        uint32_t li = local_alloc(F, vt);
        vmap_set(F, r, li);
        buf_putc(F->body, 0x21);  // local.set
        leb_u(F->body, li);
        return true;
    }

    // ---- llvm.sext ---------------------------------------------------------
    if (name_eq(name, "llvm.sext")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (in_vt == 0 || out_vt == 0) return false;
        if (!emit_use(F, s)) return false;
        if (in_vt == VT_I32 && out_vt == VT_I64) {
            buf_putc(F->body, 0xac);   // i64.extend_i32_s
        } else if (in_vt == VT_I32 && out_vt == VT_I32) {
            // small-to-i32 sign-extend via dedicated opcodes (sign-ext
            // proposal). For our scope we only need 32->64, but we leave
            // a small i32->i32 case as a no-op.
        } else {
            return false;
        }
        uint32_t li = local_alloc(F, out_vt);
        vmap_set(F, r, li);
        buf_putc(F->body, 0x21);
        leb_u(F->body, li);
        return true;
    }

    // ---- llvm.return -------------------------------------------------------
    if (name_eq(name, "llvm.return")) {
        // Restore shadow stack pointer (if any).
        size_t no = MLIR_GetOpNumOperands(op);
        // Push return value first so it sits on top after epilogue.
        if (no == 1) {
            if (!emit_use(F, MLIR_GetOpOperand(op, 0))) return false;
        } else if (no != 0) {
            return false;
        }
        if (F->frame_size > 0) {
            // global.set __stack_pointer (sp_local + frame_size)
            buf_putc(F->body, 0x20); leb_u(F->body, F->sp_local);
            buf_putc(F->body, 0x41); leb_s(F->body, (int32_t)F->frame_size);
            buf_putc(F->body, 0x6a);  // i32.add
            emit_sp_global_set(F);
        }
        buf_putc(F->body, 0x0f);  // return
        return true;
    }

    // ---- llvm.call (direct, fixed-arity) -----------------------------------
    if (name_eq(name, "llvm.call")) {
        MLIR_AttributeHandle callee = find_attr(op, "callee");
        if (callee == MLIR_INVALID_HANDLE) return false;       // indirect not supported
        MLIR_AttributeHandle var_callee_type = find_attr(op, "var_callee_type");
        if (var_callee_type != MLIR_INVALID_HANDLE) {
            MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(var_callee_type);
            if (MLIR_GetTypeFunctionIsVarArg(fty)) return false; // variadic not supported
        }
        string nm = MLIR_GetAttributeAsString(F->ctx, callee);
        // The attribute prints as `@name` for SymbolRefAttr.
        const char *cname = nm.str; size_t cn = nm.size;
        if (cn > 0 && cname[0] == '@') { cname++; cn--; }
        char *cstr = xstrdupn(cname, cn);
        int idx = func_find(&F->M->funcs, cstr);
        free(cstr);
        if (idx < 0) {
            fprintf(stderr, "wasm-emit: call to unknown function\n");
            return false;
        }
        Func *callee_func = &F->M->funcs.e[idx];

        // Push args in order.
        size_t no = MLIR_GetOpNumOperands(op);
        for (size_t i = 0; i < no; i++) {
            if (!emit_use(F, MLIR_GetOpOperand(op, i))) return false;
        }
        // call <funcidx, padded LEB, relocated>
        buf_putc(F->body, 0x10);
        uint32_t off = (uint32_t)F->body->len;
        leb_u_pad5(F->body, callee_func->func_index);
        func_add_reloc(F->F, R_WASM_FUNCTION_INDEX_LEB, off,
                       callee_func->sym_index);

        // Bind result, if any.
        size_t nr = MLIR_GetOpNumResults(op);
        if (nr == 1) {
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            if (vt == 0) return false;
            uint32_t li = local_alloc(F, vt);
            vmap_set(F, r, li);
            buf_putc(F->body, 0x21);
            leb_u(F->body, li);
        } else if (nr != 0) {
            return false;
        }
        return true;
    }

    // Anything else: unsupported in this scaffold.
    fprintf(stderr, "wasm-emit: unsupported op '%.*s'\n",
            (int)name.size, name.str);
    return false;
}

// =============================================================================
// Function emitter.
// =============================================================================

// Pre-walk: assign frame offsets for `llvm.alloca` results. Returns
// false if anything in the body is unsupported (e.g., multiple blocks).
static bool prewalk_function(FnCtx *F, MLIR_BlockHandle entry) {
    // Walk ops linearly; reject anything that introduces extra blocks
    // (we don't yet support CFG).
    size_t nops = MLIR_GetBlockNumOps(entry);
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(entry, i);
        string n = MLIR_GetOpName(op);
        if (name_eq(n, "llvm.br") || name_eq(n, "llvm.cond_br") ||
            name_eq(n, "llvm.switch")) {
            fprintf(stderr,
                    "wasm-emit: control flow not supported in scaffold\n");
            return false;
        }
        if (name_eq(n, "llvm.alloca")) {
            // Determine size from "elem_type" attribute (preferred) or
            // result type (!llvm.ptr); fall back to the operand which
            // is the element count.
            MLIR_TypeHandle et = MLIR_INVALID_HANDLE;
            MLIR_AttributeHandle ea = find_attr(op, "elem_type");
            if (ea != MLIR_INVALID_HANDLE) et = MLIR_GetAttributeTypeValue(ea);
            unsigned esz = et != MLIR_INVALID_HANDLE
                               ? type_size_bytes(F->ctx, et) : 0;
            if (esz == 0) {
                fprintf(stderr,
                        "wasm-emit: alloca of unsupported element type\n");
                return false;
            }
            // Element count: if operand 0 is a constant integer, use it;
            // otherwise we don't yet support dynamic allocas.
            int64_t cnt = 1;
            if (MLIR_GetOpNumOperands(op) >= 1) {
                MLIR_ValueHandle co = MLIR_GetOpOperand(op, 0);
                MLIR_OpHandle    cd = MLIR_GetValueDefiningOp(co);
                if (cd == MLIR_INVALID_HANDLE) return false;
                if (!name_eq(MLIR_GetOpName(cd), "llvm.mlir.constant")) {
                    fprintf(stderr,
                            "wasm-emit: dynamic alloca size not supported\n");
                    return false;
                }
                MLIR_AttributeHandle va = find_attr(cd, "value");
                cnt = MLIR_GetAttributeInteger(va);
            }
            unsigned align = esz < 4 ? 4 : esz;
            // Align frame_size up.
            F->frame_size = (F->frame_size + align - 1) & ~(align - 1);
            uint32_t off = F->frame_size;
            F->frame_size += (uint32_t)(esz * cnt);
            amap_set(F, MLIR_GetOpResult(op, 0), off);
        }
    }
    // Final 16-byte alignment (matches clang).
    F->frame_size = (F->frame_size + 15) & ~15u;
    return true;
}

// Emit a single function body (for a defined `llvm.func`). Populates
// F->body. Caller is responsible for the surrounding section framing.
static bool emit_function_body(MLIR_Context *ctx, ModCtx *M, Func *func,
                               MLIR_OpHandle fn,
                               const uint8_t *param_types, size_t n_params) {
    FnCtx F;
    memset(&F, 0, sizeof F);
    F.ctx = ctx; F.M = M; F.F = func;
    F.body = &func->body;
    F.param_types = (uint8_t *)param_types;
    F.n_params = n_params;
    F.sp_local = 0xffffffff;

    MLIR_RegionHandle body = MLIR_GetOpRegion(fn, 0);
    MLIR_BlockHandle  entry = MLIR_GetRegionBlock(body, 0);

    // Bind parameter locals (indices 0..n_params-1).
    for (size_t i = 0; i < n_params; i++) {
        vmap_set(&F, MLIR_GetBlockArg(entry, i), (uint32_t)i);
    }

    if (!prewalk_function(&F, entry)) goto fail;

    // Reserve the SP local up-front if we need a frame.
    if (F.frame_size > 0) {
        F.sp_local = local_alloc(&F, VT_I32);
        // Emit prologue:
        //   global.get __stack_pointer
        //   i32.const frame_size
        //   i32.sub
        //   local.tee sp_local
        //   global.set __stack_pointer
        buf_putc(F.body, 0x23);                     // global.get
        uint32_t off1 = (uint32_t)F.body->len;
        leb_u_pad5(F.body, M->stack_pointer_global);
        func_add_reloc(F.F, R_WASM_GLOBAL_INDEX_LEB, off1, M->stack_pointer_sym);
        buf_putc(F.body, 0x41); leb_s(F.body, (int32_t)F.frame_size);
        buf_putc(F.body, 0x6b);                     // i32.sub
        buf_putc(F.body, 0x22); leb_u(F.body, F.sp_local);  // local.tee
        buf_putc(F.body, 0x24);                     // global.set
        uint32_t off2 = (uint32_t)F.body->len;
        leb_u_pad5(F.body, M->stack_pointer_global);
        func_add_reloc(F.F, R_WASM_GLOBAL_INDEX_LEB, off2, M->stack_pointer_sym);
    }

    size_t nops = MLIR_GetBlockNumOps(entry);
    for (size_t i = 0; i < nops; i++) {
        if (!emit_op(&F, MLIR_GetBlockOp(entry, i))) goto fail;
    }
    // The terminator (llvm.return) emits the `return` opcode; the
    // function body still needs an `end` byte.
    buf_putc(F.body, 0x0b);

    // Now emit the locals prelude into a scratch buf and then prepend
    // it together with the locals count. The `body` we built above is
    // just instructions; we need to wrap it as:
    //   locals_count, [(count, type)], <body bytes>, 0x0b end
    // Group consecutive same-type locals.
    Buf prelude = {0};
    {
        size_t i = 0;
        size_t groups = 0;
        // count groups
        while (i < F.n_locals) {
            uint8_t t = F.local_types[i];
            size_t j = i + 1;
            while (j < F.n_locals && F.local_types[j] == t) j++;
            groups++;
            i = j;
        }
        leb_u(&prelude, groups);
        i = 0;
        while (i < F.n_locals) {
            uint8_t t = F.local_types[i];
            size_t j = i + 1;
            while (j < F.n_locals && F.local_types[j] == t) j++;
            leb_u(&prelude, (uint32_t)(j - i));
            buf_putc(&prelude, t);
            i = j;
        }
    }

    // Splice prelude in front of the body and shift reloc offsets.
    Buf out = {0};
    buf_append(&out, prelude.data, prelude.len);
    buf_append(&out, F.body->data, F.body->len);
    for (size_t i = 0; i < func->n_relocs; i++) {
        func->relocs[i].body_offset += (uint32_t)prelude.len;
    }
    buf_free(&prelude);
    buf_free(F.body);
    *F.body = out;

    free(F.local_types);
    free(F.vmap);
    free(F.amap);
    return true;
fail:
    free(F.local_types);
    free(F.vmap);
    free(F.amap);
    return false;
}

// =============================================================================
// Module walker: collect funcs / sigs.
// =============================================================================

// Build the param + result type vector for a `llvm.func`.
static bool sig_for_func(MLIR_Context *ctx, MLIR_OpHandle fn,
                         uint8_t **out_p, size_t *out_np,
                         uint8_t **out_r, size_t *out_nr) {
    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    if (ftya == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
    size_t ni = MLIR_GetTypeFunctionNumInputs(fty);
    size_t no = MLIR_GetTypeFunctionNumResults(fty);
    uint8_t *p = (uint8_t *)malloc(ni ? ni : 1);
    for (size_t i = 0; i < ni; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionInput(fty, i));
        if (v == 0) { free(p); return false; }
        p[i] = v;
    }
    uint8_t *r = (uint8_t *)malloc(no ? no : 1);
    for (size_t i = 0; i < no; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionResult(fty, i));
        if (v == 0) { free(p); free(r); return false; }
        r[i] = v;
    }
    *out_p = p; *out_np = ni;
    *out_r = r; *out_nr = no;
    return true;
}

// Translate the module. Returns true on success.
static bool translate_module(MLIR_Context *ctx, MLIR_OpHandle module,
                             ModCtx *M) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    // The stack-pointer global is import 0 -> global index 0. We always
    // declare it; even functions with no allocas reference it via the
    // prologue/epilogue (we only add the prologue if frame_size>0, but
    // declare the symbol unconditionally so the linker is happy).
    M->stack_pointer_global = 0;

    // Pass 1: register all imports (declared funcs) FIRST so they get
    // the low function indices. Then defs.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, op, &p, &np, &r, &nr)) return false;
        uint32_t sigi = sig_intern(&M->sigs, p, np, r, nr);
        free(p); free(r);
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        char *nm = xstrdup_str(MLIR_GetAttributeString(sa));
        if (func_find(&M->funcs, nm) < 0) {
            uint32_t fi = func_add(&M->funcs, nm, sigi, /*imported*/true);
            M->funcs.e[fi].func_index = fi;
        }
        free(nm);
    }
    uint32_t n_imports = (uint32_t)M->funcs.n;
    (void)n_imports;

    // Pass 2: register all defined functions (build sigs + emit bodies).
    // Sym indices must be known to op emitters since `call`/`global.*`
    // produce reloc records inline; assign once both passes have laid
    // down the FuncTab.
    // First: collect defs and assign indices.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (!has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, op, &p, &np, &r, &nr)) return false;
        uint32_t sigi = sig_intern(&M->sigs, p, np, r, nr);
        free(p); free(r);
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        string sym = MLIR_GetAttributeString(sa);
        const char *out_name;
        char *alloc_name = NULL;
        if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) {
            out_name = "__original_main";
        } else {
            alloc_name = xstrdup_str(sym);
            out_name = alloc_name;
        }
        uint32_t fi = func_add(&M->funcs, out_name, sigi, /*imported*/false);
        M->funcs.e[fi].func_index = fi;
        M->funcs.e[fi].exported   =
            (sym.size == 4 && memcmp(sym.str, "main", 4) == 0);
        if (alloc_name) free(alloc_name);
    }

    // Now sym_index values are stable.
    assign_sym_indices(M);

    // Pass 3: emit each defined function's body.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (!has_body) continue;
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        string sym = MLIR_GetAttributeString(sa);
        const char *out_name = (sym.size == 4 && memcmp(sym.str, "main", 4) == 0)
                                   ? "__original_main"
                                   : NULL;
        char *alloc_name = NULL;
        if (!out_name) { alloc_name = xstrdup_str(sym); out_name = alloc_name; }
        int fi = func_find(&M->funcs, out_name);
        if (alloc_name) free(alloc_name);
        if (fi < 0) return false;

        MLIR_AttributeHandle ftya = find_attr(op, "function_type");
        MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
        size_t ni = MLIR_GetTypeFunctionNumInputs(fty);
        uint8_t *params = (uint8_t *)malloc(ni ? ni : 1);
        for (size_t k = 0; k < ni; k++) {
            params[k] = wasm_vt(ctx, MLIR_GetTypeFunctionInput(fty, k));
        }
        bool ok = emit_function_body(ctx, M, &M->funcs.e[fi], op, params, ni);
        free(params);
        if (!ok) return false;
    }
    return true;
}

// =============================================================================
// Section serialization.
// =============================================================================

static void emit_section(Buf *out, uint8_t id, const Buf *payload) {
    buf_putc(out, id);
    leb_u(out, payload->len);
    buf_append(out, payload->data, payload->len);
}

static void emit_string(Buf *out, const char *s) {
    size_t n = strlen(s);
    leb_u(out, n);
    buf_append(out, s, n);
}

// Symbol table: returns the offset, in the final image, of the start of
// the function-index space symbol entries (used for reloc.CODE patching
// of `call` instructions). Also fills in M->funcs.e[i].sym_index and
// M->stack_pointer_sym.
static void build_linking_section(ModCtx *M, Buf *out_section) {
    // Header: "linking" name, version=2, then subsections.
    // We carry the name + "linking" subsection.
    Buf body = {0};
    leb_u(&body, LINK_VERSION);

    // Build SYMBOL_TABLE subsection in a scratch buffer.
    Buf sub = {0};

    // Count symbols: 1 global (__stack_pointer import) + one per function.
    uint32_t n_syms = 1 + (uint32_t)M->funcs.n;
    leb_u(&sub, n_syms);

    // Symbol 0: imported global __stack_pointer.
    // (sym_index already assigned by assign_sym_indices in translate_module.)
    buf_putc(&sub, SYM_GLOBAL);
    leb_u(&sub, SYMF_UNDEFINED);
    leb_u(&sub, M->stack_pointer_global);
    // No name field when UNDEFINED && !EXPLICIT_NAME.

    // Then all functions.
    for (size_t i = 0; i < M->funcs.n; i++) {
        Func *f = &M->funcs.e[i];
        buf_putc(&sub, SYM_FUNCTION);
        uint32_t flags = 0;
        if (f->imported) flags |= SYMF_UNDEFINED;
        leb_u(&sub, flags);
        leb_u(&sub, f->func_index);
        if (!(f->imported)) {
            // Defined symbol carries its name unconditionally.
            emit_string(&sub, f->name);
        }
        // For imports we omit the explicit name; the import entry's
        // module/field already names it.
    }

    buf_putc(&body, LINK_SUB_SYMBOL_TABLE);
    leb_u(&body, sub.len);
    buf_append(&body, sub.data, sub.len);
    buf_free(&sub);

    // Wrap as a custom section with name "linking".
    Buf payload = {0};
    emit_string(&payload, "linking");
    buf_append(&payload, body.data, body.len);
    buf_free(&body);

    emit_section(out_section, SEC_CUSTOM, &payload);
    buf_free(&payload);
}

// Emit reloc.CODE custom section. `code_payload_offset_to_func_body`:
// for each defined function, where in the CODE-section payload its
// body bytes (after the per-function size LEB) begin.
static void build_reloc_code_section(ModCtx *M, Buf *out_section,
                                     uint8_t code_section_index,
                                     const uint32_t *func_body_off) {
    // Collect all relocations across all defined functions.
    typedef struct { uint8_t type; uint32_t off; uint32_t sym; } R;
    R *all = NULL; size_t n_all = 0, c_all = 0;
    size_t k = 0;
    for (size_t i = 0; i < M->funcs.n; i++) {
        Func *f = &M->funcs.e[i];
        if (f->imported) continue;
        for (size_t j = 0; j < f->n_relocs; j++) {
            if (n_all == c_all) {
                c_all = c_all ? c_all * 2 : 8;
                all = realloc(all, c_all * sizeof(R));
            }
            all[n_all].type = f->relocs[j].type;
            all[n_all].off  = func_body_off[k] + f->relocs[j].body_offset;
            all[n_all].sym  = f->relocs[j].sym_idx;
            n_all++;
        }
        k++;
    }

    Buf body = {0};
    leb_u(&body, code_section_index);
    leb_u(&body, n_all);
    for (size_t i = 0; i < n_all; i++) {
        buf_putc(&body, all[i].type);
        leb_u(&body, all[i].off);
        leb_u(&body, all[i].sym);
        // No addend field for R_WASM_FUNCTION_INDEX_LEB.
    }
    free(all);

    Buf payload = {0};
    emit_string(&payload, "reloc.CODE");
    buf_append(&payload, body.data, body.len);
    buf_free(&body);
    emit_section(out_section, SEC_CUSTOM, &payload);
    buf_free(&payload);
}

// =============================================================================
// Top-level entry.
// =============================================================================

string mlir_translate_to_wasm_native(MLIR_Context *ctx, MLIR_OpHandle module) {
    string fail = {0};
    ModCtx M;
    memset(&M, 0, sizeof M);

    if (!translate_module(ctx, module, &M)) {
        sigtab_free(&M.sigs);
        functab_free(&M.funcs);
        return fail;
    }

    // ---- Build sections ---------------------------------------------------

    // TYPE
    Buf type_payload = {0};
    leb_u(&type_payload, M.sigs.n);
    for (size_t i = 0; i < M.sigs.n; i++) {
        buf_putc(&type_payload, 0x60);  // functype tag
        leb_u(&type_payload, M.sigs.e[i].nparams);
        buf_append(&type_payload, M.sigs.e[i].params, M.sigs.e[i].nparams);
        leb_u(&type_payload, M.sigs.e[i].nresults);
        buf_append(&type_payload, M.sigs.e[i].results, M.sigs.e[i].nresults);
    }

    // IMPORT
    // Imports = env.__linear_memory + env.__stack_pointer + each undefined func.
    Buf import_payload = {0};
    {
        uint32_t n_func_imports = 0;
        for (size_t i = 0; i < M.funcs.n; i++) if (M.funcs.e[i].imported) n_func_imports++;
        uint32_t n_imports_total = 2 + n_func_imports;
        leb_u(&import_payload, n_imports_total);
        // env.__linear_memory : memory 0 0
        emit_string(&import_payload, "env");
        emit_string(&import_payload, "__linear_memory");
        buf_putc(&import_payload, IMP_MEMORY);
        buf_putc(&import_payload, 0x00);  // limits flags = 0 (no max)
        leb_u(&import_payload, 0);        // initial pages = 0 (the linker sets actual)
        // env.__stack_pointer : global i32 mut
        emit_string(&import_payload, "env");
        emit_string(&import_payload, "__stack_pointer");
        buf_putc(&import_payload, IMP_GLOBAL);
        buf_putc(&import_payload, VT_I32);
        buf_putc(&import_payload, 0x01);  // mutable
        for (size_t i = 0; i < M.funcs.n; i++) {
            Func *f = &M.funcs.e[i];
            if (!f->imported) continue;
            emit_string(&import_payload, "env");
            emit_string(&import_payload, f->name);
            buf_putc(&import_payload, IMP_FUNC);
            leb_u(&import_payload, f->sig);
        }
    }

    // FUNCTION (defined func sig indices)
    Buf function_payload = {0};
    {
        uint32_t n_def = 0;
        for (size_t i = 0; i < M.funcs.n; i++) if (!M.funcs.e[i].imported) n_def++;
        leb_u(&function_payload, n_def);
        for (size_t i = 0; i < M.funcs.n; i++) {
            if (M.funcs.e[i].imported) continue;
            leb_u(&function_payload, M.funcs.e[i].sig);
        }
    }

    // CODE — we need per-function-body offsets within the payload
    // (after the func count LEB, after each individual size LEB) to
    // patch relocations.
    Buf code_payload = {0};
    uint32_t *body_offsets = NULL;
    {
        uint32_t n_def = 0;
        for (size_t i = 0; i < M.funcs.n; i++) if (!M.funcs.e[i].imported) n_def++;
        body_offsets = (uint32_t *)calloc(n_def ? n_def : 1, sizeof(uint32_t));
        leb_u(&code_payload, n_def);
        size_t k = 0;
        for (size_t i = 0; i < M.funcs.n; i++) {
            Func *f = &M.funcs.e[i];
            if (f->imported) continue;
            leb_u(&code_payload, f->body.len);
            body_offsets[k++] = (uint32_t)code_payload.len;
            buf_append(&code_payload, f->body.data, f->body.len);
        }
    }

    // ---- Concatenate into final image ------------------------------------
    Buf img = {0};
    static const uint8_t magic[8] = {0,'a','s','m', 1,0,0,0};
    buf_append(&img, magic, 8);
    emit_section(&img, SEC_TYPE,     &type_payload);
    emit_section(&img, SEC_IMPORT,   &import_payload);
    emit_section(&img, SEC_FUNCTION, &function_payload);
    uint8_t code_section_index = 3;  // TYPE=0, IMPORT=1, FUNCTION=2, CODE=3
    emit_section(&img, SEC_CODE,     &code_payload);
    build_linking_section(&M, &img);
    build_reloc_code_section(&M, &img, code_section_index, body_offsets);

    // ---- Cleanup + copy into arena ---------------------------------------
    free(body_offsets);
    buf_free(&type_payload);
    buf_free(&import_payload);
    buf_free(&function_payload);
    buf_free(&code_payload);

    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char  *out   = (char *)arena_alloc(arena, img.len);
    memcpy(out, img.data, img.len);
    string r;
    r.str  = out;
    r.size = img.len;
    buf_free(&img);

    sigtab_free(&M.sigs);
    functab_free(&M.funcs);
    return r;
}
