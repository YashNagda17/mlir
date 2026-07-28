// Stage 1 of the native LLVM->WASM pipeline:
// LLVM-dialect builtin.module --(walk)--> wasmssa-form `builtin.module`.
//
// Each `llvm.func` / `llvm.mlir.global` is lowered directly to `wasmssa.*`
// MLIR ops appended into the output module body. Inside a function, each
// emit-helper / inline call site builds a stack-local `wasmssa_op_t`
// describing one op and hands it to `commit_op`, which materializes the
// matching MLIR op into the function's body block and returns the
// MLIR_ValueHandle of its result. The per-function `vmap` then keys
// LLVM-side operand values to the wasmssa-side ValueHandle that supplies
// them; subsequent lookups thread those handles through `o.operands` to
// stitch up the wasmssa IR.
//
// No module-level or per-function scratch buffer is retained: ops flow
// straight from lower_op into the MLIR module under construction.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"
#include "mlir_llvm_to_wasmssa.h"
#include <base/vector.h>

#include <base/arena.h>
#include <base/string.h>

// =============================================================================
// Private working representation for this stage. `wasmssa_op_t` is a
// stack-local descriptor handed to `commit_op` once per emitted op. It
// never escapes this file and owns none of its pointer fields:
//   - `operands` points at a caller-provided buffer (stack array for
//     fixed-arity ops, arena allocation for variadic call/call_indirect);
//   - `call_target` is a non-owning view into either an MLIR attribute
//     string or a caller-owned buffer;
//   - `sig_params` / `sig_results` are non-owning views into a function
//     signature buffer.
// =============================================================================
typedef struct {
    MLIR_OpType type;
    uint8_t     valtype;

    int64_t  i_const;
    uint32_t global_idx;
    uint32_t memory_offset;
    uint32_t memory_align_log2;
    uint32_t mem_size_bytes;
    string   call_target;
    uint8_t  wasm_opcode;

    const uint8_t *sig_params;
    const uint8_t *sig_results;
    size_t         n_sig_params, n_sig_results;

    const MLIR_ValueHandle *operands;
    int                     n_operands;

    bool has_result;
} wasmssa_op_t;

// Module-level emit context: the output MLIR module's body block. The
// already-emitted `wasmssa.func` / `wasmssa.import_func` ops in that
// block are the source of truth for `is_function_symbol`, so addressof
// references can distinguish between data globals and functions.
typedef struct {
    MLIR_Context    *ctx;
    Arena           *arena;
    MLIR_BlockHandle body;
} ModCtx;

// =============================================================================
// String / type helpers (mirror the old single-stage translator).
// =============================================================================
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}
static MLIR_AttributeHandle find_attr(MLIR_OpHandle op, const char *name) {
    // First try by-name, which on the upstream backend walks both the
    // discardable attribute dictionary AND the typed property storage
    // (ODS-defined inherent attrs like scf.index_switch's `cases`,
    // arith.cmpi's `predicate`, llvm.func's `function_type`, ...).
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    if (a != MLIR_INVALID_HANDLE) return a;
    // Fall back to the indexed attribute list. The two implementations
    // agree on what's "user-set" for ops constructed via MLIR_CreateOp,
    // so this branch matters mainly for operations created by upstream
    // pass internals where the by-name lookup may also miss.
    size_t n = MLIR_GetOpNumAttributes(op);
    for (size_t i = 0; i < n; i++) {
        MLIR_AttributeHandle ai = MLIR_GetOpAttribute(op, i);
        string an = MLIR_GetAttributeName(ai);
        if (name_eq(an, name)) return ai;
    }
    return MLIR_INVALID_HANDLE;
}

// MLIR LLVM-dialect type -> WT_* value-type byte. Returns 0 if the type
// is not a primitive scalar.
static uint8_t wasm_vt(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return WT_I32;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return WT_I32;
    if (s.size == 5 && memcmp(s.str, "index", 5) == 0) return WT_I32;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return WT_F32;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return WT_F64;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8 || w == 16 || w == 32) return WT_I32;
        if (w == 64) return WT_I64;
    }
    return 0;
}

// Returns the integer bit width of `ty` (1/8/16/32/64) or 0 if `ty`
// is not an `iN` integer type.
static int int_bits(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else return 0;
        }
        return w;
    }
    return 0;
}

static unsigned type_size_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty);
static unsigned type_align_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty);

// Round x up to the nearest multiple of `align` (a power of two).
static unsigned align_up(unsigned x, unsigned align) {
    return (x + align - 1) & ~(align - 1);
}

// Sum of (padded) field sizes for an LLVM struct type, with the natural
// alignment that the LLVM data layout uses on wasm32 (each field aligned
// to its own alignment; trailing padding to the struct's max-field alignment).
static unsigned struct_size_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
    unsigned off = 0, max_align = 1;
    for (size_t i = 0; i < nf; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ty, i);
        unsigned fsz = type_size_bytes(ctx, ft);
        unsigned fal = type_align_bytes(ctx, ft);
        if (fsz == 0 || fal == 0) return 0;
        off = align_up(off, fal);
        off += fsz;
        if (fal > max_align) max_align = fal;
    }
    return align_up(off, max_align);
}
static unsigned struct_align_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
    unsigned ma = 1;
    for (size_t i = 0; i < nf; i++) {
        unsigned fa = type_align_bytes(ctx, MLIR_GetTypeLLVMStructField(ty, i));
        if (fa > ma) ma = fa;
    }
    return ma;
}
// Byte offset of the i-th field within its containing struct.
static unsigned struct_field_offset(MLIR_Context *ctx, MLIR_TypeHandle sty,
                                    size_t fld_idx) {
    unsigned off = 0;
    for (size_t i = 0; i <= fld_idx; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(sty, i);
        unsigned fal = type_align_bytes(ctx, ft);
        off = align_up(off, fal);
        if (i == fld_idx) return off;
        off += type_size_bytes(ctx, ft);
    }
    return off;
}

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
    if (MLIR_IsTypeLLVMArray(ty)) {
        unsigned esz = type_size_bytes(ctx, MLIR_GetTypeLLVMArrayElement(ty));
        if (esz == 0) return 0;
        return esz * (unsigned)MLIR_GetTypeLLVMArrayNumElements(ty);
    }
    if (MLIR_IsTypeLLVMStruct(ty)) {
        return struct_size_bytes(ctx, ty);
    }
    return 0;
}

static unsigned type_align_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_IsTypeLLVMArray(ty)) {
        return type_align_bytes(ctx, MLIR_GetTypeLLVMArrayElement(ty));
    }
    if (MLIR_IsTypeLLVMStruct(ty)) {
        return struct_align_bytes(ctx, ty);
    }
    unsigned sz = type_size_bytes(ctx, ty);
    return sz ? sz : 1;
}

// =============================================================================
// Per-function lowering state.
// =============================================================================
typedef struct { uintptr_t key; MLIR_ValueHandle val; } VMapEntry;
typedef struct { uintptr_t key; uint32_t off; } AMapEntry;

DEFINE_VECTOR_FOR_TYPE(uint8_t,    VecU8)

typedef struct {
    MLIR_Context   *ctx;
    Arena          *arena;   // for hex-encoded attribute strings
    ModCtx         *mod;     // parent module emit context (for func-name lookups)
    size_t          n_params;

    // Direct-MLIR emission state. body_block accumulates wasmssa.* ops.
    MLIR_BlockHandle  body_block;

    // Map LLVM-dialect MLIR_ValueHandle -> wasmssa MLIR_ValueHandle, as an
    // open-addressing hash table (key 0 == empty; source value handles are
    // always non-zero). Replaces an earlier linear-scan vector that made
    // lowering O(n^2) in the number of values per function.
    VMapEntry *vmap;
    size_t     vmap_cap, vmap_n;   // cap is 0 or a power of two

    // Map MLIR_ValueHandle (alloca result) -> shadow-stack frame offset.
    AMapEntry *amap;
    size_t     amap_cap, amap_n;

    uint32_t  frame_size;
    MLIR_ValueHandle  sp_value;        // post-decrement SP, or MLIR_INVALID_HANDLE

    // Variadic ABI support.
    MLIR_ValueHandle  va_list_value;   // hidden trailing va_list i32 param, or MLIR_INVALID_HANDLE
    uint32_t  va_buf_size;        // max bytes needed for variadic call buffer
    uint32_t  va_buf_offset;      // offset within shadow frame for the variadic buffer
} FnCtx;

// Open-addressing hash maps keyed by MLIR value handles. The handle is never
// zero, so key==0 marks an empty slot. Lookups are key-only (never iterated in
// an output-affecting way), and the 32-bit-safe mix below is identical on the
// wasm32 self-host (uintptr_t == 32-bit) and the 64-bit host, so the emitted
// module stays deterministic / bit-identical across the self-host cycle.
static size_t map_hash(uintptr_t k) {
    size_t h = (size_t)k;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h;
}

static void vmap_grow(FnCtx *F) {
    size_t ncap = F->vmap_cap ? F->vmap_cap * 2 : 32;
    VMapEntry *nt = (VMapEntry *)arena_alloc(F->arena, ncap * sizeof(VMapEntry));
    memset(nt, 0, ncap * sizeof(VMapEntry));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < F->vmap_cap; i++) {
        if (F->vmap[i].key == 0) continue;
        size_t j = map_hash(F->vmap[i].key) & mask;
        while (nt[j].key != 0) j = (j + 1) & mask;
        nt[j] = F->vmap[i];
    }
    F->vmap = nt;
    F->vmap_cap = ncap;
}
static void vmap_set(FnCtx *F, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    // Grow at 75% load to keep probe chains short.
    if ((F->vmap_n + 1) * 4 >= F->vmap_cap * 3) vmap_grow(F);
    size_t mask = F->vmap_cap - 1;
    size_t i = map_hash((uintptr_t)k) & mask;
    while (F->vmap[i].key != 0) {
        if (F->vmap[i].key == (uintptr_t)k) return;  // first insert wins
        i = (i + 1) & mask;
    }
    F->vmap[i].key = (uintptr_t)k;
    F->vmap[i].val = v;
    F->vmap_n++;
}
static int vmap_get(FnCtx *F, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    if (F->vmap_cap == 0) return 0;
    size_t mask = F->vmap_cap - 1;
    size_t i = map_hash((uintptr_t)k) & mask;
    while (F->vmap[i].key != 0) {
        if (F->vmap[i].key == (uintptr_t)k) { *out = F->vmap[i].val; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}
static void amap_grow(FnCtx *F) {
    size_t ncap = F->amap_cap ? F->amap_cap * 2 : 16;
    AMapEntry *nt = (AMapEntry *)arena_alloc(F->arena, ncap * sizeof(AMapEntry));
    memset(nt, 0, ncap * sizeof(AMapEntry));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < F->amap_cap; i++) {
        if (F->amap[i].key == 0) continue;
        size_t j = map_hash(F->amap[i].key) & mask;
        while (nt[j].key != 0) j = (j + 1) & mask;
        nt[j] = F->amap[i];
    }
    F->amap = nt;
    F->amap_cap = ncap;
}
static void amap_set(FnCtx *F, MLIR_ValueHandle v, uint32_t off) {
    if ((F->amap_n + 1) * 4 >= F->amap_cap * 3) amap_grow(F);
    size_t mask = F->amap_cap - 1;
    size_t i = map_hash((uintptr_t)v) & mask;
    while (F->amap[i].key != 0) {
        if (F->amap[i].key == (uintptr_t)v) return;  // first insert wins
        i = (i + 1) & mask;
    }
    F->amap[i].key = (uintptr_t)v;
    F->amap[i].off = off;
    F->amap_n++;
}
static int amap_get(FnCtx *F, MLIR_ValueHandle v, uint32_t *out) {
    if (F->amap_cap == 0) return 0;
    size_t mask = F->amap_cap - 1;
    size_t i = map_hash((uintptr_t)v) & mask;
    while (F->amap[i].key != 0) {
        if (F->amap[i].key == (uintptr_t)v) { *out = F->amap[i].off; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

// Forward decls used by commit_op (defined further down with the other
// MLIR-emit helpers).
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt);
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v);
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v);
static MLIR_AttributeHandle attr_s_cstr(MLIR_Context *ctx, const char *name, const char *v);
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v);
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen);
static MLIR_AttributeHandle attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       const char *name,
                                       const uint8_t *p, size_t n);
static MLIR_OpHandle make_op(MLIR_Context *ctx, MLIR_OpType type,
                             MLIR_AttributeHandle *attrs, size_t n_attrs,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_RegionHandle *regions, size_t n_regions,
                             uint8_t result_vt,
                             MLIR_ValueHandle *out_result);

// Materialize a wasmssa op directly into the function's MLIR body block.
// Returns the op's MLIR result value (MLIR_INVALID_HANDLE for ops with no
// result). The op's operands[] reference earlier-produced MLIR values.
static MLIR_ValueHandle commit_op(FnCtx *F, wasmssa_op_t *o) {
    MLIR_Context *ctx = F->ctx;
    size_t n_ops = (size_t)(o->n_operands < 0 ? 0 : o->n_operands);
    MLIR_AttributeHandle as[8];
    size_t nas = 0;
    as[nas++] = attr_i32(ctx, "valtype", o->valtype);
    switch (o->type) {
    case OP_TYPE_WASMSSA_CONST:
        as[nas++] = attr_i64(ctx, "value", o->i_const);
        break;
    case OP_TYPE_WASMSSA_BINOP:
    case OP_TYPE_WASMSSA_UNOP:
        as[nas++] = attr_i32(ctx, "wasm_opcode", o->wasm_opcode);
        break;
    case OP_TYPE_WASMSSA_LOAD:
    case OP_TYPE_WASMSSA_STORE:
        as[nas++] = attr_i32(ctx, "memory_offset",     (int64_t)o->memory_offset);
        as[nas++] = attr_i32(ctx, "memory_align_log2", (int64_t)o->memory_align_log2);
        as[nas++] = attr_i32(ctx, "mem_size_bytes",    (int64_t)o->mem_size_bytes);
        break;
    case OP_TYPE_WASMSSA_GLOBAL_GET:
    case OP_TYPE_WASMSSA_GLOBAL_SET:
        as[nas++] = attr_i32(ctx, "global_idx", (int64_t)o->global_idx);
        break;
    case OP_TYPE_WASMSSA_CALL:
    case OP_TYPE_WASMSSA_ADDRESSOF:
    case OP_TYPE_WASMSSA_FUNC_ADDR:
        as[nas++] = attr_s(ctx, "target", o->call_target.str, o->call_target.size);
        break;
    case OP_TYPE_WASMSSA_CALL_INDIRECT:
        as[nas++] = attr_s_hex(ctx, F->arena, "sig_params",
                               o->sig_params, o->n_sig_params);
        as[nas++] = attr_s_hex(ctx, F->arena, "sig_results",
                               o->sig_results, o->n_sig_results);
        break;
    default: break;
    }
    MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
    MLIR_OpHandle mop = make_op(ctx, o->type, as, nas,
                                (MLIR_ValueHandle *)o->operands, n_ops, NULL, 0,
                                o->has_result ? o->valtype : 0, &res);
    MLIR_AppendBlockOp(ctx, F->body_block, mop);
    return o->has_result ? res : MLIR_INVALID_HANDLE;
}

// Convenience: const-i32.
static MLIR_ValueHandle emit_const_i32(FnCtx *F, int32_t v) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_CONST;
    o.valtype = WT_I32;
    o.i_const = v;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: i32 add of two ssa-def operands.
static MLIR_ValueHandle emit_add_i32(FnCtx *F, MLIR_ValueHandle lhs, MLIR_ValueHandle rhs) {
    MLIR_ValueHandle ops[2] = { lhs, rhs };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_ADD;
    o.valtype = WT_I32;
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: i32 sub.
static MLIR_ValueHandle emit_sub_i32(FnCtx *F, MLIR_ValueHandle lhs, MLIR_ValueHandle rhs) {
    MLIR_ValueHandle ops[2] = { lhs, rhs };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_SUB;
    o.valtype = WT_I32;
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: global_get of an i32 global.
static MLIR_ValueHandle emit_global_get(FnCtx *F, uint32_t gidx) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_GLOBAL_GET;
    o.valtype = WT_I32;
    o.global_idx = gidx;
    o.has_result = true;
    return commit_op(F, &o);
}
static void emit_global_set(FnCtx *F, uint32_t gidx, MLIR_ValueHandle valv) {
    MLIR_ValueHandle ops[1] = { valv };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_GLOBAL_SET;
    o.valtype = WT_I32;
    o.global_idx = gidx;
    o.n_operands = 1;
    o.operands = ops;
    (void)commit_op(F, &o);
}

// Convenience: i32 mul.
static MLIR_ValueHandle emit_mul_i32(FnCtx *F, MLIR_ValueHandle lhs, MLIR_ValueHandle rhs) {
    MLIR_ValueHandle ops[2] = { lhs, rhs };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_BINOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0x6c;  // i32.mul
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: i32.wrap_i64 (0xa7) — narrows an i64 SSA value to i32.
static MLIR_ValueHandle emit_wrap_i64_to_i32(FnCtx *F, MLIR_ValueHandle v) {
    MLIR_ValueHandle ops[1] = { v };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_UNOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0xa7;
    o.n_operands = 1;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}

// Parse a "array<i32: v0, v1, ...>" string into an arena-allocated int32_t
// vector; returns NULL on parse failure. Sets *n_out.
static int32_t *parse_dense_i32_array(Arena *arena, string s, size_t *n_out) {
    *n_out = 0;
    const char *p = s.str;
    const char *end = s.str + s.size;
    const char *pref = "array<i32";
    size_t plen = strlen(pref);
    if (s.size < plen || memcmp(p, pref, plen) != 0) return NULL;
    p += plen;
    // Allow "array<i32>" (empty) or "array<i32: ...>".
    while (p < end && *p == ' ') p++;
    if (p < end && *p == '>') {
        return (int32_t *)arena_alloc(arena, sizeof(int32_t));
    }
    if (p >= end || *p != ':') return NULL;
    p++;
    // Upper bound the element count by counting commas + 1.
    size_t cap = 1;
    for (const char *q = p; q < end && *q != '>'; q++) if (*q == ',') cap++;
    int32_t *vs = (int32_t *)arena_alloc(arena, cap * sizeof(int32_t));
    size_t n = 0;
    while (p < end && *p != '>') {
        while (p < end && (*p == ' ' || *p == ',')) p++;
        if (p >= end || *p == '>') break;
        bool neg = false;
        if (*p == '-') { neg = true; p++; }
        if (p >= end || *p < '0' || *p > '9') return NULL;
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') { v = v*10 + (*p - '0'); p++; }
        if (neg) v = -v;
        vs[n++] = (int32_t)v;
    }
    *n_out = n;
    return vs;
}

// Look up a name as a function symbol in the module under construction
// by walking m->body for `wasmssa.func` / `wasmssa.import_func` ops
// whose `sym_name` matches. Used to disambiguate addressof @sym between
// data globals and functions. The function currently being lowered is
// not in m->body yet (its wrapper op is appended after the body is
// walked), so recursive self-address-taking falls through to the
// addressof path; that's fine because main / defined funcs that take
// their own address are rare and not exercised by the test corpus.
static bool is_function_symbol(const ModCtx *m, const char *nm, size_t nlen) {
    size_t n = MLIR_GetBlockNumOps(m->body);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(m->body, i);
        MLIR_OpType ot = MLIR_GetOpType(op);
        if (ot != OP_TYPE_WASMSSA_FUNC && ot != OP_TYPE_WASMSSA_IMPORT_FUNC) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) continue;
        string s = MLIR_GetAttributeString(sa);
        if (s.size == nlen && memcmp(s.str, nm, nlen) == 0) return true;
    }
    return false;
}

// =============================================================================
// Per-op lowering.
// =============================================================================
static bool lower_op(FnCtx *F, MLIR_OpHandle op);
static bool va_call_layout(FnCtx *F, MLIR_OpHandle op, uint32_t *total_size,
                           size_t *n_fixed, uint32_t *out_offsets);
static bool lower_block(FnCtx *F, MLIR_BlockHandle blk);

static MLIR_ValueHandle emit_eqz(FnCtx *F, MLIR_ValueHandle v) {
    MLIR_ValueHandle ops[1] = { v };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_EQZ;
    o.valtype = WT_I32;
    o.n_operands = 1;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}

// =============================================================================
// Block-arg-mode helpers (new path).
// =============================================================================

// Multi-result variant of make_op: supports ops with N>=0 results and N>=0
// regions. Does NOT auto-add a "valtype" attr (the region-bearing ops and
// the block_return / unreachable / br-with-args ops don't have one).
static MLIR_OpHandle make_op_n(MLIR_Context *ctx, MLIR_OpType type,
                               MLIR_AttributeHandle *attrs, size_t n_attrs,
                               MLIR_ValueHandle *operands, size_t n_operands,
                               MLIR_RegionHandle *regions, size_t n_regions,
                               const uint8_t *result_vts, size_t n_results,
                               MLIR_ValueHandle *out_results /* size n_results, or NULL */) {
    MLIR_TypeHandle   res_tys_inline[16];
    MLIR_ValueHandle  res_vals_inline[16];
    MLIR_TypeHandle  *res_tys  = res_tys_inline;
    MLIR_ValueHandle *res_vals = res_vals_inline;
    if (n_results > 16) {
        res_tys  = (MLIR_TypeHandle  *)arena_alloc(ctx->arena, n_results * sizeof(MLIR_TypeHandle));
        res_vals = (MLIR_ValueHandle *)arena_alloc(ctx->arena, n_results * sizeof(MLIR_ValueHandle));
    }
    for (size_t i = 0; i < n_results; i++) {
        res_tys[i] = vt_to_type(ctx, result_vts[i]);
        res_vals[i] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, (uint32_t)i,
                                               res_tys[i], (string){0},
                                               MLIR_CreateLocationUnknown(ctx, (string){0}));
    }
    MLIR_OpHandle op = MLIR_CreateOp(ctx, type, op_type_to_string(type),
        attrs, n_attrs, res_tys, n_results, res_vals, n_results,
        operands, n_operands, regions, n_regions,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    if (out_results) {
        for (size_t i = 0; i < n_results; i++) out_results[i] = res_vals[i];
    }
    return op;
}

// Emit `wasmssa.block_return (vals...)` as the terminator of F->body_block.
static void emit_block_return(FnCtx *F, MLIR_ValueHandle *vals, size_t n) {
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BLOCK_RETURN, NULL, 0,
                                 vals, n, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Emit `wasmssa.unreachable` as the terminator of F->body_block.
static void emit_unreachable(FnCtx *F) {
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_UNREACHABLE, NULL, 0,
                                 NULL, 0, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Emit `wasmssa.br {depth=N} (vals...)`.
static void emit_br_args(FnCtx *F, uint32_t depth, MLIR_ValueHandle *vals, size_t n) {
    MLIR_AttributeHandle as[1];
    as[0] = attr_i32(F->ctx, "depth", (int64_t)depth);
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BR, as, 1,
                                 vals, n, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Emit a `wasmssa.br_if {depth=N} (cond)`. Block-arg design: br_if carries
// the cond only — value-carrying conditional branches use
// `wasmssa.if (cond) { br N (vals) }` instead.
static void emit_br_if(FnCtx *F, uint32_t depth, MLIR_ValueHandle cond) {
    MLIR_AttributeHandle as[1];
    as[0] = attr_i32(F->ctx, "depth", (int64_t)depth);
    MLIR_ValueHandle ops[1] = { cond };
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BR_IF, as, 1,
                                 ops, 1, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Build a fresh detached block with `n_args` block args of the given types.
// out_args (size n_args, may be NULL) receives the new block-arg values.
static MLIR_BlockHandle make_block_with_args(MLIR_Context *ctx,
                                             const uint8_t *arg_vts, size_t n_args,
                                             MLIR_ValueHandle *out_args) {
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    for (size_t i = 0; i < n_args; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, arg_vts[i]);
        MLIR_ValueHandle a = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
                                                     MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, blk, a);
        if (out_args) out_args[i] = a;
    }
    return blk;
}

// ---- scf.if ----------------------------------------------------------------
// Block-arg form: build a `wasmssa.if (%cond) : (R*)` op with two regions,
// each terminated by `wasmssa.block_return (yield_vals:R*)`. scf.if results
// map 1:1 to the wasmssa.if's results.
static bool lower_scf_if(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumOperands(op) != 1) return false;
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(op, 0);
    MLIR_ValueHandle cond_idx;
    if (!vmap_get(F, cond_v, &cond_idx)) return false;

    size_t n_res = MLIR_GetOpNumResults(op);
    uint8_t *res_vts = NULL;
    if (n_res) {
        res_vts = (uint8_t *)arena_alloc(F->arena, n_res);
        for (size_t i = 0; i < n_res; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(MLIR_GetOpResult(op, i)));
            if (vt == 0) return false;
            res_vts[i] = vt;
        }
    }

    if (MLIR_GetOpNumRegions(op) < 1) return false;
    MLIR_RegionHandle src_then = MLIR_GetOpRegion(op, 0);
    if (MLIR_GetRegionNumBlocks(src_then) != 1) return false;
    bool has_else = MLIR_GetOpNumRegions(op) >= 2 &&
                    MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 1)) > 0;
    if (n_res && !has_else) return false;  // result-bearing scf.if needs else

    // Save outer body; walk each scf.if region into a fresh wasmssa block.
    MLIR_BlockHandle saved = F->body_block;

    // --- then-region ---
    MLIR_BlockHandle then_blk = MLIR_CreateBlock(F->ctx);
    F->body_block = then_blk;
    bool ok = lower_block(F, MLIR_GetRegionBlock(src_then, 0));
    if (!ok) { F->body_block = saved; return false; }

    // --- else-region (may be empty) ---
    MLIR_BlockHandle else_blk = MLIR_INVALID_HANDLE;
    if (has_else) {
        else_blk = MLIR_CreateBlock(F->ctx);
        F->body_block = else_blk;
        ok = lower_block(F, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 1), 0));
        if (!ok) { F->body_block = saved; return false; }
    }
    F->body_block = saved;

    // Wrap blocks in regions and build the wasmssa.if op.
    MLIR_RegionHandle then_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, then_r, then_blk);
    MLIR_RegionHandle regs[2];
    regs[0] = then_r;
    size_t n_regs = 1;
    if (has_else) {
        MLIR_RegionHandle else_r = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, else_r, else_blk);
        regs[1] = else_r;
        n_regs = 2;
    }

    MLIR_AttributeHandle as[1];
    size_t nas = 0;
    if (n_res) {
        as[nas++] = attr_s_hex(F->ctx, F->arena, "result_types", res_vts, n_res);
    }

    MLIR_ValueHandle res_vals_inline[16];
    MLIR_ValueHandle *res_vals = n_res <= 16 ? res_vals_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    MLIR_ValueHandle cond_ops[1] = { cond_idx };
    MLIR_OpHandle ifop = make_op_n(F->ctx, OP_TYPE_WASMSSA_IF, as, nas,
                                   cond_ops, 1, regs, n_regs,
                                   res_vts, n_res, res_vals);
    MLIR_AppendBlockOp(F->ctx, F->body_block, ifop);

    for (size_t i = 0; i < n_res; i++) {
        vmap_set(F, MLIR_GetOpResult(op, i), res_vals[i]);
    }
    return true;
}

// ---- scf.while -------------------------------------------------------------
// General shape (with iter args / results):
//   %res:R = scf.while (%a:T = %init:T) : (T...) -> (R...) {
//     ^before(%a:T...): ...; scf.condition(%cond) %r:R...
//   } do {
//     ^after(%b:R...): ...; scf.yield %y:T...
//   }
//
// Lowered to (block-arg form):
//   %res:R* = wasmssa.block () : (R*) {
//     wasmssa.loop (%init:T*) : () {
//     ^entry(%t:T*):                                ; bound to %a in before
//       <before-body>
//       wasmssa.if (eqz_cond) : () {
//         wasmssa.br {depth=2} (r:R*)               ; exit outer block
//       }                                            ; (no else)
//       ; fall through if cond was true.
//       <after-body>                                 ; uses r as %b
//       wasmssa.br {depth=0} (y:T*)                  ; back-edge
//     }
//     wasmssa.unreachable
//   }
static bool lower_scf_while(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumRegions(op) < 2) return false;
    MLIR_RegionHandle before_r = MLIR_GetOpRegion(op, 0);
    MLIR_RegionHandle after_r  = MLIR_GetOpRegion(op, 1);
    if (MLIR_GetRegionNumBlocks(before_r) != 1) return false;
    if (MLIR_GetRegionNumBlocks(after_r)  != 1) return false;
    MLIR_BlockHandle before_b = MLIR_GetRegionBlock(before_r, 0);
    MLIR_BlockHandle after_b  = MLIR_GetRegionBlock(after_r, 0);

    size_t n_iter = MLIR_GetOpNumOperands(op);
    size_t n_res  = MLIR_GetOpNumResults(op);

    if (n_iter != MLIR_GetBlockNumArgs(before_b)) return false;
    if (n_res  != MLIR_GetBlockNumArgs(after_b))  return false;

    // Collect iter (T*) and result (R*) wasm valtypes.
    uint8_t *iter_vts = NULL;
    if (n_iter) {
        iter_vts = (uint8_t *)arena_alloc(F->arena, n_iter);
        for (size_t i = 0; i < n_iter; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(MLIR_GetOpOperand(op, i)));
            if (vt == 0) return false;
            iter_vts[i] = vt;
        }
    }
    uint8_t *res_vts = NULL;
    if (n_res) {
        res_vts = (uint8_t *)arena_alloc(F->arena, n_res);
        for (size_t i = 0; i < n_res; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(MLIR_GetOpResult(op, i)));
            if (vt == 0) return false;
            res_vts[i] = vt;
        }
    }

    // Resolve init operands in the outer scope.
    MLIR_ValueHandle *init_vals = NULL;
    if (n_iter) {
        init_vals = (MLIR_ValueHandle *)arena_alloc(F->arena, n_iter * sizeof(MLIR_ValueHandle));
        for (size_t i = 0; i < n_iter; i++) {
            if (!vmap_get(F, MLIR_GetOpOperand(op, i), &init_vals[i])) return false;
        }
    }

    // Build the loop body block with entry block args = T*; bind before-region's
    // block args to these so the before-body sees them as %a:T*.
    MLIR_BlockHandle loop_body = MLIR_CreateBlock(F->ctx);
    for (size_t i = 0; i < n_iter; i++) {
        MLIR_TypeHandle ty = vt_to_type(F->ctx, iter_vts[i]);
        MLIR_ValueHandle a = MLIR_CreateValueBlockArg(F->ctx, (string){0}, (uint32_t)i, ty,
                                                     MLIR_CreateLocationUnknown(F->ctx, (string){0}));
        MLIR_AppendBlockArg(F->ctx, loop_body, a);
        vmap_set(F, MLIR_GetBlockArg(before_b, i), a);
    }

    MLIR_BlockHandle saved = F->body_block;
    F->body_block = loop_body;

    // Walk before-block ops. scf.condition (which must be the terminator)
    // triggers emission of the exit-if + after-body walk in the same loop_body.
    size_t nb = MLIR_GetBlockNumOps(before_b);
    bool saw_cond = false;
    for (size_t i = 0; i < nb; i++) {
        MLIR_OpHandle bop = MLIR_GetBlockOp(before_b, i);
        string n = MLIR_GetOpName(bop);
        if (name_eq(n, "scf.condition")) {
            saw_cond = true;
            if (MLIR_GetOpNumOperands(bop) < 1) { F->body_block = saved; return false; }
            if (MLIR_GetOpNumOperands(bop) - 1 != n_res) { F->body_block = saved; return false; }

            MLIR_ValueHandle cidx;
            if (!vmap_get(F, MLIR_GetOpOperand(bop, 0), &cidx)) { F->body_block = saved; return false; }
            MLIR_ValueHandle z = emit_eqz(F, cidx);

            // Resolve the scf.condition payload values in the loop_body scope.
            MLIR_ValueHandle r_vals_inline[16];
            MLIR_ValueHandle *r_vals = n_res <= 16 ? r_vals_inline
                : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
            for (size_t k = 0; k < n_res; k++) {
                if (!vmap_get(F, MLIR_GetOpOperand(bop, k + 1), &r_vals[k])) {
                    F->body_block = saved; return false;
                }
            }

            // Build the exit-if: wasmssa.if (z) : () { wasmssa.br {depth=2} (r_vals) }.
            MLIR_BlockHandle then_blk = MLIR_CreateBlock(F->ctx);
            F->body_block = then_blk;
            emit_br_args(F, /*depth*/2, r_vals, n_res);
            F->body_block = loop_body;

            MLIR_RegionHandle then_r2 = MLIR_CreateRegion(F->ctx);
            MLIR_AppendRegionBlock(F->ctx, then_r2, then_blk);
            MLIR_RegionHandle regs[1] = { then_r2 };
            MLIR_ValueHandle cond_ops[1] = { z };
            MLIR_OpHandle ifop = make_op_n(F->ctx, OP_TYPE_WASMSSA_IF, NULL, 0,
                                           cond_ops, 1, regs, 1,
                                           NULL, 0, NULL);
            MLIR_AppendBlockOp(F->ctx, F->body_block, ifop);

            // Bind after-region's block args to the scf.condition payload.
            for (size_t k = 0; k < n_res; k++) {
                vmap_set(F, MLIR_GetBlockArg(after_b, k), r_vals[k]);
            }

            // Walk after-region ops; scf.yield becomes the back-edge.
            size_t na = MLIR_GetBlockNumOps(after_b);
            bool saw_yield = false;
            for (size_t j = 0; j < na; j++) {
                MLIR_OpHandle aop = MLIR_GetBlockOp(after_b, j);
                string an = MLIR_GetOpName(aop);
                if (name_eq(an, "scf.yield")) {
                    saw_yield = true;
                    if (MLIR_GetOpNumOperands(aop) != n_iter) { F->body_block = saved; return false; }
                    MLIR_ValueHandle y_vals_inline[16];
                    MLIR_ValueHandle *y_vals = n_iter <= 16 ? y_vals_inline
                        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_iter * sizeof(MLIR_ValueHandle));
                    for (size_t k = 0; k < n_iter; k++) {
                        if (!vmap_get(F, MLIR_GetOpOperand(aop, k), &y_vals[k])) {
                            F->body_block = saved; return false;
                        }
                    }
                    emit_br_args(F, /*depth*/0, y_vals, n_iter);
                } else {
                    if (!lower_op(F, aop)) { F->body_block = saved; return false; }
                }
            }
            if (!saw_yield && n_iter) { F->body_block = saved; return false; }
            break;  // scf.condition is the terminator of before-region.
        } else {
            if (!lower_op(F, bop)) { F->body_block = saved; return false; }
        }
    }
    if (!saw_cond) { F->body_block = saved; return false; }
    F->body_block = saved;

    // Build the wasmssa.loop op (no results; never falls through).
    MLIR_RegionHandle loop_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, loop_r, loop_body);
    MLIR_OpHandle loop_op = make_op_n(F->ctx, OP_TYPE_WASMSSA_LOOP, NULL, 0,
                                      init_vals, n_iter, &loop_r, 1,
                                      NULL, 0, NULL);

    // Build the outer wasmssa.block containing [loop_op; wasmssa.unreachable].
    MLIR_BlockHandle outer_blk = MLIR_CreateBlock(F->ctx);
    MLIR_AppendBlockOp(F->ctx, outer_blk, loop_op);
    {
        MLIR_OpHandle u = make_op_n(F->ctx, OP_TYPE_WASMSSA_UNREACHABLE, NULL, 0,
                                    NULL, 0, NULL, 0, NULL, 0, NULL);
        MLIR_AppendBlockOp(F->ctx, outer_blk, u);
    }
    MLIR_RegionHandle block_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, block_r, outer_blk);
    MLIR_ValueHandle block_res_inline[16];
    MLIR_ValueHandle *block_res_vals = n_res <= 16 ? block_res_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    MLIR_OpHandle block_op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BLOCK, NULL, 0,
                                       NULL, 0, &block_r, 1,
                                       res_vts, n_res, block_res_vals);
    MLIR_AppendBlockOp(F->ctx, F->body_block, block_op);

    for (size_t i = 0; i < n_res; i++) {
        vmap_set(F, MLIR_GetOpResult(op, i), block_res_vals[i]);
    }
    return true;
}

// ---- scf.index_switch ------------------------------------------------------
// Lowered as a chain of `wasmssa.if (cond==case_i) : (R*) { case_body }
// else { recurse }` — innermost else holds the default region. Results
// propagate via block_return up the chain.
static MLIR_ValueHandle emit_i32_eq(FnCtx *F, MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle ops[2] = { a, b };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_BINOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0x46;  // i32.eq
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}

// Recursively build the case chain. The wasmssa.if op for case `i` is
// appended to F->body_block; its R* results are copied into *out_res.
static bool isw_build_chain(FnCtx *F, MLIR_OpHandle op,
                            size_t i, size_t n_cases,
                            const int64_t *case_vals,
                            MLIR_ValueHandle cond_idx,
                            size_t n_res, const uint8_t *res_vts,
                            MLIR_ValueHandle *out_res /* size n_res */) {
    MLIR_ValueHandle kc = emit_const_i32(F, (int32_t)case_vals[i]);
    MLIR_ValueHandle cmp = emit_i32_eq(F, cond_idx, kc);

    MLIR_BlockHandle saved = F->body_block;

    // then = case i body.
    MLIR_BlockHandle then_blk = MLIR_CreateBlock(F->ctx);
    F->body_block = then_blk;
    bool ok = lower_block(F, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, i + 1), 0));
    if (!ok) { F->body_block = saved; return false; }

    // else = inner chain or default region.
    MLIR_BlockHandle else_blk = MLIR_CreateBlock(F->ctx);
    F->body_block = else_blk;
    if (i + 1 == n_cases) {
        ok = lower_block(F, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0));
        if (!ok) { F->body_block = saved; return false; }
    } else {
        MLIR_ValueHandle inner_inline[16];
        MLIR_ValueHandle *inner_res = n_res <= 16 ? inner_inline
            : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
        if (!isw_build_chain(F, op, i + 1, n_cases, case_vals, cond_idx,
                             n_res, res_vts, inner_res)) {
            F->body_block = saved; return false;
        }
        emit_block_return(F, inner_res, n_res);
    }

    F->body_block = saved;

    MLIR_RegionHandle then_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, then_r, then_blk);
    MLIR_RegionHandle else_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, else_r, else_blk);
    MLIR_RegionHandle regs[2] = { then_r, else_r };

    MLIR_AttributeHandle as[1];
    size_t nas = 0;
    if (n_res) as[nas++] = attr_s_hex(F->ctx, F->arena, "result_types", res_vts, n_res);
    MLIR_ValueHandle cond_ops[1] = { cmp };
    MLIR_ValueHandle res_inline[16];
    MLIR_ValueHandle *res_vals = n_res <= 16 ? res_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    MLIR_OpHandle ifop = make_op_n(F->ctx, OP_TYPE_WASMSSA_IF, as, nas,
                                   cond_ops, 1, regs, 2,
                                   res_vts, n_res, res_vals);
    MLIR_AppendBlockOp(F->ctx, F->body_block, ifop);
    for (size_t k = 0; k < n_res; k++) out_res[k] = res_vals[k];
    return true;
}

static bool lower_scf_index_switch(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumOperands(op) != 1) return false;
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(op, 0);
    MLIR_ValueHandle cond_idx;
    if (!vmap_get(F, cond_v, &cond_idx)) return false;
    if (wasm_vt(F->ctx, MLIR_GetValueType(cond_v)) != WT_I32) return false;

    size_t n_regions = MLIR_GetOpNumRegions(op);
    if (n_regions < 1) return false;
    size_t n_cases = n_regions - 1;  // region 0 = default

    // Parse "cases" attribute: prints as "array<i64: 1, 2, 3>".
    int64_t *case_vals = NULL;
    if (n_cases > 0) {
        MLIR_AttributeHandle ca = find_attr(op, "cases");
        if (ca == MLIR_INVALID_HANDLE) return false;
        string cs = MLIR_GetAttributeAsString(F->ctx, ca);
        case_vals = (int64_t *)arena_alloc(F->arena, n_cases * sizeof(int64_t));
        size_t p = 0;
        while (p < cs.size && cs.str[p] != ':') p++;
        if (p < cs.size) p++;
        size_t n_parsed = 0;
        while (p < cs.size && n_parsed < n_cases) {
            while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
            if (p >= cs.size || cs.str[p] == '>') break;
            int64_t sign = 1;
            if (cs.str[p] == '-') { sign = -1; p++; }
            int64_t v = 0;
            while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9') {
                v = v * 10 + (cs.str[p] - '0');
                p++;
            }
            case_vals[n_parsed++] = sign * v;
        }
        if (n_parsed != n_cases) return false;
    }

    // Collect result types.
    size_t n_res = MLIR_GetOpNumResults(op);
    uint8_t *res_vts = NULL;
    if (n_res) {
        res_vts = (uint8_t *)arena_alloc(F->arena, n_res);
        for (size_t i = 0; i < n_res; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(MLIR_GetOpResult(op, i)));
            if (vt == 0) return false;
            res_vts[i] = vt;
        }
    }

    // Degenerate: no cases. Wrap default in a wasmssa.block.
    if (n_cases == 0) {
        MLIR_BlockHandle saved = F->body_block;
        MLIR_BlockHandle inner = MLIR_CreateBlock(F->ctx);
        F->body_block = inner;
        bool ok = lower_block(F, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0));
        F->body_block = saved;
        if (!ok) return false;

        MLIR_RegionHandle reg = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, reg, inner);
        MLIR_AttributeHandle as[1];
        size_t nas = 0;
        if (n_res) as[nas++] = attr_s_hex(F->ctx, F->arena, "result_types", res_vts, n_res);
        MLIR_ValueHandle res_inline[16];
        MLIR_ValueHandle *res_vals = n_res <= 16 ? res_inline
            : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
        MLIR_OpHandle bop = make_op_n(F->ctx, OP_TYPE_WASMSSA_BLOCK, as, nas,
                                      NULL, 0, &reg, 1, res_vts, n_res, res_vals);
        MLIR_AppendBlockOp(F->ctx, F->body_block, bop);
        for (size_t i = 0; i < n_res; i++) {
            vmap_set(F, MLIR_GetOpResult(op, i), res_vals[i]);
        }
        return true;
    }

    MLIR_ValueHandle outer_inline[16];
    MLIR_ValueHandle *res_vals = n_res <= 16 ? outer_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    if (!isw_build_chain(F, op, 0, n_cases, case_vals, cond_idx,
                         n_res, res_vts, res_vals)) return false;
    for (size_t i = 0; i < n_res; i++) {
        vmap_set(F, MLIR_GetOpResult(op, i), res_vals[i]);
    }
    return true;
}

static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op);
static bool lower_op(FnCtx *F, MLIR_OpHandle op) {
    bool ok = lower_op_inner(F, op);
    if (!ok) {
        string n = MLIR_GetOpName(op);
        fprintf(stderr, "wasmssa-lower: lower_op failed at '%.*s'\n",
                (int)n.size, n.str);
    }
    return ok;
}
static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op) {
    string name = MLIR_GetOpName(op);

    // ---- scf.if / scf.while / scf.index_switch ---------------------------
    if (name_eq(name, "scf.if"))    return lower_scf_if(F, op);
    if (name_eq(name, "scf.while")) return lower_scf_while(F, op);
    if (name_eq(name, "scf.index_switch")) return lower_scf_index_switch(F, op);

    // ---- scf.yield: emit wasmssa.block_return into the enclosing region's
    //                 block. The enclosing scf.if / scf.index_switch / scf.while
    //                 lowering has already set F->body_block to the right block.
    if (name_eq(name, "scf.yield")) {
        size_t no = MLIR_GetOpNumOperands(op);
        MLIR_ValueHandle vs_inline[16];
        MLIR_ValueHandle *vs = no <= 16 ? vs_inline
            : (MLIR_ValueHandle *)arena_alloc(F->arena, no * sizeof(MLIR_ValueHandle));
        for (size_t i = 0; i < no; i++) {
            if (!vmap_get(F, MLIR_GetOpOperand(op, i), &vs[i])) return false;
        }
        emit_block_return(F, vs, no);
        return true;
    }

    // ---- llvm.select -----------------------------------------------------
    if (name_eq(name, "llvm.select")) {
        if (MLIR_GetOpNumOperands(op) != 3 || MLIR_GetOpNumResults(op) != 1)
            return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        MLIR_ValueHandle ci, ai, bi;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &ci)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &ai)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 2), &bi)) return false;
        MLIR_ValueHandle ops[3] = { ai, bi, ci };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_SELECT;
        o.valtype = vt;
        o.n_operands = 3;
        // wasm select pops cond,b,a so we order operands as a,b,cond.
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.mlir.constant ------------------------------------------------
    if (name_eq(name, "llvm.mlir.constant")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        MLIR_AttributeHandle va = find_attr(op, "value");
        if (va == MLIR_INVALID_HANDLE) return false;
        MLIR_AttrKind ak = MLIR_GetAttributeKind(va);

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        if (ak == MLIR_ATTR_KIND_INTEGER) {
            o.i_const = MLIR_GetAttributeInteger(va);
        } else if (ak == MLIR_ATTR_KIND_BOOL) {
            o.i_const = MLIR_GetAttributeBool(va) ? 1 : 0;
        } else if (ak == MLIR_ATTR_KIND_FLOAT) {
            double dv = MLIR_GetAttributeFloat(va);
            // Pack the bit pattern through an integer of matching width;
            // stage 3 emits f32.const / f64.const using the same bytes.
            if (vt == WT_F32) {
                float fv = (float)dv;
                uint32_t bits;
                memcpy(&bits, &fv, 4);
                o.i_const = (int64_t)(uint64_t)bits;
            } else if (vt == WT_F64) {
                uint64_t bits;
                memcpy(&bits, &dv, 8);
                o.i_const = (int64_t)bits;
            } else {
                return false;
            }
        } else {
            return false;
        }
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- ub.poison / llvm.mlir.undef ---------------------------------------
    // Inserted by --lift-cf-to-scf for "doesn't-matter" yield values on
    // unreachable / pre-break paths. Lower either name to a typed zero
    // constant. Upstream's CFGToSCF uses ub.poison; the native port uses
    // llvm.mlir.undef (no UB dialect in the agnostic op table) — both are
    // semantically equivalent here.
    if (name_eq(name, "ub.poison") || name_eq(name, "llvm.mlir.undef")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        o.i_const = 0;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.alloca: result = sp_def + frame_offset -----------------------
    if (name_eq(name, "llvm.alloca")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint32_t off;
        if (!amap_get(F, r, &off)) return false;
        if (F->sp_value == MLIR_INVALID_HANDLE) return false;
        MLIR_ValueHandle koff = emit_const_i32(F, (int32_t)off);
        MLIR_ValueHandle addr = emit_add_i32(F, F->sp_value, koff);
        vmap_set(F, r, addr);
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
        MLIR_ValueHandle va, pa;
        if (!vmap_get(F, val, &va)) return false;
        if (!vmap_get(F, ptr, &pa)) return false;

        unsigned align_log2;
        if (vt == WT_I32) {
            align_log2 = (sz == 4) ? 2 : (sz == 2) ? 1 : 0;
        } else if (vt == WT_I64) align_log2 = 3;
        else if (vt == WT_F32)   align_log2 = 2;
        else if (vt == WT_F64)   align_log2 = 3;
        else return false;

        MLIR_ValueHandle ops[2] = { pa, va };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_STORE;
        o.valtype = vt;
        o.mem_size_bytes = sz;
        o.memory_align_log2 = align_log2;
        o.memory_offset = 0;
        o.n_operands = 2;
        o.operands = ops;
        (void)commit_op(F, &o);
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
        MLIR_ValueHandle pa;
        if (!vmap_get(F, ptr, &pa)) return false;

        unsigned align_log2;
        if (vt == WT_I32) {
            align_log2 = (sz == 4) ? 2 : (sz == 2) ? 1 : 0;
        } else if (vt == WT_I64) align_log2 = 3;
        else if (vt == WT_F32)   align_log2 = 2;
        else if (vt == WT_F64)   align_log2 = 3;
        else return false;

        MLIR_ValueHandle ops[1] = { pa };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_LOAD;
        o.valtype = vt;
        o.mem_size_bytes = sz;
        o.memory_align_log2 = align_log2;
        o.memory_offset = 0;
        o.n_operands = 1;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.sext (i8/i16/i32 -> i32; i32 -> i64; i8/i16 -> i64) ----------
    if (name_eq(name, "llvm.sext")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        int in_w  = int_bits(F->ctx, MLIR_GetValueType(s));
        int out_w = int_bits(F->ctx, MLIR_GetValueType(r));
        if (in_w == 0 || out_w == 0 || in_w > out_w) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;

        // Step 1: if narrowing source is i8/i16 stored in i32, sign-extend
        // within i32 first (extend8_s=0xc0, extend16_s=0xc1).
        MLIR_ValueHandle cur_idx = sa;
        uint8_t cur_vt = wasm_vt(F->ctx, MLIR_GetValueType(s));
        if (in_w == 8 || in_w == 16) {
            MLIR_ValueHandle ops[1] = { cur_idx };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_UNOP;
            o.valtype = WT_I32;
            o.wasm_opcode = (in_w == 8) ? 0xc0 : 0xc1;
            o.n_operands = 1;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            cur_idx = idx; cur_vt = WT_I32;
        }
        // Step 2: if widening into i64, use i64.extend_i32_s (0xac).
        if (out_w == 64) {
            MLIR_ValueHandle ops[1] = { cur_idx };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_EXTEND_I32_S;
            o.valtype = WT_I64;
            o.n_operands = 1;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            cur_idx = idx;
        }
        vmap_set(F, r, cur_idx);
        return true;
    }

    // ---- llvm.ptrtoint / llvm.inttoptr -------------------------------------
    if (name_eq(name, "llvm.ptrtoint") || name_eq(name, "llvm.inttoptr")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        // Pointers are i32. If matched, identity; if narrowing/widening, defer
        // to the existing trunc/zext-style lowering by emitting a wrap or
        // extend.
        if (in_vt == out_vt) {
            vmap_set(F, r, sa);
            return true;
        }
        if (in_vt == WT_I64 && out_vt == WT_I32) {
            MLIR_ValueHandle idx = emit_wrap_i64_to_i32(F, sa);
            vmap_set(F, r, idx);
            return true;
        }
        if (in_vt == WT_I32 && out_vt == WT_I64) {
            // unsigned extension: i64.extend_i32_u (0xad)
            MLIR_ValueHandle ops[1] = { sa };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_UNOP;
            o.valtype = WT_I64;
            o.wasm_opcode = 0xad;
            o.n_operands = 1;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            vmap_set(F, r, idx);
            return true;
        }
        return false;
    }


    // ---- llvm binary integer ops -------------------------------------------
    // Mapped to a generic wasmssa.binop carrying the actual wasm bytecode
    // byte. Op kind picks the row; valtype picks the column of the table.
    {
        // (op-name, i32-opcode, i64-opcode); 0 = unsupported
        struct { const char *nm; uint8_t i32op, i64op; } btab[] = {
            { "llvm.add",  0x6a, 0x7c },
            { "llvm.sub",  0x6b, 0x7d },
            { "llvm.mul",  0x6c, 0x7e },
            { "llvm.sdiv", 0x6d, 0x7f },
            { "llvm.udiv", 0x6e, 0x80 },
            { "llvm.srem", 0x6f, 0x81 },
            { "llvm.urem", 0x70, 0x82 },
            { "llvm.and",  0x71, 0x83 },
            { "llvm.or",   0x72, 0x84 },
            { "llvm.xor",  0x73, 0x85 },
            { "llvm.shl",  0x74, 0x86 },
            { "llvm.ashr", 0x75, 0x87 },
            { "llvm.lshr", 0x76, 0x88 },
        };
        for (size_t k = 0; k < sizeof(btab)/sizeof(btab[0]); k++) {
            if (!name_eq(name, btab[k].nm)) continue;
            if (MLIR_GetOpNumResults(op) != 1 ||
                MLIR_GetOpNumOperands(op) != 2) return false;
            MLIR_ValueHandle r  = MLIR_GetOpResult(op, 0);
            MLIR_ValueHandle a  = MLIR_GetOpOperand(op, 0);
            MLIR_ValueHandle b  = MLIR_GetOpOperand(op, 1);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            uint8_t opc = (vt == WT_I32) ? btab[k].i32op
                        : (vt == WT_I64) ? btab[k].i64op : 0;
            if (opc == 0) return false;
            MLIR_ValueHandle ai, bi;
            if (!vmap_get(F, a, &ai) || !vmap_get(F, b, &bi)) return false;
            MLIR_ValueHandle ops[2] = { ai, bi };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_BINOP;
            o.valtype = vt;
            o.wasm_opcode = opc;
            o.n_operands = 2;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            vmap_set(F, r, idx);
            return true;
        }
    }

    // ---- llvm binary float ops ---------------------------------------------
    {
        struct { const char *nm; uint8_t f32op, f64op; } ftab[] = {
            { "llvm.fadd",  0x92, 0xa0 },
            { "llvm.fsub",  0x93, 0xa1 },
            { "llvm.fmul",  0x94, 0xa2 },
            { "llvm.fdiv",  0x95, 0xa3 },
        };
        for (size_t k = 0; k < sizeof(ftab)/sizeof(ftab[0]); k++) {
            if (!name_eq(name, ftab[k].nm)) continue;
            if (MLIR_GetOpNumResults(op) != 1 ||
                MLIR_GetOpNumOperands(op) != 2) return false;
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            uint8_t opc = (vt == WT_F32) ? ftab[k].f32op
                        : (vt == WT_F64) ? ftab[k].f64op : 0;
            if (opc == 0) return false;
            MLIR_ValueHandle ai, bi;
            if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &ai)) return false;
            if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &bi)) return false;
            MLIR_ValueHandle ops[2] = { ai, bi };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_BINOP;
            o.valtype = vt;
            o.wasm_opcode = opc;
            o.n_operands = 2;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            vmap_set(F, r, idx);
            return true;
        }
    }

    // ---- llvm float compares -----------------------------------------------
    if (name_eq(name, "llvm.fcmp")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        uint8_t opvt = wasm_vt(F->ctx, MLIR_GetValueType(a));
        // FCmpPredicate enum: oeq=1, ogt=2, oge=3, olt=4, ole=5, one=6,
        // une=11, ueq=8 (unordered variants); we only support ordered.
        // wasm has eq, ne, lt, gt, le, ge (ordered). Map oeq->eq, etc.
        MLIR_AttributeHandle pa = find_attr(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE) return false;
        int64_t pred = MLIR_GetAttributeInteger(pa);
        // Map LLVM fcmp predicate -> wasm op offset (eq=0,ne=1,lt=2,gt=3,le=4,ge=5).
        int8_t off;
        switch (pred) {
        case 1: off = 0; break;  // oeq
        case 2: off = 3; break;  // ogt
        case 3: off = 5; break;  // oge
        case 4: off = 2; break;  // olt
        case 5: off = 4; break;  // ole
        case 6: off = 1; break;  // one
        default: return false;
        }
        uint8_t opc = (opvt == WT_F32) ? (uint8_t)(0x5b + off)
                    : (opvt == WT_F64) ? (uint8_t)(0x61 + off) : 0;
        if (opc == 0) return false;
        MLIR_ValueHandle ai, bi;
        if (!vmap_get(F, a, &ai)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &bi)) return false;
        MLIR_ValueHandle ops[2] = { ai, bi };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, MLIR_GetOpResult(op, 0), idx);
        return true;
    }

    // ---- llvm.fpext / llvm.fptrunc / llvm.fneg ----------------------------
    if (name_eq(name, "llvm.fpext") || name_eq(name, "llvm.fptrunc") ||
        name_eq(name, "llvm.fneg")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        uint8_t opc = 0;
        if (name_eq(name, "llvm.fpext") && in_vt == WT_F32 && out_vt == WT_F64) opc = 0xbb;
        else if (name_eq(name, "llvm.fptrunc") && in_vt == WT_F64 && out_vt == WT_F32) opc = 0xb6;
        else if (name_eq(name, "llvm.fneg") && in_vt == WT_F32 && out_vt == WT_F32) opc = 0x8c;
        else if (name_eq(name, "llvm.fneg") && in_vt == WT_F64 && out_vt == WT_F64) opc = 0x9a;
        else return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.sitofp / llvm.uitofp / llvm.fptosi / llvm.fptoui ------------
    if (name_eq(name, "llvm.sitofp") || name_eq(name, "llvm.uitofp") ||
        name_eq(name, "llvm.fptosi") || name_eq(name, "llvm.fptoui")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        bool is_signed = name_eq(name, "llvm.sitofp") || name_eq(name, "llvm.fptosi");
        uint8_t opc = 0;
        if (name_eq(name, "llvm.sitofp") || name_eq(name, "llvm.uitofp")) {
            // f{32,64}.convert_i{32,64}_{s,u}
            if (out_vt == WT_F32 && in_vt == WT_I32) opc = is_signed ? 0xb2 : 0xb3;
            else if (out_vt == WT_F32 && in_vt == WT_I64) opc = is_signed ? 0xb4 : 0xb5;
            else if (out_vt == WT_F64 && in_vt == WT_I32) opc = is_signed ? 0xb7 : 0xb8;
            else if (out_vt == WT_F64 && in_vt == WT_I64) opc = is_signed ? 0xb9 : 0xba;
        } else {
            // i{32,64}.trunc_f{32,64}_{s,u}
            if (out_vt == WT_I32 && in_vt == WT_F32) opc = is_signed ? 0xa8 : 0xa9;
            else if (out_vt == WT_I32 && in_vt == WT_F64) opc = is_signed ? 0xaa : 0xab;
            else if (out_vt == WT_I64 && in_vt == WT_F32) opc = is_signed ? 0xae : 0xaf;
            else if (out_vt == WT_I64 && in_vt == WT_F64) opc = is_signed ? 0xb0 : 0xb1;
        }
        if (opc == 0) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.icmp ---------------------------------------------------------
    // Result is i1 (encoded as i32 in wasm). Operand valtype determines
    // the i32.* vs i64.* opcode family.
    if (name_eq(name, "llvm.icmp")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(op, 1);
        uint8_t opvt = wasm_vt(F->ctx, MLIR_GetValueType(a));
        // LLVM ICmpPredicate enum: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5,
        // ult=6, ule=7, ugt=8, uge=9.
        MLIR_AttributeHandle pa = find_attr(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE) return false;
        int64_t pred = MLIR_GetAttributeInteger(pa);
        // i32: 0x46..0x4f ; i64: 0x51..0x5a (skip eqz at 0x45/0x50).
        // Map predicate -> bytecode offset within (eq, ne, lt_s, lt_u,
        // gt_s, gt_u, le_s, le_u, ge_s, ge_u).
        // LLVM order:  eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge
        // wasm order:  eq, ne, lt_s, lt_u, gt_s, gt_u, le_s, le_u, ge_s, ge_u
        static const int8_t llvm_to_wasm[10] = {0, 1, 2, 6, 4, 8, 3, 7, 5, 9};
        if (pred < 0 || pred >= 10) return false;
        int wasm_idx = llvm_to_wasm[pred];
        uint8_t opc = (opvt == WT_I32) ? (uint8_t)(0x46 + wasm_idx)
                    : (opvt == WT_I64) ? (uint8_t)(0x51 + wasm_idx) : 0;
        if (opc == 0) return false;
        MLIR_ValueHandle ai, bi;
        if (!vmap_get(F, a, &ai) || !vmap_get(F, b, &bi)) return false;
        MLIR_ValueHandle ops[2] = { ai, bi };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;  // result type
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.trunc / llvm.zext -------------------------------------------
    // i64 -> i32 trunc:    i32.wrap_i64    (0xa7)
    // i32 -> i64 zext:     i64.extend_i32_u(0xad)
    // smaller-int trunc/zext within i32: no-op (wasm has no sub-i32 reg).
    if (name_eq(name, "llvm.trunc") || name_eq(name, "llvm.zext")) {
        bool is_zext = name_eq(name, "llvm.zext");
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (in_vt == 0 || out_vt == 0) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        if (in_vt == out_vt) {
            // No-op: result alias of operand.
            vmap_set(F, r, sa);
            return true;
        }
        uint8_t opc;
        if (!is_zext && in_vt == WT_I64 && out_vt == WT_I32) opc = 0xa7;
        else if (is_zext && in_vt == WT_I32 && out_vt == WT_I64) opc = 0xad;
        else return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- builtin.unrealized_conversion_cast --------------------------------
    // Treat as identity when wasm valtypes match (e.g. ptr<->i32, index<->i32).
    // Otherwise emit a trunc/extend (i64<->i32 unsigned).
    if (name_eq(name, "builtin.unrealized_conversion_cast")) {
        if (MLIR_GetOpNumOperands(op) != 1 || MLIR_GetOpNumResults(op) != 1)
            return false;
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (in_vt == 0 || out_vt == 0) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        if (in_vt == out_vt) { vmap_set(F, r, sa); return true; }
        uint8_t opc;
        if (in_vt == WT_I64 && out_vt == WT_I32) opc = 0xa7;       // i32.wrap_i64
        else if (in_vt == WT_I32 && out_vt == WT_I64) opc = 0xad;  // i64.extend_i32_u
        else return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.return -------------------------------------------------------
    if (name_eq(name, "llvm.return")) {
        size_t no = MLIR_GetOpNumOperands(op);
        MLIR_ValueHandle retv = MLIR_INVALID_HANDLE;
        if (no == 1) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, 0);
            if (!vmap_get(F, v, &retv)) return false;
        } else if (no != 0) {
            return false;
        }

        // Epilogue: restore __stack_pointer = sp_def + frame_size.
        if (F->frame_size > 0 && F->sp_value != MLIR_INVALID_HANDLE) {
            MLIR_ValueHandle kf = emit_const_i32(F, (int32_t)F->frame_size);
            MLIR_ValueHandle restored = emit_add_i32(F, F->sp_value, kf);
            emit_global_set(F, /*sp*/0, restored);
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_RETURN;
        MLIR_ValueHandle o_ops[1] = { retv };
        if (no == 1) {
            o.n_operands = 1;
            o.operands = o_ops;
        }
        (void)commit_op(F, &o);
        return true;
    }

    // ---- llvm.call (direct or indirect) ------------------------------------
    if (name_eq(name, "llvm.call")) {
        MLIR_AttributeHandle callee = MLIR_GetOpAttributeByName(op, "callee");
        MLIR_AttributeHandle var_callee_type = MLIR_GetOpAttributeByName(op, "var_callee_type");

        // Indirect: no callee attr, operand[0] is the function pointer
        // (a !llvm.ptr SSA value), and var_callee_type carries the signature.
        if (callee == MLIR_INVALID_HANDLE) {
            // Indirect call: operand[0] is function pointer (!llvm.ptr).
            // Build the signature from operand types and result types,
            // since the upstream wasm pipeline doesn't carry var_callee_type
            // through after lowering.
            size_t no = MLIR_GetOpNumOperands(op);
            if (no < 1) return false;
            size_t snp = no - 1;
            uint8_t *sp = (uint8_t *)arena_alloc(F->arena, snp ? snp : 1);
            for (size_t i = 0; i < snp; i++) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpOperand(op, i + 1)));
                if (v == 0) return false;
                sp[i] = v;
            }
            size_t nr = MLIR_GetOpNumResults(op);
            if (nr > 1) return false;
            uint8_t *sr = (uint8_t *)arena_alloc(F->arena, 1);
            size_t snr = 0;
            if (nr == 1) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpResult(op, 0)));
                if (v == 0) return false;
                sr[0] = v;
                snr = 1;
            }
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, no * sizeof(MLIR_ValueHandle));
            // wasm call_indirect pops args first then table-index, so order
            // them in operand[] as [args..., funcptr].
            for (size_t i = 0; i < snp; i++) {
                if (!vmap_get(F, MLIR_GetOpOperand(op, i + 1), &opnds[i]))
                    return false;
            }
            if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &opnds[snp]))
                return false;
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_CALL_INDIRECT;
            o.n_operands = (int)no;
            o.operands = opnds;
            o.sig_params = sp; o.n_sig_params = snp;
            o.sig_results = sr; o.n_sig_results = snr;
            if (snr == 1) {
                o.valtype = sr[0];
                o.has_result = true;
            }
            MLIR_ValueHandle idx = commit_op(F, &o);
            if (snr == 1) {
                vmap_set(F, MLIR_GetOpResult(op, 0), idx);
            }
            return true;
        }

        if (var_callee_type != MLIR_INVALID_HANDLE) {
            MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(var_callee_type);
            if (MLIR_GetTypeFunctionIsVarArg(fty)) {
                // Variadic direct call: pack variadic args into the
                // function's variadic buffer (allocated in the shadow
                // stack frame) and pass its address as a hidden trailing
                // i32 arg in place of the `...` operands.
                size_t nfixed = MLIR_GetTypeFunctionNumInputs(fty);
                size_t no = MLIR_GetOpNumOperands(op);
                size_t nvar = no - nfixed;
                uint32_t total = 0; size_t nf2 = 0;
                uint32_t *offs = (uint32_t *)arena_alloc(F->arena, (nvar ? nvar : 1) * sizeof(uint32_t));
                if (!va_call_layout(F, op, &total, &nf2, offs)) {
                    return false;
                }
                // buf_addr = sp_def + va_buf_offset. When this call passes no
                // variadic args (total == 0) the buffer is empty and never
                // dereferenced, so it needs no shadow-stack frame — pass a null
                // pointer as the hidden va_list arg. (Without this, a variadic
                // call with zero variadic args in a function that has no other
                // stack frame — e.g. after mem2reg promotes all locals — would
                // fail for lack of sp_value.)
                MLIR_ValueHandle buf_addr;
                if (total == 0) {
                    buf_addr = emit_const_i32(F, 0);
                } else {
                    if (F->sp_value == MLIR_INVALID_HANDLE) { return false; }
                    if (F->va_buf_offset == 0) {
                        buf_addr = F->sp_value;
                    } else {
                        MLIR_ValueHandle koff = emit_const_i32(F, (int32_t)F->va_buf_offset);
                        buf_addr = emit_add_i32(F, F->sp_value, koff);
                    }
                }
                // Emit a store for each variadic arg, then the call.
                for (size_t i = 0; i < nvar; i++) {
                    MLIR_ValueHandle av = MLIR_GetOpOperand(op, nfixed + i);
                    MLIR_TypeHandle aty = MLIR_GetValueType(av);
                    uint8_t vt = wasm_vt(F->ctx, aty);
                    MLIR_ValueHandle va;
                    if (!vmap_get(F, av, &va)) { return false; }
                    uint8_t st_vt = vt;
                    unsigned sz, align_log2;
                    if (vt == WT_I32)      { sz = 4; align_log2 = 2; }
                    else if (vt == WT_I64) { sz = 8; align_log2 = 3; }
                    else if (vt == WT_F32) {
                        // Promote f32 -> f64.
                        wasmssa_op_t po = {0};
                        po.type = OP_TYPE_WASMSSA_UNOP;
                        po.valtype = WT_F64;
                        po.wasm_opcode = 0xbb; // f64.promote_f32
                        po.n_operands = 1;
                        MLIR_ValueHandle po_ops[1] = { va };
                        po.operands = po_ops;
                        po.has_result = true;
                        MLIR_ValueHandle pidx = commit_op(F, &po);
                        va = pidx;
                        st_vt = WT_F64;
                        sz = 8; align_log2 = 3;
                    }
                    else if (vt == WT_F64) { sz = 8; align_log2 = 3; }
                    else { return false; }
                    wasmssa_op_t o2 = {0};
                    o2.type = OP_TYPE_WASMSSA_STORE;
                    o2.valtype = st_vt;
                    o2.mem_size_bytes = sz;
                    o2.memory_align_log2 = align_log2;
                    o2.memory_offset = offs[i];
                    o2.n_operands = 2;
                    MLIR_ValueHandle o2_ops[2] = { buf_addr, va };
                    o2.operands = o2_ops;
                    (void)commit_op(F, &o2);
                }

                // Now emit the call with fixed args + buf_addr.
                string nm = MLIR_GetAttributeAsString(F->ctx, callee);
                string cstr = nm;
                if (cstr.size > 0 && cstr.str[0] == '@') { cstr.str++; cstr.size--; }
                size_t nout = nfixed + 1;
                MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, nout * sizeof(MLIR_ValueHandle));
                for (size_t i = 0; i < nfixed; i++) {
                    if (!vmap_get(F, MLIR_GetOpOperand(op, i), &opnds[i]))
                        return false;
                }
                opnds[nfixed] = buf_addr;

                wasmssa_op_t oc = {0};
                oc.type = OP_TYPE_WASMSSA_CALL;
                oc.call_target = cstr;
                oc.n_operands = (int)nout;
                oc.operands = opnds;
                size_t nr = MLIR_GetOpNumResults(op);
                MLIR_ValueHandle r = MLIR_INVALID_HANDLE;
                if (nr == 1) {
                    r = MLIR_GetOpResult(op, 0);
                    uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
                    if (vt == 0) return false;
                    oc.valtype = vt;
                    oc.has_result = true;
                } else if (nr != 0) {
                    return false;
                }
                MLIR_ValueHandle idx = commit_op(F, &oc);
                if (nr == 1) vmap_set(F, r, idx);
                return true;
            }
        }
        string nm = MLIR_GetAttributeAsString(F->ctx, callee);
        // SymbolRefAttr prints as `@name`.
        string cstr = nm;
        if (cstr.size > 0 && cstr.str[0] == '@') { cstr.str++; cstr.size--; }

        size_t no = MLIR_GetOpNumOperands(op);
        MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, (no ? no : 1) * sizeof(MLIR_ValueHandle));
        for (size_t i = 0; i < no; i++) {
            if (!vmap_get(F, MLIR_GetOpOperand(op, i), &opnds[i]))
                return false;
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CALL;
        o.call_target = cstr;
        o.n_operands = (int)no;
        o.operands = opnds;

        size_t nr = MLIR_GetOpNumResults(op);
        MLIR_ValueHandle r = MLIR_INVALID_HANDLE;
        if (nr == 1) {
            r = MLIR_GetOpResult(op, 0);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            if (vt == 0) return false;
            o.valtype = vt;
            o.has_result = true;
        } else if (nr != 0) {
            return false;
        }
        MLIR_ValueHandle idx = commit_op(F, &o);
        if (nr == 1) vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.getelementptr -----------------------------------------------
    if (name_eq(name, "llvm.getelementptr")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        if (MLIR_GetOpNumOperands(op) < 1) return false;
        MLIR_ValueHandle r    = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle base = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle addr;
        if (!vmap_get(F, base, &addr)) return false;
        MLIR_AttributeHandle eta = find_attr(op, "elem_type");
        if (eta == MLIR_INVALID_HANDLE) return false;
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(eta);
        MLIR_AttributeHandle ria = find_attr(op, "rawConstantIndices");
        if (ria == MLIR_INVALID_HANDLE) return false;
        size_t n_idx = 0;
        int32_t *cidx = parse_dense_i32_array(F->arena,
            MLIR_GetAttributeAsString(F->ctx, ria), &n_idx);
        if (!cidx || n_idx == 0) return false;

        size_t op_idx = 1;
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            bool is_dyn = (cidx[i] == (int32_t)0x80000000);
            MLIR_ValueHandle dyn_def = MLIR_INVALID_HANDLE;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(op)) return false;
                MLIR_ValueHandle ov = MLIR_GetOpOperand(op, op_idx++);
                if (!vmap_get(F, ov, &dyn_def)) return false;
                uint8_t ovt = wasm_vt(F->ctx, MLIR_GetValueType(ov));
                if (ovt == WT_I64) dyn_def = emit_wrap_i64_to_i32(F, dyn_def);
                else if (ovt != WT_I32) return false;
            }

            if (i == 0) {
                unsigned esz = type_size_bytes(F->ctx, elem_ty);
                // `!llvm.array<0 x T>` (zero-length array) has size 0; it is
                // used as the element type for addressof-based GEP into a
                // global whose runtime size we don't model in the type. Any
                // contribution from this index multiplied by 0 is 0, so just
                // skip it.
                if (esz == 0) {
                    // no-op
                } else if (is_dyn) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)esz);
                    MLIR_ValueHandle m = emit_mul_i32(F, dyn_def, k);
                    addr = emit_add_i32(F, addr, m);
                } else if (cidx[i] != 0) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)((int64_t)cidx[i] * (int32_t)esz));
                    addr = emit_add_i32(F, addr, k);
                }
            } else if (MLIR_IsTypeLLVMStruct(cur_ty)) {
                if (is_dyn) return false;
                unsigned off = struct_field_offset(F->ctx, cur_ty, (size_t)cidx[i]);
                if (off != 0) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)off);
                    addr = emit_add_i32(F, addr, k);
                }
                cur_ty = MLIR_GetTypeLLVMStructField(cur_ty, (size_t)cidx[i]);
            } else if (MLIR_IsTypeLLVMArray(cur_ty)) {
                MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(cur_ty);
                unsigned esz = type_size_bytes(F->ctx, et);
                if (esz == 0) return false;
                if (is_dyn) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)esz);
                    MLIR_ValueHandle m = emit_mul_i32(F, dyn_def, k);
                    addr = emit_add_i32(F, addr, m);
                } else if (cidx[i] != 0) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)((int64_t)cidx[i] * (int32_t)esz));
                    addr = emit_add_i32(F, addr, k);
                }
                cur_ty = et;
            } else {
                return false;
            }
        }
        vmap_set(F, r, addr);
        return true;
    }

    // ---- llvm.mlir.addressof ----------------------------------------------
    // Result is a pointer (i32) to a data global. We emit an ADDRESSOF op
    // that stage 3 will lower as `i32.const <padded sleb>` plus a
    // R_WASM_MEMORY_ADDR_SLEB reloc to the named global symbol.
    if (name_eq(name, "llvm.mlir.addressof")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_AttributeHandle ta = find_attr(op, "global_name");
        if (ta == MLIR_INVALID_HANDLE)
            ta = find_attr(op, "value");
        if (ta == MLIR_INVALID_HANDLE) {
            // Fallback: any string attr.
            size_t na = MLIR_GetOpNumAttributes(op);
            for (size_t i = 0; i < na; i++) {
                MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
                if (MLIR_GetAttributeKind(a) == MLIR_ATTR_KIND_STRING) {
                    ta = a; break;
                }
            }
        }
        if (ta == MLIR_INVALID_HANDLE) return false;
        string ts = MLIR_GetAttributeString(ta);
        // Strip leading '@' if present (FlatSymbolRefAttr prints with '@').
        if (ts.size && ts.str[0] == '@') {
            ts.str++;
            ts.size--;
        }

        // If this name refers to a function (not a data global), emit a
        // FUNC_ADDR (lowered as i32.const + R_WASM_TABLE_INDEX_SLEB) so the
        // wasm linker places it in __indirect_function_table.
        if (is_function_symbol(F->mod, ts.str, ts.size)) {
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_FUNC_ADDR;
            o.valtype = WT_I32;
            o.call_target = ts;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            vmap_set(F, MLIR_GetOpResult(op, 0), idx);
            return true;
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_ADDRESSOF;
        o.valtype = WT_I32;
        o.call_target = ts;  // reuse field for symbol name
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, MLIR_GetOpResult(op, 0), idx);
        return true;
    }

    // ---- llvm.mlir.zero ---------------------------------------------------
    // Pointer / integer zero. f32/f64 zero would also fall through here but
    // currently only ptr/int paths are exercised (the float test uses a
    // typed llvm.mlir.constant for 0.0).
    if (name_eq(name, "llvm.mlir.zero")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        o.i_const = 0;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.vastart / llvm.intr.vaend -------------------------------
    if (name_eq(name, "llvm.intr.vastart")) {
        // Store the function's hidden va_list i32 param into *%ap.
        if (F->va_list_value == MLIR_INVALID_HANDLE) return false;
        if (MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle pa;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &pa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_STORE;
        o.valtype = WT_I32;
        o.mem_size_bytes = 4;
        o.memory_align_log2 = 2;
        o.memory_offset = 0;
        o.n_operands = 2;
        MLIR_ValueHandle o_ops[2] = { pa, F->va_list_value };
        o.operands = o_ops;
        (void)commit_op(F, &o);
        return true;
    }
    if (name_eq(name, "llvm.intr.vaend")) {
        // No-op under the wasm32 ABI.
        return true;
    }
    if (name_eq(name, "llvm.intr.vacopy")) {
        // Copy 4 bytes (a char*) from src to dst.
        if (MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle da, sa;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &da)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &sa)) return false;
        wasmssa_op_t ol = {0};
        ol.type = OP_TYPE_WASMSSA_LOAD;
        ol.valtype = WT_I32;
        ol.mem_size_bytes = 4;
        ol.memory_align_log2 = 2;
        ol.memory_offset = 0;
        ol.n_operands = 1;
        MLIR_ValueHandle ol_ops[1] = { sa };
        ol.operands = ol_ops;
        ol.has_result = true;
        MLIR_ValueHandle lidx = commit_op(F, &ol);
        wasmssa_op_t os = {0};
        os.type = OP_TYPE_WASMSSA_STORE;
        os.valtype = WT_I32;
        os.mem_size_bytes = 4;
        os.memory_align_log2 = 2;
        os.memory_offset = 0;
        os.n_operands = 2;
        MLIR_ValueHandle os_ops[2] = { da, lidx };
        os.operands = os_ops;
        (void)commit_op(F, &os);
        return true;
    }

    // ---- llvm.intr.sqrt ---------------------------------------------------
    if (name_eq(name, "llvm.intr.sqrt")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        uint8_t opc = (vt == WT_F32) ? 0x91 : (vt == WT_F64) ? 0x9f : 0;
        if (opc == 0) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.wasm.memory.size ---------------------------------------
    // `__builtin_wasm_memory_size(0)` lowers to this. No operands (the
    // memory index byte is hard-coded to 0 in the binary encoder); the
    // single i32 result is the current memory size in 64KiB pages.
    if (name_eq(name, "llvm.intr.wasm.memory.size")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt != WT_I32) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_MEMORY_SIZE;
        o.valtype = vt;
        o.n_operands = 0;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.wasm.memory.grow ---------------------------------------
    // `__builtin_wasm_memory_grow(0, n)` lowers to this. One i32 operand
    // (pages to grow by) and one i32 result (previous size in pages or -1).
    if (name_eq(name, "llvm.intr.wasm.memory.grow")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt != WT_I32) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_MEMORY_GROW;
        o.valtype = vt;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    fprintf(stderr, "wasmssa-lower: unsupported op '%.*s'\n",
            (int)name.size, name.str);
    return false;
}

// =============================================================================
// Function lowering: pre-walk for alloca offsets, then op-by-op walk.
// =============================================================================
static bool prewalk_block(FnCtx *F, MLIR_BlockHandle blk);

// For a variadic llvm.call, compute the buffer size and per-arg offsets
// needed to pack the variadic args under the wasm32 clang ABI. Returns
// false if the call is not variadic or the variadic args have an
// unsupported wasm value type. On success, *total_size receives the total
// buffer size (in bytes), *n_fixed receives the number of fixed
// (non-variadic) operands. If out_offsets is non-NULL, it must point to a
// caller-provided array large enough for (n_variadic) entries and will be
// filled with the per-variadic-arg buffer offsets (in operand order).
static bool va_call_layout(FnCtx *F, MLIR_OpHandle op, uint32_t *total_size,
                           size_t *n_fixed, uint32_t *out_offsets) {
    MLIR_AttributeHandle vct = MLIR_GetOpAttributeByName(op, "var_callee_type");
    if (vct == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(vct);
    if (!MLIR_GetTypeFunctionIsVarArg(fty)) return false;
    size_t nf = MLIR_GetTypeFunctionNumInputs(fty);
    size_t no = MLIR_GetOpNumOperands(op);
    if (no < nf) return false;
    uint32_t cur = 0;
    for (size_t i = nf; i < no; i++) {
        MLIR_TypeHandle aty = MLIR_GetValueType(MLIR_GetOpOperand(op, i));
        uint8_t vt = wasm_vt(F->ctx, aty);
        unsigned sz = type_size_bytes(F->ctx, aty);
        unsigned al;
        if (vt == WT_I32) {
            // i8/i16 are widened to i32 in C variadic promotion; the
            // frontend currently emits all small ints as i32 already.
            sz = 4; al = 4;
        } else if (vt == WT_I64) {
            sz = 8; al = 8;
        } else if (vt == WT_F32) {
            // C variadic promotion: f32 -> f64. Pack as 8 bytes f64.
            sz = 8; al = 8;
        } else if (vt == WT_F64) {
            sz = 8; al = 8;
        } else {
            return false;
        }
        cur = align_up(cur, al);
        if (out_offsets) out_offsets[i - nf] = cur;
        cur += sz;
        (void)sz;
    }
    *total_size = cur;
    *n_fixed = nf;
    return true;
}

static bool prewalk_op(FnCtx *F, MLIR_OpHandle op) {
    string n = MLIR_GetOpName(op);
    if (name_eq(n, "llvm.br") || name_eq(n, "llvm.cond_br") ||
        name_eq(n, "llvm.switch") ||
        name_eq(n, "cf.br") || name_eq(n, "cf.cond_br") ||
        name_eq(n, "cf.switch")) {
        fprintf(stderr,
                "wasmssa-lower: control flow not lifted to scf (%.*s)\n",
                (int)n.size, n.str);
        return false;
    }
    if (name_eq(n, "llvm.call")) {
        uint32_t sz = 0; size_t nfixed = 0;
        if (va_call_layout(F, op, &sz, &nfixed, NULL)) {
            if (sz > F->va_buf_size) F->va_buf_size = sz;
        }
    }
    if (name_eq(n, "llvm.alloca")) {
        MLIR_TypeHandle et = MLIR_INVALID_HANDLE;
        MLIR_AttributeHandle ea = find_attr(op, "elem_type");
        if (ea != MLIR_INVALID_HANDLE) et = MLIR_GetAttributeTypeValue(ea);
        unsigned esz = et != MLIR_INVALID_HANDLE
                           ? type_size_bytes(F->ctx, et) : 0;
        if (esz == 0) {
            fprintf(stderr,
                    "wasmssa-lower: alloca of unsupported element type\n");
            return false;
        }
        int64_t cnt = 1;
        if (MLIR_GetOpNumOperands(op) >= 1) {
            MLIR_ValueHandle co = MLIR_GetOpOperand(op, 0);
            MLIR_OpHandle    cd = MLIR_GetValueDefiningOp(co);
            if (cd == MLIR_INVALID_HANDLE) return false;
            if (!name_eq(MLIR_GetOpName(cd), "llvm.mlir.constant")) {
                fprintf(stderr,
                        "wasmssa-lower: dynamic alloca size not supported\n");
                return false;
            }
            MLIR_AttributeHandle va = find_attr(cd, "value");
            cnt = MLIR_GetAttributeInteger(va);
        }
        unsigned ealign = type_align_bytes(F->ctx, et);
        if (ealign < 4) ealign = 4;
        F->frame_size = align_up(F->frame_size, ealign);
        uint32_t off = F->frame_size;
        F->frame_size += (uint32_t)(esz * cnt);
        amap_set(F, MLIR_GetOpResult(op, 0), off);
    }
    // Recurse into nested regions for nested allocas.
    size_t nr = MLIR_GetOpNumRegions(op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle rg = MLIR_GetOpRegion(op, r);
        size_t nblk = MLIR_GetRegionNumBlocks(rg);
        for (size_t b = 0; b < nblk; b++) {
            if (!prewalk_block(F, MLIR_GetRegionBlock(rg, b))) return false;
        }
    }
    return true;
}

static bool prewalk_block(FnCtx *F, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < nops; i++) {
        if (!prewalk_op(F, MLIR_GetBlockOp(blk, i))) return false;
    }
    return true;
}

static bool prewalk_func(FnCtx *F, MLIR_BlockHandle entry) {
    if (!prewalk_block(F, entry)) return false;
    if (F->va_buf_size > 0) {
        F->frame_size = align_up(F->frame_size, 8);
        F->va_buf_offset = F->frame_size;
        F->frame_size += F->va_buf_size;
    }
    F->frame_size = (F->frame_size + 15) & ~15u;
    return true;
}

static bool lower_block(FnCtx *F, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < nops; i++) {
        if (!lower_op(F, MLIR_GetBlockOp(blk, i))) return false;
    }
    return true;
}

// Lower a defined `llvm.func` into a `wasmssa.func` op appended to the
// module body. The function's signature attrs are passed in (param_types
// includes the synthetic trailing i32 for vararg functions); the special
// "__original_main" rename + exported flag are picked by the caller.
static bool lower_function(MLIR_Context *ctx, Arena *arena, ModCtx *mod,
                           MLIR_BlockHandle mod_body,
                           string fn_name, bool exported,
                           MLIR_OpHandle fn,
                           const uint8_t *param_types, size_t n_params,
                           const uint8_t *result_types, size_t n_results) {
    FnCtx F;
    memset(&F, 0, sizeof F);
    F.ctx = ctx;
    F.arena = arena;
    F.mod = mod;
    F.n_params = n_params;
    F.sp_value = MLIR_INVALID_HANDLE;
    F.va_list_value = MLIR_INVALID_HANDLE;

    // If this function is variadic, sig_for_func has appended a synthetic
    // i32 va_list param at index n_params-1. The MLIR entry block only has
    // the original (non-hidden) parameters as block args.
    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    bool is_vararg = false;
    if (ftya != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
        is_vararg = MLIR_GetTypeFunctionIsVarArg(fty);
    }
    size_t orig_np = is_vararg ? n_params - 1 : n_params;

    MLIR_RegionHandle body = MLIR_GetOpRegion(fn, 0);
    MLIR_BlockHandle  entry = MLIR_GetRegionBlock(body, 0);

    // Build the wasmssa.func body block. For each declared wasm param
    // (including the hidden va_list i32 for varargs), create a fresh
    // block-arg value and bind it via vmap to the corresponding LLVM
    // function entry block-arg.
    F.body_block = MLIR_CreateBlock(ctx);
    MLIR_ValueHandle last_bv = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < n_params; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, param_types[i]);
        MLIR_ValueHandle bv = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
                                                      MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, F.body_block, bv);
        if (i < orig_np) {
            vmap_set(&F, MLIR_GetBlockArg(entry, i), bv);
        }
        last_bv = bv;
    }
    if (is_vararg) F.va_list_value = last_bv;

    if (!prewalk_func(&F, entry)) goto fail;

    // Prologue: __stack_pointer -= frame_size; sp_def = result.
    if (F.frame_size > 0) {
        MLIR_ValueHandle sp_orig = emit_global_get(&F, /*sp*/0);
        MLIR_ValueHandle kf      = emit_const_i32(&F, (int32_t)F.frame_size);
        MLIR_ValueHandle sp_new  = emit_sub_i32(&F, sp_orig, kf);
        emit_global_set(&F, /*sp*/0, sp_new);
        F.sp_value = sp_new;
    }

    size_t nops = MLIR_GetBlockNumOps(entry);
    for (size_t i = 0; i < nops; i++) {
        if (!lower_op(&F, MLIR_GetBlockOp(entry, i))) goto fail;
    }

    {
        // Build the wasmssa.func wrapper op around F.body_block.
        MLIR_AttributeHandle attrs[8];
        size_t na = 0;
        attrs[na++] = attr_s(ctx, "sym_name",
                             fn_name.str ? fn_name.str : "",
                             fn_name.str ? fn_name.size : 0);
        attrs[na++] = attr_s_hex(ctx, arena, "param_types", param_types, n_params);
        attrs[na++] = attr_s_hex(ctx, arena, "result_types", result_types, n_results);
        attrs[na++] = attr_b(ctx, "exported", exported);
        // `static` C functions arrive as `llvm.func` with a
        // `llvm.linkage = "#llvm.linkage<internal>"` attribute (set in
        // emit.c). Forward that as a boolean `internal` flag on the
        // wasmssa.func so the wasm binary emitter can mark the
        // function's symbol-table entry BINDING_LOCAL — otherwise two
        // `static`s with the same name in different TUs collide at
        // link time.
        bool internal = false;
        MLIR_AttributeHandle linka = find_attr(fn, "llvm.linkage");
        if (linka != MLIR_INVALID_HANDLE) {
            string ls = MLIR_GetAttributeString(linka);
            const char *want = "#llvm.linkage<internal>";
            size_t wantn = strlen(want);
            if (ls.size == wantn && memcmp(ls.str, want, wantn) == 0) {
                internal = true;
            }
        }
        attrs[na++] = attr_b(ctx, "internal", internal);
        // Forward `wasm.export_name` (from `__attribute__((__export_name__("...")))`)
        // so the binary emitter can publish the function under the
        // user-requested export name.
        MLIR_AttributeHandle exa = find_attr(fn, "wasm.export_name");
        if (exa != MLIR_INVALID_HANDLE) {
            string es = MLIR_GetAttributeString(exa);
            attrs[na++] = attr_s(ctx, "export_name", es.str, es.size);
        }

        MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
        MLIR_AppendRegionBlock(ctx, region, F.body_block);
        MLIR_RegionHandle regs[1] = { region };
        MLIR_OpHandle op = MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_FUNC,
            op_type_to_string(OP_TYPE_WASMSSA_FUNC),
            attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
            MLIR_CreateLocationUnknown(ctx, (string){0}),
            MLIR_INVALID_HANDLE, (string){0}, -1);
        MLIR_AppendBlockOp(ctx, mod_body, op);
    }

    return true;
fail:
    return false;
}

// =============================================================================
// LLVM CFG structurization — discovery, snapshot, normalization, analysis.
//
// Target pipeline (cf-lowering.md):
//   LLVM MLIR  ->  CFGInfo (immutable snapshot)
//             ->  NormalizedCFG (mutable graph for M5/M7/M8 rewrites)
//             ->  StructuredPlan (future)
//             ->  wasmssa.if / loop / block emission (future)
//
// Upstream reference: mlir/lib/Transforms/Utils/CFGToSCF.cpp and the
// mlir_lift_cf_to_scf.c port. Phase mapping:
//   M5 — return normalization (shared return block, unify exits)  [implemented]
//   M7 — cycle normalization (SCC decomposition, loop latches)
//   M8 — branch normalization (entry/exit muxes, edge multiplexing)
//
// LLVM MLIR is never mutated during normalization; only NormalizedCFG changes.
// Discovery mirrors the cf->scf driver's post-order region walk.
// =============================================================================

// ---------------------------------------------------------------------------
// CFGInfo — immutable per-region CFG snapshot taken once from LLVM MLIR.
// Upstream: initial CFG view before CFGToSCF graph rewrites; successor_slot
// preserves llvm.cond_br / llvm.switch operand identity for edge values.
// ---------------------------------------------------------------------------

// Classifies the terminator on a CFG block (llvm.br, llvm.cond_br, etc.).
typedef enum {
    CFG_TERM_NONE,
    CFG_TERM_BR,
    CFG_TERM_COND_BR,
    CFG_TERM_SWITCH,
    CFG_TERM_RETURN,
    CFG_TERM_UNREACHABLE,
} CFGTermKind;

// Directed CFG edge in the immutable snapshot pool.
typedef struct {
    size_t from;            // dense source block index
    size_t to;              // dense destination block index
    size_t successor_slot;  // terminator successor slot on `from`
} CFGEdge;

// One basic block in the immutable snapshot with adjacency into the edge pool.
typedef struct {
    MLIR_BlockHandle block;
    MLIR_OpHandle    terminator;
    CFGTermKind      term_kind;
    bool             reachable;
    size_t          *edge_by_successor_slot;  // [n_successor_slots], SIZE_MAX if none
    size_t           n_successor_slots;
    size_t          *out_edges;   // [n_out_edges] compact valid edge pool ids
    size_t           n_out_edges;
    size_t          *in_edges;    // incoming pool indices
    size_t           n_in_edges;
} CFGBlock;

// Open-addressing hash map: MLIR_BlockHandle -> dense block index for O(1) lookup.
typedef struct {
    uintptr_t *keys;   // MLIR_BlockHandle, 0 = empty slot
    size_t    *vals;   // dense block index
    size_t     cap;    // power of two
    size_t     n;
} BlockIndexMap;

// Immutable CFG snapshot for one llvm.func region. Built once; never mutated.
typedef struct {
    MLIR_RegionHandle region;
    MLIR_BlockHandle  entry_block;
    CFGBlock         *blocks;
    size_t            n_blocks;
    CFGEdge          *edges;
    size_t            n_edges;
    BlockIndexMap     block_to_index;
} CFGInfo;

// ---------------------------------------------------------------------------
// NormalizedCFG — mutable working graph for M5/M7/M8 (CFGToSCF-style rewrites).
// Synthetic blocks, edge redirection, and mux operands live here; LLVM MLIR
// stays frozen. generation + rewrite batching invalidate cached analyses.
// ---------------------------------------------------------------------------

// Growable per-block list of NormEdge pool indices (outgoing or incoming).
typedef struct {
    size_t *edge_ids;
    size_t  n;
    size_t  cap;
} NormAdjacency;

// Semantic role of a synthetic block (why it exists in the normalized graph).
// Upstream: mux blocks, unified return, loop latch in CFGToSCF.cpp.
typedef enum {
    NORM_BLOCK_ORIGINAL,
    NORM_BLOCK_SHARED_RETURN,
    NORM_BLOCK_ENTRY_MUX,
    NORM_BLOCK_EXIT_MUX,
    NORM_BLOCK_LOOP_LATCH,
    NORM_BLOCK_LOOP_EXIT_DISPATCH,
} NormBlockKind;

// ---------------------------------------------------------------------------
// Value model — NormOperand (normalization) vs PlanValue (structured emission).
// Upstream: CFGToSCF edge/block argument forwarding; discriminators and typed
// undef padding for multiplexer edges (CFGToSCF.cpp edge multiplexing).
// ---------------------------------------------------------------------------

// Operand tag for values flowing on normalized CFG edges and terminators.
typedef enum {
    NORM_OPERAND_INVALID = 0,
    NORM_OPERAND_MLIR,
    NORM_OPERAND_BLOCK_ARG,
    NORM_OPERAND_DISCRIMINATOR,
    NORM_OPERAND_UNDEF,
} NormOperandKind;

// Tagged union for a value on a normalized edge or in a synthetic terminator.
typedef struct {
    NormOperandKind kind;
    MLIR_TypeHandle type;
    union {
        MLIR_ValueHandle mlir_value;
        struct {
            size_t block_id;
            size_t arg_index;
        } block_arg;
        int32_t discriminator;
    } as;
} NormOperand;

// Values in the structured plan layer (after structurization). Not on edges.
typedef enum {
    PLAN_VALUE_OPERAND,
    PLAN_VALUE_STRUCTURED_RESULT,
} PlanValueKind;

typedef struct {
    PlanValueKind kind;
    NormOperand   operand;
    size_t        plan_node_id;
    size_t        result_index;
} PlanValue;

// Role of a block argument in a synthetic block signature (not operand kind).
typedef enum {
    NORM_BLOCK_ARG_FORWARDED,
    NORM_BLOCK_ARG_DISCRIMINATOR,
    NORM_BLOCK_ARG_LOOP_FLAG,
    NORM_BLOCK_ARG_EXTRA,
} NormBlockArgRole;

// Type + role for one formal argument on a synthetic (or snapshot) block.
typedef struct {
    MLIR_TypeHandle   type;
    NormBlockArgRole  role;
} NormBlockArg;

// Maps one mux destination to its argument slice and discriminator constant.
// Upstream: CFGToSCF multiplexer destination layout.
typedef struct {
    size_t  destination_block;
    size_t  arg_offset;
    size_t  n_args;
    int32_t discriminator;
} NormMuxEntry;

// Synthetic terminator metadata (switch cases, return values, loop latch).
// Upstream: private terminator state in CFGToSCF before scf emission.
typedef struct {
    CFGTermKind  kind;
    NormOperand  selector;
    int32_t     *case_values;
    size_t      *case_edge_ids;
    size_t       n_cases;
    size_t       default_edge_id;
    NormOperand *return_values;
    size_t       n_return_values;
    NormOperand  should_repeat;  // loop latch cond_br
} NormTerminator;

// Whether a block's effective terminator comes from the LLVM snapshot or an
// owned NormalizedCFG override (M5 virtual branches, synthetic mux/latch).
typedef enum {
    NORM_TERM_SNAPSHOT,
    NORM_TERM_OWNED,
} NormTermSource;

// One node in the mutable normalized graph (original LLVM block or synthetic).
typedef struct {
    NormBlockKind    kind;
    MLIR_BlockHandle mlir_block;
    MLIR_OpHandle    source_terminator;
    CFGTermKind      source_term_kind;

    NormTermSource   term_source;
    NormTerminator   normalized_term;
    CFGTermKind      effective_term_kind;

    NormBlockArg    *args;
    size_t           n_args;
    NormMuxEntry    *mux_entries;
    size_t           n_mux_entries;
    NormAdjacency    outgoing;
    NormAdjacency    incoming;
    bool             active;
    bool             outgoing_finalized;
} NormBlock;

// Whether edge operands are lazy snapshot refs or owned synthetic copies.
typedef enum {
    NORM_EDGE_SNAPSHOT,
    NORM_EDGE_SYNTHETIC,
} NormEdgePayloadKind;

// SSA values forwarded across a normalized edge (snapshot or synthetic).
typedef struct {
    NormEdgePayloadKind kind;
    size_t              snapshot_edge_id;  // valid when NORM_EDGE_SNAPSHOT
    NormOperand        *operands;
    size_t              n_operands;
} NormEdgePayload;

// Mutable directed edge with adjacency positions for O(1) redirect/remove.
typedef struct {
    size_t from;
    size_t to;
    size_t successor_slot;
    size_t out_position;
    size_t in_position;
    NormEdgePayload payload;
    bool   active;
} NormEdge;

// ---------------------------------------------------------------------------
// CFG analysis caches — lazy, generation-tracked graph analyses.
// Upstream: Cooper-Harvey-Kennedy dominance (mlir_lift_cf_to_scf.c DomInfo),
// reverse postorder for SCC (M7) and branch structuring (M8).
// ---------------------------------------------------------------------------

// Reverse postorder with block->position inverse map for dominance intersect.
typedef struct {
    size_t *order;
    size_t *position;
    size_t  n;
} RPOInfo;

// Immediate-dominator tree; post_dominance uses root/has_virtual_exit for exits.
typedef struct {
    size_t *idom;
    size_t  n;
    size_t  root;
    bool    has_virtual_exit;
} DominanceInfo;

// Tarjan SCC result with component member ranges (M7 cycle normalization).
typedef struct {
    size_t *component_id;
    size_t *component_offsets;
    size_t *members;
    size_t  n;
    size_t  n_components;
} SCCInfo;

// Lazily computed analyses keyed by NormalizedCFG.generation.
typedef struct {
    size_t         generation;
    RPOInfo        rpo;
    DominanceInfo  dominance;
    DominanceInfo  post_dominance;
    SCCInfo        scc;
    bool           has_rpo;
    bool           has_dominance;
    bool           has_post_dominance;
    bool           has_scc;
} CFGAnalysisCache;

// Mutable normalized graph + arena pointers + analysis cache.
typedef struct {
    const CFGInfo *snapshot;

    NormBlock *blocks;
    size_t     n_blocks;
    size_t     blocks_cap;

    NormEdge *edges;
    size_t    n_edges;
    size_t    edges_cap;

    size_t generation;
    size_t entry_block;
    size_t rewrite_depth;
    bool   rewrite_dirty;

    Arena      *graph_arena;
    Arena      *analysis_arena;
    arena_pos_t analysis_base_pos;
    Arena      *phase_arena;
    arena_pos_t phase_base_pos;

    CFGAnalysisCache analysis;
} NormalizedCFG;

// Single-pass iterator over active outgoing/incoming edges of one block.
typedef struct {
    const NormalizedCFG *cfg;
    const NormAdjacency *adj;
    size_t               i;
} NormEdgeIter;

// DFS stack frame for iterative RPO computation.
typedef struct {
    size_t bid;
    size_t edge_i;
} NormRpoFrame;

// One llvm.func region that still needs CFG->structured lowering.
typedef struct {
    MLIR_OpHandle     fn_op;        // owning llvm.func
    MLIR_RegionHandle region;       // region whose CFG needs structurizing
    MLIR_BlockHandle  entry_block;  // first block of `region`
} LiftRegionTarget;

// Growable array of LiftRegionTarget discovered in a module walk.
typedef struct {
    LiftRegionTarget *items;
    size_t            n;
    size_t            cap;
} LiftRegionTargetList;

// Graph, analysis, and phase arenas use separate lifetimes:
//   graph_arena    — CFGInfo snapshot + persistent NormalizedCFG graph payloads
//   analysis_arena — cached RPO/dominance/SCC + temporary workspaces
//   phase_arena    — temporary M5/M7/M8 plans and worklists (reset after commit)
typedef struct {
    Arena      *graph_arena;
    arena_pos_t graph_base_pos;
    Arena      *analysis_arena;
    arena_pos_t analysis_base_pos;
    Arena      *phase_arena;
    arena_pos_t phase_base_pos;
} CFGAnalysisArena;

// M5 — return normalization (shared exit blocks for llvm.return sites).
typedef enum {
    M5_EXIT_LLVM_RETURN,
    // Later: M5_EXIT_LLVM_UNREACHABLE,
} M5ExitFlavor;

typedef struct {
    M5ExitFlavor     flavor;
    MLIR_TypeHandle *operand_types;
    size_t           n_operand_types;
    size_t           shared_exit_block;
    size_t           n_sites;
} M5ExitClass;

typedef struct {
    size_t       source_block;
    size_t       exit_class;
    NormOperand *return_values;
    size_t       n_return_values;
} M5ExitSite;

typedef struct {
    M5ExitClass *classes;
    size_t       n_classes;
    size_t       classes_cap;

    M5ExitSite *sites;
    size_t      n_sites;
    size_t      sites_cap;
} M5Plan;

// Persistent exit combiner retained after M5 for M7 loop-exit normalization.
typedef struct {
    NormalizedCFG *cfg;
    M5ExitClass    *classes;
    size_t          n_classes;
    size_t          classes_cap;
} M5ExitCombiner;

// View of one target's CFG snapshot + normalized graph in a shared arena pool.
typedef struct {
    CFGAnalysisArena *pool;
    CFGInfo           cfg;
    NormalizedCFG     norm;
    M5ExitCombiner    m5;
} CFGInfoScratch;

// ---------------------------------------------------------------------------
// BlockIndexMap — O(1) MLIR_BlockHandle -> dense index.
// Upstream: dense block numbering used throughout CFGToSCF analyses.
// ---------------------------------------------------------------------------
// Look up dense block index for an MLIR block handle; SIZE_MAX if absent.
static size_t block_index_map_probe(const BlockIndexMap *map, MLIR_BlockHandle b) {
    if (map->cap == 0) return SIZE_MAX;
    uintptr_t key = (uintptr_t)b;
    size_t mask = map->cap - 1;
    size_t i = map_hash(key) & mask;
    while (map->keys[i] != 0) {
        if (map->keys[i] == key) return map->vals[i];
        i = (i + 1) & mask;
    }
    return SIZE_MAX;
}

// Initialize an empty BlockIndexMap with at least `min_cap` slots.
static void block_index_map_init(BlockIndexMap *map, Arena *arena, size_t min_cap) {
    memset(map, 0, sizeof(*map));
    size_t cap = 16;
    while (cap < min_cap) cap *= 2;
    map->cap = cap;
    map->keys = arena_new_array(arena, uintptr_t, cap);
    map->vals = arena_new_array(arena, size_t, cap);
    memset(map->keys, 0, cap * sizeof(uintptr_t));
}

// Double BlockIndexMap capacity and rehash all entries into a new table.
static void block_index_map_grow(BlockIndexMap *map, Arena *arena) {
    size_t ocap = map->cap;
    uintptr_t *okeys = map->keys;
    size_t *ovals = map->vals;
    size_t ncap = ocap ? ocap * 2 : 16;
    map->cap = ncap;
    map->keys = arena_new_array(arena, uintptr_t, ncap);
    map->vals = arena_new_array(arena, size_t, ncap);
    memset(map->keys, 0, ncap * sizeof(uintptr_t));
    map->n = 0;
    size_t mask = ncap - 1;
    for (size_t i = 0; i < ocap; ++i) {
        uintptr_t key = okeys[i];
        if (key == 0) continue;
        size_t j = map_hash(key) & mask;
        while (map->keys[j] != 0) j = (j + 1) & mask;
        map->keys[j] = key;
        map->vals[j] = ovals[i];
        map->n++;
    }
}

// Insert block->index mapping; first insert wins on duplicate keys.
static void block_index_map_put(BlockIndexMap *map, Arena *arena,
                                MLIR_BlockHandle b, size_t index) {
    uintptr_t key = (uintptr_t)b;
    if ((map->n + 1) * 4 >= map->cap * 3) block_index_map_grow(map, arena);
    size_t mask = map->cap - 1;
    size_t i = map_hash(key) & mask;
    while (map->keys[i] != 0) {
        if (map->keys[i] == key) return;  // first insert wins
        i = (i + 1) & mask;
    }
    map->keys[i] = key;
    map->vals[i] = index;
    map->n++;
}

// ---------------------------------------------------------------------------
// CFGInfo query helpers.
// ---------------------------------------------------------------------------
// Map an LLVM terminator op to CFGTermKind for snapshot classification.
static CFGTermKind cfg_term_kind_of(MLIR_OpHandle term) {
    if (term == MLIR_INVALID_HANDLE) return CFG_TERM_NONE;
    string n = MLIR_GetOpName(term);
    if (name_eq(n, "llvm.br")) return CFG_TERM_BR;
    if (name_eq(n, "llvm.cond_br")) return CFG_TERM_COND_BR;
    if (name_eq(n, "llvm.switch")) return CFG_TERM_SWITCH;
    if (name_eq(n, "llvm.return")) return CFG_TERM_RETURN;
    if (name_eq(n, "llvm.unreachable")) return CFG_TERM_UNREACHABLE;
    return CFG_TERM_NONE;
}

// Dense index of `block` in CFGInfo, or SIZE_MAX.
static size_t cfg_info_block_index(const CFGInfo *cfg, MLIR_BlockHandle block) {
    return block_index_map_probe(&cfg->block_to_index, block);
}

// MLIR block handle at dense index `idx`.
static MLIR_BlockHandle cfg_info_block_at(const CFGInfo *cfg, size_t idx) {
    if (idx >= cfg->n_blocks) return MLIR_INVALID_HANDLE;
    return cfg->blocks[idx].block;
}

// Terminator op on block `idx` (llvm.br, llvm.cond_br, etc.).
static MLIR_OpHandle cfg_info_block_terminator(const CFGInfo *cfg, size_t idx) {
    if (idx >= cfg->n_blocks) return MLIR_INVALID_HANDLE;
    return cfg->blocks[idx].terminator;
}

// Cached CFGTermKind for block `idx`.
static CFGTermKind cfg_info_block_term_kind(const CFGInfo *cfg, size_t idx) {
    if (idx >= cfg->n_blocks) return CFG_TERM_NONE;
    return cfg->blocks[idx].term_kind;
}

static const CFGEdge *cfg_info_edge_at(const CFGInfo *cfg, size_t edge_id) {
    if (edge_id >= cfg->n_edges) return NULL;
    return &cfg->edges[edge_id];
}

// Edge pool id for successor slot `succ_slot` on block `block_idx`.
static size_t cfg_info_edge_id_for_slot(const CFGInfo *cfg, size_t block_idx,
                                        size_t succ_slot) {
    if (block_idx >= cfg->n_blocks) return SIZE_MAX;
    CFGBlock *b = &cfg->blocks[block_idx];
    if (succ_slot >= b->n_successor_slots) return SIZE_MAX;
    return b->edge_by_successor_slot[succ_slot];
}

// Incoming edge pool id at predecessor index `pred_i`.
static size_t cfg_info_in_edge_id(const CFGInfo *cfg, size_t block_idx,
                                  size_t pred_i) {
    if (block_idx >= cfg->n_blocks) return SIZE_MAX;
    CFGBlock *b = &cfg->blocks[block_idx];
    if (pred_i >= b->n_in_edges) return SIZE_MAX;
    return b->in_edges[pred_i];
}

// Number of terminator successor slots on block `idx`.
static size_t cfg_info_num_successor_slots(const CFGInfo *cfg, size_t idx) {
    if (idx >= cfg->n_blocks) return 0;
    return cfg->blocks[idx].n_successor_slots;
}

// Count of valid outgoing edges in the snapshot pool.
static size_t cfg_info_num_out_edges(const CFGInfo *cfg, size_t idx) {
    if (idx >= cfg->n_blocks) return 0;
    return cfg->blocks[idx].n_out_edges;
}

// Outgoing edge pool id at compact out-edge index `out_i`.
static size_t cfg_info_out_edge_id_at(const CFGInfo *cfg, size_t block_idx,
                                      size_t out_i) {
    if (block_idx >= cfg->n_blocks) return SIZE_MAX;
    CFGBlock *b = &cfg->blocks[block_idx];
    if (out_i >= b->n_out_edges) return SIZE_MAX;
    return b->out_edges[out_i];
}

// Dense destination block index for successor slot `succ_slot`.
static size_t cfg_info_successor_index(const CFGInfo *cfg, size_t idx,
                                       size_t succ_slot) {
    size_t eid = cfg_info_edge_id_for_slot(cfg, idx, succ_slot);
    const CFGEdge *e = cfg_info_edge_at(cfg, eid);
    return e ? e->to : SIZE_MAX;
}

// MLIR block handle for successor slot `succ_slot` on block `idx`.
static MLIR_BlockHandle cfg_info_successor_block(const CFGInfo *cfg, size_t idx,
                                                 size_t succ_slot) {
    size_t si = cfg_info_successor_index(cfg, idx, succ_slot);
    return cfg_info_block_at(cfg, si);
}

// Number of incoming edges to block `idx`.
static size_t cfg_info_num_predecessors(const CFGInfo *cfg, size_t idx) {
    if (idx >= cfg->n_blocks) return 0;
    return cfg->blocks[idx].n_in_edges;
}

// Dense source block index of predecessor `pred_i`.
static size_t cfg_info_predecessor_index(const CFGInfo *cfg, size_t idx,
                                         size_t pred_i) {
    size_t eid = cfg_info_in_edge_id(cfg, idx, pred_i);
    const CFGEdge *e = cfg_info_edge_at(cfg, eid);
    return e ? e->from : SIZE_MAX;
}

// Successor slot on the predecessor's terminator for `pred_i`.
static size_t cfg_info_predecessor_succ_slot(const CFGInfo *cfg, size_t idx,
                                             size_t pred_i) {
    size_t eid = cfg_info_in_edge_id(cfg, idx, pred_i);
    const CFGEdge *e = cfg_info_edge_at(cfg, eid);
    return e ? e->successor_slot : SIZE_MAX;
}

// Full CFGEdge record for predecessor `pred_i` of block `idx`.
static CFGEdge cfg_info_incoming_edge(const CFGInfo *cfg, size_t idx,
                                      size_t pred_i) {
    CFGEdge e;
    e.from = SIZE_MAX;
    e.to = SIZE_MAX;
    e.successor_slot = SIZE_MAX;
    size_t eid = cfg_info_in_edge_id(cfg, idx, pred_i);
    const CFGEdge *ep = cfg_info_edge_at(cfg, eid);
    if (ep) e = *ep;
    return e;
}

// MLIR block handle of predecessor `pred_i` of block `idx`.
static MLIR_BlockHandle cfg_info_predecessor_block(const CFGInfo *cfg, size_t idx,
                                                   size_t pred_i) {
    size_t pi = cfg_info_predecessor_index(cfg, idx, pred_i);
    return cfg_info_block_at(cfg, pi);
}

// Count of MLIR successor operands on snapshot edge `edge_id`.
static size_t cfg_info_edge_num_operands(const CFGInfo *cfg, size_t edge_id) {
    const CFGEdge *e = cfg_info_edge_at(cfg, edge_id);
    if (!e || e->from >= cfg->n_blocks) return 0;
    MLIR_OpHandle term = cfg->blocks[e->from].terminator;
    if (term == MLIR_INVALID_HANDLE) return 0;
    return MLIR_GetOpNumSuccessorOperands(term, e->successor_slot);
}

// MLIR successor operand at `op_idx` on snapshot edge `edge_id`.
static MLIR_ValueHandle cfg_info_edge_operand(const CFGInfo *cfg, size_t edge_id,
                                              size_t op_idx) {
    const CFGEdge *e = cfg_info_edge_at(cfg, edge_id);
    if (!e || e->from >= cfg->n_blocks) return MLIR_INVALID_HANDLE;
    MLIR_OpHandle term = cfg->blocks[e->from].terminator;
    if (term == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    return MLIR_GetOpSuccessorOperand(term, e->successor_slot, op_idx);
}

// Dense index of the region entry block.
static size_t cfg_info_entry_index(const CFGInfo *cfg) {
    size_t ei = cfg_info_block_index(cfg, cfg->entry_block);
    return ei;
}

// True when BFS from entry reached every indexed block.
static bool cfg_info_all_region_blocks_reachable(const CFGInfo *cfg) {
    for (size_t i = 0; i < cfg->n_blocks; ++i) {
        if (!cfg->blocks[i].reachable) return false;
    }
    return true;
}

// Resolve in-region successor to dense index; filters unreachable blocks.
static bool cfg_edge_dest_index(const CFGInfo *cfg, MLIR_RegionHandle region,
                                MLIR_BlockHandle succ, size_t *out_dest) {
    if (succ == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetBlockParentRegion(succ) != region) return false;
    size_t di = cfg_info_block_index(cfg, succ);
    if (di == SIZE_MAX || di >= cfg->n_blocks) return false;
    if (!cfg->blocks[di].reachable) return false;
    if (out_dest) *out_dest = di;
    return true;
}

// Build immutable CFGInfo: index blocks, BFS reachability, edge pool.
// Upstream: initial CFG snapshot before CFGToSCF rewrites.
static bool cfg_info_build(Arena *arena, MLIR_RegionHandle region,
                           CFGInfo *out) {
    memset(out, 0, sizeof(*out));
    out->region = region;
    size_t n = MLIR_GetRegionNumBlocks(region);
    if (n == 0) return false;
    out->entry_block = MLIR_GetRegionBlock(region, 0);

    out->n_blocks = n;
    out->blocks = arena_new_array(arena, CFGBlock, n);
    memset(out->blocks, 0, n * sizeof(CFGBlock));
    block_index_map_init(&out->block_to_index, arena, n * 2);

    for (size_t i = 0; i < n; ++i) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, i);
        out->blocks[i].block = b;
        out->blocks[i].terminator = MLIR_GetBlockTerminator(b);
        out->blocks[i].term_kind = cfg_term_kind_of(out->blocks[i].terminator);
        block_index_map_put(&out->block_to_index, arena, b, i);
    }

    bool *visited = arena_new_array(arena, bool, n);
    memset(visited, 0, n * sizeof(bool));
    size_t *queue = arena_new_array(arena, size_t, n);
    size_t head = 0, tail = 0;
    visited[0] = true;
    out->blocks[0].reachable = true;
    queue[tail++] = 0;
    while (head < tail) {
        size_t bi = queue[head++];
        MLIR_OpHandle term = out->blocks[bi].terminator;
        if (term == MLIR_INVALID_HANDLE) continue;
        size_t ns = MLIR_GetOpNumSuccessors(term);
        for (size_t i = 0; i < ns; ++i) {
            MLIR_BlockHandle s = MLIR_GetOpSuccessor(term, i);
            if (s == MLIR_INVALID_HANDLE) continue;
            if (MLIR_GetBlockParentRegion(s) != region) continue;
            size_t si = block_index_map_probe(&out->block_to_index, s);
            if (si == SIZE_MAX || visited[si]) continue;
            visited[si] = true;
            out->blocks[si].reachable = true;
            queue[tail++] = si;
        }
    }

    // Edge pass 1: classify terminators and count edges on reachable blocks.
    size_t total_edges = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!out->blocks[i].reachable) continue;
        CFGBlock *b = &out->blocks[i];
        MLIR_OpHandle term = b->terminator;
        size_t ns = (term == MLIR_INVALID_HANDLE) ? 0
                                                  : MLIR_GetOpNumSuccessors(term);
        b->n_successor_slots = ns;
        if (ns == 0) continue;
        for (size_t s = 0; s < ns; ++s) {
            size_t di;
            if (!cfg_edge_dest_index(out, region,
                                     MLIR_GetOpSuccessor(term, s), &di)) {
                continue;
            }
            total_edges++;
            b->n_out_edges++;
            out->blocks[di].n_in_edges++;
        }
    }

    out->n_edges = total_edges;
    out->edges = total_edges ? arena_new_array(arena, CFGEdge, total_edges) : NULL;
    for (size_t i = 0; i < n; ++i) {
        CFGBlock *b = &out->blocks[i];
        if (!b->reachable) continue;
        if (b->n_successor_slots > 0) {
            b->edge_by_successor_slot =
                arena_new_array(arena, size_t, b->n_successor_slots);
            for (size_t s = 0; s < b->n_successor_slots; ++s) {
                b->edge_by_successor_slot[s] = SIZE_MAX;
            }
        }
        if (b->n_out_edges > 0) {
            b->out_edges = arena_new_array(arena, size_t, b->n_out_edges);
        }
        if (b->n_in_edges > 0) {
            b->in_edges = arena_new_array(arena, size_t, b->n_in_edges);
        }
    }

    // Edge pass 2: populate the edge pool and per-block edge-id lists.
    size_t *in_cursor = arena_new_array(arena, size_t, n);
    memset(in_cursor, 0, n * sizeof(size_t));
    size_t edge_id = 0;
    for (size_t i = 0; i < n; ++i) {
        CFGBlock *src = &out->blocks[i];
        if (!src->reachable) continue;
        MLIR_OpHandle term = src->terminator;
        if (term == MLIR_INVALID_HANDLE) continue;
        size_t out_cursor = 0;
        for (size_t s = 0; s < src->n_successor_slots; ++s) {
            size_t di;
            if (!cfg_edge_dest_index(out, region,
                                     MLIR_GetOpSuccessor(term, s), &di)) {
                continue;
            }
            out->edges[edge_id].from = i;
            out->edges[edge_id].to = di;
            out->edges[edge_id].successor_slot = s;
            src->edge_by_successor_slot[s] = edge_id;
            src->out_edges[out_cursor++] = edge_id;
            out->blocks[di].in_edges[in_cursor[di]++] = edge_id;
            edge_id++;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// NormalizedCFG — mutable graph layered on an immutable CFGInfo snapshot.
// Upstream: CFGToSCF graph mutation layer (mux insertion, edge redirect).
// ---------------------------------------------------------------------------

// --- Generation tracking and rewrite transactions ---
static void normalized_cfg_reset_analysis_storage(NormalizedCFG *n) {
    if (n->analysis_arena) {
        arena_reset(n->analysis_arena, n->analysis_base_pos);
    }
    n->analysis.has_rpo = false;
    n->analysis.has_dominance = false;
    n->analysis.has_post_dominance = false;
    n->analysis.has_scc = false;
    memset(&n->analysis.rpo, 0, sizeof(n->analysis.rpo));
    memset(&n->analysis.dominance, 0, sizeof(n->analysis.dominance));
    memset(&n->analysis.post_dominance, 0, sizeof(n->analysis.post_dominance));
    memset(&n->analysis.scc, 0, sizeof(n->analysis.scc));
}

// Alias for reset_analysis_storage.
static void normalized_cfg_invalidate_analysis(NormalizedCFG *n) {
    normalized_cfg_reset_analysis_storage(n);
}

// Increment graph generation and discard stale analysis storage.
static void normalized_cfg_bump_generation(NormalizedCFG *n) {
    n->generation++;
    normalized_cfg_reset_analysis_storage(n);
    n->analysis.generation = n->generation;
}

// Record a graph change; defer generation bump inside rewrite batches.
static void normalized_cfg_note_mutation(NormalizedCFG *n) {
    if (n->rewrite_depth > 0) {
        n->rewrite_dirty = true;
    } else {
        normalized_cfg_bump_generation(n);
    }
}

// Invalidate analysis cache when generation lags the graph.
static void cfg_analysis_cache_sync(NormalizedCFG *n) {
    if (n->analysis.generation != n->generation) {
        normalized_cfg_reset_analysis_storage(n);
        n->analysis.generation = n->generation;
    }
}

// Open a logical rewrite transaction; forbid analysis queries until end_rewrite.
static void normalized_cfg_begin_rewrite(NormalizedCFG *n) {
    if (n->rewrite_depth == 0) {
        n->rewrite_dirty = false;
    }
    n->rewrite_depth++;
}

// False while inside begin_rewrite/end_rewrite (stale analyses).
static bool normalized_cfg_analysis_available(const NormalizedCFG *n) {
    return n->rewrite_depth == 0;
}

// Close rewrite batch; bump generation once if dirty.
static void normalized_cfg_end_rewrite(NormalizedCFG *n) {
    if (n->rewrite_depth == 0) return;
    n->rewrite_depth--;
    if (n->rewrite_depth == 0 && n->rewrite_dirty) {
        normalized_cfg_bump_generation(n);
        n->rewrite_dirty = false;
    }
}

// --- Adjacency lists, operands, and block/terminator helpers ---

static void norm_ensure_block_slots(NormalizedCFG *n, Arena *arena, size_t need);

// True when edge and both endpoint blocks are active.
static bool norm_edge_is_active(const NormalizedCFG *n, size_t edge_id);
// True when `edge_id` is an active outgoing edge of `block_id` (O(1)).
static bool norm_edge_is_outgoing_from(const NormalizedCFG *n, size_t edge_id,
                                       size_t block_id);

// Begin single-pass iteration over active outgoing edges of `block_id`.
static NormEdgeIter norm_out_edges(const NormalizedCFG *n, size_t block_id) {
    NormEdgeIter it;
    it.cfg = n;
    it.i = 0;
    if (block_id >= n->n_blocks || !n->blocks[block_id].active) {
        it.adj = NULL;
        return it;
    }
    it.adj = &n->blocks[block_id].outgoing;
    return it;
}

// Begin single-pass iteration over active incoming edges of `block_id`.
static NormEdgeIter norm_in_edges(const NormalizedCFG *n, size_t block_id) {
    NormEdgeIter it;
    it.cfg = n;
    it.i = 0;
    if (block_id >= n->n_blocks || !n->blocks[block_id].active) {
        it.adj = NULL;
        return it;
    }
    it.adj = &n->blocks[block_id].incoming;
    return it;
}

// Advance edge iterator; returns next active edge id or false.
static bool norm_edge_iter_next(NormEdgeIter *it, size_t *out_edge_id) {
    if (!it->adj) return false;
    while (it->i < it->adj->n) {
        size_t eid = it->adj->edge_ids[it->i++];
        if (!norm_edge_is_active(it->cfg, eid)) continue;
        if (out_edge_id) *out_edge_id = eid;
        return true;
    }
    return false;
}

// Allocate exact-capacity adjacency list (snapshot build path).
static void norm_adj_prealloc(Arena *arena, NormAdjacency *adj, size_t cap) {
    adj->n = 0;
    adj->cap = 0;
    adj->edge_ids = NULL;
    if (cap == 0) return;
    adj->edge_ids = arena_new_array(arena, size_t, cap);
    adj->cap = cap;
}

// Append edge id assuming adjacency has spare capacity.
static void norm_adj_append(NormAdjacency *adj, size_t edge_id) {
    adj->edge_ids[adj->n++] = edge_id;
}

// Append edge id, growing adjacency geometrically when needed.
static void norm_adj_push(Arena *arena, NormAdjacency *adj, size_t edge_id) {
    if (adj->cap > 0 && adj->n < adj->cap) {
        norm_adj_append(adj, edge_id);
        return;
    }
    if (adj->n >= adj->cap) {
        size_t ncap = adj->cap ? adj->cap * 2 : 4;
        size_t *next = arena_new_array(arena, size_t, ncap);
        if (adj->n > 0) memcpy(next, adj->edge_ids, adj->n * sizeof(size_t));
        adj->edge_ids = next;
        adj->cap = ncap;
    }
    adj->edge_ids[adj->n++] = edge_id;
}

// Grow block storage so `n_blocks + extra` slots exist without realloc during commit.
static bool normalized_cfg_reserve_blocks(NormalizedCFG *n, size_t extra) {
    if (!n->graph_arena) return false;
    norm_ensure_block_slots(n, n->graph_arena, n->n_blocks + extra);
    return n->blocks_cap >= n->n_blocks + extra;
}

// Grow edge pool so `n_edges + extra` slots exist without realloc during commit.
static bool normalized_cfg_reserve_edges(NormalizedCFG *n, size_t extra) {
    if (!n->graph_arena) return false;
    size_t need = n->n_edges + extra;
    while (n->edges_cap < need) {
        size_t ncap = n->edges_cap ? n->edges_cap * 2 : 16;
        if (ncap < need) ncap = need;
        NormEdge *ne = arena_new_array(n->graph_arena, NormEdge, ncap);
        if (!ne) return false;
        if (n->n_edges > 0) {
            memcpy(ne, n->edges, n->n_edges * sizeof(NormEdge));
        }
        n->edges = ne;
        n->edges_cap = ncap;
    }
    return true;
}

// Reserve outgoing adjacency capacity on `block_id` for `extra` more edges.
static bool normalized_cfg_reserve_outgoing(NormalizedCFG *n, size_t block_id,
                                            size_t extra) {
    if (block_id >= n->n_blocks) return false;
    if (!n->graph_arena) return false;
    NormAdjacency *adj = &n->blocks[block_id].outgoing;
    size_t need = adj->n + extra;
    if (adj->cap >= need) return true;
    size_t *next = arena_new_array(n->graph_arena, size_t, need);
    if (!next) return false;
    if (adj->n > 0) {
        memcpy(next, adj->edge_ids, adj->n * sizeof(size_t));
    }
    adj->edge_ids = next;
    adj->cap = need;
    return true;
}

// Reserve incoming adjacency capacity on `block_id` for `extra` more edges.
static bool normalized_cfg_reserve_incoming(NormalizedCFG *n, size_t block_id,
                                            size_t extra) {
    if (block_id >= n->n_blocks) return false;
    if (!n->graph_arena) return false;
    NormAdjacency *adj = &n->blocks[block_id].incoming;
    size_t need = adj->n + extra;
    if (adj->cap >= need) return true;
    size_t *next = arena_new_array(n->graph_arena, size_t, need);
    if (!next) return false;
    if (adj->n > 0) {
        memcpy(next, adj->edge_ids, adj->n * sizeof(size_t));
    }
    adj->edge_ids = next;
    adj->cap = need;
    return true;
}

// O(1) swap-remove at `pos`; updates moved edge's position field.
static void norm_adj_remove_swap(NormAdjacency *adj, size_t pos,
                                 NormEdge *edges, bool is_outgoing) {
    if (pos >= adj->n) return;
    size_t last = adj->n - 1;
    if (pos != last) {
        size_t moved_eid = adj->edge_ids[last];
        adj->edge_ids[pos] = moved_eid;
        if (is_outgoing) {
            edges[moved_eid].out_position = pos;
        } else {
            edges[moved_eid].in_position = pos;
        }
    }
    adj->n--;
}

// Zero-init NormTerminator; default_edge_id = SIZE_MAX.
static void norm_terminator_init(NormTerminator *t) {
    memset(t, 0, sizeof(*t));
    t->default_edge_id = SIZE_MAX;
}

// Zero-init NormOperand (NORM_OPERAND_INVALID).
static void norm_operand_init(NormOperand *o) {
    memset(o, 0, sizeof(*o));
}

// Construct NORM_OPERAND_MLIR with type and MLIR value.
static NormOperand norm_operand_mlir(MLIR_TypeHandle ty, MLIR_ValueHandle v) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_MLIR;
    o.type = ty;
    o.as.mlir_value = v;
    return o;
}

// Construct NORM_OPERAND_BLOCK_ARG referencing a block formal.
static NormOperand norm_operand_block_arg(MLIR_TypeHandle ty, size_t block_id,
                                          size_t arg_index) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_BLOCK_ARG;
    o.type = ty;
    o.as.block_arg.block_id = block_id;
    o.as.block_arg.arg_index = arg_index;
    return o;
}

// Construct NORM_OPERAND_DISCRIMINATOR mux constant.
static NormOperand norm_operand_discriminator(MLIR_TypeHandle ty, int32_t disc) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_DISCRIMINATOR;
    o.type = ty;
    o.as.discriminator = disc;
    return o;
}

// Construct typed NORM_OPERAND_UNDEF padding value.
static NormOperand norm_operand_undef(MLIR_TypeHandle ty) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_UNDEF;
    o.type = ty;
    return o;
}

// False for zero-initialized / NORM_OPERAND_INVALID operands.
static bool norm_operand_is_valid(const NormOperand *o) {
    return o && o->kind != NORM_OPERAND_INVALID;
}

// Copy MLIR block argument types into NormBlockArg signature.
static void norm_block_init_original_signature(Arena *arena, NormBlock *nb,
                                               MLIR_BlockHandle b) {
    size_t nargs = MLIR_GetBlockNumArgs(b);
    if (nargs == 0) return;
    nb->args = arena_new_array(arena, NormBlockArg, nargs);
    nb->n_args = nargs;
    for (size_t i = 0; i < nargs; ++i) {
        MLIR_ValueHandle arg = MLIR_GetBlockArg(b, i);
        nb->args[i].type = MLIR_GetValueType(arg);
        nb->args[i].role = NORM_BLOCK_ARG_FORWARDED;
    }
}

// Attach NormMuxEntry destination layout table to a block.
static bool norm_block_set_mux_entries(NormBlock *nb, Arena *arena,
                                       const NormMuxEntry *entries, size_t n) {
    nb->mux_entries = NULL;
    nb->n_mux_entries = 0;
    if (n == 0) return true;
    nb->mux_entries = arena_new_array(arena, NormMuxEntry, n);
    if (!nb->mux_entries) return false;
    nb->n_mux_entries = n;
    memcpy(nb->mux_entries, entries, n * sizeof(NormMuxEntry));
    return true;
}

// False after outgoing adjacency/terminator finalization on a synthetic block.
static bool norm_synthetic_block_allows_outgoing_mutation(const NormBlock *b) {
    if (b->kind == NORM_BLOCK_ORIGINAL) return true;
    return !b->outgoing_finalized;
}

// True when an active edge may be removed (source block still allows outgoing edits).
static bool norm_edge_deactivation_allowed(const NormalizedCFG *n, size_t edge_id) {
    if (edge_id >= n->n_edges) return false;
    const NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (e->from < n->n_blocks &&
        !norm_synthetic_block_allows_outgoing_mutation(&n->blocks[e->from])) {
        return false;
    }
    return true;
}

// Arena-copy variable-length NormTerminator arrays.
static bool norm_terminator_deep_copy(Arena *arena, const NormTerminator *src,
                                      NormTerminator *dst) {
    norm_terminator_init(dst);
    if (!src) return true;
    dst->kind = src->kind;
    dst->selector = src->selector;
    dst->should_repeat = src->should_repeat;
    dst->default_edge_id = src->default_edge_id;
    dst->n_cases = src->n_cases;
    if (src->n_cases > 0) {
        if (!src->case_values || !src->case_edge_ids) return false;
        dst->case_values = arena_new_array(arena, int32_t, src->n_cases);
        dst->case_edge_ids = arena_new_array(arena, size_t, src->n_cases);
        if (!dst->case_values || !dst->case_edge_ids) return false;
        memcpy(dst->case_values, src->case_values, src->n_cases * sizeof(int32_t));
        memcpy(dst->case_edge_ids, src->case_edge_ids, src->n_cases * sizeof(size_t));
    }
    dst->n_return_values = src->n_return_values;
    if (src->n_return_values > 0) {
        if (!src->return_values) return false;
        dst->return_values = arena_new_array(arena, NormOperand, src->n_return_values);
        if (!dst->return_values) return false;
        memcpy(dst->return_values, src->return_values,
               src->n_return_values * sizeof(NormOperand));
    }
    return true;
}

// Arena-copy NormBlockArg formal signature onto a block.
static bool norm_block_set_signature(NormBlock *nb, Arena *arena,
                                     const NormBlockArg *args, size_t n_args) {
    nb->args = NULL;
    nb->n_args = 0;
    if (n_args == 0) return true;
    nb->args = arena_new_array(arena, NormBlockArg, n_args);
    if (!nb->args) return false;
    nb->n_args = n_args;
    memcpy(nb->args, args, n_args * sizeof(NormBlockArg));
    return true;
}

// Allocate block signature directly from MLIR type handles (one owned array).
static bool norm_block_set_signature_from_types(NormBlock *nb, Arena *arena,
                                                const MLIR_TypeHandle *types,
                                                size_t n_types) {
    nb->args = NULL;
    nb->n_args = 0;
    if (n_types == 0) return true;
    nb->args = arena_new_array(arena, NormBlockArg, n_types);
    if (!nb->args) return false;
    nb->n_args = n_types;
    for (size_t i = 0; i < n_types; ++i) {
        nb->args[i].type = types[i];
        nb->args[i].role = NORM_BLOCK_ARG_FORWARDED;
    }
    return true;
}

// Zero-init NormBlock including owned-terminator defaults.
static void norm_block_init(NormBlock *b) {
    memset(b, 0, sizeof(*b));
    norm_terminator_init(&b->normalized_term);
    b->term_source = NORM_TERM_SNAPSHOT;
    b->effective_term_kind = CFG_TERM_NONE;
}

// Grow NormalizedCFG blocks array, initializing new NormBlock slots.
static void norm_ensure_block_slots(NormalizedCFG *n, Arena *arena, size_t need) {
    while (n->blocks_cap < need) {
        size_t ocap = n->blocks_cap;
        size_t ncap = ocap ? ocap * 2 : 8;
        NormBlock *nb = arena_new_array(arena, NormBlock, ncap);
        if (ocap > 0) {
            memcpy(nb, n->blocks, ocap * sizeof(NormBlock));
        }
        for (size_t i = ocap; i < ncap; ++i) {
            norm_block_init(&nb[i]);
        }
        n->blocks = nb;
        n->blocks_cap = ncap;
    }
}

// Replace edge payload with owned synthetic NormOperand array.
static bool norm_edge_payload_set_operands(Arena *arena, NormEdgePayload *payload,
                                           const NormOperand *operands,
                                           size_t n_operands) {
    payload->kind = NORM_EDGE_SYNTHETIC;
    payload->snapshot_edge_id = SIZE_MAX;
    payload->operands = NULL;
    payload->n_operands = 0;
    if (n_operands == 0) return true;
    payload->operands = arena_new_array(arena, NormOperand, n_operands);
    if (!payload->operands) return false;
    payload->n_operands = n_operands;
    memcpy(payload->operands, operands, n_operands * sizeof(NormOperand));
    return true;
}

// Adopt an existing operand array (must live in graph_arena); no copy.
static void norm_edge_payload_adopt_operands(NormEdgePayload *payload,
                                           NormOperand *operands,
                                           size_t n_operands) {
    memset(payload, 0, sizeof(*payload));
    payload->kind = NORM_EDGE_SYNTHETIC;
    payload->snapshot_edge_id = SIZE_MAX;
    payload->operands = operands;
    payload->n_operands = n_operands;
}

// Mark payload as lazy snapshot reference (no operand copy).
static void norm_edge_payload_init_snapshot(NormEdgePayload *out,
                                            size_t snapshot_edge_id) {
    memset(out, 0, sizeof(*out));
    out->kind = NORM_EDGE_SNAPSHOT;
    out->snapshot_edge_id = snapshot_edge_id;
}

// Initialize synthetic edge payload from operand array.
static bool norm_edge_payload_init_synthetic_operands(
        Arena *arena, const NormOperand *operands, size_t n_operands,
        NormEdgePayload *out) {
    memset(out, 0, sizeof(*out));
    return norm_edge_payload_set_operands(arena, out, operands, n_operands);
}

// Operand count; lazy-resolve snapshot edges via CFGInfo.
static size_t norm_edge_payload_num_operands(const NormalizedCFG *n,
                                             const NormEdgePayload *payload) {
    if (!payload) return 0;
    if (payload->kind == NORM_EDGE_SNAPSHOT && n->snapshot) {
        return cfg_info_edge_num_operands(n->snapshot, payload->snapshot_edge_id);
    }
    return payload->n_operands;
}

// Resolve one edge operand into caller-owned NormOperand.
static bool norm_edge_payload_operand(const NormalizedCFG *n,
                                      const NormEdgePayload *payload,
                                      size_t op_idx, NormOperand *out) {
    if (!payload || !out) return false;
    if (payload->kind == NORM_EDGE_SNAPSHOT && n->snapshot) {
        size_t sid = payload->snapshot_edge_id;
        if (op_idx >= cfg_info_edge_num_operands(n->snapshot, sid)) return false;
        MLIR_ValueHandle v = cfg_info_edge_operand(n->snapshot, sid, op_idx);
        *out = norm_operand_mlir(MLIR_GetValueType(v), v);
        return true;
    }
    if (op_idx >= payload->n_operands) return false;
    *out = payload->operands[op_idx];
    return true;
}

// Append edge to normalized graph; internal builder primitive.
static size_t normalized_cfg_add_edge(NormalizedCFG *n, Arena *arena,
                                      size_t from, size_t to,
                                      size_t succ_slot, size_t snapshot_edge_id) {
    if (n->n_edges >= n->edges_cap) {
        size_t ncap = n->edges_cap ? n->edges_cap * 2 : 16;
        NormEdge *ne = arena_new_array(arena, NormEdge, ncap);
        if (n->n_edges > 0) {
            memcpy(ne, n->edges, n->n_edges * sizeof(NormEdge));
        }
        n->edges = ne;
        n->edges_cap = ncap;
    }
    size_t eid = n->n_edges++;
    NormEdge *e = &n->edges[eid];
    memset(e, 0, sizeof(*e));
    e->from = from;
    e->to = to;
    e->successor_slot = succ_slot;
    e->active = true;
    if (snapshot_edge_id != SIZE_MAX && n->snapshot) {
        norm_edge_payload_init_snapshot(&e->payload, snapshot_edge_id);
    } else {
        if (!norm_edge_payload_set_operands(arena, &e->payload, NULL, 0)) {
            n->n_edges--;
            return SIZE_MAX;
        }
    }
    norm_ensure_block_slots(n, arena, from + 1);
    norm_ensure_block_slots(n, arena, to + 1);
    norm_adj_push(arena, &n->blocks[from].outgoing, eid);
    e->out_position = n->blocks[from].outgoing.n - 1;
    norm_adj_push(arena, &n->blocks[to].incoming, eid);
    e->in_position = n->blocks[to].incoming.n - 1;
    normalized_cfg_note_mutation(n);
    return eid;
}

// --- Graph construction and mutation (M5/M7/M8 rewrite primitives) ---

// Copy reachable CFGInfo into NormalizedCFG with exact adjacency.
// Upstream: graph copy before CFGToSCF mux/latch insertion.
static bool normalized_cfg_build_from_snapshot(CFGAnalysisArena *pool,
                                               const CFGInfo *cfg,
                                               NormalizedCFG *out) {
    memset(out, 0, sizeof(*out));
    if (!pool || !pool->graph_arena || !pool->analysis_arena ||
        !pool->phase_arena) {
        return false;
    }
    if (!cfg || cfg->n_blocks == 0) return false;
    out->snapshot = cfg;
    out->graph_arena = pool->graph_arena;
    out->analysis_arena = pool->analysis_arena;
    out->analysis_base_pos = pool->analysis_base_pos;
    out->phase_arena = pool->phase_arena;
    out->phase_base_pos = pool->phase_base_pos;
    arena_reset(out->analysis_arena, out->analysis_base_pos);
    out->entry_block = cfg_info_entry_index(cfg);
    if (out->entry_block == SIZE_MAX) return false;

    Arena *arena = pool->graph_arena;
    out->n_blocks = cfg->n_blocks;
    norm_ensure_block_slots(out, arena, cfg->n_blocks);
    for (size_t i = 0; i < cfg->n_blocks; ++i) {
        const CFGBlock *sb = &cfg->blocks[i];
        NormBlock *nb = &out->blocks[i];
        norm_block_init(nb);
        nb->kind = NORM_BLOCK_ORIGINAL;
        nb->source_term_kind = sb->term_kind;
        nb->source_terminator = sb->terminator;
        nb->term_source = NORM_TERM_SNAPSHOT;
        nb->effective_term_kind = sb->term_kind;
        nb->mlir_block = sb->block;
        nb->active = sb->reachable;
        norm_block_init_original_signature(arena, nb, sb->block);
        if (!sb->reachable) continue;
        norm_adj_prealloc(arena, &nb->outgoing, sb->n_out_edges);
        norm_adj_prealloc(arena, &nb->incoming, sb->n_in_edges);
    }

    normalized_cfg_begin_rewrite(out);
    for (size_t eid = 0; eid < cfg->n_edges; ++eid) {
        const CFGEdge *se = &cfg->edges[eid];
        if (!out->blocks[se->from].active || !out->blocks[se->to].active) {
            continue;
        }
        normalized_cfg_add_edge(out, arena, se->from, se->to, se->successor_slot,
                                eid);
    }
    normalized_cfg_end_rewrite(out);
    if (out->generation == 0) out->generation = 1;
    out->analysis.generation = out->generation;
    return true;
}

// Append synthetic block (mux, latch, shared return, etc.).
static size_t normalized_cfg_add_synthetic_block(NormalizedCFG *n, Arena *arena,
                                                 NormBlockKind kind,
                                                 CFGTermKind term_kind,
                                                 const NormBlockArg *args,
                                                 size_t n_args) {
    size_t id = n->n_blocks++;
    norm_ensure_block_slots(n, arena, n->n_blocks);
    NormBlock *b = &n->blocks[id];
    norm_block_init(b);
    b->kind = kind;
    b->term_source = NORM_TERM_OWNED;
    b->effective_term_kind = term_kind;
    b->normalized_term.kind = term_kind;
    b->mlir_block = MLIR_INVALID_HANDLE;
    b->source_terminator = MLIR_INVALID_HANDLE;
    b->source_term_kind = CFG_TERM_NONE;
    b->active = true;
    if (!norm_block_set_signature(b, arena, args, n_args)) {
        n->n_blocks--;
        return SIZE_MAX;
    }
    normalized_cfg_note_mutation(n);
    return id;
}

// Install an owned terminator on any active block (original or synthetic).
static bool normalized_cfg_set_block_owned_terminator(NormalizedCFG *n,
                                                      Arena *arena,
                                                      size_t block_id,
                                                      const NormTerminator *term) {
    if (!term) return false;
    if (block_id >= n->n_blocks) return false;
    NormBlock *b = &n->blocks[block_id];
    if (!b->active) return false;
    if (b->kind != NORM_BLOCK_ORIGINAL && b->outgoing_finalized) return false;
    if (!norm_terminator_deep_copy(arena, term, &b->normalized_term)) return false;
    b->term_source = NORM_TERM_OWNED;
    b->effective_term_kind = term->kind;
    normalized_cfg_note_mutation(n);
    return true;
}

// Deep-copy owned terminator onto a synthetic block whose outgoing edges are
// not yet finalized.
static bool normalized_cfg_set_synthetic_terminator(NormalizedCFG *n, Arena *arena,
                                                    size_t block_id,
                                                    const NormTerminator *term) {
    if (block_id >= n->n_blocks) return false;
    NormBlock *b = &n->blocks[block_id];
    if (b->kind == NORM_BLOCK_ORIGINAL) return false;
    return normalized_cfg_set_block_owned_terminator(n, arena, block_id, term);
}

// Install mux destination->argument layout on a synthetic block.
static bool normalized_cfg_set_mux_entries(NormalizedCFG *n, Arena *arena,
                                           size_t block_id,
                                           const NormMuxEntry *entries,
                                           size_t n_entries) {
    if (block_id >= n->n_blocks) return false;
    NormBlock *b = &n->blocks[block_id];
    if (b->kind == NORM_BLOCK_ORIGINAL) return false;
    if (!b->active || b->outgoing_finalized) return false;
    if (!norm_block_set_mux_entries(b, arena, entries, n_entries)) return false;
    normalized_cfg_note_mutation(n);
    return true;
}

// True when every forwarded operand on `edge_id` matches the destination signature.
static bool norm_edge_operands_match_destination(const NormalizedCFG *n,
                                                 size_t edge_id) {
    if (!norm_edge_is_active(n, edge_id)) return false;
    const NormEdge *e = &n->edges[edge_id];
    const NormBlock *dest = &n->blocks[e->to];
    size_t n_ops = norm_edge_payload_num_operands(n, &e->payload);
    if (n_ops != dest->n_args) return false;
    if (n_ops > 0 && !dest->args) return false;
    for (size_t i = 0; i < n_ops; ++i) {
        NormOperand op;
        if (!norm_edge_payload_operand(n, &e->payload, i, &op)) return false;
        if (!norm_operand_is_valid(&op)) return false;
        if (op.type != dest->args[i].type) return false;
    }
    return true;
}

// Validate operand counts/types for every active outgoing edge of `block_id`.
static bool norm_validate_block_outgoing_edges(const NormalizedCFG *n,
                                               size_t block_id) {
    NormEdgeIter it = norm_out_edges(n, block_id);
    size_t eid;
    while (norm_edge_iter_next(&it, &eid)) {
        if (!norm_edge_operands_match_destination(n, eid)) return false;
    }
    return true;
}

// SWITCH terminator: unique cases/default and bijection with outgoing edges.
static bool norm_switch_covers_all_outgoing(const NormalizedCFG *n,
                                            size_t block_id,
                                            const NormTerminator *t) {
    if (!norm_edge_is_outgoing_from(n, t->default_edge_id, block_id)) {
        return false;
    }
    if (!t->case_values || !t->case_edge_ids) return false;

    size_t n_out = 0;
    NormEdgeIter it = norm_out_edges(n, block_id);
    size_t eid;
    while (norm_edge_iter_next(&it, &eid)) {
        n_out++;
        if (eid == t->default_edge_id) continue;
        bool found = false;
        for (size_t i = 0; i < t->n_cases; ++i) {
            if (t->case_edge_ids[i] == eid) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    if (n_out != t->n_cases + 1) return false;

    for (size_t i = 0; i < t->n_cases; ++i) {
        if (!norm_edge_is_outgoing_from(n, t->case_edge_ids[i], block_id)) {
            return false;
        }
        if (t->case_edge_ids[i] == t->default_edge_id) return false;
        for (size_t j = 0; j < i; ++j) {
            if (t->case_edge_ids[i] == t->case_edge_ids[j]) return false;
            if (t->case_values[i] == t->case_values[j]) return false;
        }
    }
    return true;
}

// Validate synthetic terminator metadata against outgoing-edge topology.
static bool norm_validate_synthetic_terminator(const NormalizedCFG *n,
                                               size_t block_id,
                                               const NormBlock *b) {
    const NormTerminator *t = &b->normalized_term;
    if (b->effective_term_kind != t->kind) return false;

    size_t n_out = 0;
    NormEdgeIter it = norm_out_edges(n, block_id);
    size_t eid;
    while (norm_edge_iter_next(&it, &eid)) n_out++;

    switch (t->kind) {
    case CFG_TERM_BR:
        return n_out == 1;
    case CFG_TERM_COND_BR:
        if (!norm_operand_is_valid(&t->selector)) return false;
        return n_out == 2;
    case CFG_TERM_SWITCH:
        if (!norm_operand_is_valid(&t->selector)) return false;
        if (t->default_edge_id == SIZE_MAX) return false;
        return norm_switch_covers_all_outgoing(n, block_id, t);
    case CFG_TERM_RETURN:
        if (n_out != 0) return false;
        if (t->n_return_values > 0 && !t->return_values) return false;
        for (size_t i = 0; i < t->n_return_values; ++i) {
            if (!norm_operand_is_valid(&t->return_values[i])) return false;
        }
        return true;
    case CFG_TERM_UNREACHABLE:
        return n_out == 0;
    default:
        return false;
    }
}

// Validate mux destination layout: bounds, active targets, unique discriminators.
static bool norm_validate_mux_entries(const NormalizedCFG *n,
                                      const NormBlock *mux_block) {
    if (mux_block->n_mux_entries == 0) return true;
    if (!mux_block->mux_entries) return false;

    for (size_t i = 0; i < mux_block->n_mux_entries; ++i) {
        const NormMuxEntry *me = &mux_block->mux_entries[i];
        if (me->arg_offset + me->n_args > mux_block->n_args) return false;
        if (me->n_args > 0 && !mux_block->args) return false;
        if (me->destination_block >= n->n_blocks) return false;
        const NormBlock *dest = &n->blocks[me->destination_block];
        if (!dest->active) return false;
        if (me->n_args > dest->n_args) return false;
        if (me->n_args > 0 && !dest->args) return false;
        for (size_t j = 0; j < me->n_args; ++j) {
            if (mux_block->args[me->arg_offset + j].type != dest->args[j].type) {
                return false;
            }
        }
        for (size_t k = 0; k < i; ++k) {
            if (mux_block->mux_entries[k].discriminator == me->discriminator) {
                return false;
            }
        }
    }
    return true;
}

// Validate outgoing topology/terminator/mux, then lock outgoing mutation.
// Incoming edges may still be added or redirected (M5 sites, M7/M8 exits).
static bool normalized_cfg_finalize_outgoing(NormalizedCFG *n, size_t block_id) {
    if (block_id >= n->n_blocks) return false;
    NormBlock *b = &n->blocks[block_id];
    if (b->kind == NORM_BLOCK_ORIGINAL) return false;
    if (!b->active || b->outgoing_finalized) return false;
    if (!norm_validate_block_outgoing_edges(n, block_id)) return false;
    if (!norm_validate_synthetic_terminator(n, block_id, b)) return false;
    if (!norm_validate_mux_entries(n, b)) return false;
    b->outgoing_finalized = true;
    normalized_cfg_note_mutation(n);
    return true;
}

// Add edge with owned synthetic operand payload.
static bool normalized_cfg_add_synthetic_edge(NormalizedCFG *n, Arena *arena,
                                              size_t from, size_t to,
                                              size_t succ_slot,
                                              const NormOperand *operands,
                                              size_t n_operands,
                                              size_t *out_edge_id) {
    if (out_edge_id) *out_edge_id = SIZE_MAX;
    if (from >= n->n_blocks || to >= n->n_blocks) return false;
    if (!n->blocks[from].active || !n->blocks[to].active) return false;
    if (!norm_synthetic_block_allows_outgoing_mutation(&n->blocks[from])) {
        return false;
    }
    if (n->n_edges >= n->edges_cap) {
        size_t ncap = n->edges_cap ? n->edges_cap * 2 : 16;
        NormEdge *ne = arena_new_array(arena, NormEdge, ncap);
        if (n->n_edges > 0) {
            memcpy(ne, n->edges, n->n_edges * sizeof(NormEdge));
        }
        n->edges = ne;
        n->edges_cap = ncap;
    }
    size_t eid = n->n_edges++;
    NormEdge *e = &n->edges[eid];
    memset(e, 0, sizeof(*e));
    e->from = from;
    e->to = to;
    e->successor_slot = succ_slot;
    e->active = true;
    if (!norm_edge_payload_init_synthetic_operands(arena, operands, n_operands,
                                                   &e->payload)) {
        n->n_edges--;
        return false;
    }
    norm_ensure_block_slots(n, arena, from + 1);
    norm_ensure_block_slots(n, arena, to + 1);
    norm_adj_push(arena, &n->blocks[from].outgoing, eid);
    e->out_position = n->blocks[from].outgoing.n - 1;
    norm_adj_push(arena, &n->blocks[to].incoming, eid);
    e->in_position = n->blocks[to].incoming.n - 1;
    normalized_cfg_note_mutation(n);
    if (out_edge_id) *out_edge_id = eid;
    return true;
}

// Add edge adopting pre-owned operands in graph_arena (no operand copy).
static bool normalized_cfg_add_synthetic_edge_adopt(NormalizedCFG *n,
                                                    size_t from, size_t to,
                                                    size_t succ_slot,
                                                    NormOperand *operands,
                                                    size_t n_operands,
                                                    size_t *out_edge_id) {
    Arena *arena = n->graph_arena;
    if (!arena) return false;
    if (out_edge_id) *out_edge_id = SIZE_MAX;
    if (from >= n->n_blocks || to >= n->n_blocks) return false;
    if (!n->blocks[from].active || !n->blocks[to].active) return false;
    if (!norm_synthetic_block_allows_outgoing_mutation(&n->blocks[from])) {
        return false;
    }
    if (n->n_edges >= n->edges_cap) return false;
    size_t eid = n->n_edges++;
    NormEdge *e = &n->edges[eid];
    memset(e, 0, sizeof(*e));
    e->from = from;
    e->to = to;
    e->successor_slot = succ_slot;
    e->active = true;
    norm_edge_payload_adopt_operands(&e->payload, operands, n_operands);
    norm_ensure_block_slots(n, arena, from + 1);
    norm_ensure_block_slots(n, arena, to + 1);
    norm_adj_append(&n->blocks[from].outgoing, eid);
    e->out_position = n->blocks[from].outgoing.n - 1;
    norm_adj_append(&n->blocks[to].incoming, eid);
    e->in_position = n->blocks[to].incoming.n - 1;
    normalized_cfg_note_mutation(n);
    if (out_edge_id) *out_edge_id = eid;
    return true;
}

// Replace only the operand payload of an active edge.
static bool normalized_cfg_replace_edge_payload(NormalizedCFG *n, Arena *arena,
                                              size_t edge_id,
                                              const NormOperand *operands,
                                              size_t n_operands) {
    if (edge_id >= n->n_edges) return false;
    NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (!norm_edge_payload_set_operands(arena, &e->payload, operands, n_operands)) {
        return false;
    }
    normalized_cfg_note_mutation(n);
    return true;
}

// Soft-delete edge and remove from both adjacency lists.
static bool normalized_cfg_deactivate_edge(NormalizedCFG *n, size_t edge_id) {
    if (!norm_edge_deactivation_allowed(n, edge_id)) return false;
    NormEdge *e = &n->edges[edge_id];
    if (e->from < n->n_blocks) {
        norm_adj_remove_swap(&n->blocks[e->from].outgoing, e->out_position,
                             n->edges, true);
    }
    if (e->to < n->n_blocks) {
        norm_adj_remove_swap(&n->blocks[e->to].incoming, e->in_position,
                             n->edges, false);
    }
    e->active = false;
    normalized_cfg_note_mutation(n);
    return true;
}

// Change edge destination only; keep snapshot/synthetic operands.
static bool normalized_cfg_retarget_edge_preserving_payload(NormalizedCFG *n,
                                                            Arena *arena,
                                                            size_t edge_id,
                                                            size_t new_target) {
    if (edge_id >= n->n_edges) return false;
    NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (new_target >= n->n_blocks || !n->blocks[new_target].active) return false;
    size_t old_to = e->to;
    if (old_to == new_target) return true;
    norm_ensure_block_slots(n, arena, new_target + 1);
    if (old_to < n->n_blocks) {
        norm_adj_remove_swap(&n->blocks[old_to].incoming, e->in_position,
                             n->edges, false);
    }
    e->to = new_target;
    norm_adj_push(arena, &n->blocks[new_target].incoming, edge_id);
    e->in_position = n->blocks[new_target].incoming.n - 1;
    normalized_cfg_note_mutation(n);
    return true;
}

// Atomically retarget edge and replace payload (NULL,0 = empty). Upstream: CFGToSCF mux edge rewrite.
static bool normalized_cfg_redirect_edge_with_payload(NormalizedCFG *n, Arena *arena,
                                                    size_t edge_id,
                                                    size_t new_target,
                                                    const NormOperand *operands,
                                                    size_t n_operands) {
    if (edge_id >= n->n_edges) return false;
    NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (new_target >= n->n_blocks || !n->blocks[new_target].active) return false;
    if (!norm_edge_payload_set_operands(arena, &e->payload, operands, n_operands)) {
        return false;
    }
    size_t old_to = e->to;
    bool changed = true;
    if (old_to != new_target) {
        norm_ensure_block_slots(n, arena, new_target + 1);
        if (old_to < n->n_blocks) {
            norm_adj_remove_swap(&n->blocks[old_to].incoming, e->in_position,
                                 n->edges, false);
        }
        e->to = new_target;
        norm_adj_push(arena, &n->blocks[new_target].incoming, edge_id);
        e->in_position = n->blocks[new_target].incoming.n - 1;
    }
    normalized_cfg_note_mutation(n);
    return changed;
}

// Deactivate all incident edges then mark block inactive.
// Returns false without mutating when any incident edge originates from a
// synthetic block whose outgoing adjacency is finalized.
static bool normalized_cfg_deactivate_block(NormalizedCFG *n, size_t block_id) {
    if (block_id >= n->n_blocks) return false;
    if (!n->blocks[block_id].active) return true;

    NormBlock *block = &n->blocks[block_id];
    for (size_t i = 0; i < block->outgoing.n; ++i) {
        if (!norm_edge_deactivation_allowed(n, block->outgoing.edge_ids[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < block->incoming.n; ++i) {
        if (!norm_edge_deactivation_allowed(n, block->incoming.edge_ids[i])) {
            return false;
        }
    }

    normalized_cfg_begin_rewrite(n);
    while (block->outgoing.n > 0) {
        size_t eid = block->outgoing.edge_ids[block->outgoing.n - 1];
        if (!normalized_cfg_deactivate_edge(n, eid)) {
            normalized_cfg_end_rewrite(n);
            return false;
        }
    }
    while (block->incoming.n > 0) {
        size_t eid = block->incoming.edge_ids[block->incoming.n - 1];
        if (!normalized_cfg_deactivate_edge(n, eid)) {
            normalized_cfg_end_rewrite(n);
            return false;
        }
    }
    block->active = false;
    normalized_cfg_note_mutation(n);
    normalized_cfg_end_rewrite(n);
    return true;
}

// True when edge and both endpoint blocks are active.
static bool norm_edge_is_active(const NormalizedCFG *n, size_t edge_id) {
    if (edge_id >= n->n_edges) return false;
    const NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (e->from >= n->n_blocks || e->to >= n->n_blocks) return false;
    if (!n->blocks[e->from].active || !n->blocks[e->to].active) return false;
    return true;
}

// True when `edge_id` is an active outgoing edge of `block_id` (O(1)).
static bool norm_edge_is_outgoing_from(const NormalizedCFG *n, size_t edge_id,
                                       size_t block_id) {
    return norm_edge_is_active(n, edge_id) && n->edges[edge_id].from == block_id;
}

// Count active outgoing edges (O(degree); prefer norm_out_edges loop).
static size_t normalized_cfg_num_active_out_edges(const NormalizedCFG *n,
                                                  size_t block_id) {
    NormEdgeIter it = norm_out_edges(n, block_id);
    size_t count = 0, eid;
    while (norm_edge_iter_next(&it, &eid)) count++;
    return count;
}

// Nth active outgoing edge id (O(degree); prefer iterator).
static size_t normalized_cfg_active_out_edge_at(const NormalizedCFG *n,
                                                size_t block_id, size_t out_i) {
    NormEdgeIter it = norm_out_edges(n, block_id);
    size_t eid, seen = 0;
    while (norm_edge_iter_next(&it, &eid)) {
        if (seen++ == out_i) return eid;
    }
    return SIZE_MAX;
}

// Count active incoming edges.
static size_t normalized_cfg_num_active_in_edges(const NormalizedCFG *n,
                                                 size_t block_id) {
    NormEdgeIter it = norm_in_edges(n, block_id);
    size_t count = 0, eid;
    while (norm_edge_iter_next(&it, &eid)) count++;
    return count;
}

// Nth active incoming edge id.
static size_t normalized_cfg_active_in_edge_at(const NormalizedCFG *n,
                                             size_t block_id, size_t in_i) {
    NormEdgeIter it = norm_in_edges(n, block_id);
    size_t eid, seen = 0;
    while (norm_edge_iter_next(&it, &eid)) {
        if (seen++ == in_i) return eid;
    }
    return SIZE_MAX;
}

// --- NormalizedCFG read-only queries ---

// Const pointer to edge pool entry `edge_id`.
static const NormEdge *normalized_cfg_edge_at(const NormalizedCFG *n,
                                              size_t edge_id) {
    if (edge_id >= n->n_edges) return NULL;
    return &n->edges[edge_id];
}

// Operand count on normalized edge (lazy for snapshot).
static size_t normalized_cfg_edge_num_operands(const NormalizedCFG *n,
                                               size_t edge_id) {
    const NormEdge *e = normalized_cfg_edge_at(n, edge_id);
    return norm_edge_payload_num_operands(n, e ? &e->payload : NULL);
}

// Resolve edge operand into caller-owned NormOperand (reentrant).
static bool normalized_cfg_edge_operand_resolve(const NormalizedCFG *n,
                                                size_t edge_id, size_t op_idx,
                                                NormOperand *out) {
    const NormEdge *e = normalized_cfg_edge_at(n, edge_id);
    if (!e) return false;
    return norm_edge_payload_operand(n, &e->payload, op_idx, out);
}

// Formal block argument metadata for synthetic or original blocks.
static const NormBlockArg *normalized_cfg_block_arg(const NormalizedCFG *n,
                                                    size_t block_id,
                                                    size_t arg_idx) {
    if (block_id >= n->n_blocks) return NULL;
    const NormBlock *b = &n->blocks[block_id];
    if (arg_idx >= b->n_args) return NULL;
    return &b->args[arg_idx];
}

// Effective terminator kind for any block (snapshot or owned override).
static CFGTermKind normalized_cfg_block_term_kind(const NormalizedCFG *n,
                                                  size_t block_id) {
    if (block_id >= n->n_blocks) return CFG_TERM_NONE;
    const NormBlock *b = &n->blocks[block_id];
    if (!b->active) return CFG_TERM_NONE;
    return b->effective_term_kind;
}

// Owned terminator metadata when term_source is NORM_TERM_OWNED; else NULL.
static const NormTerminator *normalized_cfg_block_owned_terminator(
        const NormalizedCFG *n, size_t block_id) {
    if (block_id >= n->n_blocks) return NULL;
    const NormBlock *b = &n->blocks[block_id];
    if (!b->active || b->term_source != NORM_TERM_OWNED) return NULL;
    return &b->normalized_term;
}

// Owned terminator on non-ORIGINAL blocks (legacy alias).
static const NormTerminator *normalized_cfg_synthetic_terminator(
        const NormalizedCFG *n, size_t block_id) {
    if (block_id >= n->n_blocks) return NULL;
    const NormBlock *b = &n->blocks[block_id];
    if (b->kind == NORM_BLOCK_ORIGINAL) return NULL;
    return normalized_cfg_block_owned_terminator(n, block_id);
}

// --- Analysis cache (lazy, generation-tracked) ---

// Compute reverse postorder + position inverse map.
// Upstream: DFS postorder input for dominance and SCC (CFGToSCF DomInfo).
static bool norm_compute_rpo(const NormalizedCFG *n, RPOInfo *out) {
    memset(out, 0, sizeof(*out));
    if (!n->analysis_arena || n->n_blocks == 0) return true;
    if (n->entry_block >= n->n_blocks || !n->blocks[n->entry_block].active) {
        return true;
    }

    Arena *arena = n->analysis_arena;
    bool *visited = arena_new_array(arena, bool, n->n_blocks);
    memset(visited, 0, n->n_blocks * sizeof(bool));
    size_t *order = arena_new_array(arena, size_t, n->n_blocks);
    size_t post_n = 0;

    NormRpoFrame *stack = arena_new_array(arena, NormRpoFrame, n->n_blocks);
    size_t sp = 0;

    visited[n->entry_block] = true;
    stack[sp].bid = n->entry_block;
    stack[sp].edge_i = 0;
    sp++;

    while (sp > 0) {
        NormRpoFrame *f = &stack[sp - 1];
        const NormAdjacency *out_adj = &n->blocks[f->bid].outgoing;
        bool pushed = false;
        while (f->edge_i < out_adj->n) {
            size_t eid = out_adj->edge_ids[f->edge_i++];
            if (!norm_edge_is_active(n, eid)) continue;
            size_t succ = n->edges[eid].to;
            if (succ >= n->n_blocks || visited[succ]) continue;
            visited[succ] = true;
            stack[sp].bid = succ;
            stack[sp].edge_i = 0;
            sp++;
            pushed = true;
            break;
        }
        if (!pushed) {
            order[post_n++] = stack[--sp].bid;
        }
    }

    out->n = post_n;
    if (post_n == 0) return true;

    for (size_t i = 0, j = post_n - 1; i < j; ++i, --j) {
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    out->order = order;
    out->position = arena_new_array(arena, size_t, n->n_blocks);
    for (size_t i = 0; i < n->n_blocks; ++i) out->position[i] = SIZE_MAX;
    for (size_t i = 0; i < post_n; ++i) {
        out->position[out->order[i]] = i;
    }
    return true;
}

// Lazy RPO cache; computes on first use after generation sync.
static const RPOInfo *normalized_cfg_try_get_cached_rpo(NormalizedCFG *n) {
    if (!normalized_cfg_analysis_available(n)) return NULL;
    cfg_analysis_cache_sync(n);
    if (!n->analysis.has_rpo) {
        if (!norm_compute_rpo(n, &n->analysis.rpo)) return NULL;
        n->analysis.has_rpo = true;
    }
    return &n->analysis.rpo;
}

// Lazy dominance cache (NULL until Cooper-Harvey-Kennedy is ported).
static const DominanceInfo *normalized_cfg_try_get_cached_dominance(
        NormalizedCFG *n) {
    if (!normalized_cfg_analysis_available(n)) return NULL;
    cfg_analysis_cache_sync(n);
    if (!n->analysis.has_dominance) return NULL;
    return &n->analysis.dominance;
}

// Lazy post-dominance cache (NULL until compute is ported).
static const DominanceInfo *normalized_cfg_try_get_cached_post_dominance(
        NormalizedCFG *n) {
    if (!normalized_cfg_analysis_available(n)) return NULL;
    cfg_analysis_cache_sync(n);
    if (!n->analysis.has_post_dominance) return NULL;
    return &n->analysis.post_dominance;
}

// Lazy Tarjan SCC cache (NULL until M7 cycle detection is ported).
static const SCCInfo *normalized_cfg_try_get_cached_scc(NormalizedCFG *n) {
    if (!normalized_cfg_analysis_available(n)) return NULL;
    cfg_analysis_cache_sync(n);
    if (!n->analysis.has_scc) return NULL;
    return &n->analysis.scc;
}

// Explicitly drop all cached analyses at a phase boundary.
static void normalized_cfg_refresh_analysis(NormalizedCFG *n) {
    normalized_cfg_invalidate_analysis(n);
    n->analysis.generation = n->generation;
}

// ---------------------------------------------------------------------------
// M5 — return normalization (CFGToSCF.cpp ReturnLikeExitCombiner port).
// Replace llvm.return terminators with virtual branches to shared exit blocks.
// Operates only on NormalizedCFG; LLVM MLIR is never mutated.
// ---------------------------------------------------------------------------

static void m5_exit_combiner_init(M5ExitCombiner *c) {
    memset(c, 0, sizeof(*c));
}

static bool m5_return_signature_matches_class(const M5ExitClass *cls,
                                              M5ExitFlavor flavor,
                                              MLIR_OpHandle ret) {
    if (cls->flavor != flavor) return false;
    size_t n = MLIR_GetOpNumOperands(ret);
    if (cls->n_operand_types != n) return false;
    for (size_t i = 0; i < n; ++i) {
        MLIR_ValueHandle v = MLIR_GetOpOperand(ret, i);
        if (cls->operand_types[i] != MLIR_GetValueType(v)) return false;
    }
    return true;
}

static bool m5_plan_grow_classes(M5Plan *plan, Arena *phase) {
    size_t new_cap = plan->classes_cap ? plan->classes_cap * 2 : 4;
    M5ExitClass *next = arena_new_array(phase, M5ExitClass, new_cap);
    if (!next) return false;
    if (plan->n_classes > 0) {
        memcpy(next, plan->classes, plan->n_classes * sizeof(M5ExitClass));
    }
    plan->classes = next;
    plan->classes_cap = new_cap;
    return true;
}

static size_t m5_plan_find_class_from_return(const M5Plan *plan,
                                             M5ExitFlavor flavor,
                                             MLIR_OpHandle ret) {
    for (size_t i = 0; i < plan->n_classes; ++i) {
        if (m5_return_signature_matches_class(&plan->classes[i], flavor, ret)) {
            return i;
        }
    }
    return SIZE_MAX;
}

static bool m5_plan_add_class_from_return(M5Plan *plan, Arena *phase, Arena *graph,
                                          M5ExitFlavor flavor, MLIR_OpHandle ret,
                                          size_t *out_class_id) {
    if (out_class_id) *out_class_id = SIZE_MAX;
    if (plan->n_classes >= plan->classes_cap && !m5_plan_grow_classes(plan, phase)) {
        return false;
    }

    size_t n_types = MLIR_GetOpNumOperands(ret);
    M5ExitClass candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.flavor = flavor;
    candidate.n_operand_types = n_types;
    candidate.n_sites = 0;
    candidate.shared_exit_block = SIZE_MAX;
    if (n_types > 0) {
        candidate.operand_types = arena_new_array(graph, MLIR_TypeHandle, n_types);
        if (!candidate.operand_types) return false;
        for (size_t i = 0; i < n_types; ++i) {
            candidate.operand_types[i] =
                MLIR_GetValueType(MLIR_GetOpOperand(ret, i));
        }
    }

    size_t class_id = plan->n_classes;
    plan->classes[class_id] = candidate;
    plan->n_classes++;
    if (out_class_id) *out_class_id = class_id;
    return true;
}

static bool m5_plan_grow_sites(M5Plan *plan, Arena *phase) {
    size_t new_cap = plan->sites_cap ? plan->sites_cap * 2 : 8;
    M5ExitSite *next = arena_new_array(phase, M5ExitSite, new_cap);
    if (!next) return false;
    if (plan->n_sites > 0) {
        memcpy(next, plan->sites, plan->n_sites * sizeof(M5ExitSite));
    }
    plan->sites = next;
    plan->sites_cap = new_cap;
    return true;
}

static bool m5_plan_add_site_from_return(M5Plan *plan, Arena *phase, Arena *graph,
                                       size_t source_block, size_t exit_class,
                                       MLIR_OpHandle ret) {
    if (plan->n_sites >= plan->sites_cap && !m5_plan_grow_sites(plan, phase)) {
        return false;
    }

    size_t n_values = MLIR_GetOpNumOperands(ret);
    M5ExitSite *site = &plan->sites[plan->n_sites++];
    site->source_block = source_block;
    site->exit_class = exit_class;
    site->n_return_values = n_values;
    site->return_values = NULL;
    if (n_values > 0) {
        site->return_values = arena_new_array(graph, NormOperand, n_values);
        if (!site->return_values) {
            plan->n_sites--;
            return false;
        }
        for (size_t i = 0; i < n_values; ++i) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(ret, i);
            site->return_values[i] = norm_operand_mlir(MLIR_GetValueType(v), v);
        }
    }
    return true;
}

// Discover llvm.return sites; no NormalizedCFG mutation.
static bool m5_prepare(NormalizedCFG *cfg, M5Plan *plan) {
    memset(plan, 0, sizeof(*plan));

    Arena *phase = cfg->phase_arena;
    Arena *graph = cfg->graph_arena;
    if (!phase || !graph) return false;

    size_t original_n_blocks = cfg->n_blocks;

    for (size_t bid = 0; bid < original_n_blocks; ++bid) {
        const NormBlock *block = &cfg->blocks[bid];

        if (!block->active ||
            block->kind != NORM_BLOCK_ORIGINAL ||
            block->term_source != NORM_TERM_SNAPSHOT ||
            block->effective_term_kind != CFG_TERM_RETURN) {
            continue;
        }

        if (block->outgoing.n != 0) return false;

        MLIR_OpHandle ret = block->source_terminator;
        if (ret == MLIR_INVALID_HANDLE) return false;

        size_t class_id = m5_plan_find_class_from_return(
            plan, M5_EXIT_LLVM_RETURN, ret);

        if (class_id == SIZE_MAX) {
            if (!m5_plan_add_class_from_return(plan, phase, graph,
                                               M5_EXIT_LLVM_RETURN, ret,
                                               &class_id)) {
                return false;
            }
        }

        if (!m5_plan_add_site_from_return(plan, phase, graph, bid, class_id,
                                          ret)) {
            return false;
        }

        plan->classes[class_id].n_sites++;
    }

    return true;
}

static bool m5_reserve_commit_capacity(NormalizedCFG *cfg, const M5Plan *plan) {
    if (!normalized_cfg_reserve_blocks(cfg, plan->n_classes)) return false;
    if (!normalized_cfg_reserve_edges(cfg, plan->n_sites)) return false;

    for (size_t i = 0; i < plan->n_sites; ++i) {
        size_t source = plan->sites[i].source_block;
        if (!normalized_cfg_reserve_outgoing(cfg, source, 1)) return false;
    }

    return true;
}

// Create NORM_BLOCK_SHARED_RETURN with a virtual return on block arguments.
static bool m5_create_shared_exit_block(NormalizedCFG *cfg, M5ExitFlavor flavor,
                                        const MLIR_TypeHandle *types,
                                        size_t n_types, size_t n_incoming_sites,
                                        size_t *out_block_id) {
    if (out_block_id) *out_block_id = SIZE_MAX;
    Arena *arena = cfg->graph_arena;
    if (!arena) return false;

    size_t shared_id = normalized_cfg_add_synthetic_block(
        cfg, arena, NORM_BLOCK_SHARED_RETURN, CFG_TERM_RETURN, NULL, 0);
    if (shared_id == SIZE_MAX) return false;

    NormBlock *nb = &cfg->blocks[shared_id];
    if (!norm_block_set_signature_from_types(nb, arena, types, n_types)) {
        return false;
    }

    norm_adj_prealloc(arena, &nb->incoming, n_incoming_sites);

    norm_terminator_init(&nb->normalized_term);
    nb->normalized_term.kind = CFG_TERM_RETURN;
    nb->normalized_term.n_return_values = n_types;
    if (n_types > 0) {
        nb->normalized_term.return_values =
            arena_new_array(arena, NormOperand, n_types);
        if (!nb->normalized_term.return_values) return false;
        for (size_t i = 0; i < n_types; ++i) {
            nb->normalized_term.return_values[i] =
                norm_operand_block_arg(types[i], shared_id, i);
        }
    }
    nb->term_source = NORM_TERM_OWNED;
    nb->effective_term_kind = CFG_TERM_RETURN;

    if (!normalized_cfg_finalize_outgoing(cfg, shared_id)) {
        return false;
    }
    (void)flavor;
    if (out_block_id) *out_block_id = shared_id;
    return true;
}

static bool m5_commit_classes(NormalizedCFG *cfg, M5Plan *plan) {
    for (size_t i = 0; i < plan->n_classes; ++i) {
        M5ExitClass *cls = &plan->classes[i];
        if (!m5_create_shared_exit_block(
                cfg, cls->flavor, cls->operand_types, cls->n_operand_types,
                cls->n_sites, &cls->shared_exit_block)) {
            return false;
        }
    }
    return true;
}

static bool m5_rewrite_return_site(NormalizedCFG *cfg, const M5ExitSite *site,
                                   size_t shared_exit_id) {
    Arena *arena = cfg->graph_arena;
    if (!arena) return false;

    size_t edge_id;
    if (!normalized_cfg_add_synthetic_edge_adopt(
            cfg, site->source_block, shared_exit_id, 0,
            site->return_values, site->n_return_values, &edge_id)) {
        return false;
    }

    NormTerminator branch;
    norm_terminator_init(&branch);
    branch.kind = CFG_TERM_BR;

    if (!normalized_cfg_set_block_owned_terminator(
            cfg, arena, site->source_block, &branch)) {
        return false;
    }
    return true;
}

static bool m5_combiner_adopt_classes(M5ExitCombiner *combiner, Arena *graph,
                                    const M5Plan *plan) {
    if (!combiner || !graph) return false;
    if (plan->n_classes == 0) return true;

    M5ExitClass *owned = arena_new_array(graph, M5ExitClass, plan->n_classes);
    if (!owned) return false;
    memcpy(owned, plan->classes, plan->n_classes * sizeof(M5ExitClass));
    combiner->classes = owned;
    combiner->n_classes = plan->n_classes;
    combiner->classes_cap = plan->n_classes;
    return true;
}

static void normalized_cfg_reset_phase_storage(NormalizedCFG *cfg) {
    if (cfg->phase_arena) {
        arena_reset(cfg->phase_arena, cfg->phase_base_pos);
    }
}

static bool m5_normalize_returns(NormalizedCFG *cfg, M5ExitCombiner *out_combiner) {
    M5Plan plan;

    m5_exit_combiner_init(out_combiner);
    out_combiner->cfg = cfg;

    if (!m5_prepare(cfg, &plan)) return false;

    if (plan.n_sites == 0) return true;

    if (!m5_reserve_commit_capacity(cfg, &plan)) return false;

    normalized_cfg_begin_rewrite(cfg);

    bool ok = m5_commit_classes(cfg, &plan);

    for (size_t i = 0; ok && i < plan.n_sites; ++i) {
        M5ExitSite *site = &plan.sites[i];
        M5ExitClass *cls = &plan.classes[site->exit_class];
        ok = m5_rewrite_return_site(cfg, site, cls->shared_exit_block);
    }

    normalized_cfg_end_rewrite(cfg);

    if (!ok) return false;

    if (!m5_combiner_adopt_classes(out_combiner, cfg->graph_arena, &plan)) {
        return false;
    }

    normalized_cfg_reset_phase_storage(cfg);
    return true;
}

// --- Arena pool and per-target scratch ---

// Create separate graph, analysis, and phase bump allocators.
static bool cfg_analysis_arena_init(CFGAnalysisArena *pool) {
    memset(pool, 0, sizeof(*pool));
    pool->graph_arena = arena_create(64 * 1024);
    pool->analysis_arena = arena_create(16 * 1024);
    pool->phase_arena = arena_create(16 * 1024);
    if (!pool->graph_arena || !pool->analysis_arena || !pool->phase_arena) {
        if (pool->graph_arena) arena_destroy(pool->graph_arena);
        if (pool->analysis_arena) arena_destroy(pool->analysis_arena);
        if (pool->phase_arena) arena_destroy(pool->phase_arena);
        memset(pool, 0, sizeof(*pool));
        return false;
    }
    pool->graph_base_pos = arena_get_pos(pool->graph_arena);
    pool->analysis_base_pos = arena_get_pos(pool->analysis_arena);
    pool->phase_base_pos = arena_get_pos(pool->phase_arena);
    return true;
}

// Reset phase arena to scratch base position.
static void cfg_phase_arena_reset(CFGAnalysisArena *pool) {
    if (pool->phase_arena) {
        arena_reset(pool->phase_arena, pool->phase_base_pos);
    }
}

// Reset graph arena to snapshot+norm graph base position.
static void cfg_graph_arena_reset(CFGAnalysisArena *pool) {
    if (pool->graph_arena) arena_reset(pool->graph_arena, pool->graph_base_pos);
}

// Reset graph, analysis, and phase arenas (new target build).
static void cfg_analysis_arena_reset(CFGAnalysisArena *pool) {
    cfg_graph_arena_reset(pool);
    if (pool->analysis_arena) {
        arena_reset(pool->analysis_arena, pool->analysis_base_pos);
    }
    cfg_phase_arena_reset(pool);
}

// Destroy all arenas in a CFGAnalysisArena pool.
static void cfg_analysis_arena_destroy(CFGAnalysisArena *pool) {
    if (pool->graph_arena) arena_destroy(pool->graph_arena);
    if (pool->analysis_arena) arena_destroy(pool->analysis_arena);
    if (pool->phase_arena) arena_destroy(pool->phase_arena);
    memset(pool, 0, sizeof(*pool));
}

// Zero-init CFGInfoScratch view.
static void cfg_info_scratch_init(CFGInfoScratch *scratch) {
    memset(scratch, 0, sizeof(*scratch));
}

// Clear scratch view without freeing pool arenas.
static void cfg_info_scratch_clear(CFGInfoScratch *scratch) {
    memset(&scratch->cfg, 0, sizeof(scratch->cfg));
    memset(&scratch->norm, 0, sizeof(scratch->norm));
    m5_exit_combiner_init(&scratch->m5);
    scratch->pool = NULL;
}

// Build CFGInfo + NormalizedCFG for one region in the shared pool.
static bool cfg_info_scratch_build(CFGInfoScratch *scratch,
                                   CFGAnalysisArena *pool,
                                   MLIR_RegionHandle region) {
    cfg_info_scratch_clear(scratch);
    if (!pool || !pool->graph_arena || !pool->analysis_arena ||
        !pool->phase_arena) {
        return false;
    }
    cfg_analysis_arena_reset(pool);
    scratch->pool = pool;
    if (!cfg_info_build(pool->graph_arena, region, &scratch->cfg)) {
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (cfg_info_entry_index(&scratch->cfg) != 0) {
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (!cfg_info_all_region_blocks_reachable(&scratch->cfg)) {
        fprintf(stderr,
                "wasmssa-cfg: unreachable blocks in llvm.func region\n");
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (!normalized_cfg_build_from_snapshot(pool, &scratch->cfg,
                                            &scratch->norm)) {
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (!m5_normalize_returns(&scratch->norm, &scratch->m5)) {
        fprintf(stderr, "wasmssa-cfg: M5 return normalization failed\n");
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    return true;
}

// --- Lift-target discovery and preflight ---

// Preflight all lift targets: build snapshot+norm per region.
static bool lift_targets_validate_cfg_scratch(LiftRegionTarget *targets,
                                              size_t n) {
    CFGAnalysisArena pool;
    if (!cfg_analysis_arena_init(&pool)) return false;
    for (size_t i = 0; i < n; ++i) {
        CFGInfoScratch scratch;
        cfg_info_scratch_init(&scratch);
        if (!cfg_info_scratch_build(&scratch, &pool, targets[i].region)) {
            cfg_analysis_arena_destroy(&pool);
            return false;
        }
        cfg_analysis_arena_reset(&pool);
        cfg_info_scratch_clear(&scratch);
    }
    cfg_analysis_arena_destroy(&pool);
    return true;
}

// True when op is llvm.func.
static bool op_is_llvm_func(MLIR_OpHandle op) {
    return MLIR_GetOpType(op) == OP_TYPE_LLVM_FUNC;
}

// True when op is builtin.module.
static bool op_is_builtin_module(MLIR_OpHandle op) {
    return MLIR_GetOpType(op) == OP_TYPE_MODULE;
}

// llvm.br / llvm.cond_br / llvm.switch — the LLVM CFG terminators we lift.
static bool op_is_llvm_cfg_branch(MLIR_OpHandle op) {
    string n = MLIR_GetOpName(op);
    return name_eq(n, "llvm.br") || name_eq(n, "llvm.cond_br") ||
           name_eq(n, "llvm.switch");
}

// Append one LiftRegionTarget to the discovery list.
static void lift_target_list_push(LiftRegionTargetList *list, Arena *arena,
                                  MLIR_OpHandle fn_op, MLIR_RegionHandle region,
                                  MLIR_BlockHandle entry_block) {
    if (list->n >= list->cap) {
        size_t new_cap = list->cap ? list->cap * 2 : 8;
        LiftRegionTarget *items =
            arena_new_array(arena, LiftRegionTarget, new_cap);
        if (list->n > 0) {
            memcpy(items, list->items, list->n * sizeof(LiftRegionTarget));
        }
        list->items = items;
        list->cap = new_cap;
    }
    LiftRegionTarget t;
    memset(&t, 0, sizeof t);
    t.fn_op = fn_op;
    t.region = region;
    t.entry_block = entry_block;
    list->items[list->n++] = t;
}

// Single walk: detect nested-region CFG branches and record lift targets.
static void collect_llvm_cfg_lift_regions(LiftRegionTargetList *out,
                                          Arena *arena, MLIR_OpHandle fn_op,
                                          MLIR_RegionHandle region) {
    bool has_branch = false;
    size_t nb = MLIR_GetRegionNumBlocks(region);

    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
        if (term != MLIR_INVALID_HANDLE && op_is_llvm_cfg_branch(term)) {
            has_branch = true;
        }

        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            size_t nr = MLIR_GetOpNumRegions(op);
            for (size_t ri = 0; ri < nr; ++ri) {
                collect_llvm_cfg_lift_regions(out, arena, fn_op,
                                              MLIR_GetOpRegion(op, ri));
            }
        }
    }

    if (!has_branch || nb == 0) return;
    lift_target_list_push(out, arena, fn_op, region,
                          MLIR_GetRegionBlock(region, 0));
}

// True when llvm.func has a non-empty entry block body.
static bool llvm_func_has_defined_body(MLIR_OpHandle fn) {
    if (!op_is_llvm_func(fn)) return false;
    if (MLIR_GetOpNumRegions(fn) == 0) return false;
    return MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;
}

// Recurse into nested modules; collect lift targets from defined llvm.func.
static void collect_llvm_cfg_lift_targets_in_region(LiftRegionTargetList *out,
                                                    Arena *arena,
                                                    MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    collect_llvm_cfg_lift_targets_in_region(
                        out, arena, MLIR_GetOpRegion(op, 0));
                }
                continue;
            }
            if (!llvm_func_has_defined_body(op)) continue;
            collect_llvm_cfg_lift_regions(out, arena, op,
                                          MLIR_GetOpRegion(op, 0));
        }
    }
}

// Entry: discover all lift targets in a module tree.
static size_t collect_llvm_cfg_lift_targets_in_module(
    Arena *arena, MLIR_OpHandle module, LiftRegionTarget **out_items) {
    LiftRegionTargetList list = {0};
    if (module != MLIR_INVALID_HANDLE && MLIR_GetOpNumRegions(module) > 0) {
        collect_llvm_cfg_lift_targets_in_region(
            &list, arena, MLIR_GetOpRegion(module, 0));
    }
    if (out_items) *out_items = list.items;
    return list.n;
}

// =============================================================================
// Module walker.
// =============================================================================
static bool sig_for_func(MLIR_Context *ctx, Arena *arena, MLIR_OpHandle fn,
                         uint8_t **out_p, size_t *out_np,
                         uint8_t **out_r, size_t *out_nr) {
    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    if (ftya == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
    size_t ni = MLIR_GetTypeFunctionNumInputs(fty);
    size_t no = MLIR_GetTypeFunctionNumResults(fty);
    bool is_vararg = MLIR_GetTypeFunctionIsVarArg(fty);
    size_t pn = ni + (is_vararg ? 1 : 0);
    uint8_t *p = (uint8_t *)arena_alloc(arena, pn ? pn : 1);
    for (size_t i = 0; i < ni; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionInput(fty, i));
        if (v == 0) return false;
        p[i] = v;
    }
    if (is_vararg) p[ni] = WT_I32;  // hidden va_list pointer
    uint8_t *r = (uint8_t *)arena_alloc(arena, no ? no : 1);
    for (size_t i = 0; i < no; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionResult(fty, i));
        if (v == 0) return false;
        r[i] = v;
    }
    *out_p = p; *out_np = pn;
    *out_r = r; *out_nr = no;
    return true;
}

// Return alignment requirement (power-of-2 exponent) for a global of
// the given LLVM type. 0 = byte-aligned.
static uint32_t align_pow_for_type(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    unsigned a = type_align_bytes(ctx, ty);
    uint32_t p = 0;
    while ((1u << p) < a) p++;
    return p;
}

// Helper: emit a wasmssa.import_global op into `body` with the given
// fields. `data` may be NULL when `size == 0`.
static void emit_import_global(MLIR_Context *ctx, Arena *arena,
                               MLIR_BlockHandle body,
                               string sym_name, uint32_t size,
                               uint32_t align_pow, bool is_const,
                               const uint8_t *data, const char *relocs_str,
                               size_t relocs_len) {
    MLIR_AttributeHandle attrs[16];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",
                         sym_name.str ? sym_name.str : "",
                         sym_name.str ? sym_name.size : 0);
    attrs[na++] = attr_i32(ctx, "size", size);
    attrs[na++] = attr_i32(ctx, "align_pow", align_pow);
    attrs[na++] = attr_b(ctx, "is_const", is_const);
    if (size) {
        attrs[na++] = attr_s(ctx, "init_data", (const char *)data, size);
    } else {
        attrs[na++] = attr_s(ctx, "init_data", "", 0);
    }
    if (relocs_str) {
        attrs[na++] = attr_s(ctx, "relocs", relocs_str, relocs_len);
    }
    MLIR_OpHandle op = make_op(ctx, OP_TYPE_WASMSSA_IMPORT_GLOBAL,
                               attrs, na, NULL, 0, NULL, 0, 0, NULL);
    MLIR_AppendBlockOp(ctx, body, op);
    (void)arena;
}

// Per-slot scratch used by the insertvalue-chain folder in lower_global.
// Declared at file scope so tinyc (which lacks function-local struct
// declarations) accepts it.
typedef struct LowerGlobalCEntry {
    MLIR_ValueHandle v;
    uint64_t bits;
} LowerGlobalCEntry;

// Lower one llvm.mlir.global op directly into a wasmssa.import_global op
// appended to `body`. Returns false on failure (no op is appended).
static bool lower_global(MLIR_Context *ctx, Arena *arena,
                         MLIR_BlockHandle body, MLIR_OpHandle op) {
    MLIR_AttributeHandle sa = find_attr(op, "sym_name");
    if (sa == MLIR_INVALID_HANDLE) return false;
    string sym = MLIR_GetAttributeString(sa);

    uint32_t size = 0;
    uint32_t align_pow = 0;
    MLIR_AttributeHandle ga = find_attr(op, "global_type");
    if (ga != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle gty = MLIR_GetAttributeTypeValue(ga);
        size = type_size_bytes(ctx, gty);
        align_pow = align_pow_for_type(ctx, gty);
    }

    bool is_const = (find_attr(op, "constant") != MLIR_INVALID_HANDLE);

    uint8_t *data = NULL;

    MLIR_AttributeHandle va = find_attr(op, "value");
    if (va != MLIR_INVALID_HANDLE) {
        MLIR_AttrKind ak = MLIR_GetAttributeKind(va);
        if (ak == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(va);
            size = (uint32_t)s.size;
            emit_import_global(ctx, arena, body, sym, size, align_pow,
                               is_const, (const uint8_t *)s.str, NULL, 0);
            return true;
        }
        if (ak == MLIR_ATTR_KIND_INTEGER || ak == MLIR_ATTR_KIND_FLOAT) {
            if (size == 0) return false;
            data = (uint8_t *)arena_alloc(arena, size);
            memset(data, 0, size);
            uint64_t bits = 0;
            if (ak == MLIR_ATTR_KIND_INTEGER) {
                bits = (uint64_t)MLIR_GetAttributeInteger(va);
            } else {
                double d = MLIR_GetAttributeFloat(va);
                if (size == 4) {
                    float f = (float)d; uint32_t b32;
                    memcpy(&b32, &f, 4); bits = b32;
                } else if (size == 8) {
                    memcpy(&bits, &d, 8);
                } else { return false; }
            }
            for (uint32_t i = 0; i < size && i < 8; i++)
                data[i] = (uint8_t)(bits >> (8 * i));
            emit_import_global(ctx, arena, body, sym, size, align_pow,
                               is_const, data, NULL, 0);
            return true;
        }
        // Unknown attr kind -- fall through to zero-init.
    }

    // Region-init globals: walk for an llvm.mlir.addressof + llvm.return.
    // The only pattern produced by tinyc is:
    //   %0 = llvm.mlir.addressof @other : !llvm.ptr
    //   llvm.return %0 : !llvm.ptr
    // which becomes a 4-byte pointer slot with an R_WASM_MEMORY_ADDR reloc.
    if (MLIR_GetOpNumRegions(op) > 0) {
        MLIR_RegionHandle rgn = MLIR_GetOpRegion(op, 0);
        if (MLIR_GetRegionNumBlocks(rgn) > 0) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(rgn, 0);
            size_t nb = MLIR_GetBlockNumOps(blk);
            string pending_target = (string){0};
            for (size_t bi = 0; bi < nb; bi++) {
                MLIR_OpHandle bop = MLIR_GetBlockOp(blk, bi);
                string bn = MLIR_GetOpName(bop);
                if (name_eq(bn, "llvm.mlir.addressof")) {
                    MLIR_AttributeHandle ta = find_attr(bop, "global_name");
                    if (ta == MLIR_INVALID_HANDLE)
                        ta = find_attr(bop, "value");  // upstream attr name
                    if (ta == MLIR_INVALID_HANDLE) {
                        // Fallback: any FlatSymbolRef-shaped attr.
                        size_t na = MLIR_GetOpNumAttributes(bop);
                        for (size_t i = 0; i < na; i++) {
                            MLIR_AttributeHandle a = MLIR_GetOpAttribute(bop, i);
                            if (MLIR_GetAttributeKind(a) == MLIR_ATTR_KIND_STRING) {
                                ta = a; break;
                            }
                        }
                    }
                    if (ta == MLIR_INVALID_HANDLE) return false;
                    string ts = MLIR_GetAttributeString(ta);
                    if (ts.size && ts.str[0] == '@') { ts.str++; ts.size--; }
                    pending_target = ts;
                } else if (name_eq(bn, "llvm.return")) {
                    if (!pending_target.str) break;
                    size = 4;
                    if (align_pow == 0) align_pow = 2;
                    data = (uint8_t *)arena_alloc(arena, size);
                    memset(data, 0, size);
                    // Build the relocs string inline: "<off>:<target>:<addend>".
                    size_t cap = pending_target.size + 32;
                    char *buf = (char *)arena_alloc(arena, cap);
                    int n = snprintf(buf, cap, "%u:%.*s:%d", 0u,
                                     (int)pending_target.size,
                                     pending_target.str, 0);
                    size_t off = (n > 0) ? (size_t)n : 0;
                    emit_import_global(ctx, arena, body, sym, size, align_pow,
                                       is_const, data, buf, off);
                    return true;
                }
            }
        }
    }

    // Region-init globals (alternative pattern): a chain of
    //   %u = llvm.mlir.undef : !llvm.array<N x iX>
    //   %c0 = llvm.mlir.constant : iX
    //   %1 = llvm.insertvalue %u, %c0 [0]
    //   ...
    //   llvm.return %k : !llvm.array<N x iX>
    // produced by MLIR_CreateLLVMGlobalArrayInit's upstream impl. Unpack
    // the constants into a raw byte buffer matching the global's layout.
    if (MLIR_GetOpNumRegions(op) > 0 && ga != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle gty = MLIR_GetAttributeTypeValue(ga);
        if (gty != MLIR_INVALID_HANDLE && MLIR_IsTypeLLVMArray(gty)) {
            uint64_t arr_n = MLIR_GetTypeLLVMArrayNumElements(gty);
            MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(gty);
            unsigned esz = type_size_bytes(ctx, et);
            if (esz > 0 && esz <= 8 && arr_n > 0 && size > 0 &&
                size == arr_n * esz) {
                MLIR_RegionHandle rgn = MLIR_GetOpRegion(op, 0);
                if (MLIR_GetRegionNumBlocks(rgn) > 0) {
                    MLIR_BlockHandle blk = MLIR_GetRegionBlock(rgn, 0);
                    size_t nb = MLIR_GetBlockNumOps(blk);
                    // Collect (value, int) pairs for llvm.mlir.constant ops.
                    LowerGlobalCEntry *cs = (LowerGlobalCEntry *)arena_alloc(arena,
                        sizeof(LowerGlobalCEntry) * (nb + 1));
                    size_t ncs = 0;
                    uint8_t *buf = (uint8_t *)arena_alloc(arena, size);
                    memset(buf, 0, size);
                    bool ok = true;
                    bool saw_undef_or_zero = false;
                    bool saw_return = false;
                    for (size_t bi = 0; bi < nb && ok; bi++) {
                        MLIR_OpHandle bop = MLIR_GetBlockOp(blk, bi);
                        string bn = MLIR_GetOpName(bop);
                        if (name_eq(bn, "llvm.mlir.undef") ||
                            name_eq(bn, "llvm.mlir.zero") ||
                            name_eq(bn, "llvm.mlir.poison")) {
                            saw_undef_or_zero = true;
                        } else if (name_eq(bn, "llvm.mlir.constant")) {
                            MLIR_AttributeHandle ca = find_attr(bop, "value");
                            if (ca == MLIR_INVALID_HANDLE) { ok = false; break; }
                            MLIR_AttrKind ck = MLIR_GetAttributeKind(ca);
                            uint64_t bits = 0;
                            if (ck == MLIR_ATTR_KIND_INTEGER) {
                                bits = (uint64_t)MLIR_GetAttributeInteger(ca);
                            } else if (ck == MLIR_ATTR_KIND_FLOAT) {
                                double d = MLIR_GetAttributeFloat(ca);
                                if (esz == 4) {
                                    float f = (float)d; uint32_t b32;
                                    memcpy(&b32, &f, 4); bits = b32;
                                } else if (esz == 8) {
                                    memcpy(&bits, &d, 8);
                                } else { ok = false; break; }
                            } else { ok = false; break; }
                            if (MLIR_GetOpNumResults(bop) > 0) {
                                cs[ncs].v = MLIR_GetOpResult(bop, 0);
                                cs[ncs].bits = bits;
                                ncs++;
                            }
                        } else if (name_eq(bn, "llvm.insertvalue")) {
                            if (MLIR_GetOpNumOperands(bop) < 2) { ok = false; break; }
                            // Position attr — printed as "array<i64: N, ...>".
                            MLIR_AttributeHandle pa = find_attr(bop, "position");
                            if (pa == MLIR_INVALID_HANDLE) { ok = false; break; }
                            string ps = MLIR_GetAttributeAsString(ctx, pa);
                            // Skip past first ':' (or '[').
                            const char *p = ps.str;
                            const char *end = ps.str + ps.size;
                            const char *colon = NULL;
                            for (const char *q = p; q < end; q++) {
                                if (*q == ':') { colon = q; break; }
                            }
                            if (colon) p = colon + 1;
                            else if (p < end && *p == '[') p++;
                            while (p < end && (*p == ' ' || *p == '\t')) p++;
                            int64_t idx = 0;
                            bool neg = false;
                            if (p < end && (*p == '-' || *p == '+')) { neg = (*p == '-'); p++; }
                            if (p >= end || *p < '0' || *p > '9') { ok = false; break; }
                            while (p < end && *p >= '0' && *p <= '9') { idx = idx * 10 + (*p - '0'); p++; }
                            if (neg) idx = -idx;
                            if (idx < 0 || (uint64_t)idx >= arr_n) { ok = false; break; }
                            // Look up the scalar value's integer literal.
                            MLIR_ValueHandle sv = MLIR_GetOpOperand(bop, 1);
                            bool found = false;
                            uint64_t bits = 0;
                            for (size_t i = 0; i < ncs; i++) {
                                if (cs[i].v == sv) { bits = cs[i].bits; found = true; break; }
                            }
                            if (!found) { ok = false; break; }
                            // Encode `bits` into buf at offset idx*esz (LE).
                            for (unsigned b2 = 0; b2 < esz; b2++)
                                buf[(uint64_t)idx * esz + b2] = (uint8_t)(bits >> (8 * b2));
                            // Track the insertvalue's result too so
                            // nested chains could be supported.
                            if (MLIR_GetOpNumResults(bop) > 0) {
                                cs[ncs].v = MLIR_GetOpResult(bop, 0);
                                cs[ncs].bits = 0;  // not a scalar
                                ncs++;
                            }
                        } else if (name_eq(bn, "llvm.return")) {
                            saw_return = true;
                            break;
                        }
                    }
                    if (ok && saw_undef_or_zero && saw_return) {
                        emit_import_global(ctx, arena, body, sym, size, align_pow,
                                           is_const, buf, NULL, 0);
                        return true;
                    }
                }
            }
        }
    }

    // No initializer found (e.g. uninitialized scalar global). Zero-init.
    if (size == 0) return false;
    data = (uint8_t *)arena_alloc(arena, size);
    memset(data, 0, size);
    emit_import_global(ctx, arena, body, sym, size, align_pow,
                       is_const, data, NULL, 0);
    return true;
}

// =============================================================================
// MLIR emit helpers for `wasmssa.*` ops. Used both by `commit_op` (per
// in-function op) and by the module walker (for import_func / globals).
// =============================================================================
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
        case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
        case WT_F32: return MLIR_CreateTypeFloat(ctx, 32, false);
        case WT_F64: return MLIR_CreateTypeFloat(ctx, 64, false);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
}
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 64, true));
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_s_cstr(MLIR_Context *ctx, const char *name,
                                        const char *v) {
    return attr_s(ctx, name, v, v ? strlen(v) : 0);
}
static char *hex_encode(Arena *arena, const uint8_t *p, size_t n) {
    char *out = (char *)arena_alloc(arena, n ? n * 2 : 1);
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i*2 + 0] = d[(p[i] >> 4) & 0xf];
        out[i*2 + 1] = d[p[i] & 0xf];
    }
    return out;
}
static MLIR_AttributeHandle attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       const char *name,
                                       const uint8_t *p, size_t n) {
    return attr_s(ctx, name, hex_encode(arena, p, n), n * 2);
}

static MLIR_OpHandle make_op(MLIR_Context *ctx, MLIR_OpType type,
                             MLIR_AttributeHandle *attrs, size_t n_attrs,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_RegionHandle *regions, size_t n_regions,
                             uint8_t result_vt,
                             MLIR_ValueHandle *out_result) {
    MLIR_TypeHandle res_tys[1];
    MLIR_ValueHandle res_vals[1];
    size_t n_res = 0;
    if (result_vt) {
        res_tys[0] = vt_to_type(ctx, result_vt);
        res_vals[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
                                               res_tys[0], (string){0},
                                               MLIR_CreateLocationUnknown(ctx, (string){0}));
        n_res = 1;
    }
    MLIR_OpHandle op = MLIR_CreateOp(ctx, type, op_type_to_string(type),
        attrs, n_attrs, res_tys, n_res, res_vals, n_res,
        operands, n_operands, regions, n_regions,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    if (out_result) *out_result = n_res ? res_vals[0] : MLIR_INVALID_HANDLE;
    return op;
}

// Emit an imported (body-less) wasmssa.func op directly into the module body.
static void emit_import_func(MLIR_Context *ctx, Arena *arena,
                             MLIR_BlockHandle body, string sym_name,
                             string import_module, string import_name,
                             const uint8_t *param_types, size_t n_params,
                             const uint8_t *result_types, size_t n_results) {
    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",
                         sym_name.str ? sym_name.str : "",
                         sym_name.str ? sym_name.size : 0);
    attrs[na++] = attr_s_hex(ctx, arena, "param_types", param_types, n_params);
    attrs[na++] = attr_s_hex(ctx, arena, "result_types", result_types, n_results);
    attrs[na++] = attr_b(ctx, "exported", false);
    if (import_module.size > 0) {
        attrs[na++] = attr_s(ctx, "import_module", import_module.str, import_module.size);
    }
    if (import_name.size > 0) {
        attrs[na++] = attr_s(ctx, "import_name", import_name.str, import_name.size);
    }
    MLIR_OpHandle op = make_op(ctx, OP_TYPE_WASMSSA_IMPORT_FUNC,
                               attrs, na, NULL, 0, NULL, 0, 0, NULL);
    MLIR_AppendBlockOp(ctx, body, op);
}

// Recursively emit body-less llvm.func ops as wasmssa.import_func into
// `body`, including those under nested builtin.module ops.
static bool emit_import_funcs_in_region(MLIR_Context *ctx, Arena *arena,
                                        MLIR_BlockHandle body,
                                        MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    if (!emit_import_funcs_in_region(
                            ctx, arena, body, MLIR_GetOpRegion(op, 0))) {
                        return false;
                    }
                }
                continue;
            }
            if (!op_is_llvm_func(op)) continue;
            if (llvm_func_has_defined_body(op)) continue;
            uint8_t *p, *r;
            size_t np, nr;
            if (!sig_for_func(ctx, arena, op, &p, &np, &r, &nr)) return false;
            MLIR_AttributeHandle sa = find_attr(op, "sym_name");
            string nm = MLIR_GetAttributeString(sa);
            string imod = {0}, iname = {0};
            MLIR_AttributeHandle iam = find_attr(op, "wasm.import_module");
            if (iam != MLIR_INVALID_HANDLE) imod = MLIR_GetAttributeString(iam);
            MLIR_AttributeHandle ian = find_attr(op, "wasm.import_name");
            if (ian != MLIR_INVALID_HANDLE) iname = MLIR_GetAttributeString(ian);
            emit_import_func(ctx, arena, body, nm, imod, iname, p, np, r, nr);
        }
    }
    return true;
}

// Recursively lower defined llvm.func ops into wasmssa.func ops appended to
// `body`, including those under nested builtin.module ops.
static bool emit_defined_funcs_in_region(MLIR_Context *ctx, Arena *arena,
                                         ModCtx *mod, MLIR_BlockHandle body,
                                         MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    if (!emit_defined_funcs_in_region(
                            ctx, arena, mod, body,
                            MLIR_GetOpRegion(op, 0))) {
                        return false;
                    }
                }
                continue;
            }
            if (!llvm_func_has_defined_body(op)) continue;
            uint8_t *p, *r;
            size_t np, nr;
            if (!sig_for_func(ctx, arena, op, &p, &np, &r, &nr)) return false;
            MLIR_AttributeHandle sa = find_attr(op, "sym_name");
            string sym = MLIR_GetAttributeString(sa);
            bool is_main = (sym.size == 4 && memcmp(sym.str, "main", 4) == 0);
            string nm = is_main ? str_lit("__original_main") : sym;
            if (!lower_function(ctx, arena, mod, body, nm, is_main,
                                op, p, np, r, nr)) {
                return false;
            }
        }
    }
    return true;
}

// Recursively lower llvm.mlir.global ops into wasmssa.import_global ops
// appended to `body`, including those under nested builtin.module ops.
static bool emit_globals_in_region(MLIR_Context *ctx, Arena *arena,
                                   MLIR_BlockHandle body,
                                   MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    if (!emit_globals_in_region(
                            ctx, arena, body, MLIR_GetOpRegion(op, 0))) {
                        return false;
                    }
                }
                continue;
            }
            if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.global")) continue;
            if (!lower_global(ctx, arena, body, op)) {
                fprintf(stderr, "wasmssa-lower: failed to lower global\n");
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Public stage 1 entry point: walk the LLVM-dialect module and emit a
// wasmssa-form `builtin.module` directly. Each function/global is emitted
// straight into the output module body — no module-level intermediate is
// retained. Walks nested builtin.module ops recursively so every llvm.func
// and llvm.mlir.global is lowered. Output ordering per scope: import_funcs,
// then defined funcs, then import_globals.
// =============================================================================
MLIR_OpHandle mlir_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);

    Arena            *arena = MLIR_GetArenaAllocator(ctx);

    // Discover llvm.func regions that still carry llvm.br / llvm.cond_br /
    // llvm.switch (including under nested builtin.module ops). CFG analysis
    // uses a reusable scratch arena (CFGAnalysisArena) reset per target.
    LiftRegionTarget *lift_targets = NULL;
    size_t n_lift_targets =
        collect_llvm_cfg_lift_targets_in_module(arena, module, &lift_targets);
    if (n_lift_targets > 0) {
        if (!lift_targets_validate_cfg_scratch(lift_targets, n_lift_targets)) {
            return MLIR_INVALID_HANDLE;
        }
    }
    (void)lift_targets;
    (void)n_lift_targets;
    MLIR_BlockHandle  body  = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, body);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    ModCtx mod = {0};
    mod.ctx = ctx;
    mod.arena = arena;
    mod.body = body;

    // Pass 1: imported funcs (including nested builtin.module bodies).
    if (!emit_import_funcs_in_region(ctx, arena, body, mr)) {
        return MLIR_INVALID_HANDLE;
    }

    // Pass 2: defined funcs in source order, recursing into nested modules.
    if (!emit_defined_funcs_in_region(ctx, arena, &mod, body, mr)) {
        return MLIR_INVALID_HANDLE;
    }

    // Pass 3: globals last, recursing into nested modules.
    if (!emit_globals_in_region(ctx, arena, body, mr)) {
        return MLIR_INVALID_HANDLE;
    }

    return out_module;
}
