// Stage 1 of the native LLVM->WASM pipeline:
// LLVM-dialect builtin.module --(walk)--> wasmssa-form `builtin.module`.
//
// Each `llvm.func` / `llvm.mlir.global` is lowered directly to `wasmssa.*`
// MLIR ops appended into the output module body. Function bodies are walked
// as flat LLVM CFG (llvm.br / llvm.cond_br) and reconstructed into
// wasmssa.if / wasmssa.loop / switch trees. Straight-line ops use
// emit-helper / inline call sites that build a stack-local `wasmssa_op_t`
// describing one op and hands it to `commit_op`, which materializes the
// matching MLIR op into the function's body block and returns the
// MLIR_ValueHandle of its result. The per-function `vmap` then keys
// LLVM-side operand values to the wasmssa-side ValueHandle that supplies
// them; subsequent lookups thread those handles through `o.operands` to
// stitch up the wasmssa IR.
//
// No module-level scratch buffer is retained: ops flow straight from
// lower_op into the MLIR module under construction. Per-function vmap and
// CFG-walk temporaries use a scratch arena reset after each llvm.func.

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
    Arena          *arena;    // persistent wasmssa attrs (output arena)
    Arena          *scratch;  // per-function temporaries (reset each llvm.func)
    ModCtx         *mod;     // parent module emit context (for func-name lookups)
    string          fn_name;
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
    VMapEntry *nt = (VMapEntry *)arena_alloc(F->scratch, ncap * sizeof(VMapEntry));
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
    AMapEntry *nt = (AMapEntry *)arena_alloc(F->scratch, ncap * sizeof(AMapEntry));
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

static MLIR_ValueHandle emit_i32_eq(FnCtx *F, MLIR_ValueHandle a,
                                    MLIR_ValueHandle b) {
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

// Build a `wasmssa.if (%cond) : (R*)` op with pre-lowered then/else wasmssa
// blocks (used by LLVM CFG if and switch lowering).
static bool emit_wasmssa_if(FnCtx *F, MLIR_ValueHandle cond_wasm,
                            MLIR_BlockHandle then_wasm,
                            MLIR_BlockHandle else_wasm,
                            bool has_else,
                            const uint8_t *res_vts, size_t n_res,
                            MLIR_ValueHandle *out_results) {
    MLIR_RegionHandle then_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, then_r, then_wasm);
    MLIR_RegionHandle regs[2];
    regs[0] = then_r;
    size_t n_regs = 1;
    if (has_else) {
        MLIR_RegionHandle else_r = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, else_r, else_wasm);
        regs[1] = else_r;
        n_regs = 2;
    }
    MLIR_AttributeHandle as[1];
    size_t nas = 0;
    if (n_res) {
        as[nas++] = attr_s_hex(F->ctx, F->arena, "result_types", res_vts, n_res);
    }
    MLIR_ValueHandle cond_ops[1] = { cond_wasm };
    MLIR_OpHandle ifop = make_op_n(F->ctx, OP_TYPE_WASMSSA_IF, as, nas,
                                   cond_ops, 1, regs, n_regs,
                                   res_vts, n_res, out_results);
    MLIR_AppendBlockOp(F->ctx, F->body_block, ifop);
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

    if (name.size >= 4 && memcmp(name.str, "scf.", 4) == 0) {
        fprintf(stderr,
                "wasmssa-lower: '%.*s' should have been lowered to LLVM "
                "before wasmssa\n",
                (int)name.size, name.str);
        return false;
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
            uint8_t *sp = (uint8_t *)arena_alloc(F->scratch, snp ? snp : 1);
            for (size_t i = 0; i < snp; i++) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpOperand(op, i + 1)));
                if (v == 0) return false;
                sp[i] = v;
            }
            size_t nr = MLIR_GetOpNumResults(op);
            if (nr > 1) return false;
            uint8_t *sr = (uint8_t *)arena_alloc(F->scratch, 1);
            size_t snr = 0;
            if (nr == 1) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpResult(op, 0)));
                if (v == 0) return false;
                sr[0] = v;
                snr = 1;
            }
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->scratch, no * sizeof(MLIR_ValueHandle));
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
                uint32_t *offs = (uint32_t *)arena_alloc(F->scratch, (nvar ? nvar : 1) * sizeof(uint32_t));
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
                MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->scratch, nout * sizeof(MLIR_ValueHandle));
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
        MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->scratch, (no ? no : 1) * sizeof(MLIR_ValueHandle));
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
        int32_t *cidx = parse_dense_i32_array(F->scratch,
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

static bool prewalk_func(FnCtx *F, MLIR_RegionHandle body) {
    size_t nb = MLIR_GetRegionNumBlocks(body);
    for (size_t b = 0; b < nb; b++) {
        if (!prewalk_block(F, MLIR_GetRegionBlock(body, b))) return false;
    }
    if (F->va_buf_size > 0) {
        F->frame_size = align_up(F->frame_size, 8);
        F->va_buf_offset = F->frame_size;
        F->frame_size += F->va_buf_size;
    }
    F->frame_size = (F->frame_size + 15) & ~15u;
    return true;
}

// =============================================================================
// LLVM CFG lowering.
//
// This is deliberately an inverse for the CFG shapes produced by
// mlir_lower_to_llvm.c, not a generic CFG relooper.  Every recursive walk has
// explicit stop blocks, so a nested construct cannot run through its merge and
// lower the enclosing continuation twice.
// =============================================================================

typedef enum {
    CFG_FLOW_ERROR = 0,
    CFG_FLOW_STOP,  // hit a stop boundary (arm completed)
    CFG_FLOW_TERMINATED, //llvm.return
} CfgFlow;

typedef enum {
    CFG_STOP_BLOCK_RETURN = 0, // arm ends -> wasmsssa.block_return (vals)
    CFG_STOP_BR, // arm ends -> wasmsssa.br (depth, vals)
    CFG_STOP_REACH, // arm ends -> bind phi / stop
} CfgStopKind;

typedef struct {
    MLIR_BlockHandle block;
    CfgStopKind kind;
    uint32_t depth;
} CfgStop;

typedef struct CfgStops {
    const struct CfgStops *parent;
    CfgStop stop;
    bool has_stop;
} CfgStops;

typedef struct {
    FnCtx *F;
    MLIR_RegionHandle region;
    MLIR_BlockHandle *blocks; // all blocks in llvm.func
    uint8_t *state;  // 0 unseen, 1 active, 2 consumed
    size_t n_blocks;
    // Reused BFS workspace (allocated once per function in cfg_init).
    size_t *bfs_dist;
    size_t *bfs_queue;
    size_t *merge_dist; // n_roots * n_blocks for cfg_find_common_forward
    size_t merge_cap;   // merge_dist length in size_t elements
} CfgCtx;

typedef struct {
    MLIR_OpHandle term;
    size_t succ_idx;
    MLIR_BlockHandle target;
} CfgEdge;

static CfgFlow lower_block_cfg(CfgCtx *C, MLIR_BlockHandle blk,
                               const CfgStops *stops);

static bool cfg_init(CfgCtx *C, FnCtx *F, MLIR_RegionHandle region) {
    memset(C, 0, sizeof *C);
    C->F = F;
    C->region = region;
    C->n_blocks = MLIR_GetRegionNumBlocks(region);
    if (C->n_blocks == 0) return false;
    C->blocks = (MLIR_BlockHandle *)arena_alloc(
        F->scratch, C->n_blocks * sizeof(MLIR_BlockHandle));
    C->state = (uint8_t *)arena_alloc(F->scratch, C->n_blocks);
    memset(C->state, 0, C->n_blocks);
    C->bfs_dist = (size_t *)arena_alloc(
        F->scratch, C->n_blocks * sizeof(size_t));
    C->bfs_queue = (size_t *)arena_alloc(
        F->scratch, C->n_blocks * sizeof(size_t));
    if (!C->blocks || !C->state || !C->bfs_dist || !C->bfs_queue)
        return false;
    for (size_t i = 0; i < C->n_blocks; i++)
        C->blocks[i] = MLIR_GetRegionBlock(region, i);
    return true;
}

static bool cfg_ensure_merge_dist(CfgCtx *C, size_t n_roots) {
    size_t need = n_roots * C->n_blocks;
    if (need <= C->merge_cap) return true;
    C->merge_dist = (size_t *)arena_alloc(
        C->F->scratch, need * sizeof(size_t));
    if (!C->merge_dist) return false;
    C->merge_cap = need;
    return true;
}

static size_t cfg_block_index(CfgCtx *C, MLIR_BlockHandle blk) {
    for (size_t i = 0; i < C->n_blocks; i++)
        if (C->blocks[i] == blk) return i;
    return SIZE_MAX;
}

static MLIR_OpHandle cfg_terminator(MLIR_BlockHandle blk) {
    size_t n = MLIR_GetBlockNumOps(blk);
    return n ? MLIR_GetBlockOp(blk, n - 1) : MLIR_INVALID_HANDLE;
}

static void cfg_dump(CfgCtx *C) {
    if (!getenv("WASMSSA_DUMP_CFG")) return;
    fprintf(stderr, "wasmssa-lower: CFG for function '%.*s':\n",
            (int)C->F->fn_name.size, C->F->fn_name.str);
    for (size_t i = 0; i < C->n_blocks; i++) {
        MLIR_BlockHandle blk = C->blocks[i];
        MLIR_OpHandle term = cfg_terminator(blk);
        string name = term != MLIR_INVALID_HANDLE
            ? MLIR_GetOpName(term) : str_lit("<empty>");
        fprintf(stderr, "  block %zu state=%u args=%zu ops=%zu term=%.*s",
                i, (unsigned)C->state[i], MLIR_GetBlockNumArgs(blk),
                MLIR_GetBlockNumOps(blk),
                (int)name.size, name.str);
        if (term != MLIR_INVALID_HANDLE) {
            for (size_t s = 0; s < MLIR_GetOpNumSuccessors(term); s++) {
                fprintf(stderr, " %s%zu(%zu)", s ? "," : "->",
                        cfg_block_index(C, MLIR_GetOpSuccessor(term, s)),
                        MLIR_GetOpNumSuccessorOperands(term, s));
            }
        }
        fputc('\n', stderr);
    }
}

static void cfg_stops_with(CfgStops *out, const CfgStops *base,
                           MLIR_BlockHandle block, CfgStopKind kind,
                           uint32_t depth) {
    out->parent = base;
    out->stop = (CfgStop){block, kind, depth};
    out->has_stop = true;
}

static const CfgStop *cfg_find_stop(const CfgStops *stops,
                                    MLIR_BlockHandle block) {
    for (const CfgStops *it = stops; it; it = it->parent)
        if (it->has_stop && it->stop.block == block) return &it->stop;
    return NULL;
}

// Resolve successor operands only after checking that they match the target
// block arguments.  MLIR models LLVM phi nodes through precisely this edge.
static bool cfg_edge_values(CfgCtx *C, MLIR_OpHandle term, size_t succ_idx,
                            MLIR_ValueHandle **out, size_t *out_n) {
    FnCtx *F = C->F;
    if (succ_idx >= MLIR_GetOpNumSuccessors(term)) return false;
    MLIR_BlockHandle dst = MLIR_GetOpSuccessor(term, succ_idx);
    size_t n = MLIR_GetOpNumSuccessorOperands(term, succ_idx);
    size_t na = MLIR_GetBlockNumArgs(dst);
    if (n != na) {
        fprintf(stderr,
                "wasmssa-lower: edge/block-argument arity mismatch "
                "(%zu operands, %zu arguments)\n", n, na);
        return false;
    }
    MLIR_ValueHandle *vals = n ? (MLIR_ValueHandle *)arena_alloc(
        F->scratch, n * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < n; i++) {
        if (!vmap_get(F, MLIR_GetOpSuccessorOperand(term, succ_idx, i),
                      &vals[i])) {
            fprintf(stderr,
                    "wasmssa-lower: successor operand is not mapped\n");
            return false;
        }
    }
    *out = vals;
    *out_n = n;
    return true;
}

static bool cfg_bind_edge(CfgCtx *C, MLIR_OpHandle term, size_t succ_idx) {
    MLIR_ValueHandle *vals = NULL;
    size_t n = 0;
    if (!cfg_edge_values(C, term, succ_idx, &vals, &n)) return false;
    MLIR_BlockHandle dst = MLIR_GetOpSuccessor(term, succ_idx);
    for (size_t i = 0; i < n; i++) {
        MLIR_ValueHandle arg = MLIR_GetBlockArg(dst, i);
        MLIR_ValueHandle old;
        if (vmap_get(C->F, arg, &old)) {
            if (old != vals[i]) {
                fprintf(stderr,
                        "wasmssa-lower: edge block %zu -> block %zu has "
                        "a different value for argument %zu outside a "
                        "structured merge\n",
                        cfg_block_index(C, MLIR_GetOpParentBlock(term)),
                        cfg_block_index(C, dst), i);
                return false;
            }
        } else {
            vmap_set(C->F, arg, vals[i]);
        }
    }
    return true;
}

static CfgFlow cfg_emit_stop_edge(CfgCtx *C, MLIR_OpHandle term,
                                  size_t succ_idx, const CfgStop *stop) {
    if (stop->kind == CFG_STOP_REACH)
        return cfg_bind_edge(C, term, succ_idx)
            ? CFG_FLOW_STOP : CFG_FLOW_ERROR;
    MLIR_ValueHandle *vals = NULL;
    size_t n = 0;
    if (!cfg_edge_values(C, term, succ_idx, &vals, &n))
        return CFG_FLOW_ERROR;
    if (stop->kind == CFG_STOP_BLOCK_RETURN)
        emit_block_return(C->F, vals, n);
    else
        emit_br_args(C->F, stop->depth, vals, n);
    return CFG_FLOW_STOP;
}

static CfgFlow cfg_emit_stop_block(CfgCtx *C, MLIR_BlockHandle blk,
                                   const CfgStop *stop) {
    if (stop->kind == CFG_STOP_REACH) return CFG_FLOW_STOP;
    size_t n = MLIR_GetBlockNumArgs(blk);
    MLIR_ValueHandle *vals = n ? (MLIR_ValueHandle *)arena_alloc(
        C->F->scratch, n * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < n; i++) {
        if (!vmap_get(C->F, MLIR_GetBlockArg(blk, i), &vals[i]))
            return CFG_FLOW_ERROR;
    }
    if (stop->kind == CFG_STOP_BLOCK_RETURN)
        emit_block_return(C->F, vals, n);
    else
        emit_br_args(C->F, stop->depth, vals, n);
    return CFG_FLOW_STOP;
}

static bool cfg_distances(CfgCtx *C, MLIR_BlockHandle start,
                          const CfgStops *bounds, size_t *dist) {
    if (!dist) dist = C->bfs_dist;
    for (size_t i = 0; i < C->n_blocks; i++) dist[i] = SIZE_MAX;
    size_t si = cfg_block_index(C, start);
    if (si == SIZE_MAX) return false;
    size_t head = 0, tail = 0;
    dist[si] = 0;
    C->bfs_queue[tail++] = si;
    while (head < tail) {
        size_t bi = C->bfs_queue[head++];
        // A boundary remains reachable (and can itself be selected as a
        // merge), but paths belonging to an enclosing construct must not make
        // blocks beyond it appear to be local joins.
        if (cfg_find_stop(bounds, C->blocks[bi])) continue;
        MLIR_OpHandle term = cfg_terminator(C->blocks[bi]);
        if (term == MLIR_INVALID_HANDLE) continue;
        for (size_t s = 0; s < MLIR_GetOpNumSuccessors(term); s++) {
            size_t ti = cfg_block_index(C, MLIR_GetOpSuccessor(term, s));
            if (ti == SIZE_MAX || dist[ti] != SIZE_MAX) continue;
            dist[ti] = dist[bi] + 1;
            C->bfs_queue[tail++] = ti;
        }
    }
    return true;
}

// Pick the nearest common forward-reachable block.  Requiring a forward block
// keeps a loop backedge from being misclassified as an if merge.
static bool cfg_find_common_forward(CfgCtx *C, MLIR_BlockHandle parent,
                                    MLIR_BlockHandle *roots, size_t n_roots,
                                    const CfgStops *bounds,
                                    MLIR_BlockHandle *out) {
    if (n_roots == 0) return false;
    size_t pi = cfg_block_index(C, parent);
    if (pi == SIZE_MAX) return false;
    if (!cfg_ensure_merge_dist(C, n_roots)) return false;
    size_t *all = C->merge_dist;
    for (size_t r = 0; r < n_roots; r++)
        if (!cfg_distances(C, roots[r], bounds,
                           all + r * C->n_blocks))
            return false;
    size_t best = SIZE_MAX, best_max = SIZE_MAX, best_sum = SIZE_MAX;
    for (size_t b = pi + 1; b < C->n_blocks; b++) {
        if (C->state[b] != 0) continue;
        size_t mx = 0, sum = 0;
        bool common = true;
        for (size_t r = 0; r < n_roots; r++) {
            size_t d = all[r * C->n_blocks + b];
            if (d == SIZE_MAX) { common = false; break; }
            if (d > mx) mx = d;
            sum += d;
        }
        if (common && (mx < best_max ||
                       (mx == best_max && sum < best_sum))) {
            best = b;
            best_max = mx;
            best_sum = sum;
        }
    }
    if (best == SIZE_MAX) return false;
    *out = C->blocks[best];
    return true;
}

static bool cfg_block_ends_br(MLIR_BlockHandle blk,
                              MLIR_BlockHandle *target) {
    MLIR_OpHandle term = cfg_terminator(blk);
    if (term == MLIR_INVALID_HANDLE ||
        !name_eq(MLIR_GetOpName(term), "llvm.br") ||
        MLIR_GetOpNumSuccessors(term) != 1)
        return false;
    *target = MLIR_GetOpSuccessor(term, 0);
    return true;
}

// A return arm does not reach the continuation, so common reachability alone
// cannot find its merge.  In that case use the shared direct exit of the
// non-returning arms.
static bool cfg_find_arm_merge(CfgCtx *C, MLIR_BlockHandle parent,
                               MLIR_BlockHandle *roots, size_t n_roots,
                               const CfgStops *bounds,
                               MLIR_BlockHandle *merge) {
    if (cfg_find_common_forward(C, parent, roots, n_roots, bounds, merge))
        return true;
    MLIR_BlockHandle candidate = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < n_roots; i++) {
        MLIR_BlockHandle target;
        if (!cfg_block_ends_br(roots[i], &target)) continue;
        if (candidate == MLIR_INVALID_HANDLE) candidate = target;
        else if (candidate != target) return false;
    }
    size_t pi = cfg_block_index(C, parent);
    size_t ci = cfg_block_index(C, candidate);
    if (candidate == MLIR_INVALID_HANDLE || pi == SIZE_MAX ||
        ci == SIZE_MAX || ci <= pi)
        return false;
    *merge = candidate;
    return true;
}

static bool cfg_block_arg_vts(CfgCtx *C, MLIR_BlockHandle blk,
                              uint8_t **out_vts, size_t *out_n) {
    size_t n = MLIR_GetBlockNumArgs(blk);
    uint8_t *vts = n ? (uint8_t *)arena_alloc(C->F->scratch, n) : NULL;
    for (size_t i = 0; i < n; i++) {
        vts[i] = wasm_vt(C->F->ctx,
            MLIR_GetValueType(MLIR_GetBlockArg(blk, i)));
        if (!vts[i]) return false;
    }
    *out_vts = vts;
    *out_n = n;
    return true;
}

static bool cfg_map_results(CfgCtx *C, MLIR_BlockHandle blk,
                            MLIR_ValueHandle *vals, size_t n) {
    if (MLIR_GetBlockNumArgs(blk) != n) return false;
    for (size_t i = 0; i < n; i++)
        vmap_set(C->F, MLIR_GetBlockArg(blk, i), vals[i]);
    return true;
}

static bool cfg_lower_edge_arm(CfgCtx *C, CfgEdge edge,
                               const CfgStops *stops,
                               MLIR_BlockHandle *out_wasm,
                               CfgFlow *out_flow) {
    MLIR_BlockHandle saved = C->F->body_block;
    MLIR_BlockHandle wasm = MLIR_CreateBlock(C->F->ctx);
    C->F->body_block = wasm;
    const CfgStop *stop = cfg_find_stop(stops, edge.target);
    CfgFlow flow;
    if (stop)
        flow = cfg_emit_stop_edge(C, edge.term, edge.succ_idx, stop);
    else if (!cfg_bind_edge(C, edge.term, edge.succ_idx))
        flow = CFG_FLOW_ERROR;
    else
        flow = lower_block_cfg(C, edge.target, stops);
    C->F->body_block = saved;
    if (flow == CFG_FLOW_ERROR) return false;
    *out_wasm = wasm;
    *out_flow = flow;
    return true;
}

static bool cfg_emit_loop(FnCtx *F,
                          MLIR_ValueHandle *init_vals, size_t n_iter,
                          MLIR_BlockHandle loop_body,
                          const uint8_t *res_vts, size_t n_res,
                          MLIR_ValueHandle *out_results) {
    MLIR_RegionHandle loop_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, loop_r, loop_body);
    MLIR_OpHandle loop = make_op_n(F->ctx, OP_TYPE_WASMSSA_LOOP, NULL, 0,
                                   init_vals, n_iter, &loop_r, 1,
                                   NULL, 0, NULL);
    MLIR_BlockHandle outer = MLIR_CreateBlock(F->ctx);
    MLIR_AppendBlockOp(F->ctx, outer, loop);
    MLIR_OpHandle unreach = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_UNREACHABLE, NULL, 0,
        NULL, 0, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, outer, unreach);
    MLIR_RegionHandle block_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, block_r, outer);
    MLIR_OpHandle block = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_BLOCK, NULL, 0,
        NULL, 0, &block_r, 1, res_vts, n_res, out_results);
    MLIR_AppendBlockOp(F->ctx, F->body_block, block);
    return true;
}

typedef struct {
    MLIR_BlockHandle header, condition, body, exit, cont;
    MLIR_OpHandle header_cbr, exit_br;
} CfgWhile;

static bool cfg_has_edge_to(CfgCtx *C, MLIR_BlockHandle start,
                            MLIR_BlockHandle target) {
    if (!cfg_distances(C, start, NULL, NULL)) return false;
    size_t ti = cfg_block_index(C, target);
    return ti != SIZE_MAX && C->bfs_dist[ti] != SIZE_MAX;
}

static size_t cfg_distance_to_avoiding(CfgCtx *C, MLIR_BlockHandle start,
                                       MLIR_BlockHandle target,
                                       MLIR_BlockHandle avoid) {
    size_t si = cfg_block_index(C, start);
    size_t ti = cfg_block_index(C, target);
    size_t ai = cfg_block_index(C, avoid);
    if (si == SIZE_MAX || ti == SIZE_MAX || ai == SIZE_MAX || si == ai)
        return SIZE_MAX;
    size_t *dist = C->bfs_dist;
    for (size_t i = 0; i < C->n_blocks; i++) dist[i] = SIZE_MAX;
    size_t head = 0, tail = 0;
    dist[si] = 0;
    C->bfs_queue[tail++] = si;
    while (head < tail) {
        size_t bi = C->bfs_queue[head++];
        MLIR_OpHandle term = cfg_terminator(C->blocks[bi]);
        if (term == MLIR_INVALID_HANDLE) continue;
        for (size_t s = 0; s < MLIR_GetOpNumSuccessors(term); s++) {
            size_t ni = cfg_block_index(C, MLIR_GetOpSuccessor(term, s));
            if (ni == ti) {
                // The inverse lowering expects the yield/backedge emitted by
                // the while back-edge, not a forward predecessor that happens
                // to enter this block after traversing an enclosing cycle.
                if (bi >= ti) return dist[bi] + 1;
                continue;
            }
            if (ni == SIZE_MAX || ni == ai || dist[ni] != SIZE_MAX) continue;
            dist[ni] = dist[bi] + 1;
            C->bfs_queue[tail++] = ni;
        }
    }
    return SIZE_MAX;
}

// Match the br/header/body/exit-stub/continuation shape emitted by
// mlir_lower_to_llvm.c's lower_scf_while.  The header and body regions may
// contain nested lowered ifs, so the loop condition need not be the
// terminator of the entry target.  Identify it by the exact condition-edge
// the true edge can reach the header, the false edge cannot, and the false
// edge goes through the phi-forwarding exit stub.
static bool cfg_match_while(CfgCtx *C, MLIR_OpHandle entry_br, CfgWhile *W,
                            const char **why) {
    if (why) *why = "not a single-successor entry";
    if (MLIR_GetOpNumSuccessors(entry_br) != 1) return false;
    MLIR_BlockHandle header = MLIR_GetOpSuccessor(entry_br, 0);
    size_t ni = MLIR_GetOpNumSuccessorOperands(entry_br, 0);
    if (why) *why = "entry edge and header argument arities differ";
    if (ni != MLIR_GetBlockNumArgs(header)) return false;
    size_t hi = cfg_block_index(C, header);
    if (hi == SIZE_MAX) return false;

    if (why) *why = "no reachable structured loop condition";
    size_t best_back_distance = SIZE_MAX;
    CfgWhile best = {0};
    for (size_t bi = hi; bi < C->n_blocks; bi++) {
        MLIR_BlockHandle condition = C->blocks[bi];
        if (!cfg_has_edge_to(C, header, condition)) continue;
        MLIR_OpHandle cbr = cfg_terminator(condition);
        if (cbr == MLIR_INVALID_HANDLE ||
            !name_eq(MLIR_GetOpName(cbr), "llvm.cond_br") ||
            MLIR_GetOpNumOperands(cbr) < 1 ||
            MLIR_GetOpNumSuccessors(cbr) != 2)
            continue;
        MLIR_BlockHandle body = MLIR_GetOpSuccessor(cbr, 0);
        MLIR_BlockHandle exit = MLIR_GetOpSuccessor(cbr, 1);
        size_t back_distance = cfg_distance_to_avoiding(
            C, body, header, exit);
        if (back_distance == SIZE_MAX) continue;
        MLIR_OpHandle exit_br = cfg_terminator(exit);
        if (exit_br == MLIR_INVALID_HANDLE ||
            MLIR_GetBlockNumOps(exit) != 1 ||
            !name_eq(MLIR_GetOpName(exit_br), "llvm.br") ||
            MLIR_GetOpNumSuccessors(exit_br) != 1)
            continue;
        MLIR_BlockHandle cont = MLIR_GetOpSuccessor(exit_br, 0);
        size_t nt = MLIR_GetOpNumSuccessorOperands(cbr, 0);
        size_t nf = MLIR_GetOpNumSuccessorOperands(cbr, 1);
        size_t nx = MLIR_GetOpNumSuccessorOperands(exit_br, 0);
        if (nt != MLIR_GetBlockNumArgs(body) ||
            nf != MLIR_GetBlockNumArgs(exit) ||
            nx != MLIR_GetBlockNumArgs(cont) ||
            nt != nf || nf != nx)
            continue;
        bool same = true;
        for (size_t i = 0; i < nt; i++) {
            if (MLIR_GetOpSuccessorOperand(cbr, 0, i) !=
                MLIR_GetOpSuccessorOperand(cbr, 1, i)) {
                same = false;
                break;
            }
        }
        if (!same) continue;
        if (back_distance < best_back_distance) {
            best_back_distance = back_distance;
            best = (CfgWhile){
                header, condition, body, exit, cont, cbr, exit_br};
        }
    }
    if (best_back_distance == SIZE_MAX) return false;
    if (why) *why = NULL;
    *W = best;
    return true;
}

static CfgFlow cfg_lower_while(CfgCtx *C, MLIR_OpHandle entry_br,
                               const CfgWhile *W,
                               const CfgStops *outer_stops) {
    FnCtx *F = C->F;
    size_t hi = cfg_block_index(C, W->header);
    size_t ei = cfg_block_index(C, W->exit);
    if (hi == SIZE_MAX || ei == SIZE_MAX || C->state[hi] ||
        C->state[ei])
        return CFG_FLOW_ERROR;

    MLIR_ValueHandle *inits = NULL;
    size_t n_iter = 0;
    if (!cfg_edge_values(C, entry_br, 0, &inits, &n_iter))
        return CFG_FLOW_ERROR;
    uint8_t *iter_vts = n_iter ? (uint8_t *)arena_alloc(F->scratch, n_iter)
                               : NULL;
    for (size_t i = 0; i < n_iter; i++) {
        iter_vts[i] = wasm_vt(F->ctx,
            MLIR_GetValueType(MLIR_GetBlockArg(W->header, i)));
        if (!iter_vts[i]) return CFG_FLOW_ERROR;
    }
    uint8_t *res_vts = NULL;
    size_t n_res = 0;
    if (!cfg_block_arg_vts(C, W->cont, &res_vts, &n_res))
        return CFG_FLOW_ERROR;

    MLIR_BlockHandle loop_body = make_block_with_args(
        F->ctx, iter_vts, n_iter, NULL);
    for (size_t i = 0; i < n_iter; i++)
        vmap_set(F, MLIR_GetBlockArg(W->header, i),
                 MLIR_GetBlockArg(loop_body, i));

    MLIR_BlockHandle saved = F->body_block;
    F->body_block = loop_body;

    // Reconstruct nested if/switch CFG in the loop header region, but stop
    // before consuming the loop condition llvm.cond_br.  CFG_STOP_REACH binds
    // condition-block phis without emitting ops.
    CfgStops before_stops;
    cfg_stops_with(&before_stops, outer_stops, W->condition,
                   CFG_STOP_REACH, 0);
    if (lower_block_cfg(C, W->header, &before_stops) != CFG_FLOW_STOP) {
        fprintf(stderr,
                "wasmssa-lower: function '%.*s' failed while lowering "
                "the before-CFG for loop header %zu and condition %zu\n",
                (int)F->fn_name.size, F->fn_name.str, hi,
                cfg_block_index(C, W->condition));
        F->body_block = saved;
        return CFG_FLOW_ERROR;
    }
    size_t ci = cfg_block_index(C, W->condition);
    if (ci == SIZE_MAX || C->state[ci]) {
        fprintf(stderr,
                "wasmssa-lower: loop condition block is already consumed\n");
        F->body_block = saved;
        return CFG_FLOW_ERROR;
    }
    C->state[ci] = 1;
    size_t nc = MLIR_GetBlockNumOps(W->condition);
    for (size_t i = 0; i + 1 < nc; i++) {
        if (!lower_op(F, MLIR_GetBlockOp(W->condition, i))) {
            fprintf(stderr,
                    "wasmssa-lower: failed to lower a loop condition op\n");
            F->body_block = saved;
            return CFG_FLOW_ERROR;
        }
    }
    C->state[ci] = 2;

    MLIR_ValueHandle cond;
    if (!vmap_get(F, MLIR_GetOpOperand(W->header_cbr, 0), &cond)) {
        fprintf(stderr, "wasmssa-lower: loop condition is not mapped\n");
        F->body_block = saved;
        return CFG_FLOW_ERROR;
    }
    MLIR_ValueHandle exit_cond = emit_eqz(F, cond);
    MLIR_ValueHandle *exit_vals = NULL;
    size_t n_exit = 0;
    if (!cfg_edge_values(C, W->header_cbr, 1, &exit_vals, &n_exit) ||
        n_exit != n_res) {
        fprintf(stderr,
                "wasmssa-lower: loop exit values do not match results\n");
        F->body_block = saved;
        return CFG_FLOW_ERROR;
    }
    MLIR_BlockHandle exit_then = MLIR_CreateBlock(F->ctx);
    F->body_block = exit_then;
    emit_br_args(F, 2, exit_vals, n_exit);
    F->body_block = loop_body;
    if (!emit_wasmssa_if(F, exit_cond, exit_then, MLIR_INVALID_HANDLE,
                         false, NULL, 0, NULL) ||
        !cfg_bind_edge(C, W->header_cbr, 0)) {
        fprintf(stderr,
                "wasmssa-lower: failed to bind the loop body edge\n");
        F->body_block = saved;
        return CFG_FLOW_ERROR;
    }
    CfgStops header_stop, loop_stops;
    cfg_stops_with(&header_stop, outer_stops, W->header, CFG_STOP_BR, 0);
    cfg_stops_with(&loop_stops, &header_stop, W->exit, CFG_STOP_BR, 2);
    CfgFlow body_flow = lower_block_cfg(C, W->body, &loop_stops);
    F->body_block = saved;
    if (body_flow == CFG_FLOW_ERROR) {
        fprintf(stderr,
                "wasmssa-lower: function '%.*s' failed while lowering "
                "the body-CFG for loop header %zu\n",
                (int)F->fn_name.size, F->fn_name.str, hi);
        return CFG_FLOW_ERROR;
    }

    C->state[ei] = 2;  // consumed phi-forwarding stub
    MLIR_ValueHandle *results = n_res ? (MLIR_ValueHandle *)arena_alloc(
        F->scratch, n_res * sizeof(MLIR_ValueHandle)) : NULL;
    if (!cfg_emit_loop(F, inits, n_iter, loop_body,
                       res_vts, n_res, results) ||
        !cfg_map_results(C, W->cont, results, n_res)) {
        fprintf(stderr,
                "wasmssa-lower: failed to map structured loop results\n");
        return CFG_FLOW_ERROR;
    }
    return lower_block_cfg(C, W->cont, outer_stops);
}

// Accept native array<iN: ...> and upstream dense<[...]> spellings without
// consuming digits from the element type.
static bool cfg_parse_case_values(CfgCtx *C, MLIR_OpHandle op,
                                  size_t n, int64_t *out) {
    if (n == 0) return true;
    MLIR_AttributeHandle a = find_attr(op, "case_values");
    if (a == MLIR_INVALID_HANDLE) return false;
    string s = MLIR_GetAttributeAsString(C->F->ctx, a);
    size_t begin = 0, end = s.size;
    for (size_t i = 0; i + 6 <= s.size; i++) {
        if (memcmp(s.str + i, "array<", 6) == 0) {
            begin = i + 6;
            while (begin < s.size && s.str[begin] != ':') begin++;
            if (begin < s.size) begin++;
            end = begin;
            while (end < s.size && s.str[end] != '>') end++;
            break;
        }
        if (memcmp(s.str + i, "dense<", 6) == 0) {
            begin = i + 6;
            end = begin;
            while (end < s.size && s.str[end] != '>') end++;
            break;
        }
    }
    size_t p = begin, got = 0;
    while (p < end && got < n) {
        while (p < end && !((s.str[p] >= '0' && s.str[p] <= '9') ||
                            s.str[p] == '-')) p++;
        if (p == end) break;
        int64_t sign = 1;
        if (s.str[p] == '-') { sign = -1; p++; }
        if (p == end || s.str[p] < '0' || s.str[p] > '9') return false;
        int64_t v = 0;
        while (p < end && s.str[p] >= '0' && s.str[p] <= '9')
            v = v * 10 + (s.str[p++] - '0');
        out[got++] = sign * v;
    }
    return got == n;
}

static bool cfg_emit_switch_rec(FnCtx *F, size_t i, size_t n_cases,
                                const int64_t *case_vals,
                                MLIR_ValueHandle selector,
                                MLIR_BlockHandle *arms,
                                const uint8_t *res_vts, size_t n_res,
                                MLIR_ValueHandle *out) {
    MLIR_ValueHandle kc = emit_const_i32(F, (int32_t)case_vals[i]);
    MLIR_ValueHandle cmp = emit_i32_eq(F, selector, kc);
    MLIR_BlockHandle else_blk;
    if (i + 1 == n_cases) {
        else_blk = arms[0];
    } else {
        MLIR_BlockHandle saved = F->body_block;
        else_blk = MLIR_CreateBlock(F->ctx);
        F->body_block = else_blk;
        MLIR_ValueHandle *inner = n_res ? (MLIR_ValueHandle *)arena_alloc(
            F->scratch, n_res * sizeof(MLIR_ValueHandle)) : NULL;
        if (!cfg_emit_switch_rec(F, i + 1, n_cases, case_vals,
                                 selector, arms, res_vts, n_res, inner)) {
            F->body_block = saved;
            return false;
        }
        emit_block_return(F, inner, n_res);
        F->body_block = saved;
    }
    return emit_wasmssa_if(F, cmp, arms[i + 1], else_blk, true,
                           res_vts, n_res, out);
}

static bool cfg_emit_switch(FnCtx *F, const int64_t *case_vals,
                            size_t n_cases, MLIR_ValueHandle selector,
                            MLIR_BlockHandle *arms,
                            const uint8_t *res_vts, size_t n_res,
                            MLIR_ValueHandle *out) {
    if (n_cases)
        return cfg_emit_switch_rec(F, 0, n_cases, case_vals, selector,
                                   arms, res_vts, n_res, out);
    MLIR_RegionHandle r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, r, arms[0]);
    MLIR_OpHandle block = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_BLOCK, NULL, 0,
        NULL, 0, &r, 1, res_vts, n_res, out);
    MLIR_AppendBlockOp(F->ctx, F->body_block, block);
    return true;
}

static CfgFlow cfg_lower_switch_edges(CfgCtx *C,
                                      MLIR_ValueHandle selector,
                                      const int64_t *case_vals,
                                      size_t n_cases, CfgEdge *edges,
                                      MLIR_BlockHandle merge,
                                      const CfgStops *outer_stops,
                                      MLIR_BlockHandle continuation) {
    FnCtx *F = C->F;
    if (wasm_vt(F->ctx, MLIR_GetValueType(selector)) != WT_I32) {
        fprintf(stderr,
                "wasmssa-lower: llvm.switch requires an i32 selector\n");
        return CFG_FLOW_ERROR;
    }
    MLIR_ValueHandle selector_wasm;
    if (!vmap_get(F, selector, &selector_wasm)) return CFG_FLOW_ERROR;
    uint8_t *res_vts = NULL;
    size_t n_res = 0;
    CfgStops arm_stops;
    const CfgStops *use_stops = outer_stops;
    if (merge != MLIR_INVALID_HANDLE) {
        if (!cfg_block_arg_vts(C, merge, &res_vts, &n_res))
            return CFG_FLOW_ERROR;
        cfg_stops_with(&arm_stops, outer_stops, merge,
                       CFG_STOP_BLOCK_RETURN, 0);
        use_stops = &arm_stops;
    }
    size_t n_arms = n_cases + 1;
    MLIR_BlockHandle *arms = (MLIR_BlockHandle *)arena_alloc(
        F->scratch, n_arms * sizeof(MLIR_BlockHandle));
    CfgFlow *flows = (CfgFlow *)arena_alloc(
        F->scratch, n_arms * sizeof(CfgFlow));
    for (size_t i = 0; i < n_arms; i++) {
        if (!cfg_lower_edge_arm(C, edges[i], use_stops,
                                &arms[i], &flows[i]))
            return CFG_FLOW_ERROR;
    }
    MLIR_ValueHandle *results = n_res ? (MLIR_ValueHandle *)arena_alloc(
        F->scratch, n_res * sizeof(MLIR_ValueHandle)) : NULL;
    if (!cfg_emit_switch(F, case_vals, n_cases, selector_wasm,
                         arms, res_vts, n_res, results))
        return CFG_FLOW_ERROR;
    if (merge != MLIR_INVALID_HANDLE) {
        if (!cfg_map_results(C, merge, results, n_res))
            return CFG_FLOW_ERROR;
        if (continuation != merge) {
            MLIR_OpHandle stub = cfg_terminator(merge);
            MLIR_ValueHandle *forwarded = NULL;
            size_t nf = 0;
            if (stub == MLIR_INVALID_HANDLE ||
                MLIR_GetOpNumSuccessors(stub) != 1 ||
                !cfg_edge_values(C, stub, 0, &forwarded, &nf) ||
                nf != n_res ||
                !cfg_map_results(C, continuation, results, n_res))
                return CFG_FLOW_ERROR;
            size_t mi = cfg_block_index(C, merge);
            if (mi != SIZE_MAX) C->state[mi] = 2;
        }
        return lower_block_cfg(C, continuation, outer_stops);
    }
    for (size_t i = 0; i < n_arms; i++)
        if (flows[i] == CFG_FLOW_STOP) return CFG_FLOW_STOP;
    return CFG_FLOW_TERMINATED;
}

static CfgFlow cfg_lower_llvm_switch(CfgCtx *C, MLIR_BlockHandle parent,
                                     MLIR_OpHandle sw,
                                     const CfgStops *outer_stops) {
    size_t ns = MLIR_GetOpNumSuccessors(sw);
    if (MLIR_GetOpNumOperands(sw) != 1 || ns == 0)
        return CFG_FLOW_ERROR;
    size_t n_cases = ns - 1;
    int64_t *case_vals = n_cases ? (int64_t *)arena_alloc(
        C->F->scratch, n_cases * sizeof(int64_t)) : NULL;
    if (!cfg_parse_case_values(C, sw, n_cases, case_vals)) {
        fprintf(stderr,
                "wasmssa-lower: invalid llvm.switch case_values\n");
        return CFG_FLOW_ERROR;
    }
    CfgEdge *edges = (CfgEdge *)arena_alloc(
        C->F->scratch, ns * sizeof(CfgEdge));
    MLIR_BlockHandle *roots = (MLIR_BlockHandle *)arena_alloc(
        C->F->scratch, ns * sizeof(MLIR_BlockHandle));
    for (size_t i = 0; i < ns; i++) {
        roots[i] = MLIR_GetOpSuccessor(sw, i);
        edges[i] = (CfgEdge){sw, i, roots[i]};
    }
    MLIR_BlockHandle merge = MLIR_INVALID_HANDLE;
    (void)cfg_find_arm_merge(C, parent, roots, ns, outer_stops, &merge);
    return cfg_lower_switch_edges(
        C, MLIR_GetOpOperand(sw, 0), case_vals, n_cases, edges,
        merge, outer_stops, merge);
}

// Recognize one compare-chain step emitted by mlir_lower_to_llvm.c's
// lower_scf_index_switch:
// constant, icmp eq selector/constant, cond_br case/next.
static bool cfg_match_switch_step(CfgCtx *C, MLIR_BlockHandle blk,
                                  MLIR_ValueHandle expected_selector,
                                  MLIR_ValueHandle *selector,
                                  int64_t *case_value,
                                  MLIR_OpHandle *cbr) {
    (void)C;
    if (MLIR_GetBlockNumOps(blk) != 3) return false;
    MLIR_OpHandle constant = MLIR_GetBlockOp(blk, 0);
    MLIR_OpHandle cmp = MLIR_GetBlockOp(blk, 1);
    MLIR_OpHandle term = MLIR_GetBlockOp(blk, 2);
    if (!name_eq(MLIR_GetOpName(constant), "llvm.mlir.constant") ||
        !name_eq(MLIR_GetOpName(cmp), "llvm.icmp") ||
        !name_eq(MLIR_GetOpName(term), "llvm.cond_br") ||
        MLIR_GetOpNumOperands(cmp) != 2 ||
        MLIR_GetOpNumOperands(term) != 1 ||
        MLIR_GetOpNumSuccessors(term) != 2 ||
        MLIR_GetOpNumSuccessorOperands(term, 0) ||
        MLIR_GetOpNumSuccessorOperands(term, 1))
        return false;
    MLIR_AttributeHandle pred = find_attr(cmp, "predicate");
    if (pred == MLIR_INVALID_HANDLE || MLIR_GetAttributeInteger(pred) != 0 ||
        MLIR_GetValueDefiningOp(MLIR_GetOpOperand(term, 0)) != cmp)
        return false;
    bool c0 = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(cmp, 0)) == constant;
    bool c1 = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(cmp, 1)) == constant;
    if (c0 == c1) return false;
    MLIR_ValueHandle sel = MLIR_GetOpOperand(cmp, c0 ? 1 : 0);
    if (expected_selector != MLIR_INVALID_HANDLE && sel != expected_selector)
        return false;
    MLIR_AttributeHandle value = find_attr(constant, "value");
    if (value == MLIR_INVALID_HANDLE) return false;
    *selector = sel;
    *case_value = MLIR_GetAttributeInteger(value);
    *cbr = term;
    return true;
}

static CfgFlow cfg_try_switch_chain(CfgCtx *C, MLIR_BlockHandle parent,
                                    MLIR_OpHandle entry_br,
                                    const CfgStops *outer_stops,
                                    bool *matched) {
    *matched = false;
    if (MLIR_GetOpNumSuccessors(entry_br) != 1)
        return CFG_FLOW_ERROR;
    MLIR_BlockHandle cur = MLIR_GetOpSuccessor(entry_br, 0);
    MLIR_ValueHandle selector = MLIR_INVALID_HANDLE;
    int64_t *case_vals = (int64_t *)arena_alloc(
        C->F->scratch, C->n_blocks * sizeof(int64_t));
    CfgEdge *case_edges = (CfgEdge *)arena_alloc(
        C->F->scratch, C->n_blocks * sizeof(CfgEdge));
    MLIR_BlockHandle *chain = (MLIR_BlockHandle *)arena_alloc(
        C->F->scratch, C->n_blocks * sizeof(MLIR_BlockHandle));
    size_t n_cases = 0;
    CfgEdge default_edge = {0};
    while (n_cases < C->n_blocks) {
        MLIR_ValueHandle step_selector;
        int64_t value;
        MLIR_OpHandle cbr;
        if (!cfg_match_switch_step(C, cur, selector, &step_selector,
                                   &value, &cbr))
            break;
        if (selector == MLIR_INVALID_HANDLE) selector = step_selector;
        chain[n_cases] = cur;
        case_vals[n_cases] = value;
        case_edges[n_cases] = (CfgEdge){
            cbr, 0, MLIR_GetOpSuccessor(cbr, 0)};
        n_cases++;
        MLIR_BlockHandle next = MLIR_GetOpSuccessor(cbr, 1);
        MLIR_ValueHandle probe_selector;
        int64_t probe_value;
        MLIR_OpHandle probe_cbr;
        if (cfg_match_switch_step(C, next, selector, &probe_selector,
                                  &probe_value, &probe_cbr)) {
            cur = next;
            continue;
        }
        default_edge = (CfgEdge){cbr, 1, next};
        break;
    }
    if (n_cases == 0 || default_edge.term == MLIR_INVALID_HANDLE)
        return CFG_FLOW_ERROR;
    *matched = true;
    size_t n_arms = n_cases + 1;
    CfgEdge *edges = (CfgEdge *)arena_alloc(
        C->F->scratch, n_arms * sizeof(CfgEdge));
    MLIR_BlockHandle *roots = (MLIR_BlockHandle *)arena_alloc(
        C->F->scratch, n_arms * sizeof(MLIR_BlockHandle));
    edges[0] = default_edge;
    roots[0] = default_edge.target;
    for (size_t i = 0; i < n_cases; i++) {
        edges[i + 1] = case_edges[i];
        roots[i + 1] = case_edges[i].target;
    }
    MLIR_BlockHandle merge;
    if (!cfg_find_arm_merge(C, parent, roots, n_arms,
                            outer_stops, &merge)) {
        fprintf(stderr,
                "wasmssa-lower: switch compare chain has no merge\n");
        return CFG_FLOW_ERROR;
    }
    MLIR_OpHandle stub = cfg_terminator(merge);
    if (stub == MLIR_INVALID_HANDLE || MLIR_GetBlockNumOps(merge) != 1 ||
        !name_eq(MLIR_GetOpName(stub), "llvm.br") ||
        MLIR_GetOpNumSuccessors(stub) != 1) {
        fprintf(stderr,
                "wasmssa-lower: switch merge is not a phi stub\n");
        return CFG_FLOW_ERROR;
    }
    MLIR_BlockHandle cont = MLIR_GetOpSuccessor(stub, 0);
    for (size_t i = 0; i < n_cases; i++) {
        size_t ci = cfg_block_index(C, chain[i]);
        if (ci == SIZE_MAX || C->state[ci]) return CFG_FLOW_ERROR;
        C->state[ci] = 2;
    }
    return cfg_lower_switch_edges(C, selector, case_vals, n_cases,
                                  edges, merge, outer_stops, cont);
}

static CfgFlow cfg_lower_cond_br(CfgCtx *C, MLIR_BlockHandle parent,
                                 MLIR_OpHandle cbr,
                                 const CfgStops *outer_stops) {
    if (MLIR_GetOpNumOperands(cbr) != 1 ||
        MLIR_GetOpNumSuccessors(cbr) != 2)
        return CFG_FLOW_ERROR;
    MLIR_ValueHandle cond;
    if (!vmap_get(C->F, MLIR_GetOpOperand(cbr, 0), &cond))
        return CFG_FLOW_ERROR;
    MLIR_BlockHandle roots[2] = {
        MLIR_GetOpSuccessor(cbr, 0), MLIR_GetOpSuccessor(cbr, 1)};
    MLIR_BlockHandle merge = MLIR_INVALID_HANDLE;
    bool have_merge = cfg_find_arm_merge(
        C, parent, roots, 2, outer_stops, &merge);

    // With no normal join, retain both terminating paths inside the if.  This
    // covers early returns and else-if trees without treating return as merge.
    if (!have_merge) {
        MLIR_BlockHandle then_wasm, else_wasm;
        CfgFlow tf, ef;
        if (!cfg_lower_edge_arm(C, (CfgEdge){cbr, 0, roots[0]},
                                outer_stops, &then_wasm, &tf) ||
            !cfg_lower_edge_arm(C, (CfgEdge){cbr, 1, roots[1]},
                                outer_stops, &else_wasm, &ef) ||
            !emit_wasmssa_if(C->F, cond, then_wasm, else_wasm, true,
                             NULL, 0, NULL))
            return CFG_FLOW_ERROR;
        return (tf == CFG_FLOW_STOP || ef == CFG_FLOW_STOP)
            ? CFG_FLOW_STOP : CFG_FLOW_TERMINATED;
    }

    bool has_else = roots[0] == roots[1] || roots[1] != merge;
    uint8_t *res_vts = NULL;
    size_t n_res = 0;
    if (!cfg_block_arg_vts(C, merge, &res_vts, &n_res))
        return CFG_FLOW_ERROR;
    // Preserve false-edge phi inputs instead of silently dropping them.
    if (n_res && !has_else) has_else = true;
    CfgStops arm_stops;
    cfg_stops_with(&arm_stops, outer_stops, merge,
                   CFG_STOP_BLOCK_RETURN, 0);
    MLIR_BlockHandle then_wasm, else_wasm = MLIR_INVALID_HANDLE;
    CfgFlow tf, ef;
    if (!cfg_lower_edge_arm(C, (CfgEdge){cbr, 0, roots[0]},
                            &arm_stops, &then_wasm, &tf))
        return CFG_FLOW_ERROR;
    if (has_else &&
        !cfg_lower_edge_arm(C, (CfgEdge){cbr, 1, roots[1]},
                            &arm_stops, &else_wasm, &ef))
        return CFG_FLOW_ERROR;
    MLIR_ValueHandle *results = n_res ? (MLIR_ValueHandle *)arena_alloc(
        C->F->scratch, n_res * sizeof(MLIR_ValueHandle)) : NULL;
    if (!emit_wasmssa_if(C->F, cond, then_wasm, else_wasm, has_else,
                         res_vts, n_res, results) ||
        !cfg_map_results(C, merge, results, n_res))
        return CFG_FLOW_ERROR;
    return lower_block_cfg(C, merge, outer_stops);
}

// Classify the most specific llvm.br shapes first.  A generic forward branch
// is only a fallthrough after loop and switch entry recognition have failed.
static CfgFlow lower_block_cfg(CfgCtx *C, MLIR_BlockHandle blk,
                               const CfgStops *stops) {
    const CfgStop *entry_stop = cfg_find_stop(stops, blk);
    if (entry_stop) return cfg_emit_stop_block(C, blk, entry_stop);
    size_t bi = cfg_block_index(C, blk);
    if (bi == SIZE_MAX) return CFG_FLOW_ERROR;
    if (C->state[bi]) {
        fprintf(stderr,
                "wasmssa-lower: block %zu reached twice outside a merge\n",
                bi);
        return CFG_FLOW_ERROR;
    }
    C->state[bi] = 1;
    size_t n = MLIR_GetBlockNumOps(blk);
    if (n == 0) return CFG_FLOW_ERROR;
    for (size_t i = 0; i + 1 < n; i++) {
        if (!lower_op(C->F, MLIR_GetBlockOp(blk, i)))
            return CFG_FLOW_ERROR;
    }
    MLIR_OpHandle term = MLIR_GetBlockOp(blk, n - 1);
    string tn = MLIR_GetOpName(term);
    C->state[bi] = 2;
    if (name_eq(tn, "llvm.return"))
        return lower_op(C->F, term) ? CFG_FLOW_TERMINATED : CFG_FLOW_ERROR;
    if (name_eq(tn, "llvm.unreachable")) {
        emit_unreachable(C->F);
        return CFG_FLOW_TERMINATED;
    }
    if (name_eq(tn, "llvm.switch"))
        return cfg_lower_llvm_switch(C, blk, term, stops);
    if (name_eq(tn, "llvm.cond_br"))
        return cfg_lower_cond_br(C, blk, term, stops);
    if (name_eq(tn, "llvm.br")) {
        if (MLIR_GetOpNumSuccessors(term) != 1)
            return CFG_FLOW_ERROR;
        MLIR_BlockHandle target = MLIR_GetOpSuccessor(term, 0);
        const CfgStop *stop = cfg_find_stop(stops, target);
        if (stop) return cfg_emit_stop_edge(C, term, 0, stop);
        CfgWhile W;
        const char *while_why = NULL;
        if (cfg_match_while(C, term, &W, &while_why))
            return cfg_lower_while(C, term, &W, stops);
        bool matched_switch = false;
        CfgFlow sf = cfg_try_switch_chain(
            C, blk, term, stops, &matched_switch);
        if (matched_switch) return sf;
        size_t ti = cfg_block_index(C, target);
        if (ti == SIZE_MAX || ti <= bi) {
            cfg_dump(C);
            fprintf(stderr,
                    "wasmssa-lower: function '%.*s' has unrecognized "
                    "backward llvm.br from block %zu to block %zu (%s)\n",
                    (int)C->F->fn_name.size, C->F->fn_name.str,
                    bi, ti, while_why ? while_why : "not a while loop");
            MLIR_OpHandle ht = cfg_terminator(target);
            if (ht != MLIR_INVALID_HANDLE &&
                name_eq(MLIR_GetOpName(ht), "llvm.cond_br") &&
                MLIR_GetOpNumSuccessors(ht) == 2) {
                MLIR_BlockHandle hb = MLIR_GetOpSuccessor(ht, 0);
                MLIR_BlockHandle he = MLIR_GetOpSuccessor(ht, 1);
                MLIR_OpHandle xe = cfg_terminator(he);
                size_t nx = xe != MLIR_INVALID_HANDLE &&
                            MLIR_GetOpNumSuccessors(xe) == 1
                    ? MLIR_GetOpNumSuccessorOperands(xe, 0) : SIZE_MAX;
                size_t nc = xe != MLIR_INVALID_HANDLE &&
                            MLIR_GetOpNumSuccessors(xe) == 1
                    ? MLIR_GetBlockNumArgs(MLIR_GetOpSuccessor(xe, 0))
                    : SIZE_MAX;
                fprintf(stderr,
                        "wasmssa-lower: while candidate arities: "
                        "entry=%zu/header=%zu, true=%zu/body=%zu, "
                        "false=%zu/exit=%zu, exit=%zu/continuation=%zu\n",
                        MLIR_GetOpNumSuccessorOperands(term, 0),
                        MLIR_GetBlockNumArgs(target),
                        MLIR_GetOpNumSuccessorOperands(ht, 0),
                        MLIR_GetBlockNumArgs(hb),
                        MLIR_GetOpNumSuccessorOperands(ht, 1),
                        MLIR_GetBlockNumArgs(he), nx, nc);
            }
            return CFG_FLOW_ERROR;
        }
        if (!cfg_bind_edge(C, term, 0)) return CFG_FLOW_ERROR;
        return lower_block_cfg(C, target, stops);
    }
    fprintf(stderr,
            "wasmssa-lower: block %zu has unsupported terminator '%.*s'\n",
            bi, (int)tn.size, tn.str);
    return CFG_FLOW_ERROR;
}

// Lower a defined `llvm.func` into a `wasmssa.func` op appended to the
// module body. The function's signature attrs are passed in (param_types
// includes the synthetic trailing i32 for vararg functions); the special
// "__original_main" rename + exported flag are picked by the caller.
static bool lower_function(MLIR_Context *ctx, Arena *arena, Arena *scratch,
                           ModCtx *mod,
                           MLIR_BlockHandle mod_body,
                           string fn_name, bool exported,
                           MLIR_OpHandle fn,
                           const uint8_t *param_types, size_t n_params,
                           const uint8_t *result_types, size_t n_results) {
    FnCtx F;
    memset(&F, 0, sizeof F);
    F.ctx = ctx;
    F.arena = arena;
    F.scratch = scratch;
    F.mod = mod;
    F.fn_name = fn_name;
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

    if (!prewalk_func(&F, body)) goto fail;

    // Prologue: __stack_pointer -= frame_size; sp_def = result.
    if (F.frame_size > 0) {
        MLIR_ValueHandle sp_orig = emit_global_get(&F, /*sp*/0);
        MLIR_ValueHandle kf      = emit_const_i32(&F, (int32_t)F.frame_size);
        MLIR_ValueHandle sp_new  = emit_sub_i32(&F, sp_orig, kf);
        emit_global_set(&F, /*sp*/0, sp_new);
        F.sp_value = sp_new;
    }

    {
        CfgCtx C;
        CfgStops stops = {0};
        if (!cfg_init(&C, &F, body)) goto fail;
        if (lower_block_cfg(&C, entry, &stops) == CFG_FLOW_ERROR) {
            cfg_dump(&C);
            fprintf(stderr,
                    "wasmssa-lower: CFG reconstruction failed in "
                    "function '%.*s'\n",
                    (int)fn_name.size, fn_name.str);
            goto fail;
        }
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

// =============================================================================
// Public stage 1 entry point: walk the LLVM-dialect module and emit a
// wasmssa-form `builtin.module` directly. Each function/global is emitted
// straight into the output module body — no module-level intermediate is
// retained. Output ordering (matches the prior two-pass implementation
// byte-for-byte): import_funcs, then defined funcs, then import_globals.
// =============================================================================
MLIR_OpHandle mlir_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    Arena            *arena = MLIR_GetArenaAllocator(ctx);
    Arena *scratch = arena_create(1u * 1024u * 1024u);
    if (!scratch) return MLIR_INVALID_HANDLE;
    arena_pos_t scratch0 = arena_get_pos(scratch);
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

    // Pass 1: imported (declared, body-less) funcs first so they take the
    // low function-index space in wasm.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (has_body) continue;
        arena_reset(scratch, scratch0);
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, scratch, op, &p, &np, &r, &nr)) {
            arena_destroy(scratch);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        string nm = MLIR_GetAttributeString(sa);

        // Forward `wasm.import_module` / `wasm.import_name` annotations
        // (from `__attribute__((__import_module__("...")))`) so the
        // binary emitter can place this import in the requested module
        // (e.g. WASI's `wasi_snapshot_preview1`) instead of the default
        // `env`.
        string imod = {0}, iname = {0};
        MLIR_AttributeHandle iam = find_attr(op, "wasm.import_module");
        if (iam != MLIR_INVALID_HANDLE) imod = MLIR_GetAttributeString(iam);
        MLIR_AttributeHandle ian = find_attr(op, "wasm.import_name");
        if (ian != MLIR_INVALID_HANDLE) iname = MLIR_GetAttributeString(ian);

        emit_import_func(ctx, arena, body, nm, imod, iname, p, np, r, nr);
    }

    // Pass 2: defined funcs in source order. `is_function_symbol`
    // discovers each func via its already-emitted wasmssa.func wrapper
    // in `body`; refs between earlier defs therefore work naturally.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (!has_body) continue;
        arena_reset(scratch, scratch0);
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, scratch, op, &p, &np, &r, &nr)) {
            arena_destroy(scratch);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        string sym = MLIR_GetAttributeString(sa);
        bool is_main = (sym.size == 4 && memcmp(sym.str, "main", 4) == 0);
        string nm = is_main ? str_lit("__original_main") : sym;

        if (!lower_function(ctx, arena, scratch, &mod, body, nm, is_main,
                            op, p, np, r, nr)) {
            arena_destroy(scratch);
            return MLIR_INVALID_HANDLE;
        }
    }

    // Pass 3: globals last (matches the legacy emit ordering of
    // funcs-before-globals in the previously buffered module).
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.global")) continue;
        if (!lower_global(ctx, arena, body, op)) {
            fprintf(stderr, "wasmssa-lower: failed to lower global\n");
            arena_destroy(scratch);
            return MLIR_INVALID_HANDLE;
        }
    }

    arena_destroy(scratch);
    return out_module;
}
