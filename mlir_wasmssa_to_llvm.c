// See mlir_wasmssa_to_llvm.h for the role of this pass in the pipeline.
//
// Lifts a `wasmssa` builtin.module into the in-house `llvm` MLIR dialect so
// the single unified backend (mlir_llvm_to_aarch64.c) can serve the
// `--from-wasm` self-host path. This is the WASM-input counterpart to the
// C-frontend emit.c (which produces the `llvm` dialect directly).
//
// Coverage grows test-by-test, mirroring how mlir_wasmssa_to_wmir.c was
// built up. The current milestone handles single-block, straight-line
// integer functions (the macho_exit / macho_arith shape): module walk,
// import_func recognition, per-function locals-as-alloca lowering, and the
// const / local_get / local_set / add / sub / binop(arith) / extend_i32_s /
// call / return ops. Any unsupported op (control flow, floats, linear
// memory, globals, ...) makes the lowering fail cleanly with a diagnostic so
// the opt-in path degrades gracefully while the default `wmir` path is
// untouched.

#include "mlir_wasmssa_to_llvm.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

// wasm valtype bytes.
#define WT_I32 0x7f
#define WT_I64 0x7e
#define WT_F32 0x7d
#define WT_F64 0x7c

// =============================================================================
// Attribute helpers.
// =============================================================================
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_ty(MLIR_Context *ctx, const char *name,
                                    MLIR_TypeHandle ty) {
    return MLIR_CreateAttributeType(ctx, str_from_cstr_view((char *)name), ty);
}

static int64_t at_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static bool at_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}
static string at_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}

// Map a wasm valtype to its `llvm`-dialect carrier integer type. Milestone 1
// is integer-only; f32/f64 are deferred (the caller rejects them).
static MLIR_TypeHandle vt_to_llvm(MLIR_Context *ctx, uint8_t vt) {
    if (vt == WT_I64) return MLIR_CreateTypeInteger(ctx, 64, true);
    return MLIR_CreateTypeInteger(ctx, 32, true);
}
static bool vt_is_int(uint8_t vt) { return vt == WT_I32 || vt == WT_I64; }

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}
// Decode the i-th valtype byte from an ASCII-hex type string ("7e7f" = i64,i32).
static uint8_t type_byte_at(string s, size_t i) {
    if ((2 * i + 1) >= s.size) return 0;
    return (uint8_t)((hexval(s.str[2 * i]) << 4) | hexval(s.str[2 * i + 1]));
}

// =============================================================================
// VMap: wasmssa result value -> lifted llvm value. Open-addressing hash keyed
// on the MLIR_ValueHandle (sentinel MLIR_INVALID_HANDLE == empty). Mirrors the
// wmir lifter's map; lookups only, so iteration order never affects output.
// =============================================================================
typedef struct {
    MLIR_ValueHandle *src;
    MLIR_ValueHandle *dst;
    size_t            n, cap;
} VMap;

static size_t vmap_hash(MLIR_ValueHandle k) {
    size_t h = (size_t)k;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h;
}
static void vmap_insert_raw(MLIR_ValueHandle *src, MLIR_ValueHandle *dst,
                            size_t cap, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    size_t mask = cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (src[i] != MLIR_INVALID_HANDLE && src[i] != k) i = (i + 1) & mask;
    src[i] = k;
    dst[i] = v;
}
static void vmap_grow(VMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    MLIR_ValueHandle *nsrc = (MLIR_ValueHandle *)calloc(ncap, sizeof(*nsrc));
    MLIR_ValueHandle *ndst = (MLIR_ValueHandle *)malloc(ncap * sizeof(*ndst));
    for (size_t i = 0; i < m->cap; i++)
        if (m->src[i] != MLIR_INVALID_HANDLE)
            vmap_insert_raw(nsrc, ndst, ncap, m->src[i], m->dst[i]);
    free(m->src);
    free(m->dst);
    m->src = nsrc;
    m->dst = ndst;
    m->cap = ncap;
}
static void vmap_set(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    if ((m->n + 1) * 4 >= m->cap * 3) vmap_grow(m);
    size_t mask = m->cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (m->src[i] != MLIR_INVALID_HANDLE && m->src[i] != k) i = (i + 1) & mask;
    if (m->src[i] == MLIR_INVALID_HANDLE) m->n++;
    m->src[i] = k;
    m->dst[i] = v;
}
static int vmap_get(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    if (m->cap == 0) return 0;
    size_t mask = m->cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (m->src[i] != MLIR_INVALID_HANDLE) {
        if (m->src[i] == k) { *out = m->dst[i]; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

// =============================================================================
// Per-function lowering state.
// =============================================================================
typedef struct {
    MLIR_Context     *ctx;
    MLIR_BlockHandle  blk;       // destination block (single entry block)
    VMap             *vmap;
    MLIR_ValueHandle *local_ptr; // alloca ptr per local index (params + decls)
    uint8_t          *local_vt;  // valtype per local index
    size_t            n_locals;  // params + declared locals
} FLower;

// Create a fresh op result value of the given type.
static MLIR_ValueHandle mk_res(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    return MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, ty,
        (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}));
}
static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na,
                              MLIR_TypeHandle *rtys, size_t nr,
                              MLIR_ValueHandle *res,
                              MLIR_ValueHandle *ops, size_t no) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, rtys, nr, res, nr, ops, no, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Map a wasmssa.binop wasm opcode to an `llvm`-dialect integer op. Returns
// false for opcodes outside the milestone-1 (arithmetic / bitwise) subset.
static bool binop_to_llvm(int64_t opc, MLIR_OpType *out) {
    switch (opc) {
        // i32
        case 0x6a: *out = OP_TYPE_LLVM_ADD;  return true;
        case 0x6b: *out = OP_TYPE_LLVM_SUB;  return true;
        case 0x6c: *out = OP_TYPE_LLVM_MUL;  return true;
        case 0x6d: *out = OP_TYPE_LLVM_SDIV; return true;
        case 0x6e: *out = OP_TYPE_LLVM_UDIV; return true;
        case 0x6f: *out = OP_TYPE_LLVM_SREM; return true;
        case 0x70: *out = OP_TYPE_LLVM_UREM; return true;
        case 0x71: *out = OP_TYPE_LLVM_AND;  return true;
        case 0x72: *out = OP_TYPE_LLVM_OR;   return true;
        case 0x73: *out = OP_TYPE_LLVM_XOR;  return true;
        case 0x74: *out = OP_TYPE_LLVM_SHL;  return true;
        case 0x75: *out = OP_TYPE_LLVM_ASHR; return true;
        case 0x76: *out = OP_TYPE_LLVM_LSHR; return true;
        // i64
        case 0x7c: *out = OP_TYPE_LLVM_ADD;  return true;
        case 0x7d: *out = OP_TYPE_LLVM_SUB;  return true;
        case 0x7e: *out = OP_TYPE_LLVM_MUL;  return true;
        case 0x7f: *out = OP_TYPE_LLVM_SDIV; return true;
        case 0x80: *out = OP_TYPE_LLVM_UDIV; return true;
        case 0x81: *out = OP_TYPE_LLVM_SREM; return true;
        case 0x82: *out = OP_TYPE_LLVM_UREM; return true;
        case 0x83: *out = OP_TYPE_LLVM_AND;  return true;
        case 0x84: *out = OP_TYPE_LLVM_OR;   return true;
        case 0x85: *out = OP_TYPE_LLVM_XOR;  return true;
        case 0x86: *out = OP_TYPE_LLVM_SHL;  return true;
        case 0x87: *out = OP_TYPE_LLVM_ASHR; return true;
        case 0x88: *out = OP_TYPE_LLVM_LSHR; return true;
        default: return false;
    }
}

// Lower one wasmssa body op into `llvm`-dialect op(s) appended to L->blk.
static bool lower_op(FLower *L, MLIR_OpHandle op) {
    MLIR_Context *ctx = L->ctx;
    MLIR_OpType t = MLIR_GetOpType(op);

    switch (t) {
    case OP_TYPE_WASMSSA_CONST: {
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: float const not yet supported\n");
            return false;
        }
        int64_t v = at_i(op, "value");
        MLIR_TypeHandle ity = vt_to_llvm(ctx, vt);
        MLIR_AttributeHandle va = MLIR_CreateAttributeInteger(
            ctx, str_from_cstr_view((char *)"value"), v, ity);
        MLIR_TypeHandle rt[1] = { ity };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ity) };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_MLIR_CONSTANT,
            &va, 1, rt, 1, r, NULL, 0);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_LOCAL_GET: {
        int64_t idx = at_i(op, "local_idx");
        if (idx < 0 || (size_t)idx >= L->n_locals) {
            fprintf(stderr, "wasmssa->llvm: local_get idx %lld out of range\n",
                    (long long)idx);
            return false;
        }
        uint8_t vt = L->local_vt[idx];
        MLIR_TypeHandle ety = vt_to_llvm(ctx, vt);
        MLIR_TypeHandle rt[1] = { ety };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ety) };
        MLIR_ValueHandle ops[1] = { L->local_ptr[idx] };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_LOAD,
            NULL, 0, rt, 1, r, ops, 1);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_LOCAL_SET: {
        int64_t idx = at_i(op, "local_idx");
        if (idx < 0 || (size_t)idx >= L->n_locals) {
            fprintf(stderr, "wasmssa->llvm: local_set idx %lld out of range\n",
                    (long long)idx);
            return false;
        }
        MLIR_ValueHandle v;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &v)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on local_set\n");
            return false;
        }
        MLIR_ValueHandle ops[2] = { v, L->local_ptr[idx] };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_STORE,
            NULL, 0, NULL, 0, NULL, ops, 2);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        return true;
    }

    case OP_TYPE_WASMSSA_ADD:
    case OP_TYPE_WASMSSA_SUB: {
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: float add/sub not yet supported\n");
            return false;
        }
        MLIR_ValueHandle a, b;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on add/sub\n");
            return false;
        }
        MLIR_OpType ot = (t == OP_TYPE_WASMSSA_ADD) ? OP_TYPE_LLVM_ADD
                                                    : OP_TYPE_LLVM_SUB;
        MLIR_TypeHandle ity = vt_to_llvm(ctx, vt);
        MLIR_TypeHandle rt[1] = { ity };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ity) };
        MLIR_ValueHandle ops[2] = { a, b };
        MLIR_OpHandle out = build_op(ctx, ot, NULL, 0, rt, 1, r, ops, 2);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_BINOP: {
        int64_t opc = at_i(op, "wasm_opcode");
        MLIR_OpType ot;
        if (!binop_to_llvm(opc, &ot)) {
            fprintf(stderr,
                "wasmssa->llvm: binop opcode 0x%llx not yet supported\n",
                (unsigned long long)opc);
            return false;
        }
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        MLIR_ValueHandle a, b;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on binop\n");
            return false;
        }
        MLIR_TypeHandle ity = vt_to_llvm(ctx, vt);
        MLIR_TypeHandle rt[1] = { ity };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ity) };
        MLIR_ValueHandle ops[2] = { a, b };
        MLIR_OpHandle out = build_op(ctx, ot, NULL, 0, rt, 1, r, ops, 2);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_EXTEND_I32_S: {
        MLIR_ValueHandle a;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on extend_i32_s\n");
            return false;
        }
        MLIR_TypeHandle i64ty = MLIR_CreateTypeInteger(ctx, 64, true);
        MLIR_TypeHandle rt[1] = { i64ty };
        MLIR_ValueHandle r[1] = { mk_res(ctx, i64ty) };
        MLIR_ValueHandle ops[1] = { a };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_SEXT,
            NULL, 0, rt, 1, r, ops, 1);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_CALL: {
        string callee = at_s(op, "target");
        if (callee.size == 0) {
            fprintf(stderr, "wasmssa->llvm: call without target\n");
            return false;
        }
        // WASI imports map to native libSystem stubs.
        if (callee.size == 9 && memcmp(callee.str, "proc_exit", 9) == 0)
            callee = str_lit("_exit");
        size_t no = MLIR_GetOpNumOperands(op);
        MLIR_ValueHandle *ops = (MLIR_ValueHandle *)malloc(
            (no ? no : 1) * sizeof(MLIR_ValueHandle));
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, k), &ops[k])) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on call\n");
                free(ops);
                return false;
            }
        }
        size_t nr = MLIR_GetOpNumResults(op);
        if (nr > 1) {
            fprintf(stderr, "wasmssa->llvm: multi-result call unsupported\n");
            free(ops);
            return false;
        }
        MLIR_AttributeHandle attrs[1] = {
            attr_s(ctx, "callee", callee.str, callee.size)
        };
        MLIR_TypeHandle rt[1];
        MLIR_ValueHandle r[1];
        if (nr == 1) {
            MLIR_TypeHandle ty = MLIR_GetValueType(MLIR_GetOpResult(op, 0));
            rt[0] = ty;
            r[0] = mk_res(ctx, ty);
        }
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_CALL,
            attrs, 1, rt, nr, r, ops, no);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        if (nr == 1) vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        free(ops);
        return true;
    }

    case OP_TYPE_WASMSSA_RETURN: {
        size_t no = MLIR_GetOpNumOperands(op);
        if (no > 1) {
            fprintf(stderr, "wasmssa->llvm: multi-value return unsupported\n");
            return false;
        }
        MLIR_ValueHandle ops[1];
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, k), &ops[k])) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on return\n");
                return false;
            }
        }
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_RETURN,
            NULL, 0, NULL, 0, NULL, ops, no);
        MLIR_AppendBlockOp(ctx, L->blk, out);
        return true;
    }

    default:
        fprintf(stderr,
            "wasmssa->llvm: op '%.*s' not yet supported (milestone 1)\n",
            (int)MLIR_GetOpName(op).size, MLIR_GetOpName(op).str);
        return false;
    }
}

// Lift one wasmssa.func into an `llvm.func`. Returns MLIR_INVALID_HANDLE on
// failure (the caller aborts the whole module).
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src) {
    string name     = at_s(src, "sym_name");
    bool   exported = at_b(src, "exported");
    string pt       = at_s(src, "param_types");
    string lt       = at_s(src, "local_types");

    // The backend synthesises `_start` -> `bl main`, so the exported wasm
    // entry (wasi_start) is renamed to `main`.
    if (exported) name = str_lit("main");

    if (MLIR_GetOpNumRegions(src) < 1) {
        fprintf(stderr, "wasmssa->llvm: func '%.*s' has no region\n",
                (int)name.size, name.str);
        return MLIR_INVALID_HANDLE;
    }
    MLIR_RegionHandle sreg = MLIR_GetOpRegion(src, 0);
    if (MLIR_GetRegionNumBlocks(sreg) != 1) {
        fprintf(stderr,
            "wasmssa->llvm: func '%.*s' has a multi-block CFG "
            "(control flow not yet supported)\n",
            (int)name.size, name.str);
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle sblk = MLIR_GetRegionBlock(sreg, 0);

    size_t n_params = pt.size / 2;
    size_t n_decls  = lt.size / 2;
    size_t n_locals = n_params + n_decls;

    // Decode and validate all local valtypes up front (integer-only here).
    uint8_t *local_vt = (uint8_t *)malloc((n_locals ? n_locals : 1));
    for (size_t i = 0; i < n_params; i++) local_vt[i] = type_byte_at(pt, i);
    for (size_t i = 0; i < n_decls; i++)  local_vt[n_params + i] = type_byte_at(lt, i);
    for (size_t i = 0; i < n_locals; i++) {
        if (!vt_is_int(local_vt[i])) {
            fprintf(stderr,
                "wasmssa->llvm: func '%.*s' has a non-integer local "
                "(floats not yet supported)\n",
                (int)name.size, name.str);
            free(local_vt);
            return MLIR_INVALID_HANDLE;
        }
    }

    MLIR_RegionHandle dreg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, dreg, entry);

    VMap vmap = {0};
    FLower L = {0};
    L.ctx = ctx;
    L.blk = entry;
    L.vmap = &vmap;
    L.n_locals = n_locals;
    L.local_vt = local_vt;
    L.local_ptr = (MLIR_ValueHandle *)malloc(
        (n_locals ? n_locals : 1) * sizeof(MLIR_ValueHandle));

    // Function parameters become block args.
    MLIR_ValueHandle *param_args = (MLIR_ValueHandle *)malloc(
        (n_params ? n_params : 1) * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < n_params; i++) {
        MLIR_TypeHandle ty = vt_to_llvm(ctx, local_vt[i]);
        MLIR_ValueHandle da = MLIR_CreateValueBlockArg(ctx, (string){0},
            (uint32_t)i, ty, MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, entry, da);
        param_args[i] = da;
    }

    // One i64 const=1 to serve as the element count for every alloca.
    MLIR_TypeHandle i64ty = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_AttributeHandle cnt_va = MLIR_CreateAttributeInteger(
        ctx, str_from_cstr_view((char *)"value"), 1, i64ty);
    MLIR_TypeHandle cnt_rt[1] = { i64ty };
    MLIR_ValueHandle cnt_r[1] = { mk_res(ctx, i64ty) };
    MLIR_OpHandle cnt_op = build_op(ctx, OP_TYPE_LLVM_MLIR_CONSTANT,
        &cnt_va, 1, cnt_rt, 1, cnt_r, NULL, 0);
    MLIR_AppendBlockOp(ctx, entry, cnt_op);

    // One alloca per local index (params + declared locals).
    MLIR_TypeHandle ptr_ty = MLIR_CreateTypeLLVMPointer(ctx);
    for (size_t i = 0; i < n_locals; i++) {
        MLIR_TypeHandle ety = vt_to_llvm(ctx, local_vt[i]);
        MLIR_AttributeHandle a[1] = { attr_ty(ctx, "elem_type", ety) };
        MLIR_TypeHandle rt[1] = { ptr_ty };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ptr_ty) };
        MLIR_ValueHandle ops[1] = { cnt_r[0] };
        MLIR_OpHandle aop = build_op(ctx, OP_TYPE_LLVM_ALLOCA,
            a, 1, rt, 1, r, ops, 1);
        MLIR_AppendBlockOp(ctx, entry, aop);
        L.local_ptr[i] = r[0];
    }

    // Store incoming params into their local allocas.
    for (size_t i = 0; i < n_params; i++) {
        MLIR_ValueHandle ops[2] = { param_args[i], L.local_ptr[i] };
        MLIR_OpHandle st = build_op(ctx, OP_TYPE_LLVM_STORE,
            NULL, 0, NULL, 0, NULL, ops, 2);
        MLIR_AppendBlockOp(ctx, entry, st);
    }

    // Lower the body.
    bool ok = true;
    size_t n_ops = MLIR_GetBlockNumOps(sblk);
    for (size_t i = 0; i < n_ops; i++) {
        if (!lower_op(&L, MLIR_GetBlockOp(sblk, i))) { ok = false; break; }
    }

    free(vmap.src);
    free(vmap.dst);
    free(local_vt);
    free(L.local_ptr);
    free(param_args);
    if (!ok) return MLIR_INVALID_HANDLE;

    MLIR_AttributeHandle attrs[1] = {
        attr_s(ctx, "sym_name", name.str, name.size)
    };
    MLIR_RegionHandle regs[1] = { dreg };
    return MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

MLIR_OpHandle mlir_wasmssa_to_llvm(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ssa_module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  out_body = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);

    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(op);
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC) {
            // Imports resolve to native runtime / libSystem stubs at the
            // backend; no declaration op is required there. Milestone 1 only
            // supports proc_exit (-> _exit); reject anything else so the
            // failure is explicit rather than a link error later.
            string nm = at_s(op, "sym_name");
            if (!(nm.size == 9 && memcmp(nm.str, "proc_exit", 9) == 0)) {
                fprintf(stderr,
                    "wasmssa->llvm: import '%.*s' not yet supported "
                    "(milestone 1 handles proc_exit only)\n",
                    (int)nm.size, nm.str);
                return MLIR_INVALID_HANDLE;
            }
            continue;
        }
        if (t == OP_TYPE_WASMSSA_FUNC) {
            MLIR_OpHandle fn = lower_func(ctx, op);
            if (fn == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
            MLIR_AppendBlockOp(ctx, out_body, fn);
            continue;
        }
        // import_global / data segments / tables: deferred to later milestones.
        fprintf(stderr,
            "wasmssa->llvm: module-level op '%.*s' not yet supported\n",
            (int)MLIR_GetOpName(op).size, MLIR_GetOpName(op).str);
        return MLIR_INVALID_HANDLE;
    }

    MLIR_RegionHandle regs[1] = { out_region };
    return MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}
