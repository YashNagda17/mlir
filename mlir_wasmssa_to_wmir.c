// wasmssa -> wmir lowering. See mlir_wasmssa_to_wmir.h for the public
// API and rationale.
//
// First-light implementation: walks each wasmssa.func; routes the
// three supported op kinds (const, return, func itself) into wmir
// equivalents. Drops wasmssa.import_func ops entirely — the
// macho_exit smoke test imports malloc/free but the user-visible
// function (`__original_main`) never calls them, so they have no
// runtime effect on the emitted Mach-O. (The svc-based exit path
// also has no need for any libc symbols.) Subsequent slices will
// promote imports into proper wmir.import_func / call shims when we
// add op kinds that actually call into them.

#include "mlir_wasmssa_to_wmir.h"

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
#include "mlir_op_names.h"

// =============================================================================
// Wasm value-type byte constants (mirrors the WT_* table used elsewhere
// in the pipeline).
// =============================================================================
#define WT_I32 0x7f
#define WT_I64 0x7e
#define WT_F32 0x7d
#define WT_F64 0x7c

// =============================================================================
// Attribute / op helpers.
// =============================================================================
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

static MLIR_OpHandle build_op_simple(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_TypeHandle *result_types, size_t n_results,
                                     MLIR_ValueHandle *results,
                                     MLIR_ValueHandle *operands, size_t n_operands) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, result_types, n_results, results, n_results,
        operands, n_operands, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
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

// Translate a wasm value-type byte to the matching MLIR integer / float
// type used inside wmir. wmir collapses wasm's value-type kingdom into
// the IR's own iN/fN types so subsequent passes can reason about width
// directly instead of squinting at a `valtype` attribute.
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
        case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
        case WT_F32: return MLIR_CreateTypeFloat(ctx, 32, false);
        case WT_F64: return MLIR_CreateTypeFloat(ctx, 64, false);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

// =============================================================================
// SSA value remapping: each wasmssa value maps to its replacement in
// the wmir module. Linear scan; functions are small.
// =============================================================================
typedef struct {
    MLIR_ValueHandle *src;
    MLIR_ValueHandle *dst;
    size_t            n, cap;
} VMap;

static void vmap_set(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->src = (MLIR_ValueHandle *)realloc(m->src, m->cap * sizeof(*m->src));
        m->dst = (MLIR_ValueHandle *)realloc(m->dst, m->cap * sizeof(*m->dst));
    }
    m->src[m->n] = k;
    m->dst[m->n] = v;
    m->n++;
}
static int vmap_get(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->src[i] == k) { *out = m->dst[i]; return 1; }
    }
    return 0;
}

// =============================================================================
// Per-op lowering inside a function body. Each handler appends one (or
// zero) wmir ops to `dst_blk` and updates `vmap` with the result.
// Returns false on an unsupported op.
// =============================================================================
static bool lower_wasmssa_op(MLIR_Context *ctx, MLIR_OpHandle src_op,
                             MLIR_BlockHandle dst_blk, VMap *vmap) {
    MLIR_OpType t = MLIR_GetOpType(src_op);
    switch (t) {

    case OP_TYPE_WASMSSA_CONST: {
        // wasmssa.const has a `value` integer attribute and a `valtype`
        // byte attribute. wmir.const carries the same value attribute but
        // its result type is the real iN type. Width is implied by the
        // result type, so no `valtype` attribute is needed on wmir.const.
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        int64_t v  = at_i(src_op, "value");
        MLIR_AttributeHandle attrs[1];
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        // Carry the constant as an integer attribute typed to match the
        // result, so the parser round-trips faithfully. wmir floats will
        // need a separate handler later when we add f32/f64.
        attrs[0] = MLIR_CreateAttributeInteger(ctx, str_lit("value"), v, res_ty[0]);
        MLIR_ValueHandle res[1];
        res[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
                                          res_ty[0], (string){0},
                                          MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_CONST,
            attrs, 1, res_ty, 1, res, NULL, 0);
        MLIR_AppendBlockOp(ctx, dst_blk, out);
        vmap_set(vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_RETURN: {
        // Variadic, matches the func's result_types. For the first-light
        // slice the host function returns at most one value.
        size_t n_ops = MLIR_GetOpNumOperands(src_op);
        MLIR_ValueHandle operands[8];
        if (n_ops > 8) {
            fprintf(stderr, "wmir: wasmssa.return with >8 results unsupported\n");
            return false;
        }
        for (size_t k = 0; k < n_ops; k++) {
            MLIR_ValueHandle sv = MLIR_GetOpOperand(src_op, k);
            if (!vmap_get(vmap, sv, &operands[k])) {
                fprintf(stderr, "wmir: unbound operand on wasmssa.return\n");
                return false;
            }
        }
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_RETURN,
            NULL, 0, NULL, 0, NULL, operands, n_ops);
        MLIR_AppendBlockOp(ctx, dst_blk, out);
        return true;
    }

    default: {
        string nm = MLIR_GetOpName(src_op);
        fprintf(stderr,
            "wmir lowering: unsupported wasmssa op '%.*s' (kind=%d)\n",
            (int)nm.size, nm.str, (int)t);
        return false;
    }
    }
}

// =============================================================================
// Lower one wasmssa.func to one wmir.func.
// =============================================================================
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src) {
    string name      = at_s(src, "sym_name");
    bool   exported  = at_b(src, "exported");
    string pt        = at_s(src, "param_types");
    string rt        = at_s(src, "result_types");

    MLIR_BlockHandle dst_blk = MLIR_CreateBlock(ctx);

    // wmir doesn't model the wasm value-type kingdom in its op
    // attributes — the param/result types live as MLIR types on the
    // function signature. We still preserve the raw `param_types` /
    // `result_types` attribute strings on wmir.func so downstream
    // passes can recover ABI-relevant info without re-deriving it.
    // The first-light slice (`macho_exit`) has zero params, so we
    // don't actually need block arguments yet; we still create the
    // dst block empty so the structure matches the production layout.
    if (pt.size > 0) {
        // Will be filled in by a later slice once we have a test that
        // exercises function parameters through the wmir backend.
        fprintf(stderr,
            "wmir lowering: function '%.*s' has parameters — not yet supported\n",
            (int)name.size, name.str);
        return MLIR_INVALID_HANDLE;
    }

    if (MLIR_GetOpNumRegions(src) < 1) {
        fprintf(stderr, "wmir lowering: wasmssa.func has no region\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(src, 0);
    if (MLIR_GetRegionNumBlocks(src_region) < 1) {
        fprintf(stderr, "wmir lowering: wasmssa.func has no entry block\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, 0);

    VMap vmap = {0};
    size_t n_ops = MLIR_GetBlockNumOps(src_blk);
    bool ok = true;
    for (size_t i = 0; i < n_ops; i++) {
        MLIR_OpHandle bo = MLIR_GetBlockOp(src_blk, i);
        if (!lower_wasmssa_op(ctx, bo, dst_blk, &vmap)) { ok = false; break; }
    }
    free(vmap.src); free(vmap.dst);
    if (!ok) return MLIR_INVALID_HANDLE;

    MLIR_AttributeHandle attrs[6];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",     name.str, name.size);
    attrs[na++] = attr_s(ctx, "param_types",  pt.str,   pt.size);
    attrs[na++] = attr_s(ctx, "result_types", rt.str,   rt.size);
    attrs[na++] = attr_b(ctx, "exported",     exported);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, dst_blk);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_WMIR_FUNC,
        op_type_to_string(OP_TYPE_WMIR_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Top-level: walk the wasmssa module body, build a fresh wmir module.
// =============================================================================
MLIR_OpHandle mlir_wasmssa_to_wmir(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    if (!ssa_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(ssa_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ssa_module, 0);
    if (MLIR_GetRegionNumBlocks(mr) < 1) return MLIR_INVALID_HANDLE;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    size_t n_top = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC ||
            t == OP_TYPE_WASMSSA_IMPORT_GLOBAL) {
            // Imports never reach the AArch64 backend in the first-light
            // slice (macho_exit doesn't call malloc/free or touch globals).
            // Dropping them here keeps the wmir module noise-free; a
            // later slice will reintroduce wmir.import_func when a test
            // needs it.
            continue;
        }
        if (t == OP_TYPE_WASMSSA_FUNC) {
            MLIR_OpHandle out_op = lower_func(ctx, top);
            if (!out_op) return MLIR_INVALID_HANDLE;
            MLIR_AppendBlockOp(ctx, out_body, out_op);
            continue;
        }
        string nm = MLIR_GetOpName(top);
        fprintf(stderr,
            "wmir lowering: unexpected top-level op '%.*s'\n",
            (int)nm.size, nm.str);
        return MLIR_INVALID_HANDLE;
    }
    return out_module;
}
