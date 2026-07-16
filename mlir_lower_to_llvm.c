// Native lowering pass: rewrites a `builtin.module` from the high-level
// dialects (func, arith, cf, scf, vector, memref) down to the `llvm`
// dialect. Structured control flow (`scf.if`, `scf.while`,
// `scf.index_switch`) becomes flat `llvm.br` / `llvm.cond_br` CFG here for
// machine backends (aarch64, x64). The wasm pipeline keeps `scf.*` in place
// (`MLIR_LowerToLLVMDialectForWasm`); wasmssa lowers structured CF directly.
// Uses ONLY the public mlir_api.h surface — no upstream MLIR types — so
// this same translation unit is linked into both the upstream-backed and
// the native-backed builds.
//
// Strategy: walk every region top-down. For each op encountered, if it
// has a known lowering, build the replacement LLVM-dialect op(s),
// redirect uses of the old results, and erase the old op. Ops that are
// already in the LLVM dialect (or have no lowering needed) are left
// alone.
//
// This file is C, linked into both the upstream-backed and native-backed
// builds (no upstream MLIR headers), so the same translation unit
// supplies the agnostic `MLIR_LowerToLLVMDialect` in both binaries.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

static bool name_eq(string s, const char *cstr) {
    size_t n = 0;
    while (cstr[n]) n++;
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static MLIR_TypeHandle ty_i64(MLIR_Context *ctx) {
    return MLIR_CreateTypeInteger(ctx, 64, false);
}
static MLIR_TypeHandle ty_llvm_void(MLIR_Context *ctx) {
    return MLIR_CreateTypeLLVMVoid(ctx);
}

// Allocate a fresh value handle (op-result) that becomes the result of
// some op we are about to create.
static MLIR_ValueHandle make_result_value(MLIR_Context *ctx,
                                          MLIR_TypeHandle type,
                                          MLIR_LocationHandle loc) {
    return MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, type,
                                    str_lit(""), loc);
}

// Convenience for ops that take no successors and no special blocks. Use
// for everything except branch-like ops.
static MLIR_OpHandle create_simple_op(
        MLIR_Context *ctx, MLIR_OpType type, string opname,
        MLIR_AttributeHandle *attrs, size_t n_attrs,
        MLIR_TypeHandle *result_types, size_t n_result_types,
        MLIR_ValueHandle *results, size_t n_results,
        MLIR_ValueHandle *operands, size_t n_operands,
        MLIR_RegionHandle *regions, size_t n_regions,
        MLIR_LocationHandle loc) {
    return MLIR_CreateOp(ctx, type, opname,
                         attrs, n_attrs,
                         result_types, n_result_types,
                         results, n_results,
                         operands, n_operands,
                         regions, n_regions,
                         loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

// -----------------------------------------------------------------------------
// State shared across the whole lowering pass
// -----------------------------------------------------------------------------

typedef struct LowerState {
    MLIR_Context *ctx;
    MLIR_OpHandle module;
    MLIR_BlockHandle module_body;
    bool keep_scf;
} LowerState;

// -----------------------------------------------------------------------------
// Per-op lowering helpers
// -----------------------------------------------------------------------------

// Lower an `arith.constant value : T` to `llvm.mlir.constant(value : T) : T`.
// Same operand/result shape — just re-create with the right name and the
// "value" attribute kept as-is (the original op's attribute is already
// named "value", but its name in the user-attr table on the upstream
// backend is what matters).
static bool lower_arith_constant(LowerState *st, MLIR_OpHandle op,
                                 MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumResults(op) != 1) return false;
    MLIR_ValueHandle old_res = MLIR_GetOpResult(op, 0);
    MLIR_TypeHandle ty = MLIR_GetValueType(old_res);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);

    // Find the "value" attribute on the source op.
    size_t na = MLIR_GetOpNumAttributes(op);
    MLIR_AttributeHandle val_attr = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (name_eq(MLIR_GetAttributeName(a), "value")) { val_attr = a; break; }
    }
    if (val_attr == MLIR_INVALID_HANDLE) return false;

    MLIR_ValueHandle new_res = make_result_value(st->ctx, ty, loc);
    MLIR_AttributeHandle attrs[1] = { val_attr };
    MLIR_TypeHandle rts[1] = { ty };
    MLIR_ValueHandle results[1] = { new_res };
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_MLIR_CONSTANT, str_lit("llvm.mlir.constant"),
        attrs, 1, rts, 1, results, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    MLIR_ReplaceAllUsesOfValue(st->ctx, old_res, new_res);
    return true;
}

// Lower `func.return [%v ...]` to `llvm.return [%v ...]` (1:1).
static bool lower_func_return(LowerState *st, MLIR_OpHandle op,
                              MLIR_BlockHandle parent, size_t pos) {
    size_t no = MLIR_GetOpNumOperands(op);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_ValueHandle *operands = NULL;
    MLIR_ValueHandle stk[4];
    if (no <= 4) operands = stk;
    else operands = (MLIR_ValueHandle *)arena_alloc(
        MLIR_GetArenaAllocator(st->ctx), no * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_RETURN, str_lit("llvm.return"),
        NULL, 0, NULL, 0, NULL, 0, operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    return true;
}

// Lower a `func.func` to an `llvm.func`, taking ownership of its body
// region so block args, ops, and edges are preserved verbatim.
static bool lower_func_func(LowerState *st, MLIR_OpHandle op,
                            MLIR_BlockHandle parent, size_t pos) {
    // Pull the body region off (may be empty for declarations).
    size_t nr = MLIR_GetOpNumRegions(op);
    MLIR_RegionHandle body = (nr > 0)
        ? MLIR_TakeOpRegion(st->ctx, op, 0)
        : MLIR_CreateRegion(st->ctx);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);

    // Copy attributes that matter: sym_name, function_type, sym_visibility,
    // arg_attrs, res_attrs.
    //
    // We also keep the `wasm.import_module`/`wasm.import_name`/
    // `wasm.export_name` attributes (used by the native LLVM->wasmssa
    // backend in `mlir_llvm_to_wasmssa.c`), and additionally synthesize
    // an upstream-MLIR `passthrough` array attribute carrying the
    // standard LLVM IR function-attribute strings
    // (`wasm-import-module`, `wasm-import-name`, `wasm-export-name`).
    //
    // Upstream MLIR's `translateModuleToLLVMIR` ignores unknown MLIR
    // attributes, so without this the imports end up in module `env`
    // instead of `wasi_snapshot_preview1` when going through the
    // upstream WebAssembly backend. Clang attaches these same string
    // attributes when compiling `__attribute__((import_module(...)))`.
    size_t na = MLIR_GetOpNumAttributes(op);
    MLIR_AttributeHandle attrs_buf[18];
    size_t n_attrs = 0;
    string imp_module = {0}, imp_name = {0}, exp_name = {0};
    for (size_t i = 0; i < na && n_attrs < 16; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        string an = MLIR_GetAttributeName(a);
        if (name_eq(an, "sym_name") ||
            name_eq(an, "sym_visibility") ||
            name_eq(an, "arg_attrs") ||
            name_eq(an, "res_attrs") ||
            name_eq(an, "llvm.linkage")) {
            attrs_buf[n_attrs++] = a;
        } else if (name_eq(an, "wasm.import_module")) {
            attrs_buf[n_attrs++] = a;
            imp_module = MLIR_GetAttributeString(a);
        } else if (name_eq(an, "wasm.import_name")) {
            attrs_buf[n_attrs++] = a;
            imp_name = MLIR_GetAttributeString(a);
        } else if (name_eq(an, "wasm.export_name")) {
            attrs_buf[n_attrs++] = a;
            exp_name = MLIR_GetAttributeString(a);
        } else if (name_eq(an, "function_type")) {
            // Convert FunctionType -> LLVMFunctionType (which carries
            // is_var_arg and uses LLVM void instead of zero-results).
            MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(a);
            size_t ni = MLIR_GetTypeFunctionNumInputs(ft);
            size_t nr_ = MLIR_GetTypeFunctionNumResults(ft);
            MLIR_TypeHandle ins_stk[16];
            MLIR_TypeHandle *ins = ins_stk;
            if (ni > 16) ins = (MLIR_TypeHandle *)arena_alloc(
                MLIR_GetArenaAllocator(st->ctx),
                ni * sizeof(MLIR_TypeHandle));
            for (size_t k = 0; k < ni; k++) ins[k] = MLIR_GetTypeFunctionInput(ft, k);
            MLIR_TypeHandle ret_ty = (nr_ == 0)
                ? ty_llvm_void(st->ctx)
                : MLIR_GetTypeFunctionResult(ft, 0);
            MLIR_TypeHandle llvmft = MLIR_CreateTypeLLVMFunction(
                st->ctx, ret_ty, ins, ni, false);
            attrs_buf[n_attrs++] = MLIR_CreateAttributeType(
                st->ctx, str_lit("function_type"), llvmft);
        }
    }

    // Synthesize `passthrough` from the wasm.* attributes so the upstream
    // MLIR -> LLVM IR translator emits the corresponding LLVM IR function
    // attribute strings on the declaration. LLVM's WebAssembly backend
    // reads these to decide the import-module / import-name / export-name
    // of each function.
    if (imp_module.size > 0 || imp_name.size > 0 || exp_name.size > 0) {
        MLIR_AttributeHandle pt_elems[3];
        size_t n_pt = 0;
        string keys[3] = {
            str_lit("wasm-import-module"),
            str_lit("wasm-import-name"),
            str_lit("wasm-export-name"),
        };
        string vals[3] = { imp_module, imp_name, exp_name };
        for (size_t i = 0; i < 3; i++) {
            if (vals[i].size == 0) continue;
            MLIR_AttributeHandle pair[2] = {
                MLIR_CreateAttributeString(st->ctx, str_lit(""), keys[i]),
                MLIR_CreateAttributeString(st->ctx, str_lit(""), vals[i]),
            };
            pt_elems[n_pt++] = MLIR_CreateAttributeArray(
                st->ctx, str_lit(""), pair, 2);
        }
        if (n_pt > 0 && n_attrs < 18) {
            attrs_buf[n_attrs++] = MLIR_CreateAttributeArray(
                st->ctx, str_lit("passthrough"), pt_elems, n_pt);
        }
    }

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
        attrs_buf, n_attrs, NULL, 0, NULL, 0, NULL, 0,
        &body, 1, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    return true;
}

// Lower `func.call @callee(args) : (...) -> (...)` to `llvm.call`.
static bool lower_func_call(LowerState *st, MLIR_OpHandle op,
                            MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t nr = MLIR_GetOpNumResults(op);

    MLIR_ValueHandle stk[8];
    MLIR_ValueHandle *operands = stk;
    if (no > 8) operands = (MLIR_ValueHandle *)arena_alloc(
        MLIR_GetArenaAllocator(st->ctx), no * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_TypeHandle rts_stk[2];
    MLIR_ValueHandle res_stk[2];
    MLIR_TypeHandle *rts = (nr <= 2) ? rts_stk
        : (MLIR_TypeHandle *)arena_alloc(MLIR_GetArenaAllocator(st->ctx),
                                         nr * sizeof(MLIR_TypeHandle));
    MLIR_ValueHandle *results = (nr <= 2) ? res_stk
        : (MLIR_ValueHandle *)arena_alloc(MLIR_GetArenaAllocator(st->ctx),
                                          nr * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        rts[i] = MLIR_GetValueType(old);
        results[i] = make_result_value(st->ctx, rts[i], loc);
    }

    // Copy `callee` attribute through.
    size_t na = MLIR_GetOpNumAttributes(op);
    MLIR_AttributeHandle attrs_buf[2];
    size_t n_attrs = 0;
    for (size_t i = 0; i < na && n_attrs < 2; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (name_eq(MLIR_GetAttributeName(a), "callee")) {
            attrs_buf[n_attrs++] = a;
        }
    }

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
        attrs_buf, n_attrs, rts, nr, results, nr,
        operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, results[i]);
    }
    return true;
}

// Generic "rename op": build a new op with `new_name` that copies all
// operands, results, and attributes from `op` 1:1. Used for the family
// of arith→llvm conversions where the semantics are identical and only
// the op name changes (arith.addi→llvm.add, arith.cmpi→llvm.icmp, ...).
static bool lower_rename(LowerState *st, MLIR_OpHandle op,
                         MLIR_BlockHandle parent, size_t pos,
                         string new_name, MLIR_OpType new_type) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t nr = MLIR_GetOpNumResults(op);
    size_t na = MLIR_GetOpNumAttributes(op);

    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_ValueHandle *operands = no ? (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_TypeHandle *rts = nr ? (MLIR_TypeHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_TypeHandle)) : NULL;
    MLIR_ValueHandle *results = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        rts[i] = MLIR_GetValueType(old);
        results[i] = make_result_value(st->ctx, rts[i], loc);
    }

    MLIR_AttributeHandle *attrs = na ? (MLIR_AttributeHandle *)arena_alloc(
        alloc, na * sizeof(MLIR_AttributeHandle)) : NULL;
    for (size_t i = 0; i < na; i++) attrs[i] = MLIR_GetOpAttribute(op, i);

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, new_type, new_name,
        attrs, na, rts, nr, results, nr,
        operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, results[i]);
    }
    return true;
}

// Lower `cf.br` / `cf.cond_br` to `llvm.br` / `llvm.cond_br`. These ops
// carry block successors plus per-successor operand lists; we have to
// use the successor-aware op constructor.
static bool lower_cf_branch(LowerState *st, MLIR_OpHandle op,
                            MLIR_BlockHandle parent, size_t pos,
                            string new_name) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t na = MLIR_GetOpNumAttributes(op);
    size_t ns = MLIR_GetOpNumSuccessors(op);

    MLIR_ValueHandle *operands = no ? (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_AttributeHandle *attrs = na ? (MLIR_AttributeHandle *)arena_alloc(
        alloc, na * sizeof(MLIR_AttributeHandle)) : NULL;
    for (size_t i = 0; i < na; i++) attrs[i] = MLIR_GetOpAttribute(op, i);

    MLIR_BlockHandle *succs = ns ? (MLIR_BlockHandle *)arena_alloc(
        alloc, ns * sizeof(MLIR_BlockHandle)) : NULL;
    MLIR_ValueHandle **succ_ops = ns ? (MLIR_ValueHandle **)arena_alloc(
        alloc, ns * sizeof(MLIR_ValueHandle *)) : NULL;
    size_t *n_succ_ops = ns ? (size_t *)arena_alloc(
        alloc, ns * sizeof(size_t)) : NULL;
    for (size_t s = 0; s < ns; s++) {
        succs[s] = MLIR_GetOpSuccessor(op, s);
        size_t k = MLIR_GetOpNumSuccessorOperands(op, s);
        n_succ_ops[s] = k;
        succ_ops[s] = k ? (MLIR_ValueHandle *)arena_alloc(
            alloc, k * sizeof(MLIR_ValueHandle)) : NULL;
        for (size_t j = 0; j < k; j++) {
            succ_ops[s][j] = MLIR_GetOpSuccessorOperand(op, s, j);
        }
    }

    MLIR_OpHandle nop = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, new_name,
        attrs, na, NULL, 0, NULL, 0,
        operands, no, NULL, 0,
        succs, ns, succ_ops, n_succ_ops,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    return true;
}

// Lower `func.constant @sym : T` to `llvm.mlir.addressof @sym : !llvm.ptr`.
static bool lower_func_constant(LowerState *st, MLIR_OpHandle op,
                                MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    if (MLIR_GetOpNumResults(op) != 1) return false;

    // Find the symbol-ref `value` attribute and extract its name as a
    // C string.
    size_t na = MLIR_GetOpNumAttributes(op);
    string sym_name = {0};
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (name_eq(MLIR_GetAttributeName(a), "value")) {
            string s = MLIR_GetAttributeAsString(st->ctx, a);
            // Format is "@name" (FlatSymbolRefAttr) — strip leading '@'.
            if (s.size > 0 && s.str[0] == '@') {
                sym_name.str = s.str + 1;
                sym_name.size = s.size - 1;
            } else {
                sym_name = s;
            }
            break;
        }
    }
    if (sym_name.size == 0) return false;

    MLIR_TypeHandle ptr_ty = MLIR_CreateTypeLLVMPointer(st->ctx);
    MLIR_ValueHandle new_res = make_result_value(st->ctx, ptr_ty, loc);
    MLIR_TypeHandle rts[1] = { ptr_ty };
    MLIR_ValueHandle results[1] = { new_res };
    MLIR_AttributeHandle attrs[1];
    attrs[0] = MLIR_CreateAttributeSymbolRef(
        st->ctx, str_lit("global_name"), sym_name);
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.mlir.addressof"),
        attrs, 1, rts, 1, results, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    MLIR_ReplaceAllUsesOfValue(st->ctx, MLIR_GetOpResult(op, 0), new_res);
    return true;
}

// Lower `builtin.unrealized_conversion_cast %x : T1 to T2`. After our
// other lowerings produce LLVM-dialect ops at both endpoints, all
// remaining casts are no-op type punning (typically !llvm.ptr↔function
// type around indirect calls). Just RAUW the result with the operand
// and let the downstream consumer use the operand directly.
static bool lower_unrealized_cast(LowerState *st, MLIR_OpHandle op,
                                  MLIR_BlockHandle parent, size_t pos) {
    (void)st; (void)parent; (void)pos;
    if (MLIR_GetOpNumOperands(op) != 1 || MLIR_GetOpNumResults(op) != 1) {
        return false;
    }
    MLIR_ReplaceAllUsesOfValue(st->ctx,
                               MLIR_GetOpResult(op, 0),
                               MLIR_GetOpOperand(op, 0));
    return true;
}

// Lower `func.call_indirect %fn(args)` to `llvm.call %fn(args)`. The
// callee is operand[0] (a !llvm.ptr after prior lowerings); remaining
// operands are the arguments.
static bool lower_func_call_indirect(LowerState *st, MLIR_OpHandle op,
                                     MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t nr = MLIR_GetOpNumResults(op);
    if (no < 1) return false;

    MLIR_ValueHandle *operands = (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_TypeHandle *rts = nr ? (MLIR_TypeHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_TypeHandle)) : NULL;
    MLIR_ValueHandle *results = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        rts[i] = MLIR_GetValueType(old);
        results[i] = make_result_value(st->ctx, rts[i], loc);
    }

    // Build a `var_callee_type` LLVMFunctionType attribute describing
    // the indirect call's signature (return + arg types).
    MLIR_TypeHandle ret_ty = (nr == 0) ? ty_llvm_void(st->ctx) : rts[0];
    size_t n_args = no - 1;
    MLIR_TypeHandle *arg_tys = n_args ? (MLIR_TypeHandle *)arena_alloc(
        alloc, n_args * sizeof(MLIR_TypeHandle)) : NULL;
    for (size_t i = 0; i < n_args; i++) {
        arg_tys[i] = MLIR_GetValueType(operands[i + 1]);
    }
    MLIR_TypeHandle llvmft = MLIR_CreateTypeLLVMFunction(
        st->ctx, ret_ty, arg_tys, n_args, false);
    MLIR_AttributeHandle attrs[1];
    attrs[0] = MLIR_CreateAttributeType(
        st->ctx, str_lit("var_callee_type"), llvmft);

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
        attrs, 1, rts, nr, results, nr,
        operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    for (size_t i = 0; i < nr; i++) {
        MLIR_ReplaceAllUsesOfValue(st->ctx, MLIR_GetOpResult(op, i), results[i]);
    }
    return true;
}

// Replace the trailing `scf.yield %vs` of `blk` with `llvm.br ^cont(%vs)`.
static void rewrite_yield_to_br(LowerState *st, MLIR_BlockHandle blk,
                                MLIR_BlockHandle cont) {
    size_t n = MLIR_GetBlockNumOps(blk);
    if (n == 0) return;
    MLIR_OpHandle term = MLIR_GetBlockOp(blk, n - 1);
    if (!name_eq(MLIR_GetOpName(term), "scf.yield")) return;
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    size_t no = MLIR_GetOpNumOperands(term);
    MLIR_ValueHandle *operands = no ? (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(term, i);
    MLIR_BlockHandle succs[1] = { cont };
    MLIR_ValueHandle *succ_ops_arr[1] = { operands };
    size_t n_succ_ops_arr[1] = { no };
    MLIR_LocationHandle term_loc = MLIR_GetOpLocation(term);
    MLIR_OpHandle br = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.br"),
        NULL, 0, NULL, 0, NULL, 0,
        NULL, 0, NULL, 0,
        succs, 1, succ_ops_arr, n_succ_ops_arr,
        term_loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(st->ctx, blk, br);
    MLIR_EraseOp(st->ctx, term);
}

// Detach `op`'s regions into `out[0..n-1]`. On failure, restores any
// regions already taken.
static bool take_op_regions(MLIR_Context *ctx, MLIR_OpHandle op,
                            MLIR_RegionHandle *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = MLIR_TakeOpRegion(ctx, op, i);
        if (out[i] == MLIR_INVALID_HANDLE) {
            for (size_t j = 0; j < i; j++)
                MLIR_SetOpRegion(ctx, op, j, out[j]);
            return false;
        }
    }
    return true;
}

static void restore_op_regions(MLIR_Context *ctx, MLIR_OpHandle op,
                               const MLIR_RegionHandle *regs, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (regs[i] != MLIR_INVALID_HANDLE)
            MLIR_SetOpRegion(ctx, op, i, regs[i]);
    }
}

// Lower `scf.if %cond -> (Ts) { thenRegion } else { elseRegion }` to a
// CFG: the parent block ends with `llvm.cond_br %cond, ^then, ^else`;
// then/else regions are inlined into the parent region; their terminating
// `scf.yield %vs` becomes `llvm.br ^cont(%vs)`; `cont` is a fresh block
// holding everything that came after the scf.if in the parent block,
// with block args matching the scf.if's results (used to RAUW them).
//
// Limitations: only handles single-block regions (which is what tinyC's
// `if`/`else` produce — scf.if regions with no nested control flow).
static bool lower_scf_if(LowerState *st, MLIR_OpHandle op,
                         MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_RegionHandle parent_region = MLIR_GetBlockParentRegion(parent);
    if (parent_region == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumOperands(op) < 1) return false;

    MLIR_ValueHandle cond = MLIR_GetOpOperand(op, 0);
    size_t nr = MLIR_GetOpNumResults(op);
    size_t n_regs = MLIR_GetOpNumRegions(op);

    MLIR_RegionHandle regs[2];
    if (n_regs > 2 || !take_op_regions(st->ctx, op, regs, n_regs))
        return false;
    if (MLIR_GetRegionNumBlocks(regs[0]) < 1) {
        restore_op_regions(st->ctx, op, regs, n_regs);
        return false;
    }

    MLIR_RegionHandle then_reg = regs[0];
    MLIR_RegionHandle else_reg = n_regs >= 2 ? regs[1] : MLIR_INVALID_HANDLE;
    bool has_else = n_regs >= 2 && MLIR_GetRegionNumBlocks(regs[1]) >= 1;

    // Entry blocks are first block of each region; the branch target.
    MLIR_BlockHandle then_blk = MLIR_GetRegionBlock(then_reg, 0);
    MLIR_BlockHandle else_blk = has_else ? MLIR_GetRegionBlock(else_reg, 0)
                                         : MLIR_INVALID_HANDLE;

    // Continuation block with one block-arg per scf.if result.
    MLIR_BlockHandle cont = MLIR_CreateBlock(st->ctx);
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_ValueHandle *cont_args = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_TypeHandle ty = MLIR_GetValueType(old);
        cont_args[i] = MLIR_CreateValueBlockArg(
            st->ctx, str_lit(""), (uint32_t)i, ty, loc);
        MLIR_AppendBlockArg(st->ctx, cont, cont_args[i]);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, cont_args[i]);
    }

    // Move all ops AFTER the scf.if from parent to cont.
    while (MLIR_GetBlockNumOps(parent) > pos + 1) {
        MLIR_OpHandle tail = MLIR_GetBlockOp(parent, pos + 1);
        MLIR_MoveOpToBlockEnd(st->ctx, tail, cont);
    }

    // Rewrite scf.yield in any block of the regions to llvm.br ^cont(args).
    // Inner control-flow lowering may have produced multi-block regions,
    // but scf.yield only ever appears as a block terminator, so walking
    // the blocks of the moved region and matching on the terminator is
    // sufficient.
    size_t nthen = MLIR_GetRegionNumBlocks(then_reg);
    for (size_t bi = 0; bi < nthen; bi++) {
        rewrite_yield_to_br(st, MLIR_GetRegionBlock(then_reg, bi), cont);
    }
    if (has_else) {
        size_t nelse = MLIR_GetRegionNumBlocks(else_reg);
        for (size_t bi = 0; bi < nelse; bi++) {
            rewrite_yield_to_br(st, MLIR_GetRegionBlock(else_reg, bi), cont);
        }
    }

    // Move all blocks of each region into parent_region (in original order).
    // Each MoveBlockToRegionEnd removes from src region, so re-fetch index 0.
    while (MLIR_GetRegionNumBlocks(then_reg) > 0) {
        MLIR_MoveBlockToRegionEnd(st->ctx,
            MLIR_GetRegionBlock(then_reg, 0), parent_region);
    }
    if (has_else) {
        while (MLIR_GetRegionNumBlocks(else_reg) > 0) {
            MLIR_MoveBlockToRegionEnd(st->ctx,
                MLIR_GetRegionBlock(else_reg, 0), parent_region);
        }
    }
    MLIR_MoveBlockToRegionEnd(st->ctx, cont, parent_region);

    MLIR_BlockHandle false_target = has_else ? else_blk : cont;
    MLIR_BlockHandle succs[2] = { then_blk, false_target };
    MLIR_ValueHandle *empty_ops[2] = { NULL, NULL };
    size_t n_empty_ops[2] = { 0, 0 };
    MLIR_ValueHandle cond_arr[1] = { cond };
    MLIR_OpHandle cbr = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.cond_br"),
        NULL, 0, NULL, 0, NULL, 0,
        cond_arr, 1, NULL, 0,
        succs, 2, empty_ops, n_empty_ops,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, cbr, pos);
    return true;
}

// Parse a `cases` attribute that prints as "array<i64: 1, 2, 3>".
static bool parse_cases(string cs, int64_t *out, size_t n) {
    size_t p = 0;
    while (p < cs.size && cs.str[p] != ':') p++;
    if (p < cs.size) p++;
    size_t got = 0;
    while (p < cs.size && got < n) {
        while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
        if (p >= cs.size || cs.str[p] == '>') break;
        int64_t sign = 1;
        if (cs.str[p] == '-') { sign = -1; p++; }
        int64_t v = 0;
        while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9')
            v = v * 10 + (cs.str[p++] - '0');
        out[got++] = sign * v;
    }
    return got == n;
}

static void rewrite_region_yields(LowerState *st, MLIR_RegionHandle reg,
                                  MLIR_BlockHandle target) {
    size_t nb = MLIR_GetRegionNumBlocks(reg);
    for (size_t bi = 0; bi < nb; bi++)
        rewrite_yield_to_br(st, MLIR_GetRegionBlock(reg, bi), target);
}

static void move_region_blocks(LowerState *st, MLIR_RegionHandle reg,
                               MLIR_RegionHandle parent_region) {
    while (MLIR_GetRegionNumBlocks(reg) > 0) {
        MLIR_MoveBlockToRegionEnd(st->ctx,
            MLIR_GetRegionBlock(reg, 0), parent_region);
    }
}

// Replace the trailing `scf.condition %cond, %fwd...` with
// `llvm.cond_br %cond, ^after(%fwd), ^exit(%fwd)`.
static void rewrite_condition_to_cbr(LowerState *st, MLIR_BlockHandle blk,
                                     MLIR_BlockHandle body_blk,
                                     MLIR_BlockHandle exit_blk) {
    size_t n = MLIR_GetBlockNumOps(blk);
    if (n == 0) return;
    MLIR_OpHandle term = MLIR_GetBlockOp(blk, n - 1);
    if (!name_eq(MLIR_GetOpName(term), "scf.condition")) return;
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_ValueHandle cond = MLIR_GetOpOperand(term, 0);
    size_t nf = MLIR_GetOpNumOperands(term) - 1;
    MLIR_ValueHandle *fwd = nf ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nf * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nf; i++) fwd[i] = MLIR_GetOpOperand(term, i + 1);
    MLIR_BlockHandle succs[2] = { body_blk, exit_blk };
    MLIR_ValueHandle *succ_ops_arr[2] = { fwd, fwd };
    size_t n_succ_ops_arr[2] = { nf, nf };
    MLIR_LocationHandle term_loc = MLIR_GetOpLocation(term);
    MLIR_ValueHandle cond_arr[1] = { cond };
    MLIR_OpHandle cbr = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.cond_br"),
        NULL, 0, NULL, 0, NULL, 0,
        cond_arr, 1, NULL, 0,
        succs, 2, succ_ops_arr, n_succ_ops_arr,
        term_loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(st->ctx, blk, cbr);
    MLIR_EraseOp(st->ctx, term);
}

static MLIR_ValueHandle emit_int_constant(LowerState *st, MLIR_BlockHandle parent,
                                          size_t *pos, MLIR_TypeHandle ty,
                                          int64_t val, MLIR_LocationHandle loc) {
    MLIR_ValueHandle res = make_result_value(st->ctx, ty, loc);
    MLIR_AttributeHandle val_attr = MLIR_CreateAttributeInteger(
        st->ctx, str_lit("value"), val, ty);
    MLIR_AttributeHandle attrs[1] = { val_attr };
    MLIR_TypeHandle rts[1] = { ty };
    MLIR_ValueHandle results[1] = { res };
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_MLIR_CONSTANT, str_lit("llvm.mlir.constant"),
        attrs, 1, rts, 1, results, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, *pos);
    (*pos)++;
    return res;
}

static MLIR_ValueHandle emit_icmp_eq(LowerState *st, MLIR_BlockHandle parent,
                                     size_t *pos, MLIR_ValueHandle a,
                                     MLIR_ValueHandle b,
                                     MLIR_LocationHandle loc) {
    MLIR_TypeHandle i1 = MLIR_CreateTypeInteger(st->ctx, 1, false);
    MLIR_ValueHandle res = make_result_value(st->ctx, i1, loc);
    MLIR_AttributeHandle pred = MLIR_CreateAttributeInteger(
        st->ctx, str_lit("predicate"), 0,
        MLIR_CreateTypeInteger(st->ctx, 64, true));
    MLIR_AttributeHandle attrs[1] = { pred };
    MLIR_TypeHandle rts[1] = { i1 };
    MLIR_ValueHandle results[1] = { res };
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_ICMP, str_lit("llvm.icmp"),
        attrs, 1, rts, 1, results, 1, ops, 2, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, *pos);
    (*pos)++;
    return res;
}

static void emit_br_at(LowerState *st, MLIR_BlockHandle parent, size_t *pos,
                       MLIR_BlockHandle target,
                       MLIR_ValueHandle *ops, size_t n_ops,
                       MLIR_LocationHandle loc) {
    MLIR_BlockHandle succs[1] = { target };
    MLIR_ValueHandle *succ_ops_arr[1] = { ops };
    size_t n_succ_ops_arr[1] = { n_ops };
    MLIR_OpHandle br = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.br"),
        NULL, 0, NULL, 0, NULL, 0,
        NULL, 0, NULL, 0,
        succs, 1, succ_ops_arr, n_succ_ops_arr,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, br, *pos);
    (*pos)++;
}

static void emit_cond_br_at(LowerState *st, MLIR_BlockHandle parent, size_t *pos,
                            MLIR_ValueHandle cond,
                            MLIR_BlockHandle true_tgt, MLIR_BlockHandle false_tgt,
                            MLIR_LocationHandle loc) {
    MLIR_BlockHandle succs[2] = { true_tgt, false_tgt };
    MLIR_ValueHandle *empty_ops[2] = { NULL, NULL };
    size_t n_empty_ops[2] = { 0, 0 };
    MLIR_ValueHandle cond_arr[1] = { cond };
    MLIR_OpHandle cbr = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.cond_br"),
        NULL, 0, NULL, 0, NULL, 0,
        cond_arr, 1, NULL, 0,
        succs, 2, empty_ops, n_empty_ops,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, cbr, *pos);
    (*pos)++;
}

// Append `llvm.br ^target(ops...)` at the end of `blk`.
static void emit_br_at_end(LowerState *st, MLIR_BlockHandle blk,
                           MLIR_BlockHandle target,
                           MLIR_ValueHandle *ops, size_t n_ops,
                           MLIR_LocationHandle loc) {
    size_t pos = MLIR_GetBlockNumOps(blk);
    emit_br_at(st, blk, &pos, target, ops, n_ops, loc);
}

// Empty switch/while arm: branch to `merge` with zero constants when the
// op carries results, else an operand-free branch.
static void emit_br_to_merge(LowerState *st, MLIR_BlockHandle blk,
                             MLIR_BlockHandle merge, MLIR_OpHandle owner) {
    size_t nr = MLIR_GetOpNumResults(owner);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(owner);
    if (nr == 0) {
        emit_br_at_end(st, blk, merge, NULL, 0, loc);
        return;
    }
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_ValueHandle *zeros = (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < nr; i++) {
        MLIR_TypeHandle ty = MLIR_GetValueType(MLIR_GetOpResult(owner, i));
        size_t pos = MLIR_GetBlockNumOps(blk);
        zeros[i] = emit_int_constant(st, blk, &pos, ty, 0, loc);
    }
    emit_br_at_end(st, blk, merge, zeros, nr, loc);
}

// Phi join block `join` receives values on its block args; pass them on to
// `cont` where post-op code lives (separate SSA names on `cont`).
static void emit_join_to_cont(LowerState *st, MLIR_BlockHandle join,
                              MLIR_BlockHandle cont, size_t nr,
                              MLIR_LocationHandle loc) {
    if (nr == 0) {
        emit_br_at_end(st, join, cont, NULL, 0, loc);
        return;
    }
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_ValueHandle *ops = (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < nr; i++)
        ops[i] = MLIR_GetBlockArg(join, i);
    emit_br_at_end(st, join, cont, ops, nr, loc);
}

// Lower `scf.while` to a header/body/exit CFG:
//   llvm.br ^header(inits)
// ^header(%iter...): ...; llvm.cond_br %cond, ^body(%fwd), ^exit(%fwd)
// ^body(%res...): ...; llvm.br ^header(yielded)
// ^exit(%results...): llvm.br ^cont(%results...)
// ^cont: post-loop code (uses cont block args, not exit's)
static bool lower_scf_while(LowerState *st, MLIR_OpHandle op,
                            MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_RegionHandle parent_region = MLIR_GetBlockParentRegion(parent);
    if (parent_region == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumRegions(op) < 2) return false;

    MLIR_RegionHandle regs[2];
    if (!take_op_regions(st->ctx, op, regs, 2)) return false;
    MLIR_RegionHandle condition_reg = regs[0];
    MLIR_RegionHandle body_reg = regs[1];

    if (MLIR_GetRegionNumBlocks(condition_reg) < 1 ||
        MLIR_GetRegionNumBlocks(body_reg) < 1) {
        restore_op_regions(st->ctx, op, regs, 2);
        return false;
    }

    MLIR_BlockHandle header_blk = MLIR_GetRegionBlock(condition_reg, 0);
    MLIR_BlockHandle body_blk = MLIR_GetRegionBlock(body_reg, 0);
    size_t n_iter = MLIR_GetOpNumOperands(op);
    size_t nr = MLIR_GetOpNumResults(op);
    if (n_iter != MLIR_GetBlockNumArgs(header_blk) ||
        nr != MLIR_GetBlockNumArgs(body_blk)) {
        restore_op_regions(st->ctx, op, regs, 2);
        return false;
    }

    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_BlockHandle exit_blk = MLIR_CreateBlock(st->ctx);
    MLIR_BlockHandle cont_blk = MLIR_CreateBlock(st->ctx);
    MLIR_ValueHandle *exit_args = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    MLIR_ValueHandle *cont_args = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_TypeHandle ty = MLIR_GetValueType(old);
        exit_args[i] = MLIR_CreateValueBlockArg(
            st->ctx, str_lit(""), (uint32_t)i, ty, loc);
        MLIR_AppendBlockArg(st->ctx, exit_blk, exit_args[i]);
        cont_args[i] = MLIR_CreateValueBlockArg(
            st->ctx, str_lit(""), (uint32_t)i, ty, loc);
        MLIR_AppendBlockArg(st->ctx, cont_blk, cont_args[i]);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, cont_args[i]);
    }

    while (MLIR_GetBlockNumOps(parent) > pos + 1) {
        MLIR_OpHandle tail = MLIR_GetBlockOp(parent, pos + 1);
        MLIR_MoveOpToBlockEnd(st->ctx, tail, cont_blk);
    }

    size_t nbefore = MLIR_GetRegionNumBlocks(condition_reg);
    for (size_t bi = 0; bi < nbefore; bi++) {
        rewrite_condition_to_cbr(st, MLIR_GetRegionBlock(condition_reg, bi),
                                 body_blk, exit_blk);
    }
    size_t nafter = MLIR_GetRegionNumBlocks(body_reg);
    for (size_t bi = 0; bi < nafter; bi++) {
        rewrite_yield_to_br(st, MLIR_GetRegionBlock(body_reg, bi), header_blk);
    }

    move_region_blocks(st, condition_reg, parent_region);
    move_region_blocks(st, body_reg, parent_region);
    MLIR_MoveBlockToRegionEnd(st->ctx, exit_blk, parent_region);
    MLIR_MoveBlockToRegionEnd(st->ctx, cont_blk, parent_region);
    emit_join_to_cont(st, exit_blk, cont_blk, nr, loc);

    MLIR_ValueHandle *inits = n_iter ? (MLIR_ValueHandle *)arena_alloc(
        alloc, n_iter * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < n_iter; i++) inits[i] = MLIR_GetOpOperand(op, i);
    emit_br_at(st, parent, &pos, header_blk, inits, n_iter, loc);
    return true;
}

// Lower `scf.index_switch` to a compare chain plus case/default blocks that
// merge at a phi block, then continue in a separate post-switch block (region
// 0 = default, regions 1..N = cases). Empty regions become synthetic blocks
// that branch to the merge with zero-filled result operands.
static bool lower_scf_index_switch(LowerState *st, MLIR_OpHandle op,
                                   MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_RegionHandle parent_region = MLIR_GetBlockParentRegion(parent);
    if (parent_region == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumOperands(op) < 1) return false;

    MLIR_ValueHandle selector = MLIR_GetOpOperand(op, 0);
    MLIR_TypeHandle sel_ty = MLIR_GetValueType(selector);
    size_t n_regions = MLIR_GetOpNumRegions(op);
    if (n_regions < 1) return false;
    size_t n_cases = n_regions - 1;

    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    int64_t case_vals_stk[64];
    int64_t *case_vals = case_vals_stk;
    if (n_cases > 64) {
        case_vals = (int64_t *)arena_alloc(alloc, n_cases * sizeof(int64_t));
    }
    if (n_cases > 0) {
        MLIR_AttributeHandle ca = MLIR_GetOpAttributeByName(op, "cases");
        if (ca == MLIR_INVALID_HANDLE ||
            !parse_cases(MLIR_GetAttributeAsString(st->ctx, ca),
                         case_vals, n_cases))
            return false;
    }

    MLIR_RegionHandle regs_stk[16];
    MLIR_RegionHandle *regs = regs_stk;
    if (n_regions > 16) {
        regs = (MLIR_RegionHandle *)arena_alloc(
            alloc, n_regions * sizeof(MLIR_RegionHandle));
    }
    if (!take_op_regions(st->ctx, op, regs, n_regions)) return false;

    size_t nr = MLIR_GetOpNumResults(op);
    MLIR_BlockHandle merge_blk = MLIR_CreateBlock(st->ctx);
    MLIR_BlockHandle cont_blk = MLIR_CreateBlock(st->ctx);
    MLIR_ValueHandle *cont_args = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_TypeHandle ty = MLIR_GetValueType(old);
        MLIR_ValueHandle marg = MLIR_CreateValueBlockArg(
            st->ctx, str_lit(""), (uint32_t)i, ty, loc);
        MLIR_AppendBlockArg(st->ctx, merge_blk, marg);
        cont_args[i] = MLIR_CreateValueBlockArg(
            st->ctx, str_lit(""), (uint32_t)i, ty, loc);
        MLIR_AppendBlockArg(st->ctx, cont_blk, cont_args[i]);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, cont_args[i]);
    }

    while (MLIR_GetBlockNumOps(parent) > pos + 1) {
        MLIR_OpHandle tail = MLIR_GetBlockOp(parent, pos + 1);
        MLIR_MoveOpToBlockEnd(st->ctx, tail, cont_blk);
    }

    MLIR_BlockHandle *arm_blks = (MLIR_BlockHandle *)arena_alloc(
        alloc, n_regions * sizeof(MLIR_BlockHandle));
    for (size_t ri = 0; ri < n_regions; ri++) {
        if (MLIR_GetRegionNumBlocks(regs[ri]) == 0) {
            MLIR_BlockHandle empty = MLIR_CreateBlock(st->ctx);
            emit_br_to_merge(st, empty, merge_blk, op);
            arm_blks[ri] = empty;
            MLIR_MoveBlockToRegionEnd(st->ctx, empty, parent_region);
            continue;
        }
        arm_blks[ri] = MLIR_GetRegionBlock(regs[ri], 0);
        rewrite_region_yields(st, regs[ri], merge_blk);
        move_region_blocks(st, regs[ri], parent_region);
    }
    MLIR_MoveBlockToRegionEnd(st->ctx, merge_blk, parent_region);
    MLIR_MoveBlockToRegionEnd(st->ctx, cont_blk, parent_region);
    emit_join_to_cont(st, merge_blk, cont_blk, nr, loc);

    MLIR_BlockHandle default_blk = arm_blks[0];
    if (n_cases == 0) {
        emit_br_at(st, parent, &pos, default_blk, NULL, 0, loc);
        return true;
    }

    MLIR_BlockHandle *case_blks = n_cases ? (MLIR_BlockHandle *)arena_alloc(
        alloc, n_cases * sizeof(MLIR_BlockHandle)) : NULL;
    for (size_t i = 0; i < n_cases; i++)
        case_blks[i] = arm_blks[i + 1];

    MLIR_BlockHandle chain = MLIR_CreateBlock(st->ctx);
    MLIR_BlockHandle chain_cur = chain;
    size_t cpos = 0;
    for (size_t i = 0; i < n_cases; i++) {
        MLIR_ValueHandle cv = emit_int_constant(st, chain_cur, &cpos, sel_ty,
                                                case_vals[i], loc);
        MLIR_ValueHandle eq = emit_icmp_eq(st, chain_cur, &cpos, selector, cv, loc);
        if (i + 1 < n_cases) {
            MLIR_BlockHandle next = MLIR_CreateBlock(st->ctx);
            emit_cond_br_at(st, chain_cur, &cpos, eq, case_blks[i], next, loc);
            MLIR_MoveBlockToRegionEnd(st->ctx, chain_cur, parent_region);
            chain_cur = next;
            cpos = 0;
        } else {
            emit_cond_br_at(st, chain_cur, &cpos, eq, case_blks[i],
                            default_blk, loc);
            MLIR_MoveBlockToRegionEnd(st->ctx, chain_cur, parent_region);
        }
    }
    emit_br_at(st, parent, &pos, chain, NULL, 0, loc);
    return true;
}

// Return values from try_lower_op:
//   LOWER_NONE      — op not handled; walker should leave it in place.
//   LOWER_REPLACED  — lowering inserted N replacement ops at position
//                     `pos` and the original op is still attached at the
//                     end of that run. Walker erases original and skips
//                     past the inserts.
//   LOWER_DONE_BLOCK — lowering rewrote the block's tail (e.g. scf.if
//                     turned the block into a cond_br); the original op
//                     has already been erased and there is nothing more
//                     to walk in this block.
typedef enum {
    LOWER_NONE = 0,
    LOWER_REPLACED,
    LOWER_DONE_BLOCK,
} LowerResult;

// Returns one of LowerResult values. See enum docs above.
static int try_lower_op(LowerState *st, MLIR_OpHandle op,
                        MLIR_BlockHandle parent, size_t pos) {
    string name = MLIR_GetOpName(op);
    if (name_eq(name, "scf.if")) {
        if (st->keep_scf) return LOWER_NONE;
        if (lower_scf_if(st, op, parent, pos)) {
            MLIR_EraseOp(st->ctx, op);
            return LOWER_DONE_BLOCK;
        }
        return LOWER_NONE;
    }
    if (name_eq(name, "scf.while")) {
        if (st->keep_scf) return LOWER_NONE;
        if (lower_scf_while(st, op, parent, pos)) {
            MLIR_EraseOp(st->ctx, op);
            return LOWER_DONE_BLOCK;
        }
        return LOWER_NONE;
    }
    if (name_eq(name, "scf.index_switch")) {
        if (st->keep_scf) return LOWER_NONE;
        if (lower_scf_index_switch(st, op, parent, pos)) {
            MLIR_EraseOp(st->ctx, op);
            return LOWER_DONE_BLOCK;
        }
        return LOWER_NONE;
    }
    bool ok = false;
    if (name_eq(name, "func.func"))      ok = lower_func_func(st, op, parent, pos);
    else if (name_eq(name, "func.return") ||
             name_eq(name, "return"))    ok = lower_func_return(st, op, parent, pos);
    else if (name_eq(name, "func.call")) ok = lower_func_call(st, op, parent, pos);
    else if (name_eq(name, "func.constant")) ok = lower_func_constant(st, op, parent, pos);
    else if (name_eq(name, "func.call_indirect")) ok = lower_func_call_indirect(st, op, parent, pos);
    else if (name_eq(name, "builtin.unrealized_conversion_cast") ||
             name_eq(name, "unrealized_conversion_cast"))
                                         ok = lower_unrealized_cast(st, op, parent, pos);
    else if (name_eq(name, "arith.constant")) ok = lower_arith_constant(st, op, parent, pos);
    else if (name_eq(name, "arith.addi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.add"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.subi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.sub"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.muli"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.mul"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.divsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.sdiv"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.divui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.udiv"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.remsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.srem"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.remui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.urem"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.andi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.and"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.ori"))   ok = lower_rename(st, op, parent, pos, str_lit("llvm.or"),   OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.xori"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.xor"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.shli"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.shl"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.shrsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.ashr"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.shrui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.lshr"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.addf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fadd"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.subf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fsub"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.mulf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fmul"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.divf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fdiv"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.cmpi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.icmp"), OP_TYPE_LLVM_ICMP);
    else if (name_eq(name, "arith.cmpf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fcmp"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.select"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.select"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.extsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.sext"),   OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.extui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.zext"),   OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.trunci"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.trunc"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.extf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fpext"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.truncf"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.fptrunc"),OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.sitofp"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.sitofp"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.uitofp"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.uitofp"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.fptosi"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.fptosi"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.fptoui"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.fptoui"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.bitcast"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.bitcast"),OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "cf.br"))      ok = st->keep_scf ? false : lower_cf_branch(st, op, parent, pos, str_lit("llvm.br"));
    else if (name_eq(name, "cf.cond_br")) ok = st->keep_scf ? false : lower_cf_branch(st, op, parent, pos, str_lit("llvm.cond_br"));

    return ok ? LOWER_REPLACED : LOWER_NONE;
}

static void walk_block(LowerState *st, MLIR_BlockHandle block);

static void walk_op(LowerState *st, MLIR_OpHandle op) {
    size_t nr = MLIR_GetOpNumRegions(op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(op, r);
        // Recompute block count each iteration: lowerings (e.g. scf.if)
        // can add sibling blocks to this region during the walk.
        for (size_t b = 0; b < MLIR_GetRegionNumBlocks(reg); b++) {
            walk_block(st, MLIR_GetRegionBlock(reg, b));
        }
    }
}

static void walk_block(LowerState *st, MLIR_BlockHandle block) {
    size_t i = 0;
    while (i < MLIR_GetBlockNumOps(block)) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        walk_op(st, op);
        size_t before = MLIR_GetBlockNumOps(block);
        int res = try_lower_op(st, op, block, i);
        if (res == LOWER_NONE) { i++; continue; }
        if (res == LOWER_DONE_BLOCK) {
            // The lowering restructured this block (and likely created
            // sibling blocks); nothing more to walk here. The new blocks
            // are appended to the parent region and will be visited by
            // the outer walker as it enumerates them.
            return;
        }
        // LOWER_REPLACED: simple in-place rewrite. Original is still
        // attached at position i + inserted; erase it and skip the inserts.
        size_t after = MLIR_GetBlockNumOps(block);
        size_t inserted = after - before;
        MLIR_EraseOp(st->ctx, op);
        i += inserted;
    }
}

bool MLIR_LowerToLLVMDialect(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (module == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumRegions(module) == 0) return false;
    MLIR_RegionHandle body_region = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(body_region) == 0) return false;

    LowerState st = {0};
    st.ctx = ctx;
    st.module = module;
    st.module_body = MLIR_GetRegionBlock(body_region, 0);

    walk_block(&st, st.module_body);
    return true;
}

// Wasm-tailored lowering: lift cf.br / cf.cond_br into scf, then run the
// regular LLVM-dialect lowering with `keep_scf = true` so structured
// `scf.*` ops survive for mlir_llvm_to_wasmssa. cf->llvm.br rewrites are
// skipped; any surviving cf op is a hard error caught by wasmssa-lower.
bool MLIR_LowerToLLVMDialectForWasm(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (module == MLIR_INVALID_HANDLE) return false;
    if (!MLIR_LiftCfToScf(ctx, module)) return false;
    if (MLIR_GetOpNumRegions(module) == 0) return false;
    MLIR_RegionHandle body_region = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(body_region) == 0) return false;

    LowerState st = {0};
    st.ctx = ctx;
    st.module = module;
    st.module_body = MLIR_GetRegionBlock(body_region, 0);
    st.keep_scf = true;

    walk_block(&st, st.module_body);
    return true;
}
