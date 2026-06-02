// llvm dialect -> aarch64 lowering. See mlir_llvm_to_aarch64.h for the
// public API, rationale, and the staged build-up plan.
//
// Status: Step 1 walking skeleton. Lowers `int main(){return 42;}` (and
// any function whose body is just `llvm.mlir.constant` defs plus a single
// `llvm.return` of a constant) straight to the physical-register
// `aarch64` dialect, then relies on the existing `mlir_aarch64_to_macho`
// encoder. A synthesised `_start` calls `main` and exits with its return
// value via libSystem `_exit` (mirrors wmir's synth_start).
//
// This step deliberately emits `aarch64.*` directly (no virtual-register
// `a64ssa.*` tier yet): a constant-return skeleton has no register
// pressure, so the vreg tier + post-isel regalloc are introduced in
// Step 2/3 once real values appear. Anything outside the supported shape
// returns MLIR_INVALID_HANDLE with a clear diagnostic so Step 2 knows
// exactly what to grow next.
//
// Nothing here is on the path of the existing wasm/wmir backends; it is
// reached only via the opt-in `--macho-backend=llvm` driver flag.

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "mlir_llvm_to_aarch64.h"
#include "mlir_op_names.h"

// ---------------------------------------------------------------------------
// Small attribute / op builders (mirrors of the wmir backend's helpers).
// ---------------------------------------------------------------------------
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name,
                                     int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static void emit_movz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVZ, a, 4));
}
static void emit_movk(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVK, a, 4));
}
static void emit_bl(MLIR_Context *ctx, MLIR_BlockHandle blk, string callee) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_s(ctx, "callee", callee.str, callee.size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_BL, a, 1));
}
static void emit_ret(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_RET, NULL, 0));
}

// Materialise a 64-bit immediate into `rd` with a movz + movk chain,
// emitting only the non-zero 16-bit lanes (movz seeds the lowest lane so
// a zero value still produces `movz rd, #0`).
static void emit_load_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint8_t rd, uint64_t v, bool sf) {
    int lanes = sf ? 4 : 2;
    emit_movz(ctx, blk, rd, (uint16_t)(v & 0xffffu), 0, sf);
    for (int hw = 1; hw < lanes; hw++) {
        uint16_t chunk = (uint16_t)((v >> (16 * hw)) & 0xffffu);
        if (chunk != 0) emit_movk(ctx, blk, rd, chunk, (uint8_t)hw, sf);
    }
}

// ---------------------------------------------------------------------------
// Helpers over the input `llvm` dialect module.
// ---------------------------------------------------------------------------
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static bool type_is_i64(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle ty = MLIR_GetValueType(v);
    string s = MLIR_GetTypeString(ctx, ty);
    return s.size == 3 && memcmp(s.str, "i64", 3) == 0;
}

static bool func_has_body(MLIR_OpHandle fn) {
    return MLIR_GetOpNumRegions(fn) > 0 &&
           MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;
}

// Small linear value->constant table (Step 1 functions are tiny).
typedef struct { uintptr_t key; int64_t val; bool i64; } ConstEnt;

static bool const_lookup(const ConstEnt *tab, size_t n, MLIR_ValueHandle k,
                         int64_t *out, bool *is64) {
    for (size_t i = 0; i < n; i++) {
        if (tab[i].key == (uintptr_t)k) {
            *out = tab[i].val;
            *is64 = tab[i].i64;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Select one defined `llvm.func` into an `aarch64.func`. Step 1 supports a
// single entry block consisting of `llvm.mlir.constant` definitions and a
// terminating `llvm.return` of a constant (or a void return).
// ---------------------------------------------------------------------------
static MLIR_OpHandle select_func(MLIR_Context *ctx, MLIR_OpHandle fn,
                                 string sym) {
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(fn, 0);
    if (MLIR_GetRegionNumBlocks(src_region) != 1) {
        fprintf(stderr,
            "llvm->aarch64: function '%.*s' has multiple blocks "
            "(control flow not yet supported in Step 1)\n",
            (int)sym.size, sym.str);
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, 0);
    size_t nops = MLIR_GetBlockNumOps(src_blk);

    MLIR_BlockHandle  out_blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_reg, out_blk);

    ConstEnt consts[64];
    size_t   n_consts = 0;

    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(src_blk, i);
        string        on = MLIR_GetOpName(op);
        if (name_eq(on, "llvm.mlir.constant")) {
            MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
            if (va == MLIR_INVALID_HANDLE || MLIR_GetOpNumResults(op) != 1) {
                fprintf(stderr, "llvm->aarch64: malformed llvm.mlir.constant\n");
                return MLIR_INVALID_HANDLE;
            }
            MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
            if (n_consts >= 64) {
                fprintf(stderr, "llvm->aarch64: too many constants in '%.*s'\n",
                        (int)sym.size, sym.str);
                return MLIR_INVALID_HANDLE;
            }
            consts[n_consts].key = (uintptr_t)res;
            consts[n_consts].val = MLIR_GetAttributeInteger(va);
            consts[n_consts].i64 = type_is_i64(ctx, res);
            n_consts++;
        } else if (name_eq(on, "llvm.return")) {
            if (MLIR_GetOpNumOperands(op) == 0) {
                // void return: w0 left as-is.
            } else if (MLIR_GetOpNumOperands(op) == 1) {
                MLIR_ValueHandle rv = MLIR_GetOpOperand(op, 0);
                int64_t cval; bool c64;
                if (!const_lookup(consts, n_consts, rv, &cval, &c64)) {
                    fprintf(stderr,
                        "llvm->aarch64: return value in '%.*s' is not a "
                        "constant (only constant returns supported in "
                        "Step 1)\n", (int)sym.size, sym.str);
                    return MLIR_INVALID_HANDLE;
                }
                emit_load_imm(ctx, out_blk, /*rd=*/0, (uint64_t)cval, c64);
            } else {
                fprintf(stderr, "llvm->aarch64: multi-value return\n");
                return MLIR_INVALID_HANDLE;
            }
            emit_ret(ctx, out_blk);
        } else {
            fprintf(stderr,
                "llvm->aarch64: unsupported op '%.*s' in '%.*s' "
                "(Step 1 supports only constant + return)\n",
                (int)MLIR_GetOpName(op).size, MLIR_GetOpName(op).str,
                (int)sym.size, sym.str);
            return MLIR_INVALID_HANDLE;
        }
    }

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", sym.str, sym.size);
    MLIR_RegionHandle regs[1] = { out_reg };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Synthesise `_start`: `bl main; bl _exit`. Exported and emitted first so
// the Mach-O encoder lands LC_MAIN.entryoff on it.
static MLIR_OpHandle synth_start(MLIR_Context *ctx, string main_name) {
    MLIR_BlockHandle  blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, reg, blk);

    emit_bl(ctx, blk, main_name);
    emit_bl(ctx, blk, str_lit("_exit"));

    MLIR_AttributeHandle attrs[2];
    attrs[0] = attr_s(ctx, "sym_name", "_start", 6);
    attrs[1] = attr_b(ctx, "exported", true);
    MLIR_RegionHandle regs[1] = { reg };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 2, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

MLIR_OpHandle mlir_llvm_to_aarch64(MLIR_Context *ctx,
                                   MLIR_OpHandle llvm_module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    MLIR_BlockHandle  out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);

    // `_start` first (entry point).
    MLIR_AppendBlockOp(ctx, out_body, synth_start(ctx, str_lit("main")));

    bool saw_main = false;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        if (!func_has_body(op)) continue;  // skip declarations (malloc/free).
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) {
            fprintf(stderr, "llvm->aarch64: llvm.func without sym_name\n");
            return MLIR_INVALID_HANDLE;
        }
        string sym = MLIR_GetAttributeString(sa);
        if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) saw_main = true;
        MLIR_OpHandle fn = select_func(ctx, op, sym);
        if (fn == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, fn);
    }

    if (!saw_main) {
        fprintf(stderr, "llvm->aarch64: no defined 'main' function\n");
        return MLIR_INVALID_HANDLE;
    }

    MLIR_RegionHandle regs[1] = { out_region };
    return MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}
