// wmir -> aarch64 lowering. First-light scaffold: just enough to
// compile `int main() { return 42; }` to a hand-coded sequence of
// AArch64 instructions that fits into the new Mach-O backend.
//
// Conventions of this first slice (will be revisited as op coverage
// grows):
//
//   * No register allocator. Every wmir SSA value is materialised
//     immediately into a small fixed register (w9 for the "current"
//     temp); on `wmir.return %v` we mov w0 from that temp and emit a
//     `ret`. This is deliberately dumb and only correct because the
//     supported wmir ops produce at most one live value at any point.
//
//   * Each wmir.func is lowered into one aarch64.func. Additionally,
//     the lowering synthesises an `_start` aarch64.func that:
//       bl __original_main      ; call user's main, return code in w0
//       mov x16, #1             ; mach trap #1 = exit
//       svc #0x80               ; direct kernel call (no proc_exit shim)
//     This keeps the Mach-O envelope free of libSystem stubs and the
//     GOT for the first-light slice. Once we add call ops to wmir we
//     can promote the exit path to a proc_exit shim like the existing
//     wasm-to-macho backend does.
//
//   * `wmir.const` always produces a 32-bit immediate (i32). The
//     unused `i64` const that wasmssa frontends often emit (see
//     macho_exit.tc) is simply not materialised — it has no user.
//
// Anything outside this slice (params, calls, memory, control flow,
// floats, register allocation, larger immediates) returns
// MLIR_INVALID_HANDLE with a clear diagnostic identifying the op.

#include "mlir_wmir_to_aarch64.h"

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
// Attribute helpers.
// =============================================================================
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
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

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Emit `movz Wd, #imm16, LSL #(hw*16)` (sf=0 → 32-bit form, sf=1 → 64-bit).
static void emit_movz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVZ, a, 4));
}
// Emit `movk Wd, #imm16, LSL #(hw*16)`.
static void emit_movk(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVK, a, 4));
}
// Emit `mov Xd, Xn` (register-to-register, X-form: ORR Xd, XZR, Xn).
static void emit_mov_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOV_X, a, 2));
}
// Emit `bl <symbol>` (PC-relative branch-and-link).
static void emit_bl(MLIR_Context *ctx, MLIR_BlockHandle blk, string callee) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_s(ctx, "callee", callee.str, callee.size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_BL, a, 1));
}
// Emit `svc #imm16`.
static void emit_svc(MLIR_Context *ctx, MLIR_BlockHandle blk, uint16_t imm16) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "imm16", imm16);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SVC, a, 1));
}
// Emit `ret` (semantically `ret x30`).
static void emit_ret(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_RET, NULL, 0));
}

// Materialise a 32-bit immediate into Wn using movz + (optional) movk.
// First-light slice: we only need 16-bit-fitting immediates for
// macho_exit, but the helper handles 32 bits so it's already
// future-proof for small follow-ups.
static void emit_mov_imm32(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, uint32_t v) {
    emit_movz(ctx, blk, rd, (uint16_t)(v & 0xffff), 0, /*sf=*/false);
    if ((v >> 16) != 0) {
        emit_movk(ctx, blk, rd, (uint16_t)((v >> 16) & 0xffff), 1, /*sf=*/false);
    }
}

// =============================================================================
// Per-function lowering. Walks a wmir.func body and produces an
// aarch64.func with a flat sequence of aarch64.* instruction ops.
// =============================================================================
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src) {
    string name = at_s(src, "sym_name");
    bool exported = at_b(src, "exported");

    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);

    if (MLIR_GetOpNumRegions(src) < 1) {
        fprintf(stderr, "wmir->aarch64: wmir.func has no region\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(src, 0), 0);
    size_t n_ops = MLIR_GetBlockNumOps(src_blk);

    // First-light SSA "allocator": one shadow register (w9) holds the
    // most-recently-defined value. Since the only supported producer
    // ops in this slice are wmir.const, and the only consumer is
    // wmir.return, this is sufficient — no liveness conflict can
    // arise. The register allocator that replaces this will live in
    // its own pass; this function will become a pure walker over
    // pre-assigned virtual-register operands.
    const uint8_t TEMP_REG = 9;

    // Per-value cache: SSA value -> physical register the value lives
    // in right now. Tiny linear scan; cap to a sensible upper bound
    // because the first-light slice never produces more than a couple
    // of live values.
    struct { MLIR_ValueHandle v; uint8_t reg; } vmap[16];
    size_t n_vmap = 0;

    for (size_t i = 0; i < n_ops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(src_blk, i);
        MLIR_OpType t = MLIR_GetOpType(op);
        switch (t) {
        case OP_TYPE_WMIR_CONST: {
            if (MLIR_GetOpNumResults(op) < 1) {
                fprintf(stderr, "wmir->aarch64: wmir.const has no result\n");
                return MLIR_INVALID_HANDLE;
            }
            int64_t v = at_i(op, "value");
            // Only the 32-bit form is supported in this slice. The
            // user-side i32 constants we care about (e.g. the 42 in
            // macho_exit) all fit. Larger / 64-bit values land here
            // later when we expand op coverage.
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            MLIR_TypeHandle ty = MLIR_GetValueType(r);
            string ts = MLIR_GetTypeString(ctx, ty);
            if (!(ts.size == 3 && memcmp(ts.str, "i32", 3) == 0)) {
                // Silently skip i64 constants that are never used —
                // the wasmssa frontend emits a stray `i64 = 1` for
                // stack-canary purposes that has no consumer. If it
                // *is* used we'll fail below at lookup time, which
                // is the right behaviour.
                if (ts.size == 3 && memcmp(ts.str, "i64", 3) == 0) {
                    continue;
                }
                fprintf(stderr,
                    "wmir->aarch64: wmir.const of unsupported type "
                    "'%.*s'\n", (int)ts.size, ts.str);
                return MLIR_INVALID_HANDLE;
            }
            emit_mov_imm32(ctx, blk, TEMP_REG, (uint32_t)v);
            if (n_vmap < 16) {
                vmap[n_vmap].v = r; vmap[n_vmap].reg = TEMP_REG; n_vmap++;
            }
            break;
        }
        case OP_TYPE_WMIR_RETURN: {
            size_t no = MLIR_GetOpNumOperands(op);
            if (no > 1) {
                fprintf(stderr,
                    "wmir->aarch64: wmir.return with %zu results not yet "
                    "supported\n", no);
                return MLIR_INVALID_HANDLE;
            }
            if (no == 1) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, 0);
                uint8_t src_reg = 0;
                bool found = false;
                for (size_t k = 0; k < n_vmap; k++) {
                    if (vmap[k].v == v) { src_reg = vmap[k].reg; found = true; break; }
                }
                if (!found) {
                    fprintf(stderr,
                        "wmir->aarch64: unbound operand on wmir.return\n");
                    return MLIR_INVALID_HANDLE;
                }
                if (src_reg != 0) emit_mov_x(ctx, blk, /*rd=*/0, src_reg);
            }
            emit_ret(ctx, blk);
            break;
        }
        default: {
            string nm = MLIR_GetOpName(op);
            fprintf(stderr,
                "wmir->aarch64: unsupported wmir op '%.*s' (kind=%d)\n",
                (int)nm.size, nm.str, (int)t);
            return MLIR_INVALID_HANDLE;
        }
        }
    }

    MLIR_AttributeHandle attrs[3];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", name.str, name.size);
    attrs[na++] = attr_b(ctx, "exported", exported);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, blk);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Synthesise a `_start` aarch64.func that calls the user's
// `__original_main`, then issues an `exit(retcode)` Mach syscall.
//
// Body:
//   bl __original_main      ; x0 := main()'s return value
//   mov x16, #1             ; mach trap #1 = exit
//   svc #0x80               ; kernel call; no return
//
// `__original_main` already returns its result in w0/x0, so we don't
// need an explicit move into the syscall argument register. The
// process is terminated by the kernel; control never returns past
// the svc.
// =============================================================================
static MLIR_OpHandle synth_start(MLIR_Context *ctx, string main_name) {
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    emit_bl(ctx, blk, main_name);
    // mach trap #1 = SYS_exit; we want the i64 form (sf=1) so the
    // assembler stays consistent with how we'd issue an x16-targeted
    // syscall number elsewhere. Either form encodes a 16-bit value
    // identically since the upper 48 bits would all be zero anyway.
    emit_movz(ctx, blk, /*rd=*/16, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_svc(ctx, blk, 0x80);

    MLIR_AttributeHandle attrs[3];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", "_start", 6);
    attrs[na++] = attr_b(ctx, "exported", true);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, blk);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Top-level pass.
// =============================================================================
MLIR_OpHandle mlir_wmir_to_aarch64(MLIR_Context *ctx, MLIR_OpHandle wmir_module) {
    if (!wmir_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(wmir_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(wmir_module, 0);
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

    // Track the user's entry function name so we can synthesise `_start`.
    // First-light convention: the (single) exported wmir.func is treated
    // as the entry. Multi-function modules will need a more careful
    // policy once we expand coverage.
    string entry_name = {0};

    size_t n_top = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t != OP_TYPE_WMIR_FUNC) {
            string nm = MLIR_GetOpName(top);
            fprintf(stderr,
                "wmir->aarch64: unexpected top-level op '%.*s'\n",
                (int)nm.size, nm.str);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_OpHandle out_op = lower_func(ctx, top);
        if (!out_op) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, out_op);

        if (at_b(top, "exported") && entry_name.size == 0) {
            entry_name = at_s(top, "sym_name");
        }
    }

    if (entry_name.size == 0) {
        fprintf(stderr,
            "wmir->aarch64: module has no exported function to use as entry\n");
        return MLIR_INVALID_HANDLE;
    }

    MLIR_OpHandle start = synth_start(ctx, entry_name);
    if (!start) return MLIR_INVALID_HANDLE;
    MLIR_AppendBlockOp(ctx, out_body, start);

    return out_module;
}
