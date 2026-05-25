// wmir -> aarch64 lowering. Second slice (arithmetic + memory + globals
// + direct calls). The first-light scaffold lived in this file; this
// version supersedes it.
//
// Strategy: trivial "stack-slot per SSA value" allocator.
//   * Pre-pass: walk each wmir.func body. Number every block argument
//     and every op result with a unique slot index (0, 1, 2, …). Each
//     slot is 8 bytes (over-aligned even for i32; matches AArch64 LDP
//     friendliness and lets us promote to i64 later without reshuffling).
//   * Function prologue:
//         stp x29, x30, [sp, #-16]!     ; save FP/LR
//         mov x29, sp                   ; FP <- SP
//         sub sp, sp, #frame_size       ; allocate slot area
//         str w0,  [sp, #slot(param0)*8] ; spill parameters into slots
//         str w1,  [sp, #slot(param1)*8] ; …
//   * Each op materialises operands into scratch registers (w9, w10,
//     w11), computes the result into w9, then stores it back to the
//     result's slot. Scratches are clobbered between ops, so there's
//     no register-allocation problem to solve.
//   * Prologue/epilogue and per-op spill code are emitted as explicit
//     aarch64.* ops; the Mach-O encoder is a 1:1 translation from op
//     to bytes.
//
// Global / linear-memory access is mediated through three callee-
// callee-saved registers set up once by `_start`:
//   * x26 = vmctx pointer (data_priv region; unused in this slice but
//     reserved to match the existing wasm-to-macho backend's ABI).
//   * x27 = globals base. global_get/set lower to ldr/str at offset
//     global_idx*8.
//   * x28 = linear-memory base. load/store lower to
//         add xT, x28, wAddr, uxtw
//         ldr/str wValue, [xT, #memory_offset]
//
// The lowering also synthesises an `_start` aarch64.func that:
//   adrp/add x26, data_priv@page       ; vmctx
//   adrp/add x27, globals@page         ; globals base
//   adrp/add x28, linmem@page          ; linear-memory base
//   bl __original_main                 ; call user entry; retval -> w0
//   movz x16, #1                       ; mach trap #1 = exit
//   svc #0x80
//
// Module-level metadata (n_globals, linmem_size, …) is attached as
// attributes on the aarch64 builtin.module so the downstream Mach-O
// backend can size the __DATA segment without re-parsing wmir.

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
// Module-wide constants (must match the wasm-link conventions so the
// new pipeline is observationally identical to the existing one).
// =============================================================================
enum {
    DEFAULT_STACK_SIZE        = 4u * 1024u * 1024u,    // 4 MiB
    DEFAULT_GLOBAL_BASE_OFFS  = 1024u,                 // wasm-ld default
    WASM_PAGE_SIZE            = 65536u,
};

// =============================================================================
// Attribute helpers.
// =============================================================================
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 64, true));
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

// =============================================================================
// Emit helpers — one per aarch64.* op kind.
// =============================================================================
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
static void emit_mov_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOV_X, a, 2));
}
static void emit_bl(MLIR_Context *ctx, MLIR_BlockHandle blk, string callee) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_s(ctx, "callee", callee.str, callee.size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_BL, a, 1));
}
static void emit_svc(MLIR_Context *ctx, MLIR_BlockHandle blk, uint16_t imm16) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "imm16", imm16);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SVC, a, 1));
}
static void emit_ret(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_RET, NULL, 0));
}

// `add Xd, Xn, #imm12` (sf=1) or `add Wd, Wn, #imm12` (sf=0).
// Also encodes `add SP, SP, #imm12` when rd=rn=31 (SP).
static void emit_add_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint16_t imm12, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "imm12", imm12);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_IMM, a, 4));
}
static void emit_sub_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint16_t imm12, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "imm12", imm12);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SUB_IMM, a, 4));
}
// 32-bit (sf=0) or 64-bit (sf=1) shifted-register add/sub.
static void emit_add_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_REG, a, 4));
}
static void emit_sub_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SUB_REG, a, 4));
}
// 32-bit word load/store with unsigned 12-bit immediate offset (in bytes;
// encoder divides by 4 to produce the scaled imm12 field).
static void emit_ldr_w(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_LDR_W, a, 3));
}
static void emit_str_w(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_STR_W, a, 3));
}
// ADRP / ADD imm12 pair, with the target identified by a symbolic
// `target` attribute. The Mach-O backend patches the imm21/imm12
// fields after laying out the __DATA segment.
//   target ∈ { "data_priv", "globals", "linmem" }
static void emit_adrp_data(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, const char *target) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_s(ctx, "target", target, strlen(target));
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADRP_DATA, a, 2));
}
static void emit_add_data_lo(MLIR_Context *ctx, MLIR_BlockHandle blk,
                             uint8_t rd, uint8_t rn, const char *target) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_s(ctx, "target", target, strlen(target));
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_DATA_LO, a, 3));
}
// Compound prologue/epilogue pseudo-ops with a single `frame_size`
// (16-byte aligned) attribute. The encoder expands each into the
// corresponding stp/mov/sub or add/ldp/ret sequence.
static void emit_prologue(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint16_t frame_size) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", frame_size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_PROLOGUE, a, 1));
}
static void emit_epilogue(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint16_t frame_size) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", frame_size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_EPILOGUE, a, 1));
}

// Materialise a 32-bit immediate into Wn.
static void emit_mov_imm32(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, uint32_t v) {
    emit_movz(ctx, blk, rd, (uint16_t)(v & 0xffff), 0, /*sf=*/false);
    if ((v >> 16) != 0) {
        emit_movk(ctx, blk, rd, (uint16_t)((v >> 16) & 0xffff), 1, /*sf=*/false);
    }
}

// =============================================================================
// Slot map: SSA value -> stack-slot index. Linear-scan, fine at this
// size; will be replaced once we have a real register allocator.
// =============================================================================
typedef struct {
    MLIR_ValueHandle *vs;
    uint16_t         *slot;
    size_t            size;
    size_t            cap;
} SlotMap;

static void sm_set(SlotMap *m, MLIR_ValueHandle v, uint16_t s) {
    if (m->size == m->cap) {
        size_t nc = m->cap ? m->cap * 2 : 16;
        m->vs   = realloc(m->vs,   nc * sizeof(MLIR_ValueHandle));
        m->slot = realloc(m->slot, nc * sizeof(uint16_t));
        m->cap  = nc;
    }
    m->vs[m->size]   = v;
    m->slot[m->size] = s;
    m->size++;
}
static int sm_get(const SlotMap *m, MLIR_ValueHandle v, uint16_t *out) {
    for (size_t i = 0; i < m->size; i++) {
        if (m->vs[i] == v) { *out = m->slot[i]; return 1; }
    }
    return 0;
}
static void sm_free(SlotMap *m) { free(m->vs); free(m->slot); memset(m, 0, sizeof(*m)); }

// =============================================================================
// Per-function lowering.
// =============================================================================
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src) {
    string name     = at_s(src, "sym_name");
    bool   exported = at_b(src, "exported");
    string pt       = at_s(src, "param_types");

    if (MLIR_GetOpNumRegions(src) < 1) {
        fprintf(stderr, "wmir->aarch64: wmir.func has no region\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(src, 0);
    if (MLIR_GetRegionNumBlocks(src_region) < 1) {
        fprintf(stderr, "wmir->aarch64: wmir.func has no entry block\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, 0);

    // Pre-pass: allocate slots for parameters and op results.
    SlotMap sm = {0};
    uint16_t next_slot = 0;
    size_t n_params = MLIR_GetBlockNumArgs(src_blk);
    for (size_t i = 0; i < n_params; i++) {
        MLIR_ValueHandle pa = MLIR_GetBlockArg(src_blk, i);
        sm_set(&sm, pa, next_slot++);
    }
    size_t n_ops = MLIR_GetBlockNumOps(src_blk);
    for (size_t i = 0; i < n_ops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(src_blk, i);
        size_t nr = MLIR_GetOpNumResults(op);
        for (size_t k = 0; k < nr; k++) {
            sm_set(&sm, MLIR_GetOpResult(op, k), next_slot++);
        }
    }
    // 8 bytes per slot; round frame up to 16 for sp alignment.
    uint32_t frame_size = (uint32_t)next_slot * 8u;
    frame_size = (frame_size + 15u) & ~15u;
    if (frame_size > 0xfff) {
        fprintf(stderr,
            "wmir->aarch64: function '%.*s' frame size %u exceeds imm12 budget\n",
            (int)name.size, name.str, frame_size);
        sm_free(&sm);
        return MLIR_INVALID_HANDLE;
    }

    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);

    // Prologue + spill params.
    emit_prologue(ctx, blk, (uint16_t)frame_size);
    for (size_t i = 0; i < n_params; i++) {
        uint16_t s; sm_get(&sm, MLIR_GetBlockArg(src_blk, i), &s);
        emit_str_w(ctx, blk, /*rt=*/(uint8_t)i, /*rn=*/31 /*SP*/,
                   (uint16_t)(s * 8u));
    }

    // Helper macros to keep the per-op code below readable.
    #define LD_OPERAND(REG, IDX)                                            \
        do {                                                                \
            MLIR_ValueHandle _v = MLIR_GetOpOperand(op, (IDX));             \
            uint16_t _s;                                                    \
            if (!sm_get(&sm, _v, &_s)) {                                    \
                fprintf(stderr,                                             \
                    "wmir->aarch64: operand %zu unbound (op idx %zu)\n",    \
                    (size_t)(IDX), i);                                      \
                sm_free(&sm); return MLIR_INVALID_HANDLE;                   \
            }                                                               \
            emit_ldr_w(ctx, blk, (REG), 31, (uint16_t)(_s * 8u));           \
        } while (0)
    #define ST_RESULT(REG, IDX)                                             \
        do {                                                                \
            MLIR_ValueHandle _v = MLIR_GetOpResult(op, (IDX));              \
            uint16_t _s;                                                    \
            sm_get(&sm, _v, &_s);                                           \
            emit_str_w(ctx, blk, (REG), 31, (uint16_t)(_s * 8u));           \
        } while (0)

    for (size_t i = 0; i < n_ops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(src_blk, i);
        MLIR_OpType   t  = MLIR_GetOpType(op);
        switch (t) {
        case OP_TYPE_WMIR_CONST: {
            // Only emit code for i32 constants; skip unused i64 stragglers.
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            MLIR_TypeHandle  ty = MLIR_GetValueType(r);
            string ts = MLIR_GetTypeString(ctx, ty);
            if (!(ts.size == 3 && memcmp(ts.str, "i32", 3) == 0)) {
                if (ts.size == 3 && memcmp(ts.str, "i64", 3) == 0) continue;
                fprintf(stderr,
                    "wmir->aarch64: wmir.const of unsupported type '%.*s'\n",
                    (int)ts.size, ts.str);
                sm_free(&sm); return MLIR_INVALID_HANDLE;
            }
            int64_t v = at_i(op, "value");
            emit_mov_imm32(ctx, blk, 9, (uint32_t)v);
            ST_RESULT(9, 0);
            break;
        }
        case OP_TYPE_WMIR_IADD: {
            LD_OPERAND(9, 0);
            LD_OPERAND(10, 1);
            emit_add_reg(ctx, blk, 9, 9, 10, /*sf=*/false);
            ST_RESULT(9, 0);
            break;
        }
        case OP_TYPE_WMIR_ISUB: {
            LD_OPERAND(9, 0);
            LD_OPERAND(10, 1);
            emit_sub_reg(ctx, blk, 9, 9, 10, /*sf=*/false);
            ST_RESULT(9, 0);
            break;
        }
        case OP_TYPE_WMIR_GLOBAL_GET: {
            int64_t gi = at_i(op, "global_idx");
            // ldr w9, [x27, #(gi * 8)]
            emit_ldr_w(ctx, blk, 9, 27, (uint16_t)(gi * 8));
            ST_RESULT(9, 0);
            break;
        }
        case OP_TYPE_WMIR_GLOBAL_SET: {
            int64_t gi = at_i(op, "global_idx");
            LD_OPERAND(9, 0);
            emit_str_w(ctx, blk, 9, 27, (uint16_t)(gi * 8));
            break;
        }
        case OP_TYPE_WMIR_LOAD: {
            int64_t off = at_i(op, "memory_offset");
            // w9 := addr; x10 := x28 + zext(w9); w9 := *(x10 + off)
            LD_OPERAND(9, 0);
            emit_add_reg(ctx, blk, 10, 28, 9, /*sf=*/true);
            // (the encoder treats add_reg with sf=1 as a 64-bit shifted-reg
            // add of the 64-bit form of w9 — fine here because globals
            // wrote a fresh 32-bit value that the W-form load already
            // zero-extended into x9.)
            emit_ldr_w(ctx, blk, 9, 10, (uint16_t)off);
            ST_RESULT(9, 0);
            break;
        }
        case OP_TYPE_WMIR_STORE: {
            int64_t off = at_i(op, "memory_offset");
            LD_OPERAND(9, 0);  // address
            LD_OPERAND(11, 1); // value
            emit_add_reg(ctx, blk, 10, 28, 9, /*sf=*/true);
            emit_str_w(ctx, blk, 11, 10, (uint16_t)off);
            break;
        }
        case OP_TYPE_WMIR_RETURN: {
            size_t no = MLIR_GetOpNumOperands(op);
            if (no > 1) {
                fprintf(stderr,
                    "wmir->aarch64: wmir.return with %zu results not yet "
                    "supported\n", no);
                sm_free(&sm); return MLIR_INVALID_HANDLE;
            }
            if (no == 1) LD_OPERAND(0, 0);
            emit_epilogue(ctx, blk, (uint16_t)frame_size);
            emit_ret(ctx, blk);
            break;
        }
        case OP_TYPE_WMIR_CALL: {
            // Marshal args into w0, w1, …, wN. Caller-saved regs are
            // not preserved across calls, but every wmir SSA value
            // lives in a stack slot — nothing is in flight in a reg
            // at this point, so we just emit loads → bl → store.
            size_t na = MLIR_GetOpNumOperands(op);
            if (na > 8) {
                fprintf(stderr,
                    "wmir->aarch64: wmir.call with %zu args (>8) not "
                    "yet supported\n", na);
                sm_free(&sm); return MLIR_INVALID_HANDLE;
            }
            for (size_t k = 0; k < na; k++) LD_OPERAND((uint8_t)k, k);
            string callee = at_s(op, "callee");
            emit_bl(ctx, blk, callee);
            if (MLIR_GetOpNumResults(op) > 0) ST_RESULT(0, 0);
            break;
        }
        default: {
            string nm = MLIR_GetOpName(op);
            fprintf(stderr,
                "wmir->aarch64: unsupported wmir op '%.*s' (kind=%d)\n",
                (int)nm.size, nm.str, (int)t);
            sm_free(&sm); return MLIR_INVALID_HANDLE;
        }
        }
    }
    #undef LD_OPERAND
    #undef ST_RESULT

    sm_free(&sm);

    MLIR_AttributeHandle attrs[4];
    size_t naf = 0;
    attrs[naf++] = attr_s(ctx, "sym_name", name.str, name.size);
    attrs[naf++] = attr_b(ctx, "exported", exported);
    attrs[naf++] = attr_s(ctx, "param_types", pt.str, pt.size);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, blk);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, naf, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// _start synthesis. Sets up x26/x27/x28 then calls main, then svc-exits.
// =============================================================================
static MLIR_OpHandle synth_start(MLIR_Context *ctx, string main_name,
                                 bool use_data_priv, bool use_globals,
                                 bool use_linmem) {
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);

    if (use_data_priv) {
        emit_adrp_data(ctx, blk, 26, "data_priv");
        emit_add_data_lo(ctx, blk, 26, 26, "data_priv");
    }
    if (use_globals) {
        emit_adrp_data(ctx, blk, 27, "globals");
        emit_add_data_lo(ctx, blk, 27, 27, "globals");
    }
    if (use_linmem) {
        emit_adrp_data(ctx, blk, 28, "linmem");
        emit_add_data_lo(ctx, blk, 28, 28, "linmem");
    }

    emit_bl(ctx, blk, main_name);
    // mach trap #1 = exit; movz w16,#1 zero-extends to x16, fine for syscall.
    emit_movz(ctx, blk, /*rd=*/16, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_svc(ctx, blk, 0x80);

    MLIR_AttributeHandle attrs[2];
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
// Walk the module to compute the highest global_idx referenced and
// whether linmem is touched. This drives __DATA sizing in the Mach-O
// backend.
// =============================================================================
typedef struct {
    int32_t n_globals;       // max global_idx + 1, capped at >=1 if any g* op seen
    bool    uses_linmem;
} ModInfo;

static void scan_block(MLIR_BlockHandle blk, ModInfo *mi) {
    size_t n = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        MLIR_OpType t = MLIR_GetOpType(op);
        if (t == OP_TYPE_WMIR_GLOBAL_GET || t == OP_TYPE_WMIR_GLOBAL_SET) {
            int32_t gi = (int32_t)at_i(op, "global_idx");
            if (gi + 1 > mi->n_globals) mi->n_globals = gi + 1;
        } else if (t == OP_TYPE_WMIR_LOAD || t == OP_TYPE_WMIR_STORE) {
            mi->uses_linmem = true;
        }
        // Recurse into regions (no nested regions in the current
        // dialect, but the loop is cheap).
        size_t nr = MLIR_GetOpNumRegions(op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_RegionHandle rh = MLIR_GetOpRegion(op, r);
            size_t nb = MLIR_GetRegionNumBlocks(rh);
            for (size_t b = 0; b < nb; b++)
                scan_block(MLIR_GetRegionBlock(rh, b), mi);
        }
    }
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

    // Scan for resource usage to size __DATA.
    ModInfo mi = {0};
    scan_block(mb, &mi);

    // Compute layout numbers that mirror what wasm-link produces.
    // n_globals is at least 1 if any global op was seen (the wasm-link
    // implicit __stack_pointer global lives at idx 0). If no globals
    // are touched at all (e.g. macho_exit), we still want zero so the
    // Mach-O backend can elide the section.
    uint32_t n_globals = (uint32_t)mi.n_globals;
    // Initial value for global 0 = STACK_SIZE; remaining globals = 0
    // (sufficient for the macho_arith level of complexity; will need
    // to thread real init values once the wmir lowering surfaces
    // module-level wasmssa.global ops).
    uint64_t global0_init = DEFAULT_STACK_SIZE;
    // linmem_size: stack + GLOBAL_BASE_OFFSET, rounded up to a wasm page.
    uint64_t lm_bytes = (uint64_t)DEFAULT_STACK_SIZE + DEFAULT_GLOBAL_BASE_OFFS;
    uint64_t lm_pages = (lm_bytes + WASM_PAGE_SIZE - 1) / WASM_PAGE_SIZE;
    uint64_t linmem_size = lm_pages * WASM_PAGE_SIZE;

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };

    // Module-level attributes carrying __DATA-layout metadata.
    MLIR_AttributeHandle mattrs[4];
    size_t nma = 0;
    mattrs[nma++] = attr_i32(ctx, "n_globals",     (int64_t)n_globals);
    mattrs[nma++] = attr_i64(ctx, "global0_init",  (int64_t)global0_init);
    mattrs[nma++] = attr_i64(ctx, "linmem_size",   (int64_t)linmem_size);
    mattrs[nma++] = attr_i32(ctx, "linmem_pages",  (int64_t)lm_pages);

    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        mattrs, nma, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

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

    // _start sets up only the data-base regs we actually need.
    bool use_globals = n_globals > 0;
    bool use_linmem  = mi.uses_linmem;
    MLIR_OpHandle start = synth_start(ctx, entry_name,
        /*use_data_priv=*/false, use_globals, use_linmem);
    if (!start) return MLIR_INVALID_HANDLE;
    MLIR_AppendBlockOp(ctx, out_body, start);

    return out_module;
}
