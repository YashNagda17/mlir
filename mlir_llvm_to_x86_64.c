// llvm dialect -> x86_64 lowering. See mlir_llvm_to_x86_64.h.
//
// Status: first-light integer codegen for static Linux ELF. Lowers single-block
// `llvm.func` bodies of integer ops to the physical-register `x86_64` dialect,
// then `mlir_x86_64_to_elf` encodes the result. A synthesised `_start` calls
// `main` and exits via SYS_exit.
//
// Supported for the macho_* ELF smoke tests: parameters (<=6 integer args),
// llvm.mlir.constant, integer binops, alloca/load/store (i32/i64), icmp,
// select, sext/zext/trunc, direct calls (<=6 args), multi-block CFG
// (llvm.br / llvm.cond_br), and synthesised Linux syscall helpers (_write).

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_llvm_to_x86_64.h"
#include "mlir_op_names.h"

// Internal register numbers (see x86.md).
#define R_RAX 0
#define R_RCX 1
#define R_RDX 2
#define R_RBP 5
#define R_RDI 7
#define R_R10 10
#define R_R11 11
#define R_R12 12

static const uint8_t k_arg_regs[6] = { R_RDI, 6, R_RDX, R_RCX, 8, 9 };

// ---------------------------------------------------------------------------
// Attribute / op builders.
// ---------------------------------------------------------------------------
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name,
                                     int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name,
                                     int64_t v) {
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

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static void emit_mov_ri(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, int64_t imm) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i64(ctx, "imm", imm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_MOV_RI, a, 2));
}
static void emit_mov_rr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_MOV_RR, a, 2));
}
static void emit_mov_rm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t base, int32_t disp, uint8_t width) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "base", base);
    a[2] = attr_i32(ctx, "disp", disp);
    a[3] = attr_i32(ctx, "width", width);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_MOV_RM, a, 4));
}
static void emit_mov_mr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t base, int32_t disp, uint8_t rs, uint8_t width) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "base", base);
    a[1] = attr_i32(ctx, "disp", disp);
    a[2] = attr_i32(ctx, "rs", rs);
    a[3] = attr_i32(ctx, "width", width);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_MOV_MR, a, 4));
}
static void emit_add_rr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_ADD_RR, a, 3));
}
static void emit_sub_rr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_SUB_RR, a, 3));
}
static void emit_imul_rr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_IMUL_RR, a, 3));
}
static void emit_shl_ri(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t imm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "imm", imm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_SHL_RI, a, 3));
}
static void emit_sar_ri(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t imm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "imm", imm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_SAR_RI, a, 3));
}
static void emit_call(MLIR_Context *ctx, MLIR_BlockHandle blk, string callee) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_s(ctx, "callee", callee.str, callee.size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_CALL, a, 1));
}
static void emit_ret(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_RET, NULL, 0));
}
static void emit_syscall(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_SYSCALL, NULL, 0));
}
static void emit_prologue(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t fs) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)fs);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_PROLOGUE, a, 1));
}
static void emit_epilogue(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t fs) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)fs);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_EPILOGUE, a, 1));
}
static void emit_cmp_rr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rn, uint8_t rm, uint8_t width) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rn", rn);
    a[1] = attr_i32(ctx, "rm", rm);
    a[2] = attr_i32(ctx, "width", width);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_CMP_RR, a, 3));
}
static void emit_cmp_ri(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rn, int64_t imm, uint8_t width) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rn", rn);
    a[1] = attr_i64(ctx, "imm", imm);
    a[2] = attr_i32(ctx, "width", width);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_CMP_RI, a, 3));
}
static void emit_and_rr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_AND_RR, a, 3));
}
static void emit_xor_rr(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_XOR_RR, a, 3));
}
static void emit_cqo(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_CQO, NULL, 0));
}
static void emit_idiv_r(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t rm, bool unsigned_div) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "unsigned_div", unsigned_div);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_IDIV_R, a, 4));
}
static void emit_setcc(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rd, uint8_t cond) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "cond", cond);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_SETCC, a, 2));
}
static void emit_cmovcc(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_i32(ctx, "cond", cond);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_X86_64_CMOVCC, a, 4));
}
static MLIR_OpHandle build_branch_op(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_BlockHandle target) {
    MLIR_BlockHandle succs[1] = { target };
    MLIR_ValueHandle *succ_ops[1] = { NULL };
    size_t n_succ_ops[1] = { 0 };
    return MLIR_CreateOpWithSuccessors(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        succs, 1, succ_ops, n_succ_ops,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}
static void emit_jmp(MLIR_Context *ctx, MLIR_BlockHandle blk,
                     MLIR_BlockHandle target) {
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_X86_64_JMP, NULL, 0, target));
}
static void emit_jcc(MLIR_Context *ctx, MLIR_BlockHandle blk,
                     uint8_t cond, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "cond", cond);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_X86_64_JCC, a, 1, target));
}

// ---------------------------------------------------------------------------
// LLVM dialect helpers.

MLIR_OpHandle mlir_llvm_to_x86_64(MLIR_Context *ctx,
                                  MLIR_OpHandle llvm_module) {
    (void)ctx;
    (void)llvm_module;
    fprintf(stderr, "llvm->x86_64: backend skeleton is not implemented yet\n");
    return MLIR_INVALID_HANDLE;
}

LlvmX86SelState *mlir_llvm_x86_sel_begin(MLIR_Context *ctx,
                                         MLIR_OpHandle llvm_module,
                                         uint8_t **out_gblob,
                                         uint32_t *out_gblob_len) {
    (void)ctx;
    (void)llvm_module;
    if (out_gblob) *out_gblob = NULL;
    if (out_gblob_len) *out_gblob_len = 0;
    fprintf(stderr, "llvm->x86_64: streaming selector skeleton is not implemented yet\n");
    return NULL;
}

size_t mlir_llvm_x86_sel_num_funcs(LlvmX86SelState *st) {
    (void)st;
    return 0;
}

MLIR_OpHandle mlir_llvm_x86_sel_synth_start(MLIR_Context *ctx,
                                            LlvmX86SelState *st) {
    (void)ctx;
    (void)st;
    return MLIR_INVALID_HANDLE;
}

MLIR_OpHandle mlir_llvm_x86_sel_func(MLIR_Context *ctx,
                                     LlvmX86SelState *st,
                                     size_t idx) {
    (void)ctx;
    (void)st;
    (void)idx;
    return MLIR_INVALID_HANDLE;
}

bool mlir_llvm_x86_sel_saw_main(LlvmX86SelState *st) {
    (void)st;
    return false;
}

void mlir_llvm_x86_sel_end(LlvmX86SelState *st) {
    (void)st;
}
