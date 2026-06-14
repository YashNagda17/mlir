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
// ---------------------------------------------------------------------------
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static unsigned x64_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return 8;
    if (name_eq(s, "ptr") || name_eq(s, "!llvm.ptr")) return 8;
    if (name_eq(s, "f32")) return 4;
    if (name_eq(s, "f64")) return 8;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] < '0' || s.str[i] > '9') { w = -1; break; }
            w = w * 10 + (s.str[i] - '0');
        }
        if (w == 1 || w == 8) return 1;
        if (w == 16) return 2;
        if (w == 32) return 4;
        if (w == 64) return 8;
    }
    if (MLIR_IsTypeLLVMArray(ty)) {
        unsigned esz = x64_type_size(ctx, MLIR_GetTypeLLVMArrayElement(ty));
        if (esz == 0) return 0;
        return esz * (unsigned)MLIR_GetTypeLLVMArrayNumElements(ty);
    }
    return 0;
}

static int int_type_bits(MLIR_Context *ctx, MLIR_ValueHandle v) {
    string s = MLIR_GetTypeString(ctx, MLIR_GetValueType(v));
    if (s.size < 2 || s.str[0] != 'i') return 0;
    int w = 0;
    for (size_t i = 1; i < s.size; i++) {
        if (s.str[i] < '0' || s.str[i] > '9') return 0;
        w = w * 10 + (s.str[i] - '0');
    }
    return w;
}

static bool type_is_gp64(MLIR_Context *ctx, MLIR_ValueHandle v) {
    return x64_type_size(ctx, MLIR_GetValueType(v)) == 8;
}

static bool func_has_body(MLIR_OpHandle fn) {
    return MLIR_GetOpNumRegions(fn) > 0 &&
           MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;
}

static size_t parse_i32_array(string cs, int32_t *out, size_t cap) {
    size_t p = 0, got = 0;
    while (p < cs.size && cs.str[p] != ':') p++;
    if (p < cs.size) p++;
    while (p < cs.size && got < cap) {
        while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
        if (p >= cs.size || cs.str[p] == '>') break;
        int64_t sign = 1;
        if (cs.str[p] == '-') { sign = -1; p++; }
        int64_t v = 0;
        while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9')
            v = v * 10 + (cs.str[p++] - '0');
        out[got++] = (int32_t)(sign * v);
    }
    return got;
}

static bool const_int_val(MLIR_Context *ctx, MLIR_OpHandle op,
                          int64_t *val, uint8_t *is64) {
    if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.constant")) return false;
    if (MLIR_GetOpNumResults(op) != 1) return false;
    MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
    MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
    if (va == MLIR_INVALID_HANDLE) return false;
    *val = MLIR_GetAttributeInteger(va);
    *is64 = type_is_gp64(ctx, res) ? 1 : 0;
    return true;
}

#define X64_FAIL(...) do { fprintf(stderr, __VA_ARGS__); return MLIR_INVALID_HANDLE; } while (0)

// Map LLVM icmp predicate to x86 SETCC/JCC condition (0..15).
static int icmp_pred_to_x86_cond(int64_t p) {
    switch (p) {
        case 0: return 4;   // eq  -> ZF
        case 1: return 5;   // ne  -> NZ
        case 2: return 12;  // slt -> NLT
        case 3: return 14;  // sle -> NG
        case 4: return 15;  // sgt -> NGT
        case 5: return 13;  // sge -> NL
        case 6: return 12;  // ult -> LT (unsigned cmp)
        case 7: return 14;  // ule -> LE
        case 8: return 15;  // ugt -> GT
        case 9: return 13;  // uge -> GE
        default: return -1;
    }
}

// ---------------------------------------------------------------------------
// Slot map: every SSA value -> 8-byte frame slot index.
// ---------------------------------------------------------------------------
typedef struct { uintptr_t key; int32_t slot; } SlotEnt;
typedef struct { SlotEnt *t; size_t cap; size_t n; Arena *arena; } SlotMap;

static size_t sm_hash(uintptr_t k) {
    k *= 0x9E3779B97F4A7C15ull;
    return (size_t)(k >> 32);
}
static void sm_grow(SlotMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    SlotEnt *nt = (SlotEnt *)arena_alloc(m->arena, ncap * sizeof(SlotEnt));
    memset(nt, 0, ncap * sizeof(SlotEnt));
    for (size_t i = 0; i < m->cap; i++) {
        if (m->t[i].key == 0) continue;
        size_t j = sm_hash(m->t[i].key) & (ncap - 1);
        while (nt[j].key != 0) j = (j + 1) & (ncap - 1);
        nt[j] = m->t[i];
    }
    m->t = nt;
    m->cap = ncap;
}
static void sm_put(SlotMap *m, MLIR_ValueHandle k, int32_t slot) {
    if ((m->n + 1) * 4 >= m->cap * 3) sm_grow(m);
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) return;
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].slot = slot;
    m->n++;
}
static bool sm_get(SlotMap *m, MLIR_ValueHandle k, int32_t *out) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) { *out = m->t[i].slot; return true; }
        i = (i + 1) & mask;
    }
    return false;
}

static int32_t slot_disp(int32_t slot) {
    return -(int32_t)((slot + 1) * 8);
}

static void load_slot(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                      MLIR_ValueHandle v, uint8_t rd) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return;
    emit_mov_rm(ctx, blk, rd, R_RBP, slot_disp(slot), 8);
}
static bool store_slot(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                       MLIR_ValueHandle v, uint8_t rs) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return false;
    emit_mov_mr(ctx, blk, R_RBP, slot_disp(slot), rs, 8);
    return true;
}

// Constant rematerialization map.
typedef struct { uintptr_t key; int64_t val; uint8_t is64; } ConstEnt;
typedef struct { ConstEnt *t; size_t cap; size_t n; Arena *arena; } ConstMap;

static void cm_put(ConstMap *m, MLIR_ValueHandle k, int64_t val, uint8_t is64) {
    if ((m->n + 1) * 4 >= m->cap * 3) {
        size_t ncap = m->cap ? m->cap * 2 : 64;
        ConstEnt *nt = (ConstEnt *)arena_alloc(m->arena, ncap * sizeof(ConstEnt));
        memset(nt, 0, ncap * sizeof(ConstEnt));
        for (size_t i = 0; i < m->cap; i++) {
            if (m->t[i].key == 0) continue;
            size_t j = sm_hash(m->t[i].key) & (ncap - 1);
            while (nt[j].key != 0) j = (j + 1) & (ncap - 1);
            nt[j] = m->t[i];
        }
        m->t = nt;
        m->cap = ncap;
    }
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) return;
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].val = val;
    m->t[i].is64 = is64;
    m->n++;
}
static bool cm_get(ConstMap *m, MLIR_ValueHandle k, int64_t *val, uint8_t *is64) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) {
            *val = m->t[i].val;
            *is64 = m->t[i].is64;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void build_const_map(MLIR_Context *ctx, ConstMap *cm, MLIR_BlockHandle block) {
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        int64_t v; uint8_t is64;
        if (const_int_val(ctx, op, &v, &is64))
            cm_put(cm, MLIR_GetOpResult(op, 0), v, is64);
    }
}

static void assign_slots_block(MLIR_Context *ctx, SlotMap *sm,
                               MLIR_BlockHandle block, int32_t *nslots) {
    size_t na = MLIR_GetBlockNumArgs(block);
    for (size_t i = 0; i < na; i++)
        sm_put(sm, MLIR_GetBlockArg(block, i), (*nslots)++);
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        int64_t cv; uint8_t c64;
        if (const_int_val(ctx, op, &cv, &c64)) continue;
        size_t nr = MLIR_GetOpNumResults(op);
        for (size_t r = 0; r < nr; r++)
            sm_put(sm, MLIR_GetOpResult(op, r), (*nslots)++);
    }
}

static bool collect_allocas(MLIR_Context *ctx, SlotMap *am,
                            MLIR_BlockHandle block, uint32_t *bytes) {
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        if (name_eq(MLIR_GetOpName(op), "llvm.alloca") &&
            MLIR_GetOpNumResults(op) == 1) {
            MLIR_AttributeHandle eta = MLIR_GetOpAttributeByName(op, "elem_type");
            if (eta == MLIR_INVALID_HANDLE) return false;
            MLIR_TypeHandle et = MLIR_GetAttributeTypeValue(eta);
            unsigned esz = x64_type_size(ctx, et);
            if (esz == 0) return false;
            int64_t cnt = 1;
            if (MLIR_GetOpNumOperands(op) >= 1) {
                MLIR_OpHandle cd = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(op, 0));
                if (cd == MLIR_INVALID_HANDLE ||
                    !name_eq(MLIR_GetOpName(cd), "llvm.mlir.constant")) return false;
                cnt = MLIR_GetAttributeInteger(
                          MLIR_GetOpAttributeByName(cd, "value"));
            }
            unsigned al = esz < 8 ? 8u : esz;
            *bytes = (*bytes + al - 1u) & ~(al - 1u);
            sm_put(am, MLIR_GetOpResult(op, 0), (int32_t)*bytes);
            *bytes += (uint32_t)(esz * cnt);
        }
        size_t ng = MLIR_GetOpNumRegions(op);
        for (size_t g = 0; g < ng; g++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, g);
            size_t nbk = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < nbk; b++)
                if (!collect_allocas(ctx, am, MLIR_GetRegionBlock(rg, b), bytes))
                    return false;
        }
    }
    return true;
}

static void build_const_map_region(MLIR_Context *ctx, ConstMap *cm,
                                   MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t b = 0; b < nb; b++)
        build_const_map(ctx, cm, MLIR_GetRegionBlock(region, b));
}

static void assign_slots_region(MLIR_Context *ctx, SlotMap *sm,
                                MLIR_RegionHandle region, int32_t *nslots) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t b = 0; b < nb; b++)
        assign_slots_block(ctx, sm, MLIR_GetRegionBlock(region, b), nslots);
}

static bool collect_allocas_region(MLIR_Context *ctx, SlotMap *am,
                                     MLIR_RegionHandle region, uint32_t *bytes) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t b = 0; b < nb; b++)
        if (!collect_allocas(ctx, am, MLIR_GetRegionBlock(region, b), bytes))
            return false;
    return true;
}

static MLIR_BlockHandle map_block(MLIR_BlockHandle *src, MLIR_BlockHandle *dst,
                                  size_t n, MLIR_BlockHandle s) {
    for (size_t i = 0; i < n; i++)
        if (src[i] == s) return dst[i];
    return MLIR_INVALID_HANDLE;
}

typedef struct {
    MLIR_Context     *ctx;
    SlotMap          *sm;
    ConstMap         *cm;
    SlotMap          *am;
    MLIR_RegionHandle out_reg;
    MLIR_BlockHandle  cur;
    string            sym;
    uint32_t          slot_bytes;
    uint32_t          frame_size;
    bool              ok;
} LowerCtx;

static bool mat_into(LowerCtx *L, MLIR_ValueHandle v, uint8_t dst);

static void copy_slot(LowerCtx *L, MLIR_ValueHandle src, MLIR_ValueHandle dst) {
    int64_t cv; uint8_t c64;
    if (L->cm && cm_get(L->cm, src, &cv, &c64)) {
        emit_mov_ri(L->ctx, L->cur, R_R10, cv);
        if (!store_slot(L->ctx, L->cur, L->sm, dst, R_R10)) {
            fprintf(stderr, "llvm->x86_64: undefined value in copy (%.*s)\n",
                    (int)L->sym.size, L->sym.str);
            L->ok = false;
        }
        return;
    }
    if (!mat_into(L, src, R_R10) ||
        !store_slot(L->ctx, L->cur, L->sm, dst, R_R10)) {
        fprintf(stderr, "llvm->x86_64: undefined value in copy (%.*s)\n",
                (int)L->sym.size, L->sym.str);
        L->ok = false;
    }
}

static void emit_edge_copies(LowerCtx *L, MLIR_OpHandle term, size_t s) {
    MLIR_BlockHandle dst = MLIR_GetOpSuccessor(term, s);
    size_t n = MLIR_GetOpNumSuccessorOperands(term, s);
    for (size_t k = 0; k < n; k++) {
        copy_slot(L, MLIR_GetOpSuccessorOperand(term, s, k),
                  MLIR_GetBlockArg(dst, k));
        if (!L->ok) return;
    }
}

static bool mat_into(LowerCtx *L, MLIR_ValueHandle v, uint8_t dst) {
    int64_t cv; uint8_t c64;
    if (cm_get(L->cm, v, &cv, &c64)) {
        emit_mov_ri(L->ctx, L->cur, dst, cv);
        return true;
    }
    int32_t slot;
    if (sm_get(L->sm, v, &slot)) {
        unsigned w = x64_type_size(L->ctx, MLIR_GetValueType(v));
        if (w == 0) w = 8;
        if (w > 8) w = 8;
        emit_mov_rm(L->ctx, L->cur, dst, R_RBP, slot_disp(slot), (uint8_t)w);
        if (w == 1) {
            emit_mov_ri(L->ctx, L->cur, R_R11, 0xff);
            emit_and_rr(L->ctx, L->cur, dst, dst, R_R11);
        }
        return true;
    }
    return false;
}

static void use_val(LowerCtx *L, MLIR_ValueHandle v, uint8_t scratch) {
    if (!mat_into(L, v, scratch)) {
        fprintf(stderr, "llvm->x86_64: undefined operand in '%.*s'\n",
                (int)L->sym.size, L->sym.str);
        L->ok = false;
    }
}

static void fin_val(LowerCtx *L, MLIR_ValueHandle v, uint8_t produced) {
    if (!store_slot(L->ctx, L->cur, L->sm, v, produced)) L->ok = false;
}

#define LFAIL(...) do { fprintf(stderr, __VA_ARGS__); L->ok = false; return; } while (0)

static void lower_op(LowerCtx *L, MLIR_OpHandle op) {
    MLIR_Context *ctx = L->ctx;
    MLIR_BlockHandle blk = L->cur;
    string on = MLIR_GetOpName(op);

    if (name_eq(on, "llvm.mlir.constant")) return; // remat only

    if (name_eq(on, "llvm.add")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        uint8_t r0 = R_R10, r1 = R_R11;
        use_val(L, MLIR_GetOpOperand(op, 0), r0);
        use_val(L, MLIR_GetOpOperand(op, 1), r1);
        if (!L->ok) return;
        emit_add_rr(ctx, blk, r0, r0, r1);
        fin_val(L, res, r0);
    } else if (name_eq(on, "llvm.sub")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        uint8_t r0 = R_R10, r1 = R_R11;
        use_val(L, MLIR_GetOpOperand(op, 0), r0);
        use_val(L, MLIR_GetOpOperand(op, 1), r1);
        if (!L->ok) return;
        emit_sub_rr(ctx, blk, r0, r0, r1);
        fin_val(L, res, r0);
    } else if (name_eq(on, "llvm.mul")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        uint8_t r0 = R_R10, r1 = R_R11;
        use_val(L, MLIR_GetOpOperand(op, 0), r0);
        use_val(L, MLIR_GetOpOperand(op, 1), r1);
        if (!L->ok) return;
        emit_imul_rr(ctx, blk, r0, r0, r1);
        fin_val(L, res, r0);
    } else if (name_eq(on, "llvm.udiv") || name_eq(on, "llvm.sdiv")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sgn = name_eq(on, "llvm.sdiv");
        use_val(L, MLIR_GetOpOperand(op, 0), R_R10);
        use_val(L, MLIR_GetOpOperand(op, 1), R_R11);
        if (!L->ok) return;
        emit_mov_rr(ctx, blk, R_RAX, R_R10);
        if (sgn)
            emit_cqo(ctx, blk);
        else
            emit_xor_rr(ctx, blk, R_RDX, R_RDX, R_RDX);
        emit_idiv_r(ctx, blk, R_RAX, R_RAX, R_R11, !sgn);
        fin_val(L, res, R_RAX);
    } else if (name_eq(on, "llvm.urem") || name_eq(on, "llvm.srem")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sgn = name_eq(on, "llvm.srem");
        use_val(L, MLIR_GetOpOperand(op, 0), R_R10);
        use_val(L, MLIR_GetOpOperand(op, 1), R_R11);
        if (!L->ok) return;
        emit_mov_rr(ctx, blk, R_RAX, R_R10);
        if (sgn)
            emit_cqo(ctx, blk);
        else
            emit_xor_rr(ctx, blk, R_RDX, R_RDX, R_RDX);
        emit_idiv_r(ctx, blk, R_RAX, R_RAX, R_R11, !sgn);
        fin_val(L, res, R_RDX);
    } else if (name_eq(on, "llvm.getelementptr")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        MLIR_AttributeHandle eta = MLIR_GetOpAttributeByName(op, "elem_type");
        MLIR_AttributeHandle ria = MLIR_GetOpAttributeByName(op, "rawConstantIndices");
        if (eta == MLIR_INVALID_HANDLE || ria == MLIR_INVALID_HANDLE)
            LFAIL("llvm->x86_64: getelementptr missing elem_type/indices\n");
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(eta);
        int32_t cidx[64];
        size_t n_idx = parse_i32_array(
            MLIR_GetAttributeAsString(ctx, ria), cidx, 64);
        if (n_idx == 0)
            LFAIL("llvm->x86_64: getelementptr with no indices\n");
        use_val(L, MLIR_GetOpOperand(op, 0), R_R10);
        if (!L->ok) return;
        size_t op_idx = 1;
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            bool is_dyn = (cidx[i] == (int32_t)0x80000000);
            unsigned stride;
            if (i == 0)
                stride = x64_type_size(ctx, elem_ty);
            else if (MLIR_IsTypeLLVMStruct(cur_ty)) {
                if (is_dyn)
                    LFAIL("llvm->x86_64: dynamic struct gep index\n");
                LFAIL("llvm->x86_64: struct gep not supported\n");
            } else if (MLIR_IsTypeLLVMArray(cur_ty)) {
                cur_ty = MLIR_GetTypeLLVMArrayElement(cur_ty);
                stride = x64_type_size(ctx, cur_ty);
            } else {
                LFAIL("llvm->x86_64: gep into non-aggregate type\n");
            }
            if (stride == 0) continue;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(op))
                    LFAIL("llvm->x86_64: gep dynamic index missing\n");
                use_val(L, MLIR_GetOpOperand(op, op_idx++), R_R11);
                if (!L->ok) return;
                if (stride != 1) {
                    emit_mov_ri(ctx, blk, R_R12, (int64_t)stride);
                    emit_imul_rr(ctx, blk, R_R11, R_R11, R_R12);
                }
                emit_add_rr(ctx, blk, R_R10, R_R10, R_R11);
            } else if (cidx[i] != 0) {
                int64_t off = cidx[i] * (int64_t)stride;
                emit_mov_ri(ctx, blk, R_R11, off);
                emit_add_rr(ctx, blk, R_R10, R_R10, R_R11);
            }
        }
        fin_val(L, res, R_R10);
    } else if (name_eq(on, "llvm.alloca")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int32_t off;
        if (!sm_get(L->am, res, &off))
            LFAIL("llvm->x86_64: alloca without frame offset in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        int32_t addr = (int32_t)L->slot_bytes + off;
        emit_mov_rr(ctx, blk, R_R10, R_RBP);
        if (addr != 0) {
            emit_mov_ri(ctx, blk, R_R11, addr);
            emit_sub_rr(ctx, blk, R_R10, R_R10, R_R11);
        }
        fin_val(L, res, R_R10);
    } else if (name_eq(on, "llvm.load")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        unsigned sz = x64_type_size(ctx, MLIR_GetValueType(res));
        if (sz != 1 && sz != 4 && sz != 8)
            LFAIL("llvm->x86_64: unsupported load size %u in '%.*s'\n",
                  sz, (int)L->sym.size, L->sym.str);
        use_val(L, MLIR_GetOpOperand(op, 0), R_R10);
        if (!L->ok) return;
        emit_mov_rm(ctx, blk, R_R10, R_R10, 0, (uint8_t)sz);
        if (sz == 1) {
            emit_mov_ri(ctx, blk, R_R11, 0xff);
            emit_and_rr(ctx, blk, R_R10, R_R10, R_R11);
        }
        fin_val(L, res, R_R10);
    } else if (name_eq(on, "llvm.store")) {
        MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 1);
        unsigned sz = x64_type_size(ctx, MLIR_GetValueType(val));
        if (sz != 1 && sz != 4 && sz != 8)
            LFAIL("llvm->x86_64: unsupported store size %u in '%.*s'\n",
                  sz, (int)L->sym.size, L->sym.str);
        use_val(L, val, R_R10);
        use_val(L, ptr, R_R11);
        if (!L->ok) return;
        emit_mov_mr(ctx, blk, R_R11, 0, R_R10, (uint8_t)sz);
    } else if (name_eq(on, "llvm.icmp")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE)
            LFAIL("llvm->x86_64: icmp without predicate\n");
        int cond = icmp_pred_to_x86_cond(MLIR_GetAttributeInteger(pa));
        if (cond < 0)
            LFAIL("llvm->x86_64: unsupported icmp predicate\n");
        uint8_t w = (uint8_t)(type_is_gp64(ctx, MLIR_GetOpOperand(op, 0)) ? 8 : 4);
        uint8_t r0 = R_R10;
        use_val(L, MLIR_GetOpOperand(op, 0), r0);
        if (!L->ok) return;
        int64_t cv; uint8_t c64;
        if (cm_get(L->cm, MLIR_GetOpOperand(op, 1), &cv, &c64))
            emit_cmp_ri(ctx, blk, r0, cv, w);
        else {
            use_val(L, MLIR_GetOpOperand(op, 1), R_R11);
            if (!L->ok) return;
            emit_cmp_rr(ctx, blk, r0, R_R11, w);
        }
        emit_mov_ri(ctx, blk, R_R12, 0);
        emit_setcc(ctx, blk, R_R12, (uint8_t)cond);
        fin_val(L, res, R_R12);
    } else if (name_eq(on, "llvm.select")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        use_val(L, MLIR_GetOpOperand(op, 0), R_R10);
        use_val(L, MLIR_GetOpOperand(op, 1), R_R11);
        use_val(L, MLIR_GetOpOperand(op, 2), R_R12);
        if (!L->ok) return;
        emit_cmp_ri(ctx, blk, R_R10, 0, 4);
        emit_cmovcc(ctx, blk, R_R12, R_R12, R_R11, 5); // NE: r12 = r11 if cond
        fin_val(L, res, R_R12);
    } else if (name_eq(on, "llvm.call")) {
        MLIR_AttributeHandle callee = MLIR_GetOpAttributeByName(op, "callee");
        if (callee == MLIR_INVALID_HANDLE)
            LFAIL("llvm->x86_64: indirect call unsupported\n");
        string nm = MLIR_GetAttributeAsString(ctx, callee);
        if (nm.size > 0 && nm.str[0] == '@') { nm.str++; nm.size--; }
        size_t no = MLIR_GetOpNumOperands(op);
        if (no > 6)
            LFAIL("llvm->x86_64: >6 call args unsupported in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        for (size_t k = 0; k < no; k++)
            use_val(L, MLIR_GetOpOperand(op, k), k_arg_regs[k]);
        if (!L->ok) return;
        emit_call(ctx, blk, nm);
    } else if (name_eq(on, "llvm.sext")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int sw = int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        int32_t slot;
        if (sm_get(L->sm, MLIR_GetOpOperand(op, 0), &slot))
            emit_mov_rm(ctx, blk, R_R10, R_RBP, slot_disp(slot), 4);
        else if (!mat_into(L, MLIR_GetOpOperand(op, 0), R_R10))
            LFAIL("llvm->x86_64: undefined sext operand\n");
        if (!L->ok) return;
        if (sw == 32) {
            emit_shl_ri(ctx, blk, R_R10, R_R10, 32);
            emit_sar_ri(ctx, blk, R_R10, R_R10, 32);
        } else if (sw > 0 && sw < 64) {
            uint8_t sh = (uint8_t)(64 - sw);
            emit_shl_ri(ctx, blk, R_R10, R_R10, sh);
            emit_sar_ri(ctx, blk, R_R10, R_R10, sh);
        }
        fin_val(L, res, R_R10);
    } else if (name_eq(on, "llvm.zext") || name_eq(on, "llvm.trunc")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int w = name_eq(on, "llvm.trunc")
              ? int_type_bits(ctx, res)
              : int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        use_val(L, MLIR_GetOpOperand(op, 0), R_R10);
        if (!L->ok) return;
        if (w > 0 && w < 64) {
            int64_t mask = (w >= 63) ? -1 : ((1LL << w) - 1);
            emit_mov_ri(ctx, blk, R_R11, mask);
            emit_and_rr(ctx, blk, R_R10, R_R10, R_R11);
        }
        fin_val(L, res, R_R10);
    } else if (name_eq(on, "llvm.return")) {
        /* handled by caller */
    } else {
        LFAIL("llvm->x86_64: unsupported op '%.*s' in '%.*s'\n",
              (int)on.size, on.str, (int)L->sym.size, L->sym.str);
    }
}

static MLIR_OpHandle finish_x86_func(MLIR_Context *ctx, MLIR_RegionHandle out_reg,
                                     string sym, bool exported) {
    MLIR_AttributeHandle attrs[2];
    size_t na = 1;
    attrs[0] = attr_s(ctx, "sym_name", sym.str, sym.size);
    if (exported) {
        attrs[1] = attr_b(ctx, "exported", true);
        na = 2;
    }
    MLIR_RegionHandle regs[1] = { out_reg };
    return MLIR_CreateOp(ctx, OP_TYPE_X86_64_FUNC,
        op_type_to_string(OP_TYPE_X86_64_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static MLIR_OpHandle select_func_cfg(MLIR_Context *ctx, MLIR_OpHandle fn, string sym) {
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(fn, 0);
    size_t n_blocks = MLIR_GetRegionNumBlocks(src_region);
    MLIR_BlockHandle entry_src = MLIR_GetRegionBlock(src_region, 0);
    size_t nargs = MLIR_GetBlockNumArgs(entry_src);
    if (nargs > 6) {
        X64_FAIL("llvm->x86_64: function '%.*s' has %zu parameters (>6 not supported)\n",
                 (int)sym.size, sym.str, nargs);
    }

    SlotMap sm = {0};
    sm.arena = MLIR_GetArenaAllocator(ctx);
    int32_t nslots = 0;
    assign_slots_region(ctx, &sm, src_region, &nslots);
    uint32_t slot_bytes = (uint32_t)nslots * 8u;

    SlotMap am = {0};
    am.arena = MLIR_GetArenaAllocator(ctx);
    uint32_t alloca_bytes = 0;
    if (!collect_allocas_region(ctx, &am, src_region, &alloca_bytes)) {
        X64_FAIL("llvm->x86_64: unsupported alloca in '%.*s'\n",
                 (int)sym.size, sym.str);
    }
    uint32_t frame_size = (slot_bytes + alloca_bytes + 15u) & ~15u;

    MLIR_RegionHandle out_reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle *src_blks = (MLIR_BlockHandle *)malloc(
        n_blocks * sizeof(MLIR_BlockHandle));
    MLIR_BlockHandle *out_blks = (MLIR_BlockHandle *)malloc(
        n_blocks * sizeof(MLIR_BlockHandle));
    if (!src_blks || !out_blks) {
        free(src_blks);
        free(out_blks);
        X64_FAIL("llvm->x86_64: allocation failed\n");
    }
    for (size_t b = 0; b < n_blocks; b++) {
        src_blks[b] = MLIR_GetRegionBlock(src_region, b);
        out_blks[b] = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, out_reg, out_blks[b]);
    }

    ConstMap cm = {0};
    cm.arena = MLIR_GetArenaAllocator(ctx);
    build_const_map_region(ctx, &cm, src_region);

    LowerCtx L = { ctx, &sm, &cm, &am, out_reg, out_blks[0], sym,
                   slot_bytes, frame_size, true };

    emit_prologue(ctx, out_blks[0], frame_size);
    for (size_t i = 0; i < nargs; i++)
        store_slot(ctx, out_blks[0], &sm, MLIR_GetBlockArg(entry_src, i),
                   k_arg_regs[i]);

    for (size_t b = 0; b < n_blocks; b++) {
        L.cur = out_blks[b];
        MLIR_BlockHandle sb = src_blks[b];
        size_t no = MLIR_GetBlockNumOps(sb);
        if (no == 0) {
            fprintf(stderr, "llvm->x86_64: empty block in '%.*s'\n",
                    (int)sym.size, sym.str);
            free(src_blks);
            free(out_blks);
            return MLIR_INVALID_HANDLE;
        }
        for (size_t i = 0; i + 1 < no; i++) {
            lower_op(&L, MLIR_GetBlockOp(sb, i));
            if (!L.ok) {
                free(src_blks);
                free(out_blks);
                return MLIR_INVALID_HANDLE;
            }
        }
        MLIR_OpHandle term = MLIR_GetBlockOp(sb, no - 1);
        string tn = MLIR_GetOpName(term);
        if (name_eq(tn, "llvm.return")) {
            size_t nr = MLIR_GetOpNumOperands(term);
            if (nr == 1) {
                if (!mat_into(&L, MLIR_GetOpOperand(term, 0), R_RAX)) {
                    fprintf(stderr, "llvm->x86_64: undefined return value in '%.*s'\n",
                            (int)sym.size, sym.str);
                    free(src_blks);
                    free(out_blks);
                    return MLIR_INVALID_HANDLE;
                }
            } else if (nr > 1) {
                fprintf(stderr, "llvm->x86_64: multi-value return in '%.*s'\n",
                        (int)sym.size, sym.str);
                free(src_blks);
                free(out_blks);
                return MLIR_INVALID_HANDLE;
            }
            emit_epilogue(ctx, L.cur, frame_size);
            emit_ret(ctx, L.cur);
        } else if (name_eq(tn, "llvm.br") || name_eq(tn, "cf.br")) {
            MLIR_BlockHandle real_d = map_block(src_blks, out_blks, n_blocks,
                                                MLIR_GetOpSuccessor(term, 0));
            size_t nso = MLIR_GetOpNumSuccessorOperands(term, 0);
            if (nso) {
                MLIR_BlockHandle land = MLIR_CreateBlock(ctx);
                MLIR_AppendRegionBlock(ctx, out_reg, land);
                emit_jmp(ctx, L.cur, land);
                L.cur = land;
                emit_edge_copies(&L, term, 0);
                if (!L.ok) {
                    free(src_blks);
                    free(out_blks);
                    return MLIR_INVALID_HANDLE;
                }
                emit_jmp(ctx, land, real_d);
            } else {
                emit_jmp(ctx, L.cur, real_d);
            }
        } else if (name_eq(tn, "llvm.cond_br") || name_eq(tn, "cf.cond_br")) {
            if (MLIR_GetOpNumOperands(term) < 1) {
                fprintf(stderr, "llvm->x86_64: cond_br without condition in '%.*s'\n",
                        (int)sym.size, sym.str);
                free(src_blks);
                free(out_blks);
                return MLIR_INVALID_HANDLE;
            }
            use_val(&L, MLIR_GetOpOperand(term, 0), R_R10);
            if (!L.ok) {
                free(src_blks);
                free(out_blks);
                return MLIR_INVALID_HANDLE;
            }
            emit_cmp_ri(ctx, L.cur, R_R10, 0, 4);
            MLIR_BlockHandle real_t = map_block(src_blks, out_blks, n_blocks,
                                                MLIR_GetOpSuccessor(term, 0));
            MLIR_BlockHandle real_f = map_block(src_blks, out_blks, n_blocks,
                                                MLIR_GetOpSuccessor(term, 1));
            size_t nso_t = MLIR_GetOpNumSuccessorOperands(term, 0);
            size_t nso_f = MLIR_GetOpNumSuccessorOperands(term, 1);
            MLIR_BlockHandle br_t = real_t, br_f = real_f;
            MLIR_BlockHandle cur = L.cur;
            if (nso_t) {
                br_t = MLIR_CreateBlock(ctx);
                MLIR_AppendRegionBlock(ctx, out_reg, br_t);
            }
            if (nso_f) {
                br_f = MLIR_CreateBlock(ctx);
                MLIR_AppendRegionBlock(ctx, out_reg, br_f);
            }
            emit_jcc(ctx, cur, 5, br_t); // NE
            emit_jmp(ctx, cur, br_f);
            if (nso_t) {
                L.cur = br_t;
                emit_edge_copies(&L, term, 0);
                if (!L.ok) {
                    free(src_blks);
                    free(out_blks);
                    return MLIR_INVALID_HANDLE;
                }
                emit_jmp(ctx, br_t, real_t);
            }
            if (nso_f) {
                L.cur = br_f;
                emit_edge_copies(&L, term, 1);
                if (!L.ok) {
                    free(src_blks);
                    free(out_blks);
                    return MLIR_INVALID_HANDLE;
                }
                emit_jmp(ctx, br_f, real_f);
            }
        } else {
            fprintf(stderr, "llvm->x86_64: block in '%.*s' ends in '%.*s'\n",
                    (int)sym.size, sym.str, (int)tn.size, tn.str);
            free(src_blks);
            free(out_blks);
            return MLIR_INVALID_HANDLE;
        }
    }

    free(src_blks);
    free(out_blks);
    return finish_x86_func(ctx, out_reg, sym, false);
}

static MLIR_OpHandle select_func(MLIR_Context *ctx, MLIR_OpHandle fn, string sym) {
    return select_func_cfg(ctx, fn, sym);
}

static MLIR_OpHandle synth_start(MLIR_Context *ctx, string main_name) {
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, reg, blk);

    emit_call(ctx, blk, main_name);
    emit_mov_rr(ctx, blk, R_RDI, R_RAX);
    emit_mov_ri(ctx, blk, R_RAX, 60); // SYS_exit
    emit_syscall(ctx, blk);

    return finish_x86_func(ctx, reg, str_lit("_start"), true);
}

static MLIR_OpHandle synth_write(MLIR_Context *ctx) {
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, reg, blk);

    // SysV AMD64: fd=rdi, buf=rsi, count=rdx already in place.
    emit_mov_ri(ctx, blk, R_RAX, 1); // SYS_write
    emit_syscall(ctx, blk);
    emit_ret(ctx, blk);

    return finish_x86_func(ctx, reg, str_lit("_write"), false);
}

static bool sym_eq(string a, string b) {
    return a.size == b.size && memcmp(a.str, b.str, a.size) == 0;
}

static bool sym_list_has(string *syms, size_t n, string sym) {
    for (size_t i = 0; i < n; i++)
        if (sym_eq(syms[i], sym)) return true;
    return false;
}

static void sym_list_add(string **syms, size_t *n, size_t *cap, string sym) {
    if (sym_list_has(*syms, *n, sym)) return;
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 8;
        *syms = (string *)realloc(*syms, *cap * sizeof(string));
    }
    (*syms)[(*n)++] = sym;
}

static void collect_calls_region(MLIR_Context *ctx, MLIR_RegionHandle region,
                                string **syms, size_t *n, size_t *cap) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(region, b);
        size_t no = MLIR_GetBlockNumOps(blk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle o = MLIR_GetBlockOp(blk, i);
            if (!name_eq(MLIR_GetOpName(o), "llvm.call")) continue;
            MLIR_AttributeHandle ca = MLIR_GetOpAttributeByName(o, "callee");
            if (ca == MLIR_INVALID_HANDLE) continue;
            string nm = MLIR_GetAttributeAsString(ctx, ca);
            if (nm.size > 0 && nm.str[0] == '@') { nm.str++; nm.size--; }
            sym_list_add(syms, n, cap, nm);
        }
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle o = MLIR_GetBlockOp(blk, i);
            size_t ng = MLIR_GetOpNumRegions(o);
            for (size_t g = 0; g < ng; g++)
                collect_calls_region(ctx, MLIR_GetOpRegion(o, g), syms, n, cap);
        }
    }
}

static MLIR_OpHandle find_llvm_func(MLIR_BlockHandle mb, size_t nops, string sym) {
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) continue;
        if (sym_eq(MLIR_GetAttributeString(sa), sym)) return op;
    }
    return MLIR_INVALID_HANDLE;
}

static bool func_is_reachable(MLIR_Context *ctx, MLIR_BlockHandle mb, size_t nops,
                              string sym) {
    if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) return true;
    string *queue = NULL;
    size_t nq = 0, cq = 0, head = 0;
    sym_list_add(&queue, &nq, &cq, str_lit("main"));
    while (head < nq) {
        string cur = queue[head++];
        if (sym_eq(cur, sym)) {
            free(queue);
            return true;
        }
        MLIR_OpHandle fn = find_llvm_func(mb, nops, cur);
        if (fn == MLIR_INVALID_HANDLE || !func_has_body(fn)) continue;
        collect_calls_region(ctx, MLIR_GetOpRegion(fn, 0), &queue, &nq, &cq);
    }
    free(queue);
    return false;
}

MLIR_OpHandle mlir_llvm_to_x86_64(MLIR_Context *ctx, MLIR_OpHandle llvm_module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);

    MLIR_AppendBlockOp(ctx, out_body, synth_start(ctx, str_lit("main")));
    MLIR_AppendBlockOp(ctx, out_body, synth_write(ctx));

    bool saw_main = false;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        if (!func_has_body(op)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) {
            X64_FAIL("llvm->x86_64: llvm.func without sym_name\n");
        }
        string sym = MLIR_GetAttributeString(sa);
        if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) saw_main = true;
        if (!func_is_reachable(ctx, mb, nops, sym)) continue;
        MLIR_OpHandle fn = select_func(ctx, op, sym);
        if (fn == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, fn);
    }

    if (!saw_main) {
        fprintf(stderr, "llvm->x86_64: no defined 'main' function\n");
        return MLIR_INVALID_HANDLE;
    }

    MLIR_RegionHandle regs[1] = { out_region };
    return MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// ---------------------------------------------------------------------------
// Streaming selection API (mirrors AArch64; used by mlir_llvm_to_elf).
// ---------------------------------------------------------------------------
struct LlvmX86SelState {
    MLIR_BlockHandle mb;
    MLIR_OpHandle   *funcs;
    string          *syms;
    size_t           n_funcs;
    bool             saw_main;
};

LlvmX86SelState *mlir_llvm_x86_sel_begin(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                                         uint8_t **out_gblob, uint32_t *out_gblob_len) {
    if (out_gblob) *out_gblob = NULL;
    if (out_gblob_len) *out_gblob_len = 0;

    MLIR_RegionHandle mr = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    LlvmX86SelState *st = (LlvmX86SelState *)calloc(1, sizeof(LlvmX86SelState));
    if (!st) return NULL;
    st->mb = mb;
    st->funcs = (MLIR_OpHandle *)calloc(nops ? nops : 1, sizeof(MLIR_OpHandle));
    st->syms = (string *)calloc(nops ? nops : 1, sizeof(string));
    if (!st->funcs || !st->syms) {
        free(st->funcs);
        free(st->syms);
        free(st);
        return NULL;
    }

    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        if (!func_has_body(op)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) {
            fprintf(stderr, "llvm->x86_64: llvm.func without sym_name\n");
            free(st->funcs);
            free(st->syms);
            free(st);
            return NULL;
        }
        string sym = MLIR_GetAttributeString(sa);
        if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) st->saw_main = true;
        st->funcs[st->n_funcs] = op;
        st->syms[st->n_funcs] = sym;
        st->n_funcs++;
    }
    return st;
}

size_t mlir_llvm_x86_sel_num_funcs(LlvmX86SelState *st) {
    return st ? st->n_funcs : 0;
}

bool mlir_llvm_x86_sel_saw_main(LlvmX86SelState *st) {
    return st ? st->saw_main : false;
}

MLIR_OpHandle mlir_llvm_x86_sel_synth_start(MLIR_Context *ctx, LlvmX86SelState *st) {
    (void)st;
    return synth_start(ctx, str_lit("main"));
}

MLIR_OpHandle mlir_llvm_x86_sel_func(MLIR_Context *ctx, LlvmX86SelState *st,
                                     size_t idx) {
    if (!st || idx >= st->n_funcs) return MLIR_INVALID_HANDLE;
    return select_func(ctx, st->funcs[idx], st->syms[idx]);
}

void mlir_llvm_x86_sel_end(LlvmX86SelState *st) {
    if (!st) return;
    free(st->funcs);
    free(st->syms);
    free(st);
}
