// x86_64 (MLIR dialect) -> ELF64 Linux executable.

#include "mlir_x86_64_to_elf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/string.h>

#include "mlir_llvm_to_x86_64.h"
#include "mlir_api.h"
#include "mlir_op_names.h"

// =============================================================================
// Growable byte buffer + endian helpers.
// =============================================================================
typedef struct { uint8_t *data; size_t len, cap; } Buf;

static void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 1024;
    while (b->len + add > nc) nc *= 2;
    b->data = (uint8_t *)realloc(b->data, nc);
    b->cap = nc;
}
static void buf_u8(Buf *b, uint8_t v) { buf_grow(b, 1); b->data[b->len++] = v; }
static void buf_le16(Buf *b, uint16_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
}
static void buf_le32(Buf *b, uint32_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
    buf_u8(b, (uint8_t)((v >> 16) & 0xff));
    buf_u8(b, (uint8_t)((v >> 24) & 0xff));
}
static void buf_le64(Buf *b, uint64_t v) {
    for (int i = 0; i < 8; i++) buf_u8(b, (uint8_t)((v >> (8 * i)) & 0xff));
}
static void buf_patch_le32(Buf *b, size_t pos, uint32_t v) {
    b->data[pos + 0] = (uint8_t)(v & 0xff);
    b->data[pos + 1] = (uint8_t)((v >> 8) & 0xff);
    b->data[pos + 2] = (uint8_t)((v >> 16) & 0xff);
    b->data[pos + 3] = (uint8_t)((v >> 24) & 0xff);
}

#define ELF_BASE   0x400000u
#define PAGE_SIZE  0x1000u
#define EH_SIZE    64u
#define PH_SIZE    56u

// =============================================================================
// x86-64 instruction encoders (64-bit integer ops only for first-light).
// Register numbers 0..15 per x86.md.
// =============================================================================
static uint8_t rex(bool w, uint8_t r, uint8_t x, uint8_t b) {
    return 0x40u | (w ? 0x08u : 0) | ((r & 8) ? 0x04u : 0)
                 | ((x & 8) ? 0x02u : 0) | ((b & 8) ? 0x01u : 0);
}
static uint8_t modrm(uint8_t mod, uint8_t reg, uint8_t rm) {
    return (uint8_t)((mod << 6) | ((reg & 7) << 3) | (rm & 7));
}

static void enc_rex(Buf *buf, bool w, uint8_t r, uint8_t x, uint8_t rb) {
    uint8_t v = rex(w, r, x, rb);
    if (v != 0x40 || w) buf_u8(buf, v);
}
static void enc_push_rbp(Buf *b) { buf_u8(b, 0x55); }
static void enc_pop_rbp(Buf *b)  { buf_u8(b, 0x5D); }
static void enc_mov_rbp_rsp(Buf *b) {
    enc_rex(b, true, 4, 0, 5); // mov rbp, rsp
    buf_u8(b, 0x89);
    buf_u8(b, modrm(3, 4, 5));
}
static void enc_sub_rsp_imm32(Buf *b, uint32_t imm) {
    enc_rex(b, true, 0, 0, 4);
    buf_u8(b, 0x81);
    buf_u8(b, modrm(3, 5, 4)); // sub rsp, imm32
    buf_le32(b, imm);
}
static void enc_add_rsp_imm32(Buf *b, uint32_t imm) {
    enc_rex(b, true, 0, 0, 4);
    buf_u8(b, 0x81);
    buf_u8(b, modrm(3, 0, 4)); // add rsp, imm32
    buf_le32(b, imm);
}
static void enc_mov_rr(Buf *b, uint8_t dst, uint8_t src) {
    enc_rex(b, true, src, 0, dst);
    buf_u8(b, 0x89);
    buf_u8(b, modrm(3, src, dst));
}
static void enc_mov_ri64(Buf *b, uint8_t rd, uint64_t imm) {
    enc_rex(b, true, 0, 0, rd);
    buf_u8(b, 0xB8u + (rd & 7));
    buf_le64(b, imm);
}
static void enc_mov_rm(Buf *b, uint8_t rd, uint8_t base, int32_t disp, uint8_t width) {
    bool w64 = width == 8;
    enc_rex(b, w64, rd, 0, base);
    buf_u8(b, 0x8B); // mov r, r/m
    if (disp >= -128 && disp <= 127) {
        buf_u8(b, modrm(1, rd, base));
        buf_u8(b, (uint8_t)disp);
    } else {
        buf_u8(b, modrm(2, rd, base));
        buf_le32(b, (uint32_t)disp);
    }
}
static void enc_mov_mr(Buf *b, uint8_t base, int32_t disp, uint8_t rs, uint8_t width) {
    bool w64 = width == 8;
    enc_rex(b, w64, rs, 0, base);
    buf_u8(b, 0x89); // mov r/m, r
    if (disp >= -128 && disp <= 127) {
        buf_u8(b, modrm(1, rs, base));
        buf_u8(b, (uint8_t)disp);
    } else {
        buf_u8(b, modrm(2, rs, base));
        buf_le32(b, (uint32_t)disp);
    }
}
static void enc_add_rr(Buf *b, uint8_t rd, uint8_t rn, uint8_t rm) {
    enc_rex(b, true, rm, 0, rd);
    buf_u8(b, 0x01); // add rd, rm  (add r/m, r encoding: dest is rd)
    buf_u8(b, modrm(3, rm, rd));
    (void)rn;
}
static void enc_sub_rr(Buf *b, uint8_t rd, uint8_t rn, uint8_t rm) {
    enc_rex(b, true, rm, 0, rd);
    buf_u8(b, 0x29); // sub rd, rm
    buf_u8(b, modrm(3, rm, rd));
    (void)rn;
}
static void enc_imul_rr(Buf *b, uint8_t rd, uint8_t rn, uint8_t rm) {
  // imul rd, rn, rm  -> imul rd, rm (3-operand form needs SIB; use 2-operand)
    enc_rex(b, true, rm, 0, rd);
    buf_u8(b, 0x0F);
    buf_u8(b, 0xAF);
    buf_u8(b, modrm(3, rm, rd));
    (void)rn;
}
static void enc_call_rel32(Buf *b, uint32_t rel) {
    buf_u8(b, 0xE8);
    buf_le32(b, rel);
}
static void enc_ret(Buf *b) { buf_u8(b, 0xC3); }
static void enc_syscall(Buf *b) { buf_u8(b, 0x0F); buf_u8(b, 0x05); }
static void enc_cmp_rr(Buf *b, uint8_t rn, uint8_t rm, uint8_t width) {
    bool w64 = width == 8;
    enc_rex(b, w64, rm, 0, rn);
    buf_u8(b, 0x39);
    buf_u8(b, modrm(3, rm, rn));
}
static void enc_cmp_ri(Buf *b, uint8_t rn, int64_t imm, uint8_t width) {
    bool w64 = width == 8;
    enc_rex(b, w64, 0, 0, rn);
    if (imm >= -128 && imm <= 127) {
        buf_u8(b, 0x83);
        buf_u8(b, modrm(3, 7, rn));
        buf_u8(b, (uint8_t)imm);
    } else {
        buf_u8(b, w64 ? 0x81 : 0x83);
        buf_u8(b, modrm(3, 7, rn));
        buf_le32(b, (uint32_t)imm);
    }
}
static void enc_and_rr(Buf *b, uint8_t rd, uint8_t rn, uint8_t rm) {
    enc_rex(b, true, rm, 0, rd);
    buf_u8(b, 0x21);
    buf_u8(b, modrm(3, rm, rd));
    (void)rn;
}
static void enc_xor_rr(Buf *b, uint8_t rd, uint8_t rn, uint8_t rm) {
    enc_rex(b, true, rm, 0, rd);
    buf_u8(b, 0x31);
    buf_u8(b, modrm(3, rm, rd));
    (void)rn;
}
static void enc_shl_ri(Buf *b, uint8_t rd, uint8_t rn, uint8_t imm) {
    enc_rex(b, true, 0, 0, rd);
    buf_u8(b, 0xC1);
    buf_u8(b, modrm(3, 4, rd));
    buf_u8(b, imm);
    (void)rn;
}
static void enc_sar_ri(Buf *b, uint8_t rd, uint8_t rn, uint8_t imm) {
    enc_rex(b, true, 0, 0, rd);
    buf_u8(b, 0xC1);
    buf_u8(b, modrm(3, 7, rd));
    buf_u8(b, imm);
    (void)rn;
}
static void enc_setcc(Buf *b, uint8_t rd, uint8_t cond) {
    enc_rex(b, false, 0, 0, rd);
    buf_u8(b, 0x0F);
    buf_u8(b, (uint8_t)(0x90 + (cond & 0xF)));
    buf_u8(b, modrm(3, 0, rd));
}
static void enc_cmovcc(Buf *b, uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
    enc_rex(b, true, rm, 0, rd);
    buf_u8(b, 0x0F);
    buf_u8(b, (uint8_t)(0x40 + (cond & 0xF)));
    buf_u8(b, modrm(3, rm, rd));
    (void)rn;
}
static void enc_cqo(Buf *b) {
    enc_rex(b, true, 0, 0, 0);
    buf_u8(b, 0x48); // cdq/cqo in 64-bit mode
}
static void enc_idiv_r(Buf *b, uint8_t rm) {
    enc_rex(b, true, 0, 0, rm);
    buf_u8(b, 0xF7);
    buf_u8(b, modrm(3, 7, rm));
}
static void enc_div_r(Buf *b, uint8_t rm) {
    enc_rex(b, true, 0, 0, rm);
    buf_u8(b, 0xF7);
    buf_u8(b, modrm(3, 6, rm));
}
static void enc_jmp_rel32(Buf *b, uint32_t rel) {
    buf_u8(b, 0xE9);
    buf_le32(b, rel);
}
static void enc_jcc_rel32(Buf *b, uint8_t cond, uint32_t rel) {
    buf_u8(b, 0x0F);
    buf_u8(b, (uint8_t)(0x80 + (cond & 0xF)));
    buf_le32(b, rel);
}

// =============================================================================
// Emitted function + relocation tracking.
// =============================================================================
typedef struct { string name; uint32_t off; } CallReloc;
typedef struct { MLIR_BlockHandle blk; uint32_t fn_off; } BlockPos;
typedef struct { uint32_t fn_off; MLIR_BlockHandle target; uint8_t cond; bool is_jcc; } BranchReloc;
typedef struct {
    string   name;
    bool     exported;
    Buf      code;
    CallReloc *relocs;
    size_t   n_relocs, c_relocs;
    BlockPos *bp;
    size_t   n_bp, c_bp;
    BranchReloc *br;
    size_t   n_br, c_br;
} EmittedFunc;

static void ef_add_bp(EmittedFunc *e, MLIR_BlockHandle blk, uint32_t off) {
    if (e->n_bp == e->c_bp) {
        e->c_bp = e->c_bp ? e->c_bp * 2 : 8;
        e->bp = (BlockPos *)realloc(e->bp, e->c_bp * sizeof(BlockPos));
    }
    e->bp[e->n_bp].blk = blk;
    e->bp[e->n_bp].fn_off = off;
    e->n_bp++;
}
static void ef_add_br(EmittedFunc *e, uint32_t off, MLIR_BlockHandle tgt,
                      uint8_t cond, bool is_jcc) {
    if (e->n_br == e->c_br) {
        e->c_br = e->c_br ? e->c_br * 2 : 8;
        e->br = (BranchReloc *)realloc(e->br, e->c_br * sizeof(BranchReloc));
    }
    e->br[e->n_br].fn_off = off;
    e->br[e->n_br].target = tgt;
    e->br[e->n_br].cond = cond;
    e->br[e->n_br].is_jcc = is_jcc;
    e->n_br++;
}

static void ef_add_reloc(EmittedFunc *e, string callee, uint32_t off) {
    if (e->n_relocs == e->c_relocs) {
        e->c_relocs = e->c_relocs ? e->c_relocs * 2 : 4;
        e->relocs = (CallReloc *)realloc(e->relocs, e->c_relocs * sizeof(CallReloc));
    }
    e->relocs[e->n_relocs].name = callee;
    e->relocs[e->n_relocs].off = off;
    e->n_relocs++;
}

static uint32_t bp_find(EmittedFunc *e, MLIR_BlockHandle blk) {
    for (size_t i = 0; i < e->n_bp; i++)
        if (e->bp[i].blk == blk) return e->bp[i].fn_off;
    return (uint32_t)-1;
}

static bool patch_branches(EmittedFunc *e) {
    for (size_t i = 0; i < e->n_br; i++) {
        BranchReloc *r = &e->br[i];
        uint32_t tgt = bp_find(e, r->target);
        if (tgt == (uint32_t)-1) {
            fprintf(stderr, "x86_64->elf: branch target block has no offset\n");
            return false;
        }
        uint32_t rip = r->fn_off + (r->is_jcc ? 6u : 5u);
        int32_t rel = (int32_t)(tgt - rip);
        if (r->is_jcc)
            buf_patch_le32(&e->code, r->fn_off + 2, (uint32_t)rel);
        else
            buf_patch_le32(&e->code, r->fn_off + 1, (uint32_t)rel);
    }
    return true;
}

static int64_t attr_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static string attr_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}
static bool attr_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}

static bool emit_x86_func(MLIR_OpHandle fn, EmittedFunc *out) {
    out->name = attr_s(fn, "sym_name");
    out->exported = attr_b(fn, "exported");

    if (MLIR_GetOpNumRegions(fn) < 1) {
        fprintf(stderr, "x86_64->elf: x86_64.func has no region\n");
        return false;
    }
    MLIR_RegionHandle reg = MLIR_GetOpRegion(fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(reg);
    Buf *code = &out->code;

    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(reg, bi);
        ef_add_bp(out, blk, (uint32_t)code->len);
        size_t n = MLIR_GetBlockNumOps(blk);
        for (size_t i = 0; i < n; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
            MLIR_OpType t = MLIR_GetOpType(op);
            switch (t) {
            case OP_TYPE_X86_64_PROLOGUE:
                enc_push_rbp(code);
                enc_mov_rbp_rsp(code);
                enc_sub_rsp_imm32(code, (uint32_t)attr_i(op, "frame_size"));
                break;
            case OP_TYPE_X86_64_EPILOGUE:
                enc_add_rsp_imm32(code, (uint32_t)attr_i(op, "frame_size"));
                enc_pop_rbp(code);
                break;
            case OP_TYPE_X86_64_MOV_RI:
                enc_mov_ri64(code, (uint8_t)attr_i(op, "rd"), (uint64_t)attr_i(op, "imm"));
                break;
            case OP_TYPE_X86_64_MOV_RR:
                enc_mov_rr(code, (uint8_t)attr_i(op, "rd"), (uint8_t)attr_i(op, "rn"));
                break;
            case OP_TYPE_X86_64_MOV_RM:
                enc_mov_rm(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "base"), (int32_t)attr_i(op, "disp"),
                           (uint8_t)attr_i(op, "width"));
                break;
            case OP_TYPE_X86_64_MOV_MR:
                enc_mov_mr(code, (uint8_t)attr_i(op, "base"),
                           (int32_t)attr_i(op, "disp"), (uint8_t)attr_i(op, "rs"),
                           (uint8_t)attr_i(op, "width"));
                break;
            case OP_TYPE_X86_64_ADD_RR:
                enc_add_rr(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "rm"));
                break;
            case OP_TYPE_X86_64_SUB_RR:
                enc_sub_rr(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "rm"));
                break;
            case OP_TYPE_X86_64_IMUL_RR:
                enc_imul_rr(code, (uint8_t)attr_i(op, "rd"),
                            (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "rm"));
                break;
            case OP_TYPE_X86_64_AND_RR:
                enc_and_rr(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "rm"));
                break;
            case OP_TYPE_X86_64_XOR_RR:
                enc_xor_rr(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "rm"));
                break;
            case OP_TYPE_X86_64_SHL_RI:
                enc_shl_ri(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "imm"));
                break;
            case OP_TYPE_X86_64_SAR_RI:
                enc_sar_ri(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "imm"));
                break;
            case OP_TYPE_X86_64_CMP_RR:
                enc_cmp_rr(code, (uint8_t)attr_i(op, "rn"),
                           (uint8_t)attr_i(op, "rm"), (uint8_t)attr_i(op, "width"));
                break;
            case OP_TYPE_X86_64_CMP_RI:
                enc_cmp_ri(code, (uint8_t)attr_i(op, "rn"),
                           attr_i(op, "imm"), (uint8_t)attr_i(op, "width"));
                break;
            case OP_TYPE_X86_64_SETCC:
                enc_setcc(code, (uint8_t)attr_i(op, "rd"), (uint8_t)attr_i(op, "cond"));
                break;
            case OP_TYPE_X86_64_CMOVCC:
                enc_cmovcc(code, (uint8_t)attr_i(op, "rd"),
                           (uint8_t)attr_i(op, "rn"), (uint8_t)attr_i(op, "rm"),
                           (uint8_t)attr_i(op, "cond"));
                break;
            case OP_TYPE_X86_64_CQO:
                enc_cqo(code);
                break;
            case OP_TYPE_X86_64_IDIV_R:
                if (attr_b(op, "unsigned_div"))
                    enc_div_r(code, (uint8_t)attr_i(op, "rm"));
                else
                    enc_idiv_r(code, (uint8_t)attr_i(op, "rm"));
                break;
            case OP_TYPE_X86_64_JMP: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint32_t off = (uint32_t)code->len;
                enc_jmp_rel32(code, 0);
                ef_add_br(out, off, tgt, 0, false);
                break;
            }
            case OP_TYPE_X86_64_JCC: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint32_t off = (uint32_t)code->len;
                enc_jcc_rel32(code, (uint8_t)attr_i(op, "cond"), 0);
                ef_add_br(out, off, tgt, (uint8_t)attr_i(op, "cond"), true);
                break;
            }
            case OP_TYPE_X86_64_CALL: {
                string callee = attr_s(op, "callee");
                uint32_t off = (uint32_t)code->len + 1u; // rel32 follows E8
                enc_call_rel32(code, 0);
                ef_add_reloc(out, callee, off);
                break;
            }
            case OP_TYPE_X86_64_RET:
                enc_ret(code);
                break;
            case OP_TYPE_X86_64_SYSCALL:
                enc_syscall(code);
                break;
            default:
                fprintf(stderr, "x86_64->elf: unsupported op in '%.*s'\n",
                        (int)out->name.size, out->name.str);
                return false;
            }
        }
    }
    return patch_branches(out);
}

static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static bool name_eq_s(string a, string b) {
    return a.size == b.size && memcmp(a.str, b.str, a.size) == 0;
}

static int find_func(EmittedFunc *efs, size_t n, string name) {
    for (size_t i = 0; i < n; i++)
        if (name_eq_s(efs[i].name, name)) return (int)i;
    return -1;
}

static void ef_free(EmittedFunc *e) {
    free(e->code.data);
    free(e->relocs);
    free(e->bp);
    free(e->br);
    *e = (EmittedFunc){0};
}

static bool finalize_elf(EmittedFunc *efs, size_t n_funcs, size_t start_idx,
                         uint8_t **out_data, size_t *out_size) {
    // Lay out functions: _start first, then others in emission order.
    size_t text_size = 0;
    uint32_t *fn_off = (uint32_t *)calloc(n_funcs, sizeof(uint32_t));
    for (size_t i = 0; i < n_funcs; i++) {
        fn_off[i] = (uint32_t)text_size;
        text_size += efs[i].code.len;
    }

    // Patch call relocs.
    for (size_t i = 0; i < n_funcs; i++) {
        EmittedFunc *e = &efs[i];
        for (size_t r = 0; r < e->n_relocs; r++) {
            int tgt = find_func(efs, n_funcs, e->relocs[r].name);
            if (tgt < 0) {
                fprintf(stderr, "x86_64->elf: unresolved call to '%.*s' from '%.*s'\n",
                        (int)e->relocs[r].name.size, e->relocs[r].name.str,
                        (int)e->name.size, e->name.str);
                free(fn_off);
                return false;
            }
            uint32_t call_pos = fn_off[i] + e->relocs[r].off;
            uint32_t tgt_pos = fn_off[tgt];
            int32_t rel = (int32_t)(tgt_pos - (call_pos + 4));
            buf_patch_le32(&e->code, e->relocs[r].off, (uint32_t)rel);
        }
    }

    uint32_t text_file_off = PAGE_SIZE;
    uint64_t text_vaddr = ELF_BASE + text_file_off;
    uint64_t entry = text_vaddr + fn_off[start_idx];

    Buf out = {0};
    // Elf64_Ehdr
    uint8_t ident[16] = {
        0x7F, 'E', 'L', 'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    for (int i = 0; i < 16; i++) buf_u8(&out, ident[i]);
    buf_le16(&out, 2);           // ET_EXEC
    buf_le16(&out, 62);          // EM_X86_64
    buf_le32(&out, 1);           // version
    buf_le64(&out, entry);
    buf_le64(&out, EH_SIZE);     // phoff
    buf_le64(&out, 0);           // shoff
    buf_le32(&out, 0);
    buf_le16(&out, EH_SIZE);
    buf_le16(&out, PH_SIZE);
    buf_le16(&out, 1);           // one PT_LOAD for first milestone
    buf_le16(&out, 0);
    buf_le16(&out, 0);
    buf_le16(&out, 0);

    // PT_LOAD: headers + text in one RX segment.
    uint64_t filesz = text_file_off + text_size;
    buf_le32(&out, 1);           // PT_LOAD
    buf_le32(&out, 7);           // PF_R | PF_X (also covers header pages as R)
    buf_le64(&out, 0);           // offset
    buf_le64(&out, ELF_BASE);
    buf_le64(&out, ELF_BASE);
    buf_le64(&out, filesz);
    buf_le64(&out, filesz);
    buf_le64(&out, PAGE_SIZE);

    // Pad to text offset.
    while (out.len < text_file_off) buf_u8(&out, 0);

  // Copy function bodies in order.
    for (size_t i = 0; i < n_funcs; i++) {
        buf_grow(&out, efs[i].code.len);
        memcpy(out.data + out.len, efs[i].code.data, efs[i].code.len);
        out.len += efs[i].code.len;
    }

    *out_data = out.data;
    *out_size = out.len;
    free(fn_off);
    return true;
}

bool mlir_x86_64_to_elf(MLIR_Context *ctx, MLIR_OpHandle x86_64_module,
                        uint8_t **out_data, size_t *out_size) {
    (void)ctx;
    if (out_data) *out_data = NULL;
    if (out_size) *out_size = 0;

    MLIR_RegionHandle mr = MLIR_GetOpRegion(x86_64_module, 0);
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    size_t cap = nops ? nops : 1;
    EmittedFunc *efs = (EmittedFunc *)calloc(cap, sizeof(EmittedFunc));
    size_t n_funcs = 0;
    size_t start_idx = (size_t)-1;

    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "x86_64.func")) continue;
        EmittedFunc *e = &efs[n_funcs];
        if (!emit_x86_func(op, e)) {
            for (size_t k = 0; k <= n_funcs; k++) ef_free(&efs[k]);
            free(efs);
            return false;
        }
        if (e->name.size == 6 && memcmp(e->name.str, "_start", 6) == 0)
            start_idx = n_funcs;
        n_funcs++;
    }

    if (n_funcs == 0 || start_idx == (size_t)-1) {
        fprintf(stderr, "x86_64->elf: no `_start` function found\n");
        for (size_t k = 0; k < n_funcs; k++) ef_free(&efs[k]);
        free(efs);
        return false;
    }

    bool ok = finalize_elf(efs, n_funcs, start_idx, out_data, out_size);
    for (size_t k = 0; k < n_funcs; k++) ef_free(&efs[k]);
    free(efs);
    return ok;
}

bool mlir_llvm_to_elf(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                      uint8_t **out_data, size_t *out_size) {
    if (out_data) *out_data = NULL;
    if (out_size) *out_size = 0;

    MLIR_OpHandle x86 = mlir_llvm_to_x86_64(ctx, llvm_module);
    if (x86 == MLIR_INVALID_HANDLE) return false;
    return mlir_x86_64_to_elf(ctx, x86, out_data, out_size);
}
