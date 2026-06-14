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

bool mlir_x86_64_to_elf(MLIR_Context *ctx, MLIR_OpHandle x86_64_module,
                        uint8_t **out_data, size_t *out_size) {
    (void)ctx;
    (void)x86_64_module;
    if (out_data) *out_data = NULL;
    if (out_size) *out_size = 0;
    fprintf(stderr, "x86_64->elf: backend skeleton is not implemented yet\n");
    return false;
}

bool mlir_llvm_to_elf(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                      uint8_t **out_data, size_t *out_size) {
    (void)ctx;
    (void)llvm_module;
    if (out_data) *out_data = NULL;
    if (out_size) *out_size = 0;
    fprintf(stderr, "llvm->elf: streaming backend skeleton is not implemented yet\n");
    return false;
}
