// llvm dialect -> x86_64 lowering skeleton.

#include "mlir_llvm_to_x86_64.h"

#include <stdio.h>

struct LlvmX86SelState {
    int unused;
};

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
