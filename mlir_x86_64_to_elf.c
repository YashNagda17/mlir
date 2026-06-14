// x86_64 (MLIR dialect) -> ELF64 Linux executable skeleton.

#include "mlir_x86_64_to_elf.h"

#include <stdio.h>

#include "mlir_llvm_to_x86_64.h"

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
