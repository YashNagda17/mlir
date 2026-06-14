// llvm dialect -> x86_64 lowering skeleton.
//
// This is the planned native Linux/x86-64 backend entry point. It mirrors the
// llvm -> aarch64 -> Mach-O split: this layer owns instruction selection,
// ABI lowering, frame layout, and synthetic entry helpers, while
// mlir_x86_64_to_elf.c owns byte encoding and ELF layout.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lower an `llvm`-dialect module to an `x86_64`-dialect module suitable
// for `mlir_x86_64_to_elf`. This is a skeleton entry point for now and
// returns MLIR_INVALID_HANDLE until lowering is implemented.
MLIR_OpHandle mlir_llvm_to_x86_64(MLIR_Context *ctx,
                                  MLIR_OpHandle llvm_module);

// Streaming selection API placeholder. This mirrors the AArch64 streaming
// interface so the ELF finalizer can later lower one function at a time.
typedef struct LlvmX86SelState LlvmX86SelState;

LlvmX86SelState *mlir_llvm_x86_sel_begin(MLIR_Context *ctx,
                                         MLIR_OpHandle llvm_module,
                                         uint8_t **out_gblob,
                                         uint32_t *out_gblob_len);

size_t mlir_llvm_x86_sel_num_funcs(LlvmX86SelState *st);
MLIR_OpHandle mlir_llvm_x86_sel_synth_start(MLIR_Context *ctx,
                                            LlvmX86SelState *st);
MLIR_OpHandle mlir_llvm_x86_sel_func(MLIR_Context *ctx,
                                     LlvmX86SelState *st,
                                     size_t idx);
bool mlir_llvm_x86_sel_saw_main(LlvmX86SelState *st);
void mlir_llvm_x86_sel_end(LlvmX86SelState *st);

// Stream an `llvm`-dialect module straight to an ELF image. Skeleton only.
bool mlir_llvm_to_elf(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                      uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif
