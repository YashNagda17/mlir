// Stage 1 of the native LLVM-dialect MLIR -> WASM pipeline.
//
//   LLVM-dialect builtin.module  -->  wasmssa builtin.module
//
// Also declares the WebAssembly value-type byte encodings shared by
// all three pipeline stages.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// WebAssembly value-type byte encodings.
enum {
    WT_I32 = 0x7f,
    WT_I64 = 0x7e,
    WT_F32 = 0x7d,
    WT_F64 = 0x7c,
};

// Lower an LLVM-dialect module to wasmssa form. On success
// `MLIR_SetArenaAllocator` is switched to a fresh output arena holding the
// wasmssa module; the input `module` and its arena must stay alive until all
// later wasm pipeline stages finish (the type intern table still references
// LLVM types allocated there). The caller should destroy the input arena
// after the full wasm emit completes, then refresh via
// `MLIR_GetArenaAllocator(ctx)`. Returns MLIR_INVALID_HANDLE on failure.
MLIR_OpHandle mlir_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module);

#ifdef __cplusplus
}
#endif
