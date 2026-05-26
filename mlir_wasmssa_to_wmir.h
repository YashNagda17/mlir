// Stage 2-alt of the native LLVM-dialect MLIR -> Mach-O pipeline:
//
//   wasmssa builtin.module  -->  wmir builtin.module
//
// wmir ("wasm middle IR") is a flat-CFG, side-effect-explicit IR with
// real iN/fN/ptr types — the optimization-ready representation that
// feeds the AArch64 instruction selector. This is the first-light
// scaffold; the initial supported op subset is just enough to lower
// `int main() { return 42; }` (only `wmir.func`, `wmir.const`,
// `wmir.return`). Op coverage will grow test-by-test.
//
// On unsupported wasmssa ops the lowering returns MLIR_INVALID_HANDLE
// and prints a diagnostic to stderr identifying the op kind.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_OpHandle mlir_wasmssa_to_wmir(MLIR_Context *ctx,
                                   MLIR_OpHandle ssa_module);

#ifdef __cplusplus
}
#endif
