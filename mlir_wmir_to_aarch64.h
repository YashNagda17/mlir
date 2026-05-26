// wmir -> aarch64 lowering. See mlir_wmir_to_aarch64.h for the public
// API and rationale.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns MLIR_INVALID_HANDLE on failure (with diagnostics on stderr).
MLIR_OpHandle mlir_wmir_to_aarch64(MLIR_Context *ctx,
                                   MLIR_OpHandle wmir_module);

#ifdef __cplusplus
}
#endif
