// Internal dispatch helpers — backend-specific implementations of the
// UPSTREAM kinds for MLIR_PrintOperation / MLIR_ParseText. The CLASSIC /
// GENERIC kinds go through the regular MLIR_* API and don't need this.

#ifndef MLIR_API_INTERNAL_H
#define MLIR_API_INTERNAL_H

#include <base/string.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Print an operation using the upstream pretty-printer. Native backend
// returns an "error: ..." string.
string MLIR_PrintOperation_upstream_impl(MLIR_Context *ctx, MLIR_OpHandle op);

// Parse a top-level module via the upstream parser. Native backend returns
// MLIR_INVALID_HANDLE.
MLIR_OpHandle MLIR_ParseText_upstream_impl(MLIR_Context *ctx, string text);

#ifdef __cplusplus
}
#endif

#endif
