// See mlir_wasmssa_to_llvm.h for the role of this pass in the pipeline.
//
// First-light scaffold: the entry point is wired into the build and the
// driver (`--from-wasm ... --macho-backend=llvm`) but the lowering itself is
// not yet implemented. It returns MLIR_INVALID_HANDLE with a diagnostic so
// the opt-in path fails cleanly while the default `wmir` path is untouched.
// Op coverage grows test-by-test (see mlir_wasmssa_to_wmir.c for the analog).

#include "mlir_wasmssa_to_llvm.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

MLIR_OpHandle mlir_wasmssa_to_llvm(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    (void)ctx;
    (void)ssa_module;
    fprintf(stderr,
        "wasmssa->llvm: lowering not yet implemented "
        "(use --macho-backend=wmir for the --from-wasm path)\n");
    return MLIR_INVALID_HANDLE;
}
