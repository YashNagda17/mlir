// tests/cross/driver.c
//
// Cross-implementation smoke test for the public MLIR C API in
// `mlir_api.h`. Builds a minimal module containing one unregistered op,
// then prints it via the generic printer. Compiled twice:
//
//   * `cross_native` — backed by mlir_api_impl.c (corec arena, no libc).
//   * `cross_upstream` — backed by mlir_api_impl_upstream.cpp (upstream
//     LLVM/MLIR C++, system libc).
//
// Both binaries must produce byte-identical output.

#include <base/arena.h>
#include <base/string.h>
#include <base/io.h>
#include <platform/platform.h>
#include "mlir_api.h"
#include "mlir_generic_printer.h"

static void print_string(string s) {
    ciovec_t iov;
    iov.buf = s.str;
    iov.buf_len = s.size;
    write_all(PLATFORM_STDOUT_FD, &iov, 1);
}

int app_main(void) {
    Arena *arena = arena_create(1 * 1024 * 1024);
    MLIR_Context ctx;
    MLIR_SetArenaAllocator(&ctx, arena);

    // Build:
    //   module {
    //     "custom.empty"() : () -> ()
    //   }
    MLIR_RegionHandle module_region = MLIR_CreateRegion(&ctx);
    MLIR_BlockHandle module_block   = MLIR_CreateBlock(&ctx);
    MLIR_AppendRegionBlock(&ctx, module_region, module_block);

    MLIR_RegionHandle *module_regions = arena_new_array(arena, MLIR_RegionHandle, 1);
    module_regions[0] = module_region;

    MLIR_OpHandle module_op = MLIR_CreateOp(
        &ctx, OP_TYPE_MODULE, str_lit("module"),
        /*attrs*/ NULL, 0,
        /*result_types*/ NULL, 0,
        /*results*/ NULL, 0,
        /*operands*/ NULL, 0,
        /*regions*/ module_regions, 1,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

    MLIR_OpHandle empty_op = MLIR_CreateOp(
        &ctx, OP_TYPE_UNREGISTERED, str_lit("custom.empty"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

    MLIR_AppendBlockOp(&ctx, module_block, empty_op);

    MLIR_InitApi(&ctx, module_op);

    string out = print_operation_generic(&ctx, 0, module_op);
    print_string(out);

    arena_destroy(arena);
    return 0;
}
