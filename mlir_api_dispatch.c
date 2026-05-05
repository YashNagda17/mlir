// Native-backend implementation of the unified parse/print dispatch API
// (MLIR_PrintOperation / MLIR_ParseText). Lives in its own translation
// unit so the cross-impl native test driver — which only links the
// generic printer — does not have to pull in the classic printer +
// parser stack.

#include <base/string.h>
#include "mlir_api.h"
#include "mlir_api_internal.h"
#include "mlir_parser.h"
#include "mlir_classic_printer.h"
#include "mlir_generic_printer.h"

string MLIR_PrintOperation(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_PrintKind kind) {
    switch (kind) {
        case MLIR_PRINT_UPSTREAM:
            return MLIR_PrintOperation_upstream_impl(ctx, op);
        case MLIR_PRINT_CLASSIC:
            return print_module_classic(ctx, op, NULL);
        case MLIR_PRINT_GENERIC:
            return print_operation_generic(ctx, 0, op);
    }
    return str_lit("error: unknown MLIR_PrintKind\n");
}

MLIR_OpHandle MLIR_ParseText(MLIR_Context *ctx, string text, MLIR_ParseKind kind) {
    switch (kind) {
        case MLIR_PARSE_UPSTREAM:
            return MLIR_ParseText_upstream_impl(ctx, text);
        case MLIR_PARSE_CLASSIC: {
            MLIR_LocationMap *locmap = NULL;
            return mlir_parse_module(ctx, (const char*)text.str, text.size, &locmap);
        }
    }
    return MLIR_INVALID_HANDLE;
}
