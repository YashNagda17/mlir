// x86_64 (MLIR dialect) -> ELF64 Linux executable skeleton.
//
// Pairs with mlir_llvm_to_x86_64.{h,c}. The selector will produce a
// physical-register `x86_64` dialect; this file will eventually be the dumb
// byte emitter that encodes instructions, patches rel32/RIP-relative sites,
// and writes ELF program headers.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Translate an MLIR module whose body is x86_64.* ops into a Linux ELF64
// executable. executable.
bool mlir_x86_64_to_elf(MLIR_Context *ctx, MLIR_OpHandle x86_64_module,
                        uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif
