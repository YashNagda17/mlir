#!/usr/bin/env bash
set -e

MLIR_LIB=$(ls "$CONDA_PREFIX"/lib/libMLIR.19.1.dylib "$CONDA_PREFIX"/lib/libMLIR.so.19* 2>/dev/null | head -1)
if [ -z "$MLIR_LIB" ]; then
    echo "Could not find libMLIR shared library under $CONDA_PREFIX/lib" >&2
    exit 1
fi

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c $PLATFORM_C"
PROJ_C_FILES="tests/cross/driver.c mlir_generic_printer.c mlir_op_names.c"

$CC -c -g -I corec -I . $COREC_C_FILES $PROJ_C_FILES
$CXX -c -std=c++17 -fno-rtti -g -I corec -I . -I "$CONDA_PREFIX/include" mlir_api_impl_upstream.cpp
$CXX -g -o cross_upstream \
    driver.o mlir_generic_printer.o mlir_op_names.o mlir_api_impl_upstream.o \
    io.o buddy.o arena.o scratch.o format.o math.o string.o mem.o numconv.o assert.o exit.o $PLATFORM_OBJ \
    -L "$CONDA_PREFIX/lib" "$MLIR_LIB" -lLLVM $EXTRA_LINK_FLAGS \
    -Wl,-rpath,"$CONDA_PREFIX/lib"
