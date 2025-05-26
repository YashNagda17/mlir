#!/bin/bash

set -ex

CFLAGS="-fsanitize=address -g -Wall -ferror-limit=1"

re2c -b tokenizer.re -o tokenizer.c
clang $CFLAGS -o parser parser.c tokenizer.c arena.c mlir_parser.c
./parser
