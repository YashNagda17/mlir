#pragma once

#include <base/arena.h>
#include <base/string.h>

#include "tokenizer.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
} Parser;


typedef struct Region Region;

typedef enum ValueKind {
    BLOCK_ARG,
    OP_RESULT
} ValueKind;

typedef struct ValueRef {
    ValueKind kind;
    void* def; // Block* or Operation*
    uint64_t index;
} ValueRef;

typedef struct Operation Operation;
struct Operation {
    string opcode;
    ValueRef **operands;
    uint64_t n_operands;
    Region **regions;
    uint64_t n_regions;
};

typedef struct Block Block;
struct Block {
    Operation **operations;
    uint64_t n_operations;
};

struct Region {
    Block **blocks;
    uint64_t n_blocks;
};

void parser_init(Arena *arena, Parser *parser, string text);
Operation* parse_module(Parser *parser);

#ifdef __cplusplus
}
#endif
