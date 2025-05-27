#pragma once

#include "tokenizer.h"
#include <base/arena.h>

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

typedef struct {
    char *str;
    uint64_t size;
} string;

#define str_lit(S)  (string){.str=(char*)(S), .size=sizeof(S)-1}

string str_from_cstr_view(char *cstr); 
string str_from_cstr_len_view(char *cstr, uint64_t size);
char *str_to_cstr_copy(Arena *arena, string str);
bool str_eq(string a, string b);
string str_substr(string str, uint64_t min, uint64_t max);

typedef struct Region Region;

typedef struct Operation Operation;
struct Operation {
    string opcode;
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


#ifdef __cplusplus
}
#endif
