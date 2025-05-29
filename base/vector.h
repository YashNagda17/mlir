#pragma once

#include <base/arena.h>

#ifdef WITH_BASE_ASSERT
static int vec_called_const = 0xdeadbeef;
#endif

#define T int64_t

typedef struct vector_##T {
    VecType* p;
    size_t n, max;
#ifdef WITH_BASE_ASSERT
    int reserve_called;
#endif
} vector_##T;

void vector_##T##_reserve(Arena *arena, vector_##T vec, size_t max); 
void vector_##T##(Arena *arena, vector_##T vec, T x); 

#undef T
