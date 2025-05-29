#pragma once

#include <base/arena.h> // Provides Arena, size_t

#ifdef WITH_BASE_ASSERT
static int vec_called_const = 0xdeadbeef;
#endif

// Helper macros for token pasting with prior expansion of arguments
// These ensure that macro arguments are expanded before being pasted.
#define PASTE_IMPL(a, b) a##b
#define PASTE(a, b) PASTE_IMPL(a, b)

#define PASTE_TRIPLE_IMPL(a, b, c) a##b##c
#define PASTE_TRIPLE(a, b, c) PASTE_TRIPLE_IMPL(a, b, c)

// Define the type T (e.g., int64_t)
#define T int64_t

// Corrected macros to generate names and functions.
// 'type_name_token' (which will be T) is expanded before pasting.
#define VECTOR_T_NAME(type_name_token) PASTE(vector_, type_name_token)
// 'type_name_token' is expanded; 'suffix' is used literally with a preceding underscore.
#define VECTOR_T_FUNC(type_name_token, suffix) PASTE_TRIPLE(vector_, type_name_token, _##suffix)

// Define the struct
// Example: VECTOR_T_NAME(T) will expand to vector_int64_t
typedef struct VECTOR_T_NAME(T) {
    T* p; // T here will correctly expand to int64_t*
    size_t n, max;
#ifdef WITH_BASE_ASSERT
    int reserve_called;
#endif
} VECTOR_T_NAME(T); // The typedef name will be vector_int64_t

// Function declarations
// Example: VECTOR_T_FUNC(T, reserve) will expand to vector_int64_t_reserve
// Example: VECTOR_T_NAME(T) for the vec parameter type will expand to vector_int64_t
void VECTOR_T_FUNC(T, reserve)(Arena *arena, VECTOR_T_NAME(T) *vec, size_t max);
void VECTOR_T_FUNC(T, push_back)(Arena *arena, VECTOR_T_NAME(T) *vec, T x); // T for x type will be int64_t

// Undefine T and the generator macros, allowing this header to be potentially
// re-included with a different definition of T (a common pattern for generic containers).
#undef T
#undef VECTOR_T_NAME
#undef VECTOR_T_FUNC

// Undefine helper macros if they are local to this file's generation logic
#undef PASTE_IMPL
#undef PASTE
#undef PASTE_TRIPLE_IMPL
#undef PASTE_TRIPLE
