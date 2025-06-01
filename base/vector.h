#pragma once


#include <stddef.h> // For size_t
#include <assert.h> // For assert()
#include <string.h> // For memcpy()

#include <base/arena.h>

/*
 * IMPORTANT:
 * 1. Arena Implementation: You must include your Arena header (e.g., <base/arena.h>)
 * *before* including this "generic_vector.h" file.
 * This generic vector expects `struct Arena;` to be defined and a function
 * `void* arena_alloc_array(Arena* arena, size_t element_size, size_t element_count);`
 * or a macro like `#define arena_alloc_array(arena, type, count) (type*)arena_alloc(arena, sizeof(type) * (count))`
 * to be available. The implementation below will cast the result of arena_alloc_array to (TYPE*).
 *
 * 2. Type Names with Commas: If your TYPE contains commas (e.g., `unsigned long`),
 * you must use a typedef first:
 * typedef unsigned long my_ulong_t;
 * DEFINE_VECTOR_FOR_TYPE(my_ulong_t, ULongVec)
 *
 * 3. WITH_BASE_ASSERT: If you define WITH_BASE_ASSERT (e.g., via a compiler flag -DWITH_BASE_ASSERT),
 * additional runtime checks will be enabled. This requires the vector to be initialized
 * and then reserved before the first push_back.
 */

// --- Helper Macros (internal use) ---
#define _GV_CONCAT_IMPL(a, b) a##b
#define _GV_CONCAT(a, b) _GV_CONCAT_IMPL(a, b)

#define _GV_CONCAT3_IMPL(a, b, c) a##b##c
#define _GV_CONCAT3(a, b, c) _GV_CONCAT3_IMPL(a, b, c)

// --- Conditional Compilation Helper for WITH_BASE_ASSERT ---
#if defined(WITH_BASE_ASSERT)
    #define IF_GENERIC_VECTOR_WITH_BASE_ASSERT(code) code
    // This constant is used for assertion checks.
    // Declared static const for internal linkage, avoiding multiple definition errors.
    static const int GV_INTERNAL_RESERVE_CALLED_MAGIC = 0xDEADBEEF;
#else
    #define IF_GENERIC_VECTOR_WITH_BASE_ASSERT(code)
#endif

// --- Main Macro to Define a Vector Type and its Functions ---
// TYPE: The data type to be stored (e.g., int, MyStructA).
// NAME: A prefix for the generated struct and function names (e.g., IntVec, MyStructAVec).
//       - Struct type will be: NAME_t
//       - Functions will be: NAME_init, NAME_reserve, NAME_push_back.
#define DEFINE_VECTOR_FOR_TYPE(TYPE, NAME) \
    \
    /* Vector Struct Definition */ \
    typedef struct NAME { \
        TYPE *data; \
        size_t size; \
        size_t max; \
        /* This field is conditionally compiled based on WITH_BASE_ASSERT */ \
        IF_GENERIC_VECTOR_WITH_BASE_ASSERT(int reserve_called_flag;) \
    } NAME; \
    \
    /* Initializes an empty vector. Call this before using a new vector instance. */ \
    static inline void _GV_CONCAT3(NAME, _, init)(NAME *vec) { \
        vec->data = NULL; \
        vec->size = 0; \
        vec->max = 0; \
        IF_GENERIC_VECTOR_WITH_BASE_ASSERT(vec->reserve_called_flag = 0;) \
    } \
    \
    /* Reserves memory for at least 'new_max_capacity' elements. */ \
    /* Resets size to 0. Old data is not preserved (typical for arena-based reserve). */ \
    static inline void _GV_CONCAT3(NAME, _, reserve)(Arena *arena, NAME *vec, size_t new_max_capacity) { \
        vec->size = 0; \
        if (new_max_capacity == 0) new_max_capacity = 1; /* Minimum capacity of 1 */ \
        assert(new_max_capacity > 0 && "New maximum capacity must be greater than 0."); \
        \
        /* Assuming arena_alloc_array takes element type and count, or element_size and count. */ \
        /* The original macro was arena_alloc_array(arena, T, max), implying T is the type. */ \
        /* We adapt to this by passing TYPE as if it's the type name for a macro, */ \
        /* or rely on sizeof(TYPE) if arena_alloc_array is a function taking element_size. */ \
        /* For maximum compatibility with original pattern: assume arena_alloc_array might be a macro using the TYPE name. */ \
        /* If arena_alloc_array is `void* arena_alloc_array(Arena* arena, size_t elem_size, size_t count)` */ \
        /* then you'd use `(TYPE*)arena_alloc_array(arena, sizeof(TYPE), new_max_capacity);` */ \
        /* Let's assume the user provides an `arena_alloc_array` that works with `(arena, TYPE, count)` pattern or similar logic. */ \
        /* The cast (TYPE*) is to ensure the pointer type is correct. */ \
        TYPE* new_data = (TYPE*)arena_alloc_array(arena, TYPE, new_max_capacity); \
        assert(new_data != NULL && "Arena allocation failed in reserve."); \
        \
        vec->data = new_data; \
        vec->max = new_max_capacity; \
        \
        IF_GENERIC_VECTOR_WITH_BASE_ASSERT(vec->reserve_called_flag = GV_INTERNAL_RESERVE_CALLED_MAGIC;) \
    } \
    \
    /* Adds an element to the end of the vector, resizing if necessary. */ \
    static inline void _GV_CONCAT3(NAME, _, push_back)(Arena *arena, NAME *vec, TYPE value) { \
        IF_GENERIC_VECTOR_WITH_BASE_ASSERT( \
            assert(vec->reserve_called_flag == GV_INTERNAL_RESERVE_CALLED_MAGIC && \
                   "Vector reserve() not called or state corrupted. Call init() then reserve() first, or ensure vector is already reserved."); \
        ) \
        \
        if (vec->size == vec->max) { \
            size_t new_max_capacity; \
            if (vec->max == 0) { \
                /* This handles push_back on a vector initialized by init() but not reserve() */ \
                /* (only if WITH_BASE_ASSERT is OFF, otherwise the assert above would catch it). */ \
                new_max_capacity = 8; /* Default initial capacity */ \
                IF_GENERIC_VECTOR_WITH_BASE_ASSERT( \
                    /* This state (max=0 AND reserve_called_flag set) should not be reachable if reserve() works correctly. */ \
                    if (vec->reserve_called_flag == GV_INTERNAL_RESERVE_CALLED_MAGIC) { \
                        assert(0 && "Internal logic error: reserve_called_flag is set, but vector max capacity is 0."); \
                    } \
                ) \
            } else { \
                new_max_capacity = 2 * vec->max; /* Double the current capacity */ \
            } \
            assert(new_max_capacity > vec->max && "New capacity calculation failed or resulted in no growth."); \
            \
            TYPE* new_data = (TYPE*)arena_alloc_array(arena, TYPE, new_max_capacity); \
            assert(new_data != NULL && "Arena allocation failed during resize in push_back."); \
            \
            if (vec->data && vec->size > 0) { /* Only copy if there's existing data */ \
                memcpy(new_data, vec->data, sizeof(TYPE) * vec->size); \
            } \
            /* With typical arena allocators, the old vec->data block is not explicitly freed here. */ \
            vec->data = new_data; \
            vec->max = new_max_capacity; \
        } \
        \
        /* Structs are copied by assignment. For large structs, this is a member-wise copy. */ \
        vec->data[vec->size] = value; \
        vec->size++; \
    }



DEFINE_VECTOR_FOR_TYPE(int64_t, vector_int64_t)
