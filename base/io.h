#pragma once

#include <stdbool.h>
#include <base/arena.h>
#include <base/string.h>
#include <base/format.h>

// Returns the file contents as a null-terminated string in `text`.
// Returns `true` on success, otherwise `false`.
bool read_file(Arena *arena, const string filename, string *text);
string read_file_ok(Arena *arena, const string filename);

void println_explicit_varg(Arena *arena, string fmt, size_t arg_count,
        va_list varg);
void println_explicit(Arena *arena, string fmt, size_t arg_count, ...);

#define println(arena, fmt, ...) \
    println_explicit(arena, fmt, COUNT_ARGS(__VA_ARGS__) __VA_OPT__(,) APPLY_WITH_COUNT(COUNT_ARGS(__VA_ARGS__) __VA_OPT__(,) __VA_ARGS__))
