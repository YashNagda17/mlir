#pragma once

#include <stdbool.h>
#include <base/arena.h>
#include <base/string.h>

// Returns the file contents as a null-terminated string in `text`.
// Returns `true` on success, otherwise `false`.
bool read_file(Arena *arena, const string filename, string *text);
string read_file_ok(Arena *arena, const string filename);
