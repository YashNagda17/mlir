#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <stdarg.h>
#include <assert.h>

#include <base/format.h>


int main() {
    Arena* arena = arena_create(1024);

    string text;
    bool ok = read_file(arena, str_lit("does not exist"), &text);
    assert(!ok)

    arena_free(arena);
    return 0;
}
