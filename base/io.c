#include <stdbool.h>
#include <stdio.h>

#include <base/io.h>


// Returns the file contents as a null-terminated string in `text`.
// Returns `true` on success, otherwise `false`.
bool read_file(Arena *arena, const string filename, string *text) {
    char *cfilename = str_to_cstr_copy(arena, filename);
    if (cfilename == NULL || *cfilename == '\0') return false;
    FILE *file = fopen(cfilename, "rb");
    if (file == NULL) return false;
    fseek(file, 0, SEEK_END);
    uint64_t filesize = ftell(file);
    if (filesize < 0) {
        fclose(file);
        return false;
    }
    fseek(file, 0, SEEK_SET);
    char *bytes = arena_alloc_array(arena, char, filesize+1);
    if (bytes == NULL) {
        fclose(file);
        return false;
    }
    size_t readsize = fread(bytes, 1, filesize, file);
    fclose(file);
    if (readsize != filesize) return false;
    bytes[readsize] = '\0';
    text->str=bytes;
    text->size=filesize+1;
    return true;
}


string read_file_ok(Arena *arena, const string filename) {
    string text;
    if (read_file(arena, filename, &text)) {
        return text;
    } else {
        //std::cerr << "File '" + filename + "' cannot be opened." << std::endl;
        printf("File cannot be opened.\n");
        abort();
    }
}

void println_explicit(Arena *arena, string fmt, size_t arg_count, ...) {
    va_list varg;
    va_start(varg, arg_count);
    string text = format_explicit_varg(arena, fmt, arg_count, varg);
    va_end(varg);
    text = str_concat(arena, text, str_lit("\n"));
    printf("%s", str_to_cstr_copy(arena, text));
}
