// Native (arm64/Darwin, LP64) runtime for the `--macho-backend=llvm`
// backend. Unlike the wmir backend — which hand-emits malloc/printf/str*/
// print* as ~2500 lines of aarch64 "synth_*" MLIR — the llvm backend gets
// its runtime as ordinary tinyC-subset C, parsed into the user's module and
// lowered through the same llvm -> aarch64 path. The only primitives that
// cannot be expressed in C are the raw syscalls; those bind to libSystem
// dyld stubs by their underscore names (`_write`, `_exit`, `_mmap`, ...),
// which the Mach-O encoder already imports.
//
// Semantics mirror runtime_wasm.c exactly so output matches the wmir suite:
//   printI64(v)    -> decimal digits, NO trailing newline
//   printNewline() -> "\n"
//   printI32(v)    -> decimal digits + "\n"
//   printStr(s)    -> the string bytes + a trailing "\n"
//
// Each public runtime function is injected only if the user program does not
// already define it (forward declarations don't count), so user-provided
// implementations (e.g. selfhost's stdlib) win and we never hit a "two
// definitions" parse error.
#ifndef TINYC_NATIVE_RUNTIME_H
#define TINYC_NATIVE_RUNTIME_H

#include <base/arena.h>
#include <base/string.h>
#include "tinyc.h"

// One public runtime function per entry: its name (for the user-provides
// guard) and its tinyC-subset C source. Definitions are ordered so that a
// function only calls ones defined earlier.
typedef struct {
    const char *name;
    const char *src;
} TinycRuntimeFn;

static const char TINYC_RT_PRELUDE[] =
    "long _write(long fd, char *buf, long n);\n";

static const TinycRuntimeFn TINYC_RUNTIME_FNS[] = {
    { "printNewline",
      "void printNewline(void){ char c; c=10; _write(1,&c,1); }\n" },
    { "printI64",
      "void printI64(long v){\n"
      "  char tmp[32]; char out[34]; unsigned long u; int neg; int n; int o; int i;\n"
      "  neg=0;\n"
      "  if(v<0){ neg=1; u=(unsigned long)(-(v+1))+1; } else { u=(unsigned long)v; }\n"
      "  n=0;\n"
      "  if(u==0){ tmp[n]=48; n=n+1; }\n"
      "  while(u){ tmp[n]=(char)(48+(u%10)); n=n+1; u=u/10; }\n"
      "  o=0;\n"
      "  if(neg){ out[o]=45; o=o+1; }\n"
      "  i=n-1;\n"
      "  while(i>=0){ out[o]=tmp[i]; o=o+1; i=i-1; }\n"
      "  _write(1,out,(long)o);\n"
      "}\n" },
    { "printI32",
      "void printI32(int v){ printI64((long)v); printNewline(); }\n" },
    { "printStr",
      "void printStr(char *s){\n"
      "  long n; char c;\n"
      "  n=0;\n"
      "  if(s){ while(s[n]) n=n+1; _write(1,s,n); }\n"
      "  c=10; _write(1,&c,1);\n"
      "}\n" },
};
#define TINYC_RUNTIME_FN_COUNT \
    (sizeof(TINYC_RUNTIME_FNS) / sizeof(TINYC_RUNTIME_FNS[0]))

// True if `prog` already DEFINES (not just forward-declares) a function
// named `name`.
static bool tinyc_user_defines(Program *prog, const char *name) {
    size_t nlen = strlen(name);
    for (size_t i = 0; i < prog->funcs.size; i++) {
        Func *f = prog->funcs.data[i];
        if (f->is_forward) continue;
        if (f->name.size == nlen && memcmp(f->name.str, name, nlen) == 0)
            return true;
    }
    return false;
}

// Inject the native runtime into `prog` (parsed in-place). Only functions the
// user hasn't defined are added. No-op if nothing needs injecting.
static void tinyc_inject_native_runtime(Arena *arena, Program *prog,
                                        bool target_wasm32) {
    string combined = (string){0};
    bool any = false;
    for (size_t i = 0; i < TINYC_RUNTIME_FN_COUNT; i++) {
        if (tinyc_user_defines(prog, TINYC_RUNTIME_FNS[i].name)) continue;
        string s = str_from_cstr_view((char *)TINYC_RUNTIME_FNS[i].src);
        combined = any ? str_concat(arena, combined, s) : s;
        any = true;
    }
    if (!any) return;
    string full = str_concat(arena,
                             str_from_cstr_view((char *)TINYC_RT_PRELUDE),
                             combined);
    VecTcTok toks = tinyc_lex(arena, full);
    tinyc_parse_into(arena, prog, toks, target_wasm32);
}

#endif // TINYC_NATIVE_RUNTIME_H
