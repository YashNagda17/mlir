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
    // Variadic support. tinyC lowers `va_arg(ap, T)` to a call to one of these
    // helpers (the backend lowers va_start/va_end). On Darwin arm64 the va_list
    // buffer's first 8 bytes hold a `cur` pointer into the on-stack variadic
    // area; each helper reads the next 8-byte slot and advances cur.
    { "tinyc_va_arg_i32",
      "int tinyc_va_arg_i32(char **ap){\n"
      "  char *p; int *q; p=*ap; *ap=p+8; q=(int*)p; return *q;\n"
      "}\n" },
    { "tinyc_va_arg_i64",
      "long tinyc_va_arg_i64(char **ap){\n"
      "  char *p; long *q; p=*ap; *ap=p+8; q=(long*)p; return *q;\n"
      "}\n" },
    { "tinyc_va_arg_f64",
      "double tinyc_va_arg_f64(char **ap){\n"
      "  char *p; double *q; p=*ap; *ap=p+8; q=(double*)p; return *q;\n"
      "}\n" },
    { "tinyc_va_arg_ptr",
      "char *tinyc_va_arg_ptr(char **ap){\n"
      "  char *p; char **q; p=*ap; *ap=p+8; q=(char**)p; return *q;\n"
      "}\n" },
    // Integer formatting helpers used by printf.
    { "tinyc_rt_fmt_u64",
      "int tinyc_rt_fmt_u64(unsigned long v, char *buf, int base, int upper){\n"
      "  char tmp[32]; int n; int o; int i; char *D;\n"
      "  if(upper){ D=\"0123456789ABCDEF\"; } else { D=\"0123456789abcdef\"; }\n"
      "  n=0;\n"
      "  if(v==0){ tmp[n]='0'; n=n+1; }\n"
      "  while(v){ tmp[n]=D[v%(unsigned long)base]; n=n+1; v=v/(unsigned long)base; }\n"
      "  o=0; i=n-1;\n"
      "  while(i>=0){ buf[o]=tmp[i]; o=o+1; i=i-1; }\n"
      "  return o;\n"
      "}\n" },
    { "tinyc_rt_fmt_i64",
      "int tinyc_rt_fmt_i64(long v, char *buf){\n"
      "  unsigned long u; int neg; int k;\n"
      "  neg=0;\n"
      "  if(v<0){ neg=1; u=(unsigned long)(-(v+1))+1; } else { u=(unsigned long)v; }\n"
      "  if(neg){ buf[0]='-'; }\n"
      "  k=tinyc_rt_fmt_u64(u, buf+neg, 10, 0);\n"
      "  return k+neg;\n"
      "}\n" },
    // Minimal printf: supports %d %i %u %x %X %p %c %s %% (and length modifiers
    // l/ll). Float specifiers (%f %g) are not yet implemented. Semantics mirror
    // runtime_wasm.c's vprintf_impl for the integer/string subset.
    { "printf",
      "int printf(char *fmt, ...){\n"
      "  char buf[40]; int total; int i; int prec; int lcount; char c;\n"
      "  __builtin_va_list ap; __builtin_va_start(ap, fmt);\n"
      "  total=0; i=0;\n"
      "  while(fmt[i]){\n"
      "    if(fmt[i] != '%'){ _write(1,fmt+i,1); total=total+1; i=i+1; continue; }\n"
      "    i=i+1;\n"
      "    prec=-1;\n"
      "    if(fmt[i]=='.'){ i=i+1; prec=0; while(fmt[i]>='0' && fmt[i]<='9'){ prec=prec*10+(fmt[i]-'0'); i=i+1; } }\n"
      "    lcount=0;\n"
      "    while(fmt[i]=='l'){ lcount=lcount+1; i=i+1; }\n"
      "    c=fmt[i];\n"
      "    if(c==0){ break; }\n"
      "    i=i+1;\n"
      "    if(c=='d' || c=='i'){\n"
      "      long v; int n;\n"
      "      if(lcount>=1){ v=__builtin_va_arg(ap,long); } else { v=(long)__builtin_va_arg(ap,int); }\n"
      "      n=tinyc_rt_fmt_i64(v,buf); _write(1,buf,(long)n); total=total+n;\n"
      "    } else if(c=='u'){\n"
      "      unsigned long v; int n;\n"
      "      v=(unsigned long)__builtin_va_arg(ap,unsigned long);\n"
      "      n=tinyc_rt_fmt_u64(v,buf,10,0); _write(1,buf,(long)n); total=total+n;\n"
      "    } else if(c=='x' || c=='X'){\n"
      "      unsigned long v; int n; int up; up=0; if(c=='X'){ up=1; }\n"
      "      v=(unsigned long)__builtin_va_arg(ap,unsigned long);\n"
      "      n=tinyc_rt_fmt_u64(v,buf,16,up); _write(1,buf,(long)n); total=total+n;\n"
      "    } else if(c=='p'){\n"
      "      char *p; int n; p=__builtin_va_arg(ap,char*);\n"
      "      _write(1,\"0x\",2); total=total+2;\n"
      "      n=tinyc_rt_fmt_u64((unsigned long)p,buf,16,0); _write(1,buf,(long)n); total=total+n;\n"
      "    } else if(c=='c'){\n"
      "      int v; char ch; v=__builtin_va_arg(ap,int); ch=(char)v; _write(1,&ch,1); total=total+1;\n"
      "    } else if(c=='s'){\n"
      "      char *s; long sn; s=__builtin_va_arg(ap,char*);\n"
      "      if(!s){ s=\"(null)\"; }\n"
      "      sn=0; while(s[sn]){ sn=sn+1; } _write(1,s,sn); total=total+(int)sn;\n"
      "    } else if(c=='%'){\n"
      "      _write(1,\"%\",1); total=total+1;\n"
      "    } else {\n"
      "      _write(1,\"%\",1); _write(1,&c,1); total=total+2;\n"
      "    }\n"
      "  }\n"
      "  __builtin_va_end(ap);\n"
      "  return total;\n"
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
