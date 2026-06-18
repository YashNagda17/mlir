// Tiny runtime for tinyC: implements vector.print's lowered hooks
// (printI64, printNewline) so that programs lowered via the standard
// MLIR conversion pipeline can be linked into a real executable.

#include <stdio.h>
#include <stdint.h>

#ifndef TINYC_RUNTIME_VA_ONLY
void printI64(int64_t v) { printf("%lld", (long long)v); }
void printNewline(void)  { printf("\n"); }
// vector.print on i32 actually lowers via signext to printI64 above, but
// some MLIR versions emit a direct printI32 call — provide it for safety.
void printI32(int32_t v) { printf("%d\n", v); }
// Float printing: tinyC's `_tinyc_print(<float>)` lowers to a direct call to
// printF32 followed by printNewline. We use %g for compact, exact-enough
// output that matches typical C float formatting.
static void printF64Trimmed(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", v);
    int n = 0;
    while (buf[n]) n++;
    while (n > 0 && buf[n - 1] == '0') n--;
    if (n > 0 && buf[n - 1] == '.') n--;
    buf[n] = '\0';
    printf("%s", buf);
}
void printF32(float v) { printF64Trimmed((double)v); }
void printF64(double v) { printF64Trimmed(v); }

// tinyC's `_tinyc_print(<string>)` lowers to a direct call to @printStr. We
// emit the bytes (which are NUL-terminated by the string-literal global)
// followed by a newline to mirror the behavior of `_tinyc_print(<int>)` and
// `_tinyc_print(<float>)`.
void printStr(const char *s) {
    printf("%s\n", s ? s : "");
}
#endif


#include <stdarg.h>
// va_arg helpers used by tinyC. The MLIR LLVM dialect doesn't expose a
// `va_arg` op, and clang traditionally lowers va_arg manually anyway, so
// the tinyC compiler emits calls to these wrappers instead. The va_list
// in the caller is a 32-byte alloca passed by pointer; we dereference it
// here, run the standard va_arg macro, and return the value.
int            tinyc_va_arg_i32(va_list *ap) { return va_arg(*ap, int); }
long long      tinyc_va_arg_i64(va_list *ap) { return va_arg(*ap, long long); }
double         tinyc_va_arg_f64(va_list *ap) { return va_arg(*ap, double); }
void          *tinyc_va_arg_ptr(va_list *ap) { return va_arg(*ap, void *); }

// Generic struct va_arg: copies `size` bytes (rounded up to 8) from the
// va_list into `out` by reading consecutive 8-byte words. Works for
// Darwin ARM64 / SysV x86_64 where small structs occupy contiguous slots
// in the variadic argument area.
void tinyc_va_arg_struct(va_list *ap, void *out, long long size) {
    long long *o = (long long *)out;
    long long words = (size + 7) / 8;
    for (long long i = 0; i < words; i++) {
        o[i] = va_arg(*ap, long long);
    }
}
