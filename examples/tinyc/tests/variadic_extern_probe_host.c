// Host implementation for variadic_extern_probe.tc (compiled by clang at link time).
//
// echo_check(n_extra, a, b, ...):
//   fixed:    a == 10, b == 20
//   optional: n_extra values must be 30, then 40 (hardcoded).
// Returns 0 if all match, non-zero otherwise.

#include <stdarg.h>

int echo_check(int n_extra, int a, int b, ...) {
    if (a != 10 || b != 20)
        return 1;
    if (n_extra < 0 || n_extra > 2)
        return 2;

    static const int expected[2] = {30, 40};

    va_list ap;
    va_start(ap, b);
    for (int i = 0; i < n_extra; i++) {
        int got = va_arg(ap, int);
        if (got != expected[i]) {
            va_end(ap);
            return 10 + i;
        }
    }
    va_end(ap);
    return 0;
}
