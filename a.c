#include <stdio.h>

#define NARGS(...) __NARGS(0 __VA_OPT__(,) __VA_ARGS__, 5,4,3,2,1,0)
#define __NARGS(_0,_1,_2,_3,_4,_5,N,...) N


int main()
{
  // NARGS
  printf("%d\n", NARGS());          // Prints 0
  printf("%d\n", NARGS(1));         // Prints 1
  printf("%d\n", NARGS(1, 2));      // Prints 2
  fflush(stdout);

#ifdef _MSC_VER
  // NARGS minus 1
  printf("\n");
  printf("%d\n", NARGS_1(1));       // Prints 0
  printf("%d\n", NARGS_1(1, 2));    // Prints 1
  printf("%d\n", NARGS_1(1, 2, 3)); // Prints 2
#endif

  return 0;
}
