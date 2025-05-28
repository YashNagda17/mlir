set -ex

cl /std:c11 /Zc:preprocessor /I. /Fe:run_tests.exe tests/run_tests.c base/arena.c base/string.c base/format.c base/io.c
./run_tests
