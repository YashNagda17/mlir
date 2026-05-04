#!/usr/bin/env bash
# Run the full MLIR test suite using parser_upstream in place of parser.
# Tests marked `upstream_strict = true` in tests/tests.toml are diffed
# against the native reference (must match exactly). All other tests
# are smoke-tested: parser_upstream must parse them without error, but
# output is not compared (since the upstream backend cannot represent
# everything our native impl does — comments, register names, etc.).

set -u
cd "$(dirname "$0")/.."

cleanup() {
    rc=$?
    rm -f parser
    if [ -e parser.native.bak ]; then
        mv parser.native.bak parser
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

if [ -e parser ]; then
    mv parser parser.native.bak
fi
cp parser_upstream parser

python run_tests.py -s --upstream
