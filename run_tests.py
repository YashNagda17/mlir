#!/usr/bin/env python
"""Test runner for the MLIR repo.

Each entry in tests/tests.toml is a single input file. References are
stored under tests/reference/<stem>.canonical.out, <stem>.classic.out,
and <stem>.generic.out. References are generated using the upstream
backend (treated as ground truth):

    canonical.out  =  ./parser_upstream FILE --parse=upstream --print=upstream
    classic.out    =  ./parser_upstream FILE --parse=upstream --print=classic
    generic.out    =  ./parser_upstream FILE --parse=upstream --print=generic

When a file is marked `upstream_parser = false` (e.g. it uses the
Triton dialect which is not in upstream MLIR), references fall back to
using the classic parser, and the canonical reference is omitted.

The runner then exercises every supported (parser, printer, backend)
combination and diffs against the matching reference.

Usage:
    python run_tests.py            # run all tests against committed refs
    python run_tests.py -u         # regenerate references
    python run_tests.py --upstream # only test the upstream backend
    python run_tests.py --native   # only test the native backend
"""
import argparse
import os
import subprocess
import sys
import shutil
import toml

ROOT = os.path.abspath(os.path.dirname(__file__))
TESTS = os.path.join(ROOT, "tests")
REF = os.path.join(TESTS, "reference")
OUT = os.path.join(TESTS, "output")

NATIVE = os.path.join(ROOT, "parser")
UPSTREAM = os.path.join(ROOT, "parser_upstream")
if os.name == "nt":
    NATIVE += ".exe"
    UPSTREAM += ".exe"


class Fail(Exception):
    pass


def run_capture(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr


def diff(a_path, b_path):
    """Return None if files match, or a diff string if they differ."""
    with open(a_path, "rb") as f:
        a = f.read()
    with open(b_path, "rb") as f:
        b = f.read()
    if a == b:
        return None
    # Use system diff for nice output
    try:
        return subprocess.run(
            ["diff", "-u", a_path, b_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return f"<files differ: {a_path} vs {b_path}>"


def run_combo(exe, infile, parse_kind, print_kind):
    """Run `exe infile --parse=KIND --print=KIND`. Returns stdout bytes
    on success or raises Fail on non-zero exit."""
    cmd = [exe, infile, f"--parse={parse_kind}", f"--print={print_kind}"]
    rc, out, err = run_capture(cmd)
    if rc != 0:
        raise Fail(f"command failed (rc={rc}): {' '.join(cmd)}\nstderr:\n{err.decode('utf-8', errors='replace')}")
    return out


def stem(filename):
    return os.path.splitext(os.path.basename(filename))[0] + os.path.splitext(filename)[1].replace('.', '_')


def ref_path(filename, parse_kind, print_kind):
    # e.g. tests/reference/simple_mlir.upstream.upstream.out
    return os.path.join(REF, f"{stem(filename)}.{parse_kind}.{print_kind}.out")


# All (parse, print) combos we test. Each row is exercised on both backends
# (subject to backend support). When `upstream_parser` is false in tests.toml,
# only the classic-parser rows are exercised.
COMBOS_CLASSIC_PARSER = [
    ("classic", "classic"),
    ("classic", "generic"),
]
COMBOS_UPSTREAM_PARSER = [
    ("upstream", "upstream"),  # canonical
    ("upstream", "classic"),
    ("upstream", "generic"),
]


def ensure_refs(filename, upstream_parser):
    """Generate per-(parser,printer) reference files using parser_upstream."""
    infile = os.path.join(TESTS, filename)
    rows = list(COMBOS_CLASSIC_PARSER)
    if upstream_parser:
        rows.extend(COMBOS_UPSTREAM_PARSER)
    for parse_k, print_k in rows:
        try:
            out = run_combo(UPSTREAM, infile, parse_k, print_k)
        except Fail as e:
            raise Fail(f"failed to generate ref {parse_k}/{print_k} for {filename}: {e}")
        path = ref_path(filename, parse_k, print_k)
        with open(path, "wb") as f:
            f.write(out)


def check_combo(exe, filename, parse_kind, print_kind):
    """Run a (parse, print, backend) combo and diff against the matching
    {parse}.{print} reference file."""
    infile = os.path.join(TESTS, filename)
    ref = ref_path(filename, parse_kind, print_kind)
    if not os.path.exists(ref):
        raise Fail(f"missing reference {ref}")
    out = run_combo(exe, infile, parse_kind, print_kind)
    out_path = os.path.join(OUT, f"{stem(filename)}.{os.path.basename(exe)}.{parse_kind}.{print_kind}.out")
    with open(out_path, "wb") as f:
        f.write(out)
    d = diff(ref, out_path)
    if d is not None:
        raise Fail(
            f"{os.path.basename(exe)} {filename} --parse={parse_kind} --print={print_kind}"
            f" differs from reference {os.path.basename(ref)}:\n{d}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--update", action="store_true",
                    help="regenerate references")
    ap.add_argument("-s", "--sequential", action="store_true",
                    help="(accepted for compatibility; tests already run sequentially)")
    ap.add_argument("--upstream", action="store_true",
                    help="run upstream-backend tests only")
    ap.add_argument("--native", action="store_true",
                    help="run native-backend tests only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    do_native = (not args.upstream) or args.native
    do_upstream = (not args.native) or args.upstream

    os.makedirs(REF, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    cfg = toml.load(os.path.join(TESTS, "tests.toml"))
    tests = cfg["test"]

    if args.update:
        # Regenerate references; require parser_upstream.
        if not os.path.exists(UPSTREAM):
            print(f"error: {UPSTREAM} is required to regenerate references", file=sys.stderr)
            sys.exit(1)
        # Wipe stale refs.
        if os.path.isdir(REF):
            shutil.rmtree(REF)
        os.makedirs(REF, exist_ok=True)
        for t in tests:
            filename = t["filename"]
            upstream_parser = t.get("upstream_parser", True)
            print(f"  ref {filename} (upstream_parser={upstream_parser})")
            ensure_refs(filename, upstream_parser)
        print("References updated.")
        return

    failures = []
    n = 0
    for t in tests:
        filename = t["filename"]
        upstream_parser = t.get("upstream_parser", True)
        # Native backend: classic-parser combos only.
        native_combos = list(COMBOS_CLASSIC_PARSER)
        # Upstream backend: classic-parser combos always; upstream-parser
        # combos only if the upstream parser handles the file.
        upstream_combos = list(COMBOS_CLASSIC_PARSER)
        if upstream_parser:
            upstream_combos.extend(COMBOS_UPSTREAM_PARSER)

        if do_native and os.path.exists(NATIVE):
            for parse_k, print_k in native_combos:
                n += 1
                desc = f"native {filename} parse={parse_k} print={print_k}"
                if args.verbose:
                    print(f"  {desc}")
                try:
                    check_combo(NATIVE, filename, parse_k, print_k)
                except Fail as e:
                    failures.append((desc, str(e)))
        if do_upstream and os.path.exists(UPSTREAM):
            for parse_k, print_k in upstream_combos:
                n += 1
                desc = f"upstream {filename} parse={parse_k} print={print_k}"
                if args.verbose:
                    print(f"  {desc}")
                try:
                    check_combo(UPSTREAM, filename, parse_k, print_k)
                except Fail as e:
                    failures.append((desc, str(e)))

    if failures:
        print(f"\n{len(failures)}/{n} FAILED:\n")
        for desc, err in failures:
            print(f"--- {desc} ---")
            print(err)
            print()
        sys.exit(1)
    print(f"All {n} tests passed.")


if __name__ == "__main__":
    main()
