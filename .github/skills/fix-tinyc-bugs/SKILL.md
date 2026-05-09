---
name: fix-tinyc-bugs
description: >
  Iteratively fix all tinyC bugs blocking a target (e.g. "fix everything
  blocking pixi run -e macos test_macos_tinyc", "make this program compile
  with tinyc", "fix all tinyc test failures"). Drives a loop of
  create-tinyc-mre → fix-tinyc-mre, one bug at a time, until the target
  command succeeds. Triggers: fix all tinyc bugs, fix tinyc bootstrap, get
  test_macos_tinyc to pass, make this compile with tinyc, fix everything,
  bootstrap tinyc, fix all failures.
---

# Fix tinyC Bugs — Drive the Bug-Fix Loop to a Passing Target

Some prompts ask for "fix everything" rather than a single bug. This skill
turns that open-ended request into a deterministic loop:

1. Run the target command.
2. If it passes, done.
3. If it fails, **delegate** to `create-tinyc-mre` to capture the first
   failure as a committed MRE.
4. **Delegate** to `fix-tinyc-mre` to fix it.
5. Loop.

This skill **does not** create or fix MREs itself. It orchestrates the two
single-purpose skills and enforces the contract between them: the MRE
handoff is always a `tinyc-mre:` commit at HEAD.

## When to Use This Skill

Use when the user asks for any of:
- "Fix all bugs blocking <command>"
- "Make this program compile / run with tinyc"
- "Get `pixi run -e macos test_macos_tinyc` to pass"
- "Fix all failing tinyC tests"
- "Bootstrap mlir with tinyc"

Do **not** use this skill when the user has a single specific bug — use
`create-tinyc-mre` directly (and then `fix-tinyc-mre`).

## Inputs to Gather

Ask the user (if not already clear):

1. **Target command** — the command that must end up passing. Defaults to
   `pixi run -e macos test_macos_tinyc` if the user said "bootstrap" or
   "compile this program".
2. **Iteration cap** — how many bugs to fix before stopping to report
   back. Default: 5. The user may want to review progress before continuing.
3. **Branch policy** — whether to work on the current branch or create a
   new one (`git switch -c fix-tinyc-bugs-<date>`). Default: ask.

## Procedure

### Phase 0: Sanity Check the Starting State

```bash
# Working tree must be clean — bug-fix loop must not entangle unrelated edits.
git status --porcelain          # must be empty
test -x ./tinyc || pixi run -e upstream build_tinyc_upstream
```

If the tree is dirty, stop and ask the user to commit or stash first.

### Phase 1: Run the Target

```bash
TARGET="<target command>"
$TARGET
```

If it exits 0 with no failures: **done**, print success summary, stop.

If it fails: capture the **first** failure (filename, error message, line
number). Subsequent failures may be cascades from the same root cause —
fix one at a time.

### Phase 2: Delegate to create-tinyc-mre

Invoke the `create-tinyc-mre` skill with the captured failure as input.
That skill:
- reduces the failure into `examples/tinyc/tests/<name>.tc`,
- adds the `tests.toml` entry,
- **commits** with subject `tinyc-mre: <name>`.

When create-tinyc-mre finishes, **verify the contract**:

```bash
git log -1 --format=%s | grep -q '^tinyc-mre: ' \
  || { echo "create-tinyc-mre did not produce a tinyc-mre: commit; aborting"; exit 1; }
```

If the contract isn't met, stop and report back — do not try to fix
without a committed MRE.

### Phase 3: Delegate to fix-tinyc-mre

Invoke the `fix-tinyc-mre` skill. It:
- reads HEAD (the MRE commit),
- fixes the bug in `corec/examples/tinyc/`,
- runs `test_tinyc_upstream`,
- commits inside `corec/` and bumps the submodule pointer in `mlir`.

After it returns, verify:

```bash
# The MRE test must now pass:
NAME=$(git log --grep='^tinyc-mre: ' -1 --format=%s | sed 's/^tinyc-mre: //')
pixi run -e upstream test_tinyc_upstream 2>&1 | grep -E "^FAIL: $NAME|All [0-9]+ tinyC tests passed"
```

Expect the "All N tinyC tests passed" line and **no** `FAIL: $NAME`.

### Phase 4: Loop

Increment the iteration counter and go back to Phase 1.

Stop when **any** of:
- target command exits 0,
- iteration cap is hit (report progress, ask whether to continue),
- two consecutive iterations fail to make progress on the same failure
  signature (likely an infinite loop — stop and ask the user),
- `create-tinyc-mre` or `fix-tinyc-mre` reports a hard failure.

### Phase 5: Final Verification and Summary

When the target finally passes, run the *full* set of regressions one last
time:

```bash
pixi run -e upstream test_tinyc_upstream
pixi run -e macos    test_macos                # clang baseline still works
pixi run -e macos    test_macos_tinyc          # the actual target
```

Print a summary listing every MRE commit and every fix-bump pair:

```
Fixed N tinyC bugs to make `<target>` pass.

  1. <SHA> tinyc-mre: <name1>     +  <SHA> tinyc: fix <name1>     +  <SHA> Bump corec
  2. <SHA> tinyc-mre: <name2>     +  <SHA> tinyc: fix <name2>     +  <SHA> Bump corec
  ...

Final status:
  test_tinyc_upstream  — PASS
  test_macos           — PASS
  test_macos_tinyc     — PASS
```

## Constraints

- **Never skip create-tinyc-mre.** Even "obvious" bugs go through the MRE
  pipeline so we get a committed regression test.
- **Never bundle fixes.** One MRE → one fix → one bump. If a fix happens
  to unblock multiple test failures, that's fine and good — the next loop
  iteration will discover the next root cause has gone away. Do not
  retroactively merge MREs.
- **Never amend `tinyc-mre:` commits.** They're the permanent record.
- **Don't iterate silently for hours.** Stop at the iteration cap and
  show the user where you are. They may want to review before paying for
  more compute.
- **Don't fix unrelated failures.** If the target command's output also
  shows pre-existing non-tinyC failures, surface them but don't try to
  fix them via this loop.
- **Commit hygiene**: any commits made by this skill or the skills it
  delegates to must NOT include `Co-authored-by` trailers.

## Tips

- **Cascading fixes**: one tinyC fix often makes 3-5 tests pass at once.
  The loop naturally handles this — the next iteration just finds the
  *next* genuine bug.
- **Large preprocessed files**: when the target is the bootstrap build,
  the failing source is one of the `.c.i` files in `build_tinyc/`. That
  goes to `create-tinyc-mre` as the input.
- **Watch the iteration count**: if you're past 5 iterations on a
  bootstrap target, it usually means the program exposes a fundamentally
  hard tinyC limitation (variadic ABI, atomics, etc.). Stop and discuss
  with the user rather than grinding.
