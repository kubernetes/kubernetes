# Linting the Windows node e2e package

The Kubernetes CI runs `golangci-lint` against the Linux build (`GOOS=linux`).
Because every file under `test/e2e_node_windows/` is tagged `//go:build windows`,
the Linux lint pass treats the package as if it had no source files, so issues
inside the Windows code (unused symbols, ineffectual assignments, etc.) are
never reported there.

To catch those issues, run the linter locally with `GOOS=windows`. This document
walks through the full setup on a Linux host.

## Prerequisites

- Linux host with Go 1.26+ matching the version in `go.mod`
- Kubernetes source tree checked out (these instructions assume the working
  directory is the repo root)

## One-time setup

The Kubernetes repo pins its own version of `golangci-lint` and several custom
plugins (`logcheck.so`, `kube-api-linter.so`, `sorted.so`) under
`hack/tools/golangci-lint/`. The wrapper script
`hack/verify-golangci-lint.sh` installs them into `_output/local/bin/`.

The tricky part: `go install` refuses to cross-compile binaries when `GOBIN` is
set, which is exactly what the wrapper does. So the install pass must run
natively (no `GOOS=windows`). Once installed, the binary is reusable.

```bash
# Run the wrapper once on a Linux package so it installs all tools.
# The Linux lint pass on this package isn't useful for our purposes, but it
# completes the install step we care about. Ignore the lint output.
hack/verify-golangci-lint.sh -- ./test/e2e_node/builder/... 2>&1 | tail -20
```

After this runs, you should have:

```
_output/local/bin/
├── golangci-lint
├── kube-api-linter.so
├── logcheck.so
└── sorted.so
```

Quick sanity check:

```bash
_output/local/bin/golangci-lint --version
```

### Alternative manual install

If you'd rather not invoke the wrapper, install the same components directly
using the pinned Go toolchain from `hack/tools/golangci-lint/go.mod`:

```bash
mkdir -p _output/local/bin
GOBIN=$PWD/_output/local/bin \
  go -C hack/tools/golangci-lint install \
  github.com/golangci/golangci-lint/v2/cmd/golangci-lint

GOBIN=$PWD/_output/local/bin \
  go -C hack/tools/golangci-lint build \
  -o $PWD/_output/local/bin/logcheck.so \
  -buildmode=plugin \
  sigs.k8s.io/logtools/logcheck/plugin

GOBIN=$PWD/_output/local/bin \
  go -C hack/tools/golangci-lint build \
  -o $PWD/_output/local/bin/kube-api-linter.so \
  -buildmode=plugin \
  sigs.k8s.io/kube-api-linter/pkg/plugin

GOBIN=$PWD/_output/local/bin \
  go -C hack/tools/golangci-lint build \
  -o $PWD/_output/local/bin/sorted.so \
  -buildmode=plugin \
  k8s.io/kubernetes/hack/tools/golangci-lint/sorted/plugin
```

Note: `GOOS` and `GOARCH` must **not** be set during installation. The lint
tool and its plugins are host-native (Linux); they analyse Windows code via
Go's `types` package, which doesn't care about the target platform of the code
under analysis.

## Running the lint with `GOOS=windows`

Once `_output/local/bin/golangci-lint` exists, run the linter against the
Windows package with the target platform set:

```bash
GOOS=windows GOARCH=amd64 \
  _output/local/bin/golangci-lint run \
    --config hack/golangci.yaml \
    ./test/e2e_node_windows/...
```

Exit code:

- `0` — no issues found
- non-zero — issues were reported (printed before the prompt returns)

The first run is slow (Go has to typecheck the whole repo's transitive
imports against `GOOS=windows`). Subsequent runs are heavily cached and
complete in seconds for incremental changes.

## Convenience wrapper

If you'll re-run frequently, drop a small script into the repo (do not commit
it unless agreed) so you don't have to remember the flags:

```bash
cat > /tmp/lint-windows.sh <<'EOF'
#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel)"
GOOS=windows GOARCH=amd64 \
  _output/local/bin/golangci-lint run \
    --config hack/golangci.yaml \
    "$@" \
    ./test/e2e_node_windows/...
EOF
chmod +x /tmp/lint-windows.sh

# Run it
/tmp/lint-windows.sh
```

## Compile-only sanity check

Lint runs full static analysis, which can be slow. For quick "did I break the
build?" checks, use `go vet` or `go build` with the same env vars:

```bash
# Build only (no test files)
GOOS=windows GOARCH=amd64 go build ./test/e2e_node_windows/...

# Build the test binary too (slower, but catches test-file errors)
GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./test/e2e_node_windows

# Vet — fast, catches the most common issues
GOOS=windows GOARCH=amd64 go vet ./test/e2e_node_windows/...
```

## Common issues you may see

- **`unused`** — a function, variable, or constant is declared but never
  referenced from any reachable Windows code. Usually these are leftovers
  copied from `test/e2e_node/` (the Linux package) without porting the call
  sites. Delete them.
- **`ineffassign`** — a value is assigned to a variable but never read before
  the next assignment. Often a missed `err` check after a function call.
- **`typecheck`** — undefined symbol. Most often caused by `//go:build`
  tags excluding the file that defines the symbol. Verify build tags align
  between definition and usage.

## Why not enable this in CI?

The Kubernetes CI lint pipeline assumes `GOOS=linux`. Running a second pass
with `GOOS=windows` would require changes to the prow/golangci.yaml
infrastructure (out of scope for this PR). Until that's done, contributors
should run the steps above locally before submitting changes that touch
`test/e2e_node_windows/`.
