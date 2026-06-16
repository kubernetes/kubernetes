# Kubernetes

## Communication Preferences

- Dry, concise, low-key humor. No flattery, no forced memes. Skip preambles and postambles.
- Comments explain "why", not "what".
- Error messages: actionable and specific. No vague "something went wrong" output.

## Constraints (read first)

- **Generated files are read-only.** Never hand-edit `zz_generated.*`, `generated.pb.go`,
  `types_swagger_doc_generated.go`, or anything in `pkg/generated/`. Run `make update` to refresh.
- **`go.mod` / `go.work` are generated.** Use `hack/pin-dependency.sh` and
  `hack/update-vendor.sh`. Never run `go mod tidy` or `go get` against the main module.
- **Staging is source of truth** for `k8s.io/*` packages (`staging/src/k8s.io/`).
  Edits to a `k8s.io/api`, `k8s.io/apimachinery`, `k8s.io/client-go`, etc. type
  must happen in `staging/src/k8s.io/...`, then `make update` to re-vendor.
  Never import `k8s.io/kubernetes` from a staging module.
- **Boilerplate required.** Every `.go` and `.sh` file needs the Apache-2.0
  license header. Templates: `hack/boilerplate/boilerplate.go.txt`,
  `hack/boilerplate/boilerplate.sh.txt`. `make verify` runs
  `hack/verify-boilerplate.sh` and will fail CI otherwise.
- **API compatibility.** New API fields must be additive. See `api/api-rules/`
  and the API conventions in the [community guide].

## Build / Lint / Test

`make help` lists every target. Common workflows:

```
make test WHAT=./pkg/kubelet                       # unit tests for one package
make test WHAT=./pkg/kubelet GOFLAGS=-v -run TestX  # one test, one pkg
make test-integration WHAT=./test/integration/scheduler
make lint                                            # golangci-lint (hack/verify-golangci-lint.sh)
make quick-verify                                    # fast presubmit checks (gofmt, imports, boilerplate, spelling, pkg names)
make verify                                          # full verify, runs every hack/verify-*.sh
make verify WHAT=gofmt typecheck boilerplate        # subset of verify checks
make update                                          # regenerate everything
```

### Running a single test

The Make `test` target wraps `go test` and applies race detection, JUnit
output, and the gotestsum runner. Useful one-liners:

```
# by package + regex (uses the test target's go test plumbing)
make test WHAT=./pkg/kubelet GOFLAGS="-v -run TestSyncPod" KUBE_TIMEOUT=120s

# direct go test, fastest iteration loop once the toolchain is warm
go test -race -count=1 -run '^TestSyncPod$' ./pkg/kubelet/...

# integration tests
make test-integration WHAT=./test/integration/scheduler KUBE_TEST_ARGS='-run ^TestScheduler$'
```

### Environment

- Go version pinned in `.go-version` (1.26.x). Repo is a Go workspace
  (`go.work`); many staging modules are first-class.
- On macOS, raise `ulimit -n` to >= 1000 or some tests panic
  (`hack/make-rules/test.sh:checkFDs`).
- Tests set `KUBE_CACHE_MUTATION_DETECTOR=true` and
  `KUBE_PANIC_WATCH_DECODE_ERROR=true` by default.

## Code Style

### Files & packages

- Package name: lowercase, single word, must match its directory
  (verified by `hack/verify-pkg-names.sh`).
- File header: copy boilerplate from `hack/boilerplate/boilerplate.go.txt`.
- Imports: three groups separated by blank lines, ordered stdlib / third-party
  / k8s.io. Aliased imports are common for disambiguation
  (e.g. `apiequality "k8s.io/apimachinery/pkg/api/equality"`,
  `v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"`,
  `sysruntime "runtime"`). Use `hack/update-import-aliases.sh` after edits.
- `depguard` forbids `k8s.io/utils/pointer` (use `k8s.io/utils/ptr`) and
  forbids `github.com/google/go-cmp/cmp` outside test files.

### Linting

- `hack/golangci.yaml.in` is the source; `hack/golangci.yaml` and
  `hack/golangci-hints.yaml` are generated. Linters enabled in addition to
  `standard`: `depguard`, `forbidigo`, `ginkgolinter`, `gocritic`, `govet`,
  `ineffassign`, `kubeapilinter`, `logcheck`, `modernize`, `revive`, `sorted`,
  `staticcheck`, `testifylint`, `unused`. The "hints" config adds `errorlint`
  and `usestdlibvars`.
- `forbidigo` bans: `AnnotatedEventf` (use plain events), `md5.*`
  (use sha256), `managedfields.ExtractInto`, `applyconfigurations.Extract`,
  non-`AddVersioned` feature gates, Ginkgo `ReportBeforeSuite`/`ReportAfterSuite`
  outside SIG Testing, and bare `gomega.BeTrue`/`BeFalse` (use
  `BeTrueBecause`/`BeFalseBecause` with an explicit message).
- Conversion functions intentionally use underscores: `Convert_v1_To_…`,
  `SetDefaults_…`. Do not rename them — `staticcheck`/`revive` exclude them.

### Naming, types, errors

- Receiver names: short, consistent per type (commonly 1–2 letters),
  `staticcheck`'s ST1016 is disabled in base config.
- Error variables: name the type `errFoo`, the sentinel `ErrFoo`.
- Wrap with `fmt.Errorf("doing x: %w", err)`; messages lowercase, no trailing
  period. Use `k8s.io/apimachinery/pkg/util/errors` for joining.
- Logging: migrated packages use `klog` with contextual logging
  (see `hack/logcheck.conf`). Many packages still use structured-only
  logging. Never use `fmt.Println` / `log.Println` in production code.
- Logging helper import alias pattern: `clog "k8s.io/utils/logger"`.

### Tests

- Use Ginkgo + Gomega for integration / e2e; use stdlib `testing` + testify
  for unit tests. `testifylint` is enabled.
- New test utilities belong in `test/utils/` or `k8s.io/utils/ktesting`
  for the `TContext` pattern, not in a per-package helper.
- E2E test images live in `test/images/`; add new ones with
  `hack/update-test-images.sh`.

### Generated & API code

- Don't hand-edit `staging/src/k8s.io/api/**/types.go` field tags or json
  ordering. Modify the source struct and run `make update-codegen`.
- Feature gates are sorted (enforced by the `sorted` linter). New gates go in
  `pkg/features/kube_features.go` (or the staging equivalent for the binary
  that uses it) with `AddVersioned`. Do not use the deprecated `Add`.

## Contributor Workflow

- One logical change per PR; keep diffs reviewable.
- Add or update tests for any behavior change. Bug fixes need a regression
  test.
- Run `make quick-verify` and the relevant `make test WHAT=…` before pushing.
- Commit messages: short summary line, optional body. No `@mentions`, no
  `Fixes #…` / `Closes #…` keywords, no `Co-authored-by:` trailers. PR body
  is where issue linking lives.
- Use the PR template's `/kind` labels (`bug`, `feature`, `cleanup`, `flake`,
  `failing-test`, `api-change`, `regression`, `documentation`).
