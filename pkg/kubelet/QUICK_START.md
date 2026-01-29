# Kubelet Development Quick Start

A consolidated guide for setting up your development environment, testing changes, and submitting code for Kubernetes Kubelet development.

## Prerequisites

- **Go**: Version 1.23+ (check with `go version`)
- **Git**: For version control
- **Docker**: Optional, for containerized builds
- **GitHub Account**: Required for contributions
- **Signed CLA**: Required before first PR ([sign here](https://git.k8s.io/community/CLA.md))

## Initial Setup

### 1. Fork and Clone

```bash
# Fork kubernetes/kubernetes on GitHub, then:
git clone https://github.com/<your-username>/kubernetes.git $GOPATH/src/k8s.io/kubernetes
cd $GOPATH/src/k8s.io/kubernetes

# Add upstream remote
git remote add upstream https://github.com/kubernetes/kubernetes.git
git fetch upstream
```

### 2. Create a Feature Branch

```bash
git checkout -b my-feature upstream/master
```

## Building

### Build Kubelet Only

```bash
# Standard build
make kubelet

# Verbose build
make kubelet GOFLAGS=-v

# Debug build (keeps symbols for debugging)
make kubelet DBG=1

# Output location: _output/bin/<os>/<arch>/kubelet
```

### Build All Binaries

```bash
make all

# Or specific component
make all WHAT=cmd/kubelet
```

### Cross-Platform Build

```bash
# Build for specific platform
make all WHAT=cmd/kubelet KUBE_BUILD_PLATFORMS=linux/arm64

# Build all platforms
make cross WHAT=cmd/kubelet
```

## Testing

### Unit Tests

**Preferred method** (faster, avoids script issues on macOS):

```bash
# Run all kubelet tests
go test ./pkg/kubelet/...

# Run specific package tests
go test -v ./pkg/kubelet/nodeinfocache/...
go test -v ./pkg/kubelet/lifecycle/...

# Run with race detector
go test -race ./pkg/kubelet/...

# Run specific test by name
go test -v -run TestCacheAddRemovePod ./pkg/kubelet/nodeinfocache/...
```

**Via Make** (may fail on macOS due to ulimit issue):

```bash
make test WHAT=./pkg/kubelet GOFLAGS=-v

# With coverage
make test WHAT=./pkg/kubelet KUBE_COVER=y
```

### Integration Tests

```bash
# Requires etcd (auto-started by script)
make test-integration WHAT=./test/integration/kubelet

# Specific test
make test-integration WHAT=./test/integration/kubelet KUBE_TEST_ARGS='-run ^TestName$'
```

### Test Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `KUBE_TIMEOUT` | Test timeout | `-timeout=300s` |
| `KUBE_RACE` | Race detector | `` (empty to disable) |
| `KUBE_COVER` | Coverage | `y` |
| `KUBE_TEST_ARGS` | Extra test args | `-run ^TestFoo$` |

## Benchmarking

### Run Benchmarks

```bash
# Run all benchmarks in a package
go test -bench=. ./pkg/kubelet/nodeinfocache/...

# Run specific benchmark
go test -bench=BenchmarkSnapshot -run=^ ./pkg/kubelet/nodeinfocache/...

# With memory allocation stats
go test -bench=. -benchmem ./pkg/kubelet/nodeinfocache/...

# Extended benchmark time
go test -bench=. -benchtime=10s ./pkg/kubelet/nodeinfocache/...

# Multiple runs for statistical comparison
go test -bench=. -benchmem -count=10 ./pkg/kubelet/nodeinfocache/... > results.txt
```

### Compare Benchmarks

```bash
# Install benchstat
go install golang.org/x/perf/cmd/benchstat@latest

# Run before/after and compare
git checkout main
go test -bench=. -benchmem -count=10 ./pkg/kubelet/nodeinfocache/... > old.txt
git checkout my-feature
go test -bench=. -benchmem -count=10 ./pkg/kubelet/nodeinfocache/... > new.txt
benchstat old.txt new.txt
```

### Writing Benchmarks

```go
func BenchmarkMyOperation(b *testing.B) {
    // Setup (not timed)
    data := setupTestData()

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        // Operation to benchmark
        _ = myOperation(data)
    }
}

// With memory reporting
func BenchmarkMyOperationWithAllocs(b *testing.B) {
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        _ = myOperation()
    }
}
```

## Verification Before PR

### Run All Checks

```bash
# Full verification (slow but thorough)
hack/verify-all.sh

# Quick verification
make quick-verify

# If verification fails, run updates
hack/update-all.sh
```

### Key Individual Checks

```bash
# Go formatting
hack/verify-gofmt.sh

# Linting
make lint

# Generated code
hack/verify-codegen.sh

# Import formatting
hack/verify-imports.sh
```

## Submitting a PR

### 1. Commit Your Changes

```bash
git add -A
git commit -m "kubelet: add NodeInfo caching for admission

This improves admission performance by caching NodeInfo
and updating it incrementally rather than rebuilding
on every admission.

Fixes #132858"
```

### 2. Push and Create PR

```bash
git push origin my-feature
```

Then create a PR on GitHub with:
- Clear title: `kubelet: <short description>`
- Description with:
  - What the change does
  - Why it's needed
  - How to test
  - Related issues (use `Fixes #123` to auto-close)

### 3. PR Labels

Add appropriate labels via comments:
- `/kind bug` - Bug fix
- `/kind feature` - New feature
- `/kind cleanup` - Code cleanup
- `/sig node` - SIG Node (kubelet)
- `/area kubelet` - Kubelet component

### 4. Respond to Reviews

```bash
# After making requested changes
git add -A
git commit --amend  # Or new commit depending on preference
git push --force-with-lease origin my-feature
```

## Useful Commands

### Git Operations

```bash
# Sync with upstream
git fetch upstream
git rebase upstream/master

# Squash commits before PR
git rebase -i upstream/master
```

### GitHub CLI (gh)

```bash
# Create PR
gh pr create --title "kubelet: my feature" --body "Description"

# View PR status
gh pr status

# View CI checks
gh pr checks

# Add reviewers
gh pr edit --add-reviewer @username

# Merge PR (if you have permissions)
gh pr merge --squash
```

### Debugging

```bash
# Build with debug symbols
make kubelet DBG=1

# Run specific test with verbose output
go test -v -run TestSpecificTest ./pkg/kubelet/...

# Run with race detector
go test -race -run TestConcurrentAccess ./pkg/kubelet/...
```

## Project Structure

```
pkg/kubelet/
├── kubelet.go              # Main Kubelet struct and methods
├── kubelet_nodecache.go    # Node caching logic
├── kubelet_pods.go         # Pod handling
├── allocation/             # Resource allocation management
├── lifecycle/              # Pod lifecycle and admission
│   └── predicate.go        # Admission predicates
├── nodeinfocache/          # NodeInfo caching (issue #132858)
├── cm/                     # Container manager
├── eviction/               # Pod eviction
├── status/                 # Status management
└── */testing/              # Fake implementations for testing
```

## Key Resources

- [Kubernetes Contributor Guide](https://git.k8s.io/community/contributors/guide/)
- [Developer Guide](https://git.k8s.io/community/contributors/devel/development.md)
- [SIG Node](https://github.com/kubernetes/community/tree/master/sig-node)
- [Kubelet Code](https://github.com/kubernetes/kubernetes/tree/master/pkg/kubelet)
- [Issue Tracker](https://github.com/kubernetes/kubernetes/issues?q=is%3Aopen+label%3Asig%2Fnode)

## Common Issues

### `make test` fails on macOS

The test script has issues when `ulimit -n` returns "unlimited". Use `go test` directly instead.

### Verification failures

Run `hack/update-all.sh` to auto-fix most issues, then re-run verification.

### etcd not found for integration tests

Integration tests require etcd. The test scripts auto-start it, but ensure Docker is running if using containerized etcd.
