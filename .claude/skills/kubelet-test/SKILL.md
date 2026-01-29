---
name: kubelet-test
description: Run kubelet unit tests. Use when testing kubelet code changes, running specific test files, or validating fixes.
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Kubelet Test Runner

Run unit tests for the Kubernetes kubelet package.

## Instructions

When the user wants to run kubelet tests:

1. **Identify the test scope**:
   - Specific package: `go test -v ./pkg/kubelet/<package>/...`
   - Specific test: `go test -v -run TestName ./pkg/kubelet/<package>/...`
   - All kubelet: `go test ./pkg/kubelet/...`

2. **Run the appropriate command**:

```bash
# Run all tests in a specific package
go test -v ./pkg/kubelet/$ARGUMENTS/...

# Run with race detector (recommended for concurrency tests)
go test -race -v ./pkg/kubelet/$ARGUMENTS/...

# Run specific test by name
go test -v -run "TestSpecificName" ./pkg/kubelet/$ARGUMENTS/...
```

3. **Report results** including:
   - Pass/fail status
   - Any failures with error messages
   - Suggestions for fixing failures

## Common Test Packages

| Package | Purpose |
|---------|---------|
| `nodeinfocache` | NodeInfo caching for admission |
| `lifecycle` | Pod lifecycle and admission handlers |
| `allocation` | Resource allocation management |
| `status` | Pod status management |
| `eviction` | Pod eviction logic |
| `cm` | Container manager |

## Examples

User: "Run the nodeinfocache tests"
```bash
go test -v ./pkg/kubelet/nodeinfocache/...
```

User: "Run TestCacheAddRemovePod"
```bash
go test -v -run TestCacheAddRemovePod ./pkg/kubelet/nodeinfocache/...
```

User: "Run all kubelet tests with race detection"
```bash
go test -race ./pkg/kubelet/...
```

## Notes

- Prefer `go test` over `make test` on macOS (ulimit issues)
- Use `-v` for verbose output showing individual test results
- Use `-race` when testing concurrent code
- Tests timeout after 180 seconds by default; use `-timeout=300s` for longer tests
