# Package: coverage

## Purpose
Provides tools for collecting and flushing code coverage information from coverage-instrumented Kubernetes binaries.

## Key Functions
- `InitCoverage()` - Initializes coverage collection with a binary name
- `FlushCoverage()` - Writes collected coverage data to disk

## Configuration
- `KUBE_COVERAGE_FILE` - Environment variable to set coverage output file path
- `KUBE_COVERAGE_FLUSH_INTERVAL` - Controls how often coverage is flushed (default: 5s)

## Build Tags
- `coverage` - Enables coverage collection (build with `-tags=coverage`)
- Without tag, functions are no-ops or panic

## Design Patterns
- Uses Go's testing infrastructure to collect coverage
- Periodically flushes coverage to disk for long-running processes
- Atomic file writes using temp file + rename pattern
- Implements fake test dependencies to trigger coverage output
