---
name: kubelet-benchmark
description: Run kubelet benchmarks and compare performance. Use when measuring performance, comparing before/after changes, or validating optimizations.
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
  - Write
---

# Kubelet Benchmark Runner

Run and analyze benchmarks for the Kubernetes kubelet package.

## Instructions

When the user wants to run benchmarks:

1. **Run benchmarks with memory stats**:
```bash
go test -bench=. -benchmem ./pkg/kubelet/$ARGUMENTS/...
```

2. **For performance comparisons**, run multiple iterations:
```bash
go test -bench=. -benchmem -count=10 ./pkg/kubelet/$ARGUMENTS/... > benchmark_results.txt
```

3. **Report results** including:
   - Operations per second (ns/op)
   - Memory allocations (B/op, allocs/op)
   - Comparison with baseline if available

## Benchmark Commands

### Basic Benchmark
```bash
# Run all benchmarks in package
go test -bench=. ./pkg/kubelet/nodeinfocache/...

# Run specific benchmark
go test -bench=BenchmarkSnapshot -run=^ ./pkg/kubelet/nodeinfocache/...
```

### With Memory Analysis
```bash
# Include allocation stats
go test -bench=. -benchmem ./pkg/kubelet/nodeinfocache/...
```

### Extended Runs
```bash
# Longer benchmark time for more stable results
go test -bench=. -benchtime=10s ./pkg/kubelet/nodeinfocache/...

# Multiple runs for statistical comparison
go test -bench=. -benchmem -count=10 ./pkg/kubelet/nodeinfocache/...
```

### Comparing Before/After

```bash
# Save baseline results
go test -bench=. -benchmem -count=10 ./pkg/kubelet/nodeinfocache/... > old.txt

# After changes, save new results
go test -bench=. -benchmem -count=10 ./pkg/kubelet/nodeinfocache/... > new.txt

# Compare with benchstat
benchstat old.txt new.txt
```

## Interpreting Results

| Metric | Meaning | Goal |
|--------|---------|------|
| `ns/op` | Nanoseconds per operation | Lower is better |
| `B/op` | Bytes allocated per operation | Lower is better |
| `allocs/op` | Allocations per operation | Lower is better |

### Example Output
```
BenchmarkSnapshot-8    75000    15230 ns/op   12288 B/op   156 allocs/op
```
- Ran 75,000 iterations
- Each took ~15.2 microseconds
- Allocated ~12KB per operation
- Made 156 allocations per operation

## Common Kubelet Benchmarks

| Location | Benchmarks |
|----------|------------|
| `nodeinfocache` | Snapshot, AddPod, RemovePod |
| `images/pullmanager` | Cache hit/miss, pull decisions |
| `preemption` | Pod preemption selection |

## Tips

- Use `-run=^` to skip unit tests and only run benchmarks
- Use `-benchtime=5s` minimum for consistent results
- Always use `-count=10` when comparing
- Install `benchstat` with: `go install golang.org/x/perf/cmd/benchstat@latest`
