# traceparse

`traceparse` summarizes GC activity from Go execution trace files (`.trace.out`)
produced by `go test -trace` or the `-perf-trace` flag in scheduler_perf.

## Usage

```
go -C hack/tools build -o /tmp/traceparse ./traceparse
/tmp/traceparse <trace-file> [<trace-file> ...]
```

## Output

For each file the tool prints a table of named range events sorted by total
wall-clock duration, followed by the average live heap size:

```
=== path/to/foo-trace.out ===
range                                                    count  total duration
-----                                                    -----  --------------
GC concurrent mark phase                                    47  5.662s
GC mark assist                                           14294  743.936ms
GC incremental sweep                                    161077  321.769ms
stop-the-world (GC sweep termination)                       47  7.568ms
stop-the-world (read mem stats)                             42  5.446ms
stop-the-world (GC mark termination)                        47  2.221ms
stop-the-world (start trace)                                 1  1.792µs

avg live heap: 796 MB  (3827825 samples)
```

### Key metrics for scheduler_perf analysis

| Range | What it tells you |
|---|---|
| `GC concurrent mark phase` | Total GC work; fewer cycles / less time = less allocation pressure |
| `GC mark assist` | **Most sensitive indicator**: goroutines interrupted mid-scheduling to help mark because allocation outpaces background GC; high counts mean hot-path allocations are limiting throughput |
| `stop-the-world (GC sweep termination)` | STW pauses that stop all goroutines; directly affects scheduling tail latency |
| `stop-the-world (GC mark termination)` | Second STW pause per cycle; usually short |
| `avg live heap` | Working set size; drives GC trigger frequency |
