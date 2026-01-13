# Package: debugger

## Purpose
Provides debugging utilities for the scheduler cache. Allows comparing cache state against actual cluster state and dumping cache contents for troubleshooting.

See `../CLAUDE.md` for details on the cache architecture (pod state machine, optimistic scheduling, snapshot mechanism).

## Key Types

### CacheDebugger
Main debugger struct containing:
- **Comparer**: For comparing cache vs. actual state
- **Dumper**: For dumping cache contents

### CacheComparer
Compares cache state with API server state:
- **NodeLister**: For fetching actual nodes
- **PodLister**: For fetching actual pods
- **Cache**: The scheduler cache to compare
- **PodQueue**: The scheduling queue

### CacheDumper
Dumps cache and queue contents:
- **cache**: The scheduler cache
- **podQueue**: The scheduling queue

## Key Functions

- **New(nodeLister, podLister, cache, podQueue)**: Creates a CacheDebugger
- **ListenForSignal(ctx)**: Starts goroutine that triggers debug on signal
- **Comparer.Compare(logger)**: Compares and logs discrepancies
- **Dumper.DumpAll(logger)**: Dumps all cache state

## Signal Handling
- **SIGUSR2** (Unix): Triggers cache comparison and dump
- **SIGINT** (Windows): Triggers cache comparison and dump

## Design Pattern
- Signal-driven debugging for production troubleshooting
- Logs discrepancies between cache and actual state
- Useful for diagnosing scheduling issues caused by stale cache
- Platform-specific signal handling (signal.go, signal_windows.go)
