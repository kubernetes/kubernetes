# Package debugger

Package debugger provides debugging utilities for the scheduler cache, allowing inspection and comparison of cached state against actual cluster state.

## Key Types

- `CacheDebugger`: Main debugger combining comparer and dumper functionality
- `CacheComparer`: Compares cached nodes/pods against cluster state from listers
- `CacheDumper`: Outputs cache contents and scheduling queue to logs

## Key Functions

- `New`: Creates a CacheDebugger with node/pod listers, cache, and scheduling queue
- `ListenForSignal`: Starts goroutine that triggers debugging on SIGUSR2 (Unix) or SIGINT (Windows)
- `Compare`: Compares actual nodes/pods with cached state, logs mismatches
- `CompareNodes`: Identifies missed/redundant nodes between actual and cached
- `ComparePods`: Identifies missed/redundant pods including pending queue
- `DumpAll`: Writes complete cache and queue state to logs

## Design Notes

- Triggered via OS signal for on-demand debugging in production
- Useful for diagnosing cache consistency issues
- Comparer helps identify missing or stale entries
- Dumper provides detailed node info including pods, resources, and nominated pods
