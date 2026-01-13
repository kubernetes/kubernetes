# Package: parallelize

## Purpose
Provides utilities for running scheduler operations in parallel to improve performance and CPU utilization. Wraps the client-go workqueue parallelization with scheduler-specific optimizations.

## Key Types

### Parallelizer
Wrapper around parallel execution logic with configurable parallelism level:
- **parallelism**: Number of worker goroutines (default: 16)

### ErrorChannel
Non-blocking error channel for capturing the first error from parallel operations:
- Keeps at most one error
- Additional errors are silently dropped
- Supports cancellation via SendErrorWithCancel

## Key Constants

- **DefaultParallelism**: 16 workers (configurable via scheduler options)

## Key Functions

### Parallelizer Methods
- **NewParallelizer(p int)**: Creates a new parallelizer with specified parallelism
- **Until(ctx, pieces, doWorkPiece, operation)**: Runs work in parallel chunks with metrics tracking

### ErrorChannel Methods
- **NewErrorChannel()**: Creates a new error channel
- **SendError(err)**: Sends error without blocking (drops if channel full)
- **SendErrorWithCancel(err, cancel)**: Sends error and cancels context
- **ReceiveError()**: Receives error without blocking (returns nil if none)

## Internal Functions

- **chunkSizeFor(n, parallelism)**: Calculates optimal chunk size as max(1, min(sqrt(n), n/parallelism))
- **numWorkersForChunkSize**: Determines actual worker count based on workload

## Design Pattern
- Uses chunk-based parallelism for optimal CPU utilization
- Automatically adjusts chunk size based on workload (sqrt heuristic)
- Integrates with scheduler metrics (Goroutines metric labeled by operation)
- Non-blocking error handling to avoid deadlocks in parallel code
