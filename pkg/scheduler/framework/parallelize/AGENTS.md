# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/parallelize`

This guide provides AI agents and human contributors with an architectural overview, mathematical foundations, concurrency patterns, metric instrumentation, and developer invariants for the parallel execution engine located in `pkg/scheduler/framework/parallelize`.

---

## 1. High-Level Overview & Package Scope

The `parallelize` package provides the concurrency primitives used throughout `kube-scheduler` to evaluate large numbers of nodes, pods, or preemption candidates in parallel. Built as an optimized layer on top of `k8s.io/client-go/util/workqueue.ParallelizeUntil`, it balances goroutine scheduling overhead against core saturation through dynamic chunking and non-blocking short-circuit channels.

### Core Responsibilities:
1. **Parallel Execution Abstraction (`Parallelizer`)**: Implements `fwk.Parallelizer` to execute arbitrary piecewise functions (`workqueue.DoWorkPieceFunc`) concurrently across a bounded worker pool.
2. **Dynamic Chunking Engine (`chunkSizeFor`, `numWorkersForChunkSize`)**: Computes optimal task chunk sizes based on the total number of work pieces ($N$) and configured parallelism ($P$).
3. **Goroutine Metrics Instrumentation (`metrics.Goroutines`)**: Atomically tracks active worker goroutines via Prometheus gauges with operation labeling without introducing high-contention mutexes or atomic bottlenecks.
4. **Generic Non-Blocking Result Propagation (`ResultChannel[T]`)**: Provides a thread-safe, non-blocking single-value result channel with early context cancellation for fast-failing parallel pipelines.

---

## 2. Directory Architecture & File Map

```
pkg/scheduler/framework/parallelize/
├── parallelism.go             # Parallelizer struct, chunkSizeFor, numWorkersForChunkSize, Until
├── parallelism_test.go        # Tests for chunk sizing math, worker counts, and error handling
├── result_channel.go          # Generic ResultChannel[T comparable] with SendWithCancel
├── result_channel_test.go     # Tests for ResultChannel concurrent send/receive and non-blocking semantics
└── AGENTS.md                  # This agent guide
```

---

## 3. Worker Pool Architecture & Dynamic Chunking

### 3.1. `Parallelizer` Struct
```go
const DefaultParallelism int = 16

type Parallelizer struct {
    parallelism int
}

func NewParallelizer(p int) Parallelizer {
    return Parallelizer{parallelism: p}
}
```
The framework initializes a global `Parallelizer` instance (exposed via `fwk.Handle.Parallelizer()`) with parallelism configured from `KubeSchedulerConfiguration.Parallelism` (default: 16).

### 3.2. Mathematical Chunking Algorithm (`chunkSizeFor`)

To prevent excessive goroutine scheduling and channel synchronization overhead when iterating over thousands of nodes, `chunkSizeFor` dynamically groups work items into sequential chunks:

```go
func chunkSizeFor(n, parallelism int) int {
    s := int(math.Sqrt(float64(n)))
    if r := n/parallelism + 1; s > r {
        s = r
    } else if s < 1 {
        s = 1
    }
    return s
}
```

#### Mathematical Formulation:
For $N$ total pieces and parallelism limit $P$:

$$\text{ChunkSize}(N, P) = \max\left(1, \min\left(\lfloor\sqrt{N}\rfloor, \lfloor N/P \rfloor + 1\right)\right)$$

#### Design Rationale:
1. **Upper Bound ($\lfloor\sqrt{N}\rfloor$)**: When $N$ is large relative to $P$, capping chunk size at $\sqrt{N}$ ensures sufficient chunks are created to distribute work evenly across workers and prevent stragglers.
2. **Lower Bound ($\lfloor N/P \rfloor + 1$)**: When $N$ is small relative to $P$, setting chunk size to $\lfloor N/P \rfloor + 1$ ensures that work is split across all available $P$ workers without creating empty chunks.
3. **Boundary Clamp ($\ge 1$)**: Ensures chunk size never evaluates to 0.

#### Chunking Examples:

| Total Pieces ($N$) | Parallelism ($P$) | $\sqrt{N}$ | $N/P + 1$ | Computed `chunkSize` | Total Chunks | Effective Workers |
|---|---|---|---|---|---|---|
| **10** | 16 | 3 | 1 | **1** | 10 | 10 |
| **100** | 16 | 10 | 7 | **7** | 15 | 15 |
| **1,000** | 16 | 31 | 63 | **31** | 33 | 16 |
| **10,000** | 16 | 100 | 626 | **100** | 100 | 16 |

### 3.3. Worker Sizing (`numWorkersForChunkSize`)

```go
func numWorkersForChunkSize(parallelism, pieces, chunkSize int) int {
    chunks := (pieces + chunkSize - 1) / chunkSize
    if chunks < parallelism {
        return chunks
    }
    return parallelism
}
```

Determines the actual number of goroutines dispatched by taking $\min(P, \lceil N / S \rceil)$, avoiding the creation of idle worker goroutines.

---

## 4. Execution Flow & Metrics Lifecycle (`Until`)

`Parallelizer.Until` coordinates parallel execution and Prometheus metrics:

```
                  [ Parallelizer.Until(ctx, pieces, doWorkPiece, op) ]
                                          │
                                          ▼
                      [ Calculate chunkSize & worker count ]
                         ├── chunkSize = chunkSizeFor(pieces, p)
                         └── workers = numWorkersForChunkSize(p, pieces, chunkSize)
                                          │
                                          ▼
                      [ Instrument Active Goroutines Metric ]
                         └── metrics.Goroutines.WithLabelValues(op).Add(float64(workers))
                                          │
                                          ▼
                      [ Dispatch to client-go workqueue ]
                         └── workqueue.ParallelizeUntil(ctx, p, pieces, doWorkPiece, WithChunkSize(chunkSize))
                                          │
                                          ▼
                      [ Deferred Metric Cleanup ]
                         └── metrics.Goroutines.WithLabelValues(op).Add(float64(-workers))
```

### Key Properties:
- **Bulk Metric Updates**: Rather than each worker updating metrics individually on start and exit, `Until` performs a single bulk `.Add(+workers)` on entry and a deferred `.Add(-workers)` on exit. This minimizes atomic counter cache line bouncing.
- **Context Cancellation**: Canceling `ctx` causes `workqueue.ParallelizeUntil` to stop pulling new chunks from the queue, allowing all workers to drain and exit promptly.

---

## 5. Non-Blocking Result Propagation (`ResultChannel[T]`)

`ResultChannel[T comparable]` is a thread-safe wrapper around a buffered channel of size 1, designed for scenarios where multiple parallel workers evaluate candidate solutions and only the first result (or first error) is needed.

```go
type ResultChannel[T comparable] struct {
    ch chan T
}

func NewResultChannel[T comparable]() *ResultChannel[T] {
    return &ResultChannel[T]{
        ch: make(chan T, 1),
    }
}
```

### 5.1. Non-Blocking Send (`Send` and `SendWithCancel`)
```go
func (e *ResultChannel[T]) Send(result T) {
    select {
    case e.ch <- result:
    default:
    }
}

func (e *ResultChannel[T]) SendWithCancel(result T, cancel context.CancelFunc) {
    e.Send(result)
    cancel()
}
```
- **Drop on Full**: If a result is already present, subsequent sends non-blockingly fall through the `default` branch and are discarded.
- **Early Cancellation**: `SendWithCancel` writes the first result and immediately invokes the provided `context.CancelFunc` to stop remaining workers across all parallel chunks.

### 5.2. Non-Blocking Receive (`Receive`)
```go
func (e *ResultChannel[T]) Receive() T {
    select {
    case result := <-e.ch:
        return result
    default:
        var zeroValue T
        return zeroValue
    }
}
```
Allows reading the stored result without blocking if no worker produced a value.

---

## 6. Subsystem Usage Across `kube-scheduler`

| Subsystem / Extension Point | File Location | Operation Label | Parallelized Task |
|---|---|---|---|
| **Filter Evaluation** | `pkg/scheduler/framework/runtime/framework.go` | `"Filter"` | Evaluating `Filter` plugins across candidate nodes. |
| **Score Evaluation** | `pkg/scheduler/framework/runtime/framework.go` | `"Score"` | Evaluating `Score` plugins and calculating per-node scores. |
| **PreScore Normalization** | `pkg/scheduler/framework/runtime/framework.go` | `"NormalizeScore"` | Invoking `ScoreExtensions.NormalizeScore` across plugin score arrays. |
| **Preemption Dry-Run** | `pkg/scheduler/framework/preemption/preemption.go` | `"DefaultPreemption"` | Concurrently simulating victim selection across candidate nodes in `DryRunPreemption`. |
| **Preemption Eviction** | `pkg/scheduler/framework/preemption/executor.go` | `"PreemptionEviction"` | Concurrently issuing `DeletePod` / status patches for victim pods. |
| **Topology Spread / Affinity** | `pkg/scheduler/framework/plugins/podtopologyspread/` | `"PodTopologySpread"` | Calculating failure domain skew and matching pods across nodes. |

---

## 7. Developer Invariants & Concurrency Rules

1. **Pure Piece Functions (`DoWorkPieceFunc`)**:
   - `piece` index passed to `DoWorkPieceFunc(piece int)` corresponds to the global item index ($0 \le \text{piece} < N$).
   - Functions MUST be safe for concurrent execution across goroutines.
2. **State Sharing & Cloning**:
   - When calling framework plugins inside parallel loops, workers MUST NOT mutate shared `CycleState` concurrently. Pass cloned copies (`state.Clone()`) or ensure plugins use read-only access (WORM pattern).
3. **Error Collection Invariants**:
   - Use `ResultChannel[error]` or an explicit `sync.Mutex` guarded error slice (`[]error`) aggregated via `utilerrors.NewAggregate(errs)`. Never append to slices concurrently without synchronization.
4. **Metric Cleanup Safety**:
   - Always ensure metric decrements occur via `defer` to guarantee accurate active goroutine counts even if a panic or early return occurs.

---

## 8. Unit Testing Patterns

```bash
# Run all parallelize package tests
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/parallelize/...

# Benchmark chunk sizing and worker execution
GOTOOLCHAIN=auto go test -bench=. ./pkg/scheduler/framework/parallelize/...
```
