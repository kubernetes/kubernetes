# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/backend` Utilities

This document provides AI agents and human contributors with an architectural overview, data structures, concurrency invariants, lifecycle workflows, and testing patterns for the Kubernetes scheduler backend utilities under `pkg/scheduler/backend` — specifically **`heap`**, **`api_cache`**, and **`api_dispatcher`**.

---

## 1. High-Level Purpose & Scope

The `pkg/scheduler/backend` package contains core data structures and concurrency abstractions that isolate scheduler decision loops from internal queuing mechanics, in-memory caching, and external API server network latency.

```
+-------------------------------------------------------------------------+
|                           kube-scheduler Core                           |
|       (scheduleOne / runBindingCycle / Informer Event Handlers)         |
+--------------------+-------------------------------+--------------------+
                     |                               |
                     v                               v
+--------------------+-------------------+   +-------+--------------------+
|            api_cache                   |   |         queue              |
|  (Optimistic cache-first abstraction)  |   |    (PriorityQueue)         |
+--------------------+-------------------+   +-------+--------------------+
                     |                               |
                     v                               v
+--------------------+-------------------+   +-------+--------------------+
|          api_dispatcher                |   |          heap              |
|  (Async batching, deduplication,       |   |  (Generic min/max heap     |
|   relevance merging & execution)       |   |   with O(1) index lookup)  |
+--------------------+-------------------+   +----------------------------+
                     |
                     v
+--------------------+-------------------+
|            kube-apiserver              |
+----------------------------------------+
```

### Core Utility Responsibilities:
1. **Generic Heap (`pkg/scheduler/backend/heap`)**: A generic binary heap data structure parameterized over schedulable entities (`Item`), featuring $O(1)$ key lookups via synchronous index tracking and metric recording. Backs `activeQ` and `backoffQ` in `pkg/scheduler/backend/queue`.
2. **API Cache (`pkg/scheduler/backend/api_cache`)**: An optimistic caching coordinator implementing `fwk.APICacher`. Applies state mutations (status patches, bindings) locally to cache/queue before dispatching asynchronous API calls, eliminating API latency from scheduling loops.
3. **API Dispatcher (`pkg/scheduler/backend/api_dispatcher`)**: An asynchronous API execution engine implementing `fwk.APIDispatcher`. Serializes calls per object UID, coalesces/merges duplicate operations, prioritizes calls by relevance, synchronizes informer state changes, and enforces worker concurrency limits.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/backend/
├── heap/
│   ├── heap.go                  # Generic Heap[T Item] with keyIndex tracking & MetricRecorder
│   └── heap_test.go             # Unit tests for heap operations, re-indexing, and metrics
├── api_cache/
│   └── api_cache.go             # APICache coordinating optimistic queue & cache updates
├── api_dispatcher/
│   ├── api_dispatcher.go        # APIDispatcher worker loop, lifecycle, and metrics
│   ├── api_dispatcher_test.go   # Integration tests for APIDispatcher lifecycle
│   ├── call_queue.go            # callQueue managing per-UID FIFO ring buffer & reconciliation
│   ├── call_queue_test.go       # Comprehensive tests for merging, skipping, in-flight & sync
│   ├── goroutines_limiter.go    # Concurrency limiter using condition variables
│   └── goroutines_limiter_test.go # Tests for goroutine acquisition, blocking, and release
├── cache/                       # Scheduler node/pod cache subsystem
├── queue/                       # Scheduling queue (PriorityQueue)
└── AGENTS.md                    # This developer & agent documentation
```

---

## 3. Generic Heap Data Structure (`pkg/scheduler/backend/heap`)

### 3.1. Core Interfaces & Generic Data Model

The heap is parameterized over type `T` constrained by `Item`:

```go
type Item interface {
    Size() int                  // Size of item (e.g. 1 for Pod, N for PodGroup), used for metrics
    Type() fwk.EntityKeyType    // Entity type ("pod" or "podgroup")
}

type KeyFunc[T Item] func(obj T) string
type LessFunc[T Item] func(item1, item2 T) bool
```

Internally, `Heap[T]` wraps `data[T]`, which implements standard `container/heap.Interface`:

```go
type heapItem[T Item] struct {
    obj T
    key string
}

type data[T Item] struct {
    queue    []*heapItem[T]
    keyIndex map[string]int      // Maps object key -> slice index in queue
    keyFunc  KeyFunc[T]
    lessFunc LessFunc[T]
}
```

### 3.2. Synchronous Index Tracking

To allow $O(1)$ lookups, updates, and arbitrary element deletions in $O(\log n)$ without scanning the slice, `data[T]` synchronizes `keyIndex` on every heap mutation:

```
          queue: []*heapItem[T]                     keyIndex: map[string]int
      ┌──────────────────────────────┐              ┌───────────────┬───────┐
    0 │ key: "pod-A", obj: PodInfo-A │ <─────────── │ "pod-A"       │   0   │
      ├──────────────────────────────┤              ├───────────────┼───────┤
    1 │ key: "pod-B", obj: PodInfo-B │ <─────────── │ "pod-B"       │   1   │
      ├──────────────────────────────┤              ├───────────────┼───────┤
    2 │ key: "pod-C", obj: PodInfo-C │ <─────────── │ "pod-C"       │   2   │
      └──────────────────────────────┘              └───────────────┴───────┘
```

1. **`Swap(i, j int)`**:
   ```go
   h.queue[i], h.queue[j] = h.queue[j], h.queue[i]
   h.keyIndex[h.queue[i].key] = i
   h.keyIndex[h.queue[j].key] = j
   ```
2. **`Push(x interface{})`**:
   ```go
   item := x.(*heapItem[T])
   h.keyIndex[item.key] = len(h.queue)
   h.queue = append(h.queue, item)
   ```
3. **`Pop() interface{}`**:
   ```go
   n := len(h.queue)
   item := h.queue[n-1]
   h.queue[n-1] = nil           // Prevents memory leak
   h.queue = h.queue[:n-1]
   delete(h.keyIndex, item.key)
   return item.obj
   ```

### 3.3. Public API & Complexity

| Method | Complexity | Description |
|---|---|---|
| `AddOrUpdate(obj T)` | $O(\log n)$ | Inserts a new object via `heap.Push` or updates existing item via `heap.Fix(data, idx)`. Updates `metricRecorder`. |
| `Delete(obj T) T` | $O(\log n)$ | Deletes object at `keyIndex[key]` via `heap.Remove(data, idx)` and decrements `metricRecorder`. Returns removed object. |
| `Pop() (T, error)` | $O(\log n)$ | Pops and removes the root item via `heap.Pop`. Returns error if heap is empty. |
| `Peek() (T, bool)` | $O(1)$ | Returns root item `queue[0].obj` without removal. |
| `Get(obj T) (T, bool)` | $O(1)$ | Fetches object by key via `keyIndex` without modifying heap order. |
| `GetByKey(key string) (T, bool)` | $O(1)$ | Direct key lookup. |
| `Has(obj T) bool` | $O(1)$ | Fast membership test via `keyIndex`. |
| `List() []T` | $O(n)$ | Returns a shallow slice copy of all elements in queue order. |
| `Len() int` | $O(1)$ | Returns total item count in heap. |

### 3.4. Heap Invariants & Rules

1. **Non-Thread-Safe by Design**: `Heap[T]` does NOT perform internal locking. Synchronization is the explicit responsibility of the higher-level caller (e.g., `SchedulingQueue` lock).
2. **Deterministic Keys**: `keyFunc` must produce a unique, deterministic identity string per entity across its lifecycle.
3. **Accurate Metric Adjustments**: `metricRecorder.Remove(removed)` uses the object returned by `heap.Remove` rather than the lookup argument, ensuring metric counter accuracy when item sizes change dynamically.
4. **Memory Leak Protection**: Slices truncated in `Pop()` must explicitly zero out the discarded element (`queue[n-1] = nil`) before reslicing.

---

## 4. API Cache Subsystem (`pkg/scheduler/backend/api_cache`)

### 4.1. Purpose & Flow

When the feature gate `SchedulerAsyncAPICalls` is enabled, the scheduler avoids waiting synchronously for API server network responses during pod status patching or binding. `APICache` orchestrates optimistic updates against local state:

```
[ Pod Binding Request ]
          │
          ▼
 APICache.BindPod(binding)
   ├── 1. Apply binding optimistically to internalcache.Cache
   └── 2. Enqueue APICall in APIDispatcher for asynchronous execution
          │
          ▼
 Returns (<-chan error, error) to caller
          │
 (Optional) APICache.WaitOnFinish(ctx, onFinishCh)
```

### 4.2. Method Specifications

- **`PatchPodStatus(pod, conditions, nominatingInfo) (<-chan error, error)`**:
  - Delegates to `schedulingQueue.PatchPodStatus()`.
  - Mutates cached pod conditions / nominating info locally.
  - Returns non-blocking completion channel.
- **`BindPod(binding) (<-chan error, error)`**:
  - Delegates to `cache.BindPod()`.
  - Assumes the pod on the target node in scheduler cache.
  - Returns non-blocking completion channel.
- **`WaitOnFinish(ctx context.Context, onFinish <-chan error) error`**:
  - Blocks until API call completes or context expires (`ctx.Done()`).
  - Filters out expected lifecycle outcomes using `fwk.IsUnexpectedError(err)`:
    - `fwk.ErrCallSkipped`: Treated as success (no-op or irrelevant).
    - `fwk.ErrCallOverwritten`: Treated as success (superseded by a newer call).
    - Unrecognized errors: Returned to caller for failure handling.

---

## 5. Asynchronous API Dispatcher (`pkg/scheduler/backend/api_dispatcher`)

### 5.1. Architecture & Components

```
                      APIDispatcher.Add(call, opts)
                                    │
                                    ▼
                         callQueue.add(apiCall)
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
[ Existing call for UID? ]                             [ No existing call ]
  ├─ Less relevant  -> Skip (ErrCallSkipped)             ├─ Push UID to Ring Buffer
  ├─ Higher relev.  -> Overwrite / Wait in-flight        └─ Record Pending Metric
  └─ Same type      -> Merge (apiCall.Merge())
                                    │
                                    ▼
                          APIDispatcher.Run()
                       Worker Loop: callQueue.pop()
                                    │
                                    ▼
               goroutine: apiCall.Execute(ctx, client)
                                    │
                                    ▼
                         callQueue.finalize(call)
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
        [ callID unchanged ]                  [ New call arrived ]
        Delete from apiCalls                  Re-queue UID to Ring Buffer
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    ▼
                         apiCall.sendOnFinish(err)
```

### 5.2. `callQueue` State Management

`callQueue` synchronizes state using an internal `sync.RWMutex` and `sync.Cond`:

```go
type callQueue struct {
    lock              sync.RWMutex
    cond              *sync.Cond
    closed            bool
    apiCallRelevances fwk.APICallRelevances
    callIDCounter     int
    apiCalls          map[types.UID]*queuedAPICall
    callsQueue        buffer.Ring[types.UID]
    inFlightEntities  sets.Set[types.UID]
}
```

### 5.3. Invariant: Single Active Call Per Object UID

For any Kubernetes object (identified by `types.UID`), **at most one API call is present in `callsQueue` or executing in `inFlightEntities` at any given time**. This guarantees sequential consistency and eliminates write races against the API server for the same resource.

### 5.4. Call Reconciliation & Coalescing Logic (`cq.add`)

When a new call arrives for an object that already has an entry in `apiCalls`:

| Condition | Action | Notification (`onFinish`) | Queue State |
|---|---|---|---|
| **New call has lower relevance** (`isLessRelevant`) | Skip new call | New call receives `ErrCallSkipped` | Old call remains untouched |
| **Different call type & old call pending** | Overwrite old call | Old call receives `ErrCallOverwritten` | New call replaces old call in `apiCalls` |
| **Different call type & old call in-flight** | Defer new call | None immediately | New call updates `apiCalls`; enqueued upon old call's `finalize()` |
| **Same call type** | Merge: `newCall.Merge(oldCall)` | Old call receives `ErrCallOverwritten` | Merged call updates `apiCalls` |
| **Call is No-Op** (`IsNoOp() == true`) | Skip call (if not in-flight) | Call receives `ErrCallSkipped` | Removed from `apiCalls` & `callsQueue` |

### 5.5. Two-Way State Synchronization (`SyncObject`)

When cluster informers receive updated object events, `APIDispatcher.SyncObject(obj)` reconciles incoming API server objects with pending dispatcher state:

1. **Object State Enrichment**: Invokes `apiCall.Sync(obj)` to apply pending modifications to `obj` (optimistic preview) while updating internal call state.
2. **No-Op Cancellation**: If synchronization renders the pending API call redundant (`apiCall.IsNoOp() == true`) and the call is not yet in-flight:
   - Call is evicted from `callsQueue` and `apiCalls`.
   - `sendOnFinish(ErrCallSkipped)` notifies waiting listeners.

### 5.6. Worker Lifecycle & Finalization

1. **`pop()`**:
   - Blocks on `cond.Wait()` while `callsQueue.Len() == 0`.
   - Reads `objectUID` from FIFO ring buffer.
   - Adds `objectUID` to `inFlightEntities`.
   - Decrements `metrics.AsyncAPIPendingCalls`.
2. **`Execute()`**:
   - Runs in a separate worker goroutine: `err := apiCall.Execute(ctx, client)`.
   - Records execution metrics (`AsyncAPICallsTotal`, `AsyncAPICallDuration`).
3. **`finalize(apiCall)`**:
   - Acquires `callQueue.lock`.
   - Compares `apiCalls[uid].callID == apiCall.callID`:
     - **Match**: No intervening call arrived during execution; removes `uid` from `apiCalls`.
     - **Mismatch**: A newer call for this object arrived while in-flight; pushes `objectUID` back to `callsQueue`, increments pending metric, and calls `cond.Broadcast()`.
   - Deletes `objectUID` from `inFlightEntities`.
4. **`sendOnFinish(err)`**:
   - Performs non-blocking write to `onFinish` channel (`select { case onFinish <- err: default: }`).

### 5.7. Goroutines Limiter (`goroutinesLimiter`)

`goroutinesLimiter` provides bounded concurrency control for asynchronous worker routines:
- **`acquire() bool`**: Blocks on `cond.Wait()` while active `goroutines >= maxGoroutines`. Returns `false` if limiter is closed.
- **`release()`**: Decrements active count and invokes `cond.Broadcast()`.
- **`close()`**: Sets `closed = true` and wakes all waiting goroutines.

---

## 6. Prometheus Metrics & Observability

The backend utilities register and maintain the following Prometheus metrics under `pkg/scheduler/metrics`:

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `scheduler_async_api_pending_calls` | Gauge | `call_type` | Number of API calls currently queued in `callQueue` awaiting execution. |
| `scheduler_async_api_call_execution_total` | Counter | `call_type`, `result` (`success`/`error`) | Total count of executed asynchronous API calls. |
| `scheduler_async_api_call_execution_duration_seconds` | Histogram | `call_type`, `result` (`success`/`error`) | Latency distribution of executed asynchronous API calls. |
| `scheduler_schedule_attempts_total` / Queue gauges | Custom | `type` | Heap entity additions and removals via `metrics.MetricRecorder`. |

---

## 7. Testing Strategies & Patterns

### 7.1. Synctest Concurrency Testing

Tests in `api_dispatcher` utilize Go's `testing/synctest` package to deterministically test blocking channels, condition variables, and goroutine synchronization without flaky sleep timers:

```go
synctest.Test(t, func(t *testing.T) {
    cq := newCallQueue(mockRelevances)
    poppedCallCh := make(chan *queuedAPICall)

    go func() {
        poppedCall, _ := cq.pop()
        poppedCallCh <- poppedCall
    }()

    // Durably blocks until the goroutine is waiting in cq.cond.Wait()
    synctest.Wait()

    cq.add(call)
    poppedCall := <-poppedCallCh
    // Assert results...
})
```

### 7.2. Test Helper Utilities

- **`verifyQueueState(t, cq, expectedPendingCalls)`**: Asserts ring buffer length and verifies `AsyncAPIPendingCalls` gauge metrics per call type.
- **`verifyCalls(t, cq, calls...)`**: Asserts exact content of `cq.apiCalls` using `cmp.Diff` with unexported field options.
- **`verifyInFlight(t, cq, uids...)`**: Validates members of `inFlightEntities` set.
- **`expectOnFinish(t, onFinish, expectedErr)`**: Verifies expected non-blocking channel error or timeout.

### 7.3. Running Backend Tests

```bash
# Run tests across heap, api_cache, and api_dispatcher
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/backend/heap ./pkg/scheduler/backend/api_cache ./pkg/scheduler/backend/api_dispatcher

# Run all scheduler backend tests
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/backend/...
```

---

## 8. Critical Invariants for Developers & AI Agents

1. **Maintain Bi-Directional Heap Indexing**: Any new method altering `data.queue` in `pkg/scheduler/backend/heap` MUST update `data.keyIndex` immediately.
2. **Never Call Heap Methods Concurrently**: `Heap[T]` is not thread-safe. All calls to a `Heap` instance must be guarded by an external mutex.
3. **Always Finalize Popped API Calls**: Any caller popping a call from `callQueue` MUST call `finalize(call)` upon completion to prevent memory leaks and stalled in-flight tracking.
4. **Non-Blocking `onFinish` Notifications**: Always use `sendOnFinish(err)` helper; never perform blocking sends directly on `onFinish` channels.
5. **No-Op Pruning**: When implementing custom `fwk.APICall` types, ensure `IsNoOp()` accurately detects when a call has no side effects so `callQueue` and `SyncObject` can prune unnecessary network round-trips.
