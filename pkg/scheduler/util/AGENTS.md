# Developer & Agent Guide for `pkg/scheduler/util` and `assumecache`

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, concurrency model, thread-safety invariants, and API contracts for the shared utilities and assume cache subsystem located under `pkg/scheduler/util`.

---

## 1. Overview & Core Responsibilities

The `pkg/scheduler/util` package and its `assumecache` subpackage provide shared foundations for `kube-scheduler`, including optimistic in-memory caching, API server mutation helpers with exponential backoff, pod ordering and prioritization comparators, resource classification, network port extraction, and type-safe informer casting.

### Core Objectives:
1. **Optimistic Assume Caching (`assumecache.AssumeCache`)**: Provide a dual-pointer in-memory caching layer over client-go informers/indexers that allows scheduler plugins (e.g., Dynamic Resource Allocation / DRA, storage/volume binding) to optimistically assume object mutations and roll them back on failure without waiting for asynchronous API server watch delivery.
2. **Deadlock-Free Event Delivery**: Decouple cache lock acquisition from external event handler notifications via an unbounded ring buffer (`buffer.Ring`) and a single-worker baton-passing condition variable (`sync.Cond`).
3. **Deterministic Pod Priority & Sorting**: Provide strict weak ordering comparators (`MoreImportantPod`) and stable maximum start timestamps (`maxPodStartTime`) to prevent sort order instability during scheduling and preemption evaluation.
4. **Resilient API Mutation & Backoff**: Standardize transient network error and conflict retries (`Retriable`, `RetriableWithConflict`, `BindPod`, `PatchPodStatus`, `PatchPodGroupStatus`, `PatchCompositePodGroupStatus`).
5. **Resource Classification & Defaults**: Provide helper classifiers for scalar and DRA device class resources (`IsScalarResourceName`, `IsDRAExtendedResourceName`) along with baseline resource defaults (`DefaultMilliCPURequest`, `DefaultMemoryRequest`) for zero-request pods during scoring.
6. **Generic Informer Event Unpacking**: Safely unpack and cast Informer objects (`As[T]`) while transparently unwrapping `cache.DeletedFinalStateUnknown` deletion tombstones.

---

## 2. Directory & Subpackage Architecture

```
pkg/scheduler/util/
├── AGENTS.md                  # This agent guide
├── utils.go                   # Pod helpers, retry logic, strategic patch helpers, generics
├── utils_test.go              # Unit tests for utils.go functions
├── pod_resources.go           # Baseline default resource requests for zero-request scoring
└── assumecache/               # Optimistic assume cache implementation
    ├── assume_cache.go        # AssumeCache struct, objInfo dual-pointer store, event delivery
    └── assume_cache_test.go   # Unit tests for AssumeCache concurrency, indexing, and event handlers
```

---

## 3. Subsystem Deep Dive: `assumecache`

The `assumecache` package (`pkg/scheduler/util/assumecache`) implements an optimistic in-memory cache on top of client-go `SharedInformer` event streams. It is heavily utilized by plugins requiring fast reservation cycles (such as Dynamic Resource Allocation / DRA resource claims and volume binding).

```
                 ┌──────────────────────────────────────────────┐
                 │          client-go SharedInformer            │
                 └──────────────────────┬───────────────────────┘
                                        │ Informer Watch Events
                                        │ (Add, Update, Delete)
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │                 AssumeCache                  │
                 │  - rwMutex sync.RWMutex                      │
                 │  - cond *sync.Cond                           │
                 │  - store cache.Indexer                       │
                 │  - eventQueue buffer.Ring[func()]            │
                 │  - emittingEvents bool                       │
                 └──────────────────────┬───────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 ┌───────────────┐             ┌───────────────┐
                 │    objInfo    │             │  objInfo      │
                 │  - name       │             │  - name       │
                 │  - apiObj ────┼────┐        │  - apiObj ────┼────┐
                 │  - latestObj ─┼──┐ │        │  - latestObj ─┼──┐ │
                 └───────────────┘  │ │        └───────────────┘  │ │
                                    │ │                           │ │
    Authoritative Informer Object ◄─┘ │           Authoritative  ◄─┘ │
    Optimistic Assumed Object ◄───────┘           Assumed Object ◄───┘
```

### 3.1. The Dual-Pointer `objInfo` Model

Each entry in `AssumeCache.store` is stored as an internal `*objInfo` struct wrapping two object pointers:

```go
type objInfo struct {
    name      string      // MetaNamespaceKey ("<namespace>/<name>" or "<name>")
    latestObj interface{} // Latest version (either assumed in-memory or from informer)
    apiObj    interface{} // Authoritative version received from the API server informer
}
```

- **`latestObj`**: Returned by `Get(key)` and `List(indexObj)`. Represents the active working state (including unconfirmed optimistic reservations).
- **`apiObj`**: Returned by `GetAPIObj(key)`. Tracks the authoritative state known to the API server watch stream.
- **`Assume(obj)`**: Updates `latestObj` only. Validates that `newVersion >= storedVersion`.
- **`Restore(objName)`**: Reverts `latestObj` back to `apiObj` if a reservation fails or is aborted.
- **Informer `add` / `update`**: Updates **both** `latestObj` and `apiObj` if `newVersion > storedVersion`.

### 3.2. Lifecycle Operations & Version Reconciliation

| Operation | Trigger / Caller | Lock | Version Check Rule | State Mutation |
|---|---|---|---|---|
| **`Assume(obj)`** | Scheduler Plugin (e.g. DRA) | Write (`Lock`) | `newVersion >= storedVersion` | Updates `latestObj = obj`. Emits update event. |
| **`Restore(name)`** | Plugin reservation rollback | Write (`Lock`) | N/A | Resets `latestObj = apiObj`. Emits update event if different. |
| **`add(obj)` / `update(old, new)`** | Informer watch callback | Write (`Lock`) | `newVersion > storedVersion` | Sets `latestObj = obj`, `apiObj = obj`. Emits add/update event. Skips if resync/older. |
| **`delete(obj)`** | Informer deletion callback | Write (`Lock`) | N/A | Deletes `objInfo` from `store`. Emits delete event. |
| **`Get(key)`** | Plugin read | Read (`RLock`) | N/A | Returns `objInfo.latestObj`. |
| **`GetAPIObj(key)`**| State reconciliation read | Read (`RLock`) | N/A | Returns `objInfo.apiObj`. |
| **`List(indexObj)`**| Indexed query | Read (`RLock`) | N/A | Returns slice of `latestObj` matching index or all objects. |

#### ResourceVersion Parsing
Resource versions are converted to `int64` via `meta.Accessor(obj).GetResourceVersion()` and `strconv.ParseInt`. Non-numeric or missing versions return errors wrapped in `ObjectNameError` or format errors.

### 3.3. Thread Safety, Event Queueing & Deadlock Prevention

`AssumeCache` uses a dedicated concurrency design to prevent deadlocks with registered `eventHandlers`:

#### The Deadlock Hazard:
Event handlers registered via `AddEventHandler` may invoke `AssumeCache` read methods (e.g., `Get` or `List`) during their execution. If event handlers were invoked while holding `rwMutex`, a nested lock or handler lock inversion would result in a deadlock.

#### The Solution:
1. **Queued Execution**: While `rwMutex.Lock()` is held during cache updates (`Assume`, `Restore`, `add`, `delete`, `AddEventHandler`), notification closures `func()` are pushed into an unbounded ring buffer (`eventQueue buffer.Ring[func()]`).
2. **Deferred Dispatch via Baton Passing (`emitEvents`)**:
   - `emitEvents()` is scheduled via `defer c.emitEvents()` **outside** the lock.
   - A single active worker flag (`emittingEvents bool`) and condition variable (`cond = sync.NewCond(&c.rwMutex)`) ensure that only one goroutine drains and delivers events at any given time.
   - Events are dequeued one-by-one under lock, and the notification closure `deliver()` is invoked **without holding `rwMutex`**.
   - Strict FIFO event ordering is preserved across concurrent callers.

```go
func (c *AssumeCache) emitEvents() {
    c.rwMutex.Lock()
    for c.emittingEvents {
        c.cond.Wait()
    }
    c.emittingEvents = true
    c.rwMutex.Unlock()

    defer func() {
        c.rwMutex.Lock()
        c.emittingEvents = false
        c.cond.Signal() // Hand over baton to next waiting goroutine
        c.rwMutex.Unlock()
    }()

    for {
        c.rwMutex.Lock()
        deliver, ok := c.eventQueue.ReadOne()
        c.rwMutex.Unlock()

        if !ok {
            return
        }
        func() {
            defer utilruntime.HandleCrash()
            deliver() // Invoked without holding rwMutex!
        }()
    }
}
```

### 3.4. Error Handling & Sentinel Errors

`assumecache` provides sentinel errors and structured error types for inspection via `errors.Is`:

```go
var (
    ErrWrongType  = errors.New("object has wrong type")
    ErrNotFound   = errors.New("object not found")
    ErrObjectName = errors.New("cannot determine object name")
)
```

- **`WrongTypeError`**: Holds `TypeName` and `Object`. Satisfies `errors.Is(err, ErrWrongType)`.
- **`NotFoundError`**: Holds `TypeName` and `ObjectKey`. Satisfies `errors.Is(err, ErrNotFound)`.
- **`ObjectNameError`**: Wraps `DetailedErr`. Satisfies `errors.Is(err, ErrObjectName)`.

---

## 4. Scheduler Utilities (`pkg/scheduler/util`)

### 4.1. Pod Priority, Ordering & Timestamp Utilities

```go
var maxPodStartTime = metav1.NewTime(time.Unix(0, math.MaxInt64).UTC())
```

- **`maxPodStartTime`**: A static sentinel timestamp representing infinity (`math.MaxInt64`). Assumed pods and bound pods that have not yet started do not possess `pod.Status.StartTime`. Using `maxPodStartTime` treats them deterministically as newer than started pods without generating real-time timestamps (e.g. `time.Now()`), which would violate Go `sort.Slice` strict weak ordering invariants and cause runtime panics or non-deterministic queue reordering.
- **`GetPodFullName(pod *v1.Pod)`**: Returns `pod.Name + "_" + pod.Namespace`. The underscore delimiter (`_`) is used because DNS subdomain conventions for Kubernetes pod names prohibit underscores, guaranteeing collision-free unique keys.
- **`GetPodStartTime(pod *v1.Pod)`**: Returns `pod.Status.StartTime` if set, otherwise returns `&maxPodStartTime`.
- **`GetEarliestPodStartTime(victims *extenderv1.Victims)`**: Iterates over victim pods in preemption evaluation to find the earliest `StartTime` among all pods sharing the highest priority level.
- **`MoreImportantPod(pod1, pod2 *v1.Pod)`**: Compares two pods:
  1. Priority comparison: `corev1helpers.PodPriority(pod1) > corev1helpers.PodPriority(pod2)`
  2. Start time comparison (tie-breaker): `GetPodStartTime(pod1).Before(GetPodStartTime(pod2))` (older started pod is more important).
- **`PodPreemptionPolicy(pod *v1.Pod)`**: Returns `*pod.Spec.PreemptionPolicy` if specified, or defaults to `v1.PreemptLowerPriority`.

### 4.2. API Mutation Helpers & Exponential Backoff

The scheduler interacts with the API server during binding and status updates. Network blips, rate limiting, and optimistic concurrency conflicts require robust retry strategies.

#### Error Classification Functions:
- **`Retriable(err error) bool`**: Returns `true` for transient network/server failures:
  - `apierrors.IsInternalError(err)` (HTTP 500)
  - `apierrors.IsServiceUnavailable(err)` (HTTP 503)
  - `net.IsConnectionRefused(err)`
- **`RetriableWithConflict(err error) bool`**: Extends `Retriable(err)` by including `apierrors.IsConflict(err)` (HTTP 409). Used for patch operations where resource version collisions can be safely retried.

#### Mutation Helpers:
- **`BindPod(ctx, cs, binding)`**: Submits a `v1.Binding` subresource request to assign a pod to a node using `retry.OnError(retry.DefaultBackoff, Retriable, bindFn)`. Conflicts are not retried for binding because a bound pod cannot be rebound.
- **`PatchPodStatus(ctx, cs, name, namespace, oldStatus, newStatus)`**:
  1. Marshals `oldStatus` and `newStatus` into `v1.Pod` wrappers.
  2. Computes a 2-way strategic merge patch via `strategicpatch.CreateTwoWayMergePatch`.
  3. Short-circuits if the diff is empty (`"{}"`).
  4. Executes `cs.CoreV1().Pods(namespace).Patch` with `types.StrategicMergePatchType` under `retry.OnError(retry.DefaultBackoff, RetriableWithConflict, patchFn)`.
- **`PatchPodGroupStatus(ctx, cs, name, namespace, oldStatus, newStatus)`**: Performs 2-way strategic merge patching for `schedulingv1beta1.PodGroupStatus` with conflict retries.
- **`PatchCompositePodGroupStatus(ctx, cs, name, namespace, oldStatus, newStatus)`**: Performs 2-way strategic merge patching for `schedulingv1alpha3.CompositePodGroupStatus` with conflict retries.
- **`DeletePod(ctx, cs, pod)`**: Issues a direct pod deletion request via `cs.CoreV1().Pods(pod.Namespace).Delete`.

### 4.3. Resource Classification & Defaults

- **`IsScalarResourceName(name v1.ResourceName)`**: Identifies scalar countable resources (extended resources, hugepages, prefixed native resources `kubernetes.io/*`, and attachable volume resources `attachable-volumes-/*`).
- **`IsDRAExtendedResourceName(name v1.ResourceName)`**: Identifies extended resources or Dynamic Resource Allocation (DRA) device class names prefixed with `deviceclass.resource.kubernetes.io/`.
- **Default Resource Requests (`pod_resources.go`)**:
  - `DefaultMilliCPURequest = 100` (0.1 core)
  - `DefaultMemoryRequest = 200 * 1024 * 1024` (200 MiB)
  - *Purpose*: When scoring plugins evaluate node allocations, containers omitting explicit resource requests are treated as requesting these default quantities. This prevents zero-request pods from concentrating exclusively on nodes with the smallest in-use requests and ensures regular workloads do not perceive zero-request pods as consuming zero capacity.

### 4.4. Container Host Port Extraction

- **`GetHostPorts(pod *v1.Pod)`**: Extracts all `HostPort > 0` specifications from:
  1. Standard containers (`pod.Spec.Containers[*]`).
  2. Sidecar init containers (`pod.Spec.InitContainers[*]`) having `RestartPolicy == v1.ContainerRestartPolicyAlways`. Regular terminating init containers are excluded since they do not hold host ports for the pod's entire operational lifetime.

### 4.5. Generic Type Assertion Helper (`As[T]`)

```go
func As[T any](oldObj, newobj interface{}) (T, T, error)
```
- Safely asserts informer event handler arguments (`oldObj`, `newobj`) to target type `T`.
- Transparently unpacks `cache.DeletedFinalStateUnknown` tombstones for `oldObj` during delete events.
- Handles `nil` objects without panic, returning typed zero values.

---

## 5. Developer & Invariant Cheat Sheet

### Thread Safety Invariants:
1. **Never Invoke Callbacks While Holding `AssumeCache.rwMutex`**: All event handlers and external callbacks must be queued in `eventQueue` and dispatched via `emitEvents()`.
2. **Strict Weak Ordering in Slices**: Never call `time.Now()` inside sorting comparators or priority helpers. Always use `GetPodStartTime()` which relies on static `maxPodStartTime` for unstarted pods.
3. **Resource Version Ordering**: An assumed object must have `newVersion >= storedVersion`. Informer updates must have `newVersion > storedVersion` to supersede assumed state.
4. **Retry Idempotency**: Status patch functions (`PatchPodStatus`, `PatchPodGroupStatus`, etc.) must compute delta patches against the provided `oldStatus` to ensure retries on conflict operate cleanly.

### Testing Hooks:
- **`assumecache.AddTestObject` / `UpdateTestObject` / `DeleteTestObject`**: Helper functions exported for test packages to manipulate cache state directly without constructing full informers.
