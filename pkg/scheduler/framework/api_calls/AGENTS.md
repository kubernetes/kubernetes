# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/api_calls`

This guide provides AI agents and human contributors with an architectural overview, interface specifications, execution lifecycles, and synchronization invariants for the asynchronous API operations framework located in `pkg/scheduler/framework/api_calls`.

---

## 1. High-Level Overview & Package Scope

The `api_calls` package encapsulates deferred, batched, or asynchronous Kubernetes API interactions dispatched by the scheduling framework (e.g. `PodStatusPatchCall` and `PodBindingCall`). By abstracting raw client-go API invocations into polymorphic `fwk.APICall` objects, the scheduler enables deduplication, optimistic concurrency merging, relevance-based ordering, and thread-safe cache reconciliation.

### Core Responsibilities:
1. **API Operation Abstraction (`fwk.APICall`)**: Encapsulates specific mutation requests against API server resources with standard `Execute`, `Sync`, and `Merge` lifecycles.
2. **Relevance Hierarchy (`Relevances`)**: Assigns deterministic execution priority to API call types (e.g. status patch vs binding) to prevent out-of-order race conditions during batched execution.
3. **Optimistic Cache Synchronization (`Sync`)**: Updates in-flight API calls with newer object versions from informer caches without discarding pending mutations before execution begins.
4. **Mutation Merging & Deduplication (`Merge`, `IsNoOp`)**: Coalesces multiple condition patches or nomination updates for the same pod UID into a single atomic API call, discarding redundant no-op updates.

---

## 2. Directory Architecture & File Map

```
pkg/scheduler/framework/api_calls/
├── api_calls.go               # APICallType constants, Relevances ranking, and Constructor registry
├── api_calls_test.go          # Tests for constructor registration and type relevance ordering
├── pod_binding.go             # PodBindingCall implementation (fwk.APICall)
├── pod_binding_test.go        # Unit tests for binding creation, synchronization, and execution
├── pod_status_patch.go        # PodStatusPatchCall implementation with mutex-guarded status syncing
├── pod_status_patch_test.go   # Unit tests for status condition merging, deduplication, and no-ops
└── AGENTS.md                  # This agent guide
```

---

## 3. Architecture & Type System

### 3.1. API Call Registry & Relevance (`api_calls.go`)

```go
const (
    PodStatusPatch fwk.APICallType = "pod_status_patch"
    PodBinding     fwk.APICallType = "pod_binding"
)

var Relevances = map[fwk.APICallType]int{
    PodStatusPatch: 1,
    PodBinding:     2,
}
```

#### Relevance Rules:
- When multiple API calls are queued for the same pod, higher-relevance calls run after lower-relevance calls or supersede them during consolidation.
- `PodBinding` (relevance 2) has higher precedence than `PodStatusPatch` (relevance 1).

### 3.2. Constructor Factory (`Implementations`)
```go
var Implementations = map[fwk.APICallType]func(obj runtime.Object) (fwk.APICall, error){
    PodStatusPatch: NewPodStatusPatchCall,
    PodBinding:     NewPodBindingCall,
}
```

---

## 4. API Call Implementations

### 4.1. `PodBindingCall` (`pod_binding.go`)

Encapsulates an API request to bind a pod to a chosen node:

```go
type PodBindingCall struct {
    binding *v1.Binding
}
```

- **`Execute(ctx, client)`**: Calls `util.BindPod(ctx, client, pbc.binding)`.
- **`Merge(other)`**: Replaces the current binding with the incoming `PodBindingCall` (last-write-wins).
- **`Sync(obj)`**: No-op (binding payload contains static target node name and UID).

### 4.2. `PodStatusPatchCall` (`pod_status_patch.go`)

Thread-safe encapsulation for patching pod status conditions (e.g. `PodScheduled = False`, `PodReasonUnschedulable`, `DisruptionTarget`) and `NominatedNodeName`:

```go
type PodStatusPatchCall struct {
    mu             sync.Mutex
    executed       bool
    pod            *v1.Pod
    podStatus      *v1.PodStatus
    newConditions  []v1.PodCondition
    nominatingInfo *fwk.NominatingInfo
}
```

#### Key Lifecycle Mechanics:

1. **`Sync(obj runtime.Object) error`**:
   - Invoked when the scheduler's informer cache receives an updated `*v1.Pod` object while this call is queued.
   - If `executed == false`, updates internal `p.pod` and `p.podStatus` references to ensure the subsequent patch is computed against the latest resource version.
2. **`Merge(other fwk.APICall) error`**:
   - Merges newly added conditions from `other` into `newConditions`. If condition types collide, the newer condition overwrites the older one.
   - Merges `nominatingInfo` (e.g. setting or clearing nominated node).
3. **`IsNoOp() bool`**:
   - Inspects `newConditions` and `nominatingInfo` against current `podStatus`.
   - Returns `true` if all conditions are already present with identical status/reason/message and `NominatedNodeName` is unchanged, avoiding unnecessary API roundtrips.
4. **`Execute(ctx, client) error`**:
   - Sets `executed = true`.
   - Synchronizes final conditions and nominated node name into a cloned status object.
   - Computes JSON strategic merge patch and calls `util.PatchPodStatus(ctx, client, p.pod, newStatus)`.

---

## 5. Developer Invariants & Concurrency Rules

1. **Thread Safety via Mutex**:
   - `PodStatusPatchCall` fields are guarded by `mu sync.Mutex`. All read and write operations (`Execute`, `Sync`, `Merge`, `IsNoOp`) must acquire this lock.
2. **Post-Execution Immutability**:
   - Once `executed` is marked `true`, subsequent `Sync` calls MUST be ignored to prevent race conditions during in-flight network requests.
3. **No-Op Suppression**:
   - Callers should check `IsNoOp()` before dispatching patches over the network to prevent redundant API server load.

---

## 6. Verification Commands

```bash
# Run all api_calls tests with race detector
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/api_calls/...
```
