# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework`

This guide provides AI agents and human contributors with an architectural overview, interface specifications, lifecycle contracts, concurrency mechanics, context propagation models, and testing strategies for the Kubernetes Scheduler Framework core interfaces under `pkg/scheduler/framework` and `staging/src/k8s.io/kube-scheduler/framework`.

---

## 1. High-Level Purpose & Scope

The Scheduling Framework is a pluggable architecture for the Kubernetes scheduler (`kube-scheduler`). It defines a comprehensive set of extension points across the pod scheduling and binding lifecycles, enabling modular plugins to control queue sorting, feasibility filtering, node scoring, resource reservation, binding approval, pre-binding setup, and API binding.

### Core Responsibilities:
1. **Extension Point Abstraction**: Defines standardized Go interfaces for every phase of scheduling (from queue ingress to post-bind cleanup).
2. **Context & State Management (`CycleState`)**: Provides an isolated, thread-safe, ephemeral key-value store (`CycleState`) passed through every extension point within a single scheduling cycle.
3. **Queueing & Entity Abstractions (`QueuedPodInfo`, `QueuedEntityInfo`)**: Tracks pod and pod group queueing history, backoff expiration, attempts, unschedulable plugins, PreEnqueue gating, and opportunistic batching signatures.
4. **Cluster Snapshot & Node Representation (`NodeInfo`)**: Encapsulates point-in-time cached state of nodes (allocatable resources, consumed resources, running pods, affinities, taints, ports, PVCs, and image states) with generation tracking.
5. **Framework Handle (`fwk.Handle`)**: Exposes scheduler services to plugins (informers, client-go clientsets, event recorders, node/pod listers, parallelizer, DRA/CSI managers, pod activators/nominators, and async API dispatchers).
6. **Execution Orchestration (`pkg/scheduler/framework/runtime`)**: Manages plugin registries, profile-to-plugin bindings, parallel chunk execution, score normalization, error handling, and rollback guarantees (`Unreserve`).

---

## 2. Package Architecture & Staging Boundary

The scheduling framework types are split across two packages:
- **`k8s.io/kube-scheduler/framework` (`staging/src/k8s.io/kube-scheduler/framework`)**: The exported public API containing interface definitions (`Plugin`, `FilterPlugin`, `ScorePlugin`, etc.), `Handle`, `Status`, `Code`, `WaitingPod`, and public extension points.
- **`k8s.io/kubernetes/pkg/scheduler/framework` (`pkg/scheduler/framework/`)**: The internal implementation containing `CycleState`, `NodeInfo`, `QueuedPodInfo`, `NodeToStatus`, `Events`, heap wrappers, and subpackages (`runtime`, `plugins`, `parallelize`, `preemption`, `autoscaler_contract`, `api_calls`).

### File Map:

```
pkg/scheduler/framework/
├── interface.go                 # Internal Framework interface, NodeToStatus, PodsToActivate, SortedScoredNodes
├── cycle_state.go               # CycleState implementation, thread-safe storage, cloning, skip lists
├── types.go                     # NodeInfo, QueuedPodInfo, QueuedEntityInfo, QueueingParams, Diagnosis, FitError, Resource
├── events.go                    # ClusterEvent definitions, metrics labels, event matching, unrolling
├── sorted_nodes.go              # Min-heap and sorting primitives for scored candidate nodes
├── parallelize/                 # Parallel chunk worker pool execution (Parallelizer, Until)
├── runtime/                     # Framework initialization, plugin registry, and extension point runner pipeline
├── plugins/                     # In-tree plugin implementations (NodeResourcesFit, NodeAffinity, etc.)
├── preemption/                  # Preemption logic, candidate generation, and victim eviction algorithms
├── autoscaler_contract/         # Interfaces and contracts for Cluster Autoscaler integration
├── api_calls/                   # Helpers for asynchronous/batched API interactions
└── AGENTS.md                    # This agent guide
```

---

## 3. Core Data Structures & State Management

### 3.1. `CycleState` (`cycle_state.go`)

`CycleState` is an ephemeral, thread-safe key-value store scoped to a single scheduling cycle (and passed forward into the asynchronous binding cycle).

```go
type CycleState struct {
    storage                  sync.Map
    recordPluginMetrics      bool
    skipFilterPlugins        sets.Set[string]
    skipScorePlugins         sets.Set[string]
    skipPreBindPlugins       sets.Set[string]
    skipAllPostFilterPlugins bool
    parallelPreBindPlugins   sets.Set[string]
    podGroupCycleState       fwk.PodGroupCycleState
    placementCycleState      fwk.PlacementCycleState
}
```

#### Key Characteristics & Mechanics:
- **Write-Once, Read-Many (WORM) Pattern**: Built on `sync.Map`. In-tree plugins write precomputed metadata once during `PreFilter` or `PreScore`, and read it concurrently during parallel `Filter` or `Score` evaluations across hundreds/thousands of nodes.
- **`StateData` Interface**: Any value stored in `CycleState` must implement `fwk.StateData` (which requires a `Clone() StateData` method).
- **Deep Cloning**: `CycleState.Clone()` iterates over `storage` using `Range` and invokes `Clone()` on each `StateData` entry. This allows branching during preemption simulation without corrupting the parent cycle's state.
- **Plugin Skip Sets**:
  - `skipFilterPlugins`: Set when `PreFilter` returns `framework.NewStatus(framework.Skip)`. The runtime bypasses the corresponding `Filter` plugin on all nodes.
  - `skipScorePlugins`: Set when `PreScore` returns `framework.NewStatus(framework.Skip)`. The runtime bypasses the corresponding `Score` plugin.
  - `skipPreBindPlugins`: Set when `PreBindPreFlight` returns `framework.NewStatus(framework.Skip)`.
  - `skipAllPostFilterPlugins`: Instructs the framework to skip post-filter preemption attempts.
- **Hierarchical Workload Scoping**: Supports `podGroupCycleState` and `placementCycleState` links for generic workload gang scheduling.

### 3.2. `NodeInfo` (`types.go`)

`NodeInfo` represents the scheduler's cached point-in-time view of a single node.

#### Core Fields:
- `node *v1.Node`: The API Node object.
- `Pods []fwk.PodInfo`: All pods currently assumed or scheduled on the node.
- `PodsWithAffinity`, `PodsWithRequiredAntiAffinity`, `PodsWithRequiredNonHostScopedAntiAffinity`: Fast-path indexed slices of pods declaring affinity rules.
- `UsedPorts fwk.HostPortInfo`: Set of container ports currently allocated on host IPs.
- `Requested *Resource`: Cumulative resource requests of all pods on the node (CPU, memory, ephemeral storage, scalar/extended resources).
- `NonZeroRequested *Resource`: Requests with minimum fallback values for CPU/memory to avoid packing zero-request pods indefinitely.
- `Allocatable *Resource`: Precomputed numeric allocatable capacities (`Node.Status.Allocatable`).
- `ImageStates map[string]*fwk.ImageStateSummary`: Image presence and size map for ImageLocality scoring.
- `PVCRefCounts map[string]int`: Mapping of PVCs to active pod counts on this node.
- `Generation int64`: Monotonically increasing counter bumped on every mutation, enabling O(1) cache snapshot invalidation checks.

### 3.3. `QueuedPodInfo` & `QueuedEntityInfo` (`types.go`)

`QueuedPodInfo` wraps a `*PodInfo` alongside comprehensive queueing metadata (`QueueingParams`):

```go
type QueuedPodInfo struct {
    *PodInfo
    QueueingParams
    PodSignature fwk.PodSignature
}
```

#### `QueueingParams` Invariants:
- `Timestamp time.Time`: Enqueue timestamp used for FIFO queue ordering among equal-priority pods.
- `Attempts int`: Total number of scheduling attempts across all queues.
- `InitialAttemptTimestamp *time.Time`: Initial admission timestamp; never overwritten, used for end-to-end latency metrics.
- `BackoffExpiration time.Time`: Backoff completion timestamp for `backoffQ` calculation.
- `UnschedulableCount int`: Incremented when scheduling fails with `Unschedulable`/`UnschedulableAndUnresolvable` status. Used to compute exponential backoff intervals.
- `ConsecutiveErrorsCount int`: Incremented when scheduling fails with `Error` status (e.g. API server network failure). Uses distinct backoff semantics to protect control plane stability without penalizing pod priority.
- `UnschedulablePlugins sets.Set[string]`: Set of plugin names that rejected this pod at `PreFilter`, `Filter`, `Reserve`, `Permit`, or `PlacementFeasible`.
- `GatingPlugin string` & `GatingPluginEvents []fwk.ClusterEvent`: Tracks the plugin gating this pod at `PreEnqueue` and the events registered to ungate it.
- `PodSignature fwk.PodSignature`: Serialized cryptographic signature fragment slice used by Opportunistic Batching (KEP-5598) to reuse scheduling decisions for identical pods.

### 3.4. `Status` & `Code` (`staging/src/k8s.io/kube-scheduler/framework/interface.go`)

Every extension point returns a structured `*Status`:

| Code | Constant | Meaning | Framework Action |
|---|---|---|---|
| `0` | `Success` | Phase completed successfully. | Continue to next plugin/phase. |
| `1` | `Error` | Internal/system error (e.g. timeout, network failure). | Abort cycle; requeue pod with error backoff. |
| `2` | `Unschedulable` | Pod cannot be scheduled on the evaluated node/cluster, but cluster state changes may fix it. | Record unschedulable reason; trigger PostFilter/requeue. |
| `3` | `UnschedulableAndUnresolvable` | Pod cannot be scheduled and preemption/cluster changes cannot resolve it (e.g. architecture mismatch). | Exclude node from preemption victim evaluation. |
| `4` | `Wait` | Permit or PlacementFeasible requests holding/delaying the pod. | Hold pod in `WaitingPods` or wait for timeout. |
| `5` | `Skip` | Plugin elects not to execute downstream phase (PreFilter -> Filter, PreScore -> Score, PreBindPreFlight -> PreBind, Bind). | Skip coupled execution for this plugin only. |
| `6` | `Pending` | Asynchronous operation pending. | Tracked by async API dispatcher. |

### 3.5. `NodeToStatus` (`interface.go`)

`NodeToStatus` maintains node-to-status mappings resulting from `PreFilter` and `Filter`:
- **Absent Node Optimization**: Holds an `absentNodesStatus` (default: `UnschedulableAndUnresolvable`). If `PreFilter` rejects all nodes, it sets `absentNodesStatus` rather than populating thousands of map entries.
- **Reader Interface (`NodeToStatusReader`)**: Passed into `PostFilterPlugin.PostFilter` to allow preemption plugins to inspect why each node failed.

---

## 4. Scheduling & Binding Lifecycle Architecture

The scheduling pipeline operates in two distinct phases: the **Scheduling Cycle** (synchronous, per-pod) and the **Binding Cycle** (asynchronous, executed in a separate goroutine).

```
                     ┌───────────────────────────────────────────────┐
                     │            Scheduling Queue Ingress           │
                     └───────────────────────┬───────────────────────┘
                                             │
                                     [ PreEnqueue ] ── (Fail/Gated) ──► unschedulableQ
                                             │
                                      [ QueueSort ] ──► Ordered activeQ
                                             │
 ╔═══════════════════════════════════════════╪═══════════════════════════════════════════════╗
 ║ SCHEDULING CYCLE (Synchronous)            ▼                                               ║
 ║                                    [ SignPod ] (Opportunistic Batching)                   ║
 ║                                           │                                               ║
 ║                                     [ PreFilter ] ── (Error/Unschedulable) ─┐             ║
 ║                                           │ (Skip Filter)                   │             ║
 ║                                           ▼                                 │             ║
 ║                                       [ Filter ]                            │             ║
 ║                                (Parallel per node)                          │             ║
 ║                                           │                                 │             ║
 ║                                    All Nodes Failed?                        │             ║
 ║                                   ├── Yes ──► [ PostFilter ] (Preemption) ◄─┘             ║
 ║                                   │                  │                                    ║
 ║                                   └── No             └──► Schedulable / Nominated / Abort ║
 ║                                       │                                                   ║
 ║                                  [ PreScore ]                                             ║
 ║                                       │ (Skip Score)                                      ║
 ║                                       ▼                                                   ║
 ║                                    [ Score ] (Parallel per node)                          ║
 ║                                       │                                                   ║
 ║                               [ NormalizeScore ] & Plugin Weighting                       ║
 ║                                       │                                                   ║
 ║                               [ Select Best Node ] (Reservoir Sampling for ties)          ║
 ║                                       │                                                   ║
 ║                                   [ Reserve ] ── (Fail) ──► Unreserve All                 ║
 ║                                       │                                                   ║
 ║                                    [ Permit ] ── (Reject) ──► Unreserve All               ║
 ╚═══════════════════════════════════════════╪═══════════════════════════════════════════════╝
                                             │ (Hand off to Goroutine)
 ╔═══════════════════════════════════════════╪═══════════════════════════════════════════════╗
 ║ BINDING CYCLE (Asynchronous Goroutine)    ▼                                               ║
 ║                                  [ WaitOnPermit ] ── (Timeout/Reject) ──► Unreserve All   ║
 ║                                           │                                               ║
 ║                               [ PreBindPreFlight ] (Optional Skip check)                  ║
 ║                                           │                                               ║
 ║                                   [ PreBind ] (Parallel & Sequential)                     ║
 ║                                           │ (Fail) ──► Unreserve All + ForgetPod          ║
 ║                                           ▼                                               ║
 ║                                     [ Bind ] (First matching plugin handles)              ║
 ║                                           │ (Fail) ──► Unreserve All + ForgetPod          ║
 ║                                           ▼                                               ║
 ║                                   [ PostBind ] (Informational metrics/cleanup)            ║
 ╚═══════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 5. Plugin Extension Point Contracts & Interfaces

### 5.1. `PreEnqueuePlugin`
```go
type PreEnqueuePlugin interface {
    Plugin
    PreEnqueue(ctx context.Context, p *v1.Pod) *Status
}
```
- **Execution**: Called before adding a pod to `activeQ` or `backoffQ`.
- **Contract**: Must be lightweight and non-blocking (never make remote API calls).
- **Semantics**: Non-success status marks the pod as gated and routes it to `unschedulableQ`. The pod is only retried when events registered by the gating plugin or wildcard events occur.

### 5.2. `QueueSortPlugin`
```go
type QueueSortPlugin interface {
    Plugin
    Less(QueuedEntityInfo, QueuedEntityInfo) bool
}
```
- **Execution**: Determines priority ordering in `activeQ`.
- **Contract**: Exactly **one** `QueueSortPlugin` must be enabled across all profiles in a scheduler process.
- **Default**: `PrioritySort` (compares `.spec.priority`, breaking ties with enqueue timestamp).

### 5.3. `EnqueueExtensions`
```go
type EnqueueExtensions interface {
    Plugin
    EventsToRegister(context.Context) ([]ClusterEventWithHint, error)
}
```
- **Execution**: Evaluated once during scheduler startup.
- **Contract**: Plugins that can reject pods (`PreEnqueue`, `PreFilter`, `Filter`, `Reserve`, `Permit`, `PlacementFeasible`) must implement this to return specific cluster events (`ClusterEvent`) and queueing hint callbacks (`QueueingHintFn`).
- **Optimization**: Queueing hints allow the scheduler to filter out irrelevent cluster events before waking unscheduled pods.

### 5.4. `PreFilterPlugin` & `PreFilterExtensions`
```go
type PreFilterPlugin interface {
    Plugin
    PreFilter(ctx context.Context, state CycleState, p *v1.Pod, nodes []NodeInfo) (*PreFilterResult, *Status)
    PreFilterExtensions() PreFilterExtensions
}

type PreFilterExtensions interface {
    AddPod(ctx context.Context, state CycleState, podToSchedule *v1.Pod, podInfoToAdd PodInfo, nodeInfo NodeInfo) *Status
    RemovePod(ctx context.Context, state CycleState, podToSchedule *v1.Pod, podInfoToRemove PodInfo, nodeInfo NodeInfo) *Status
}
```
- **`PreFilter`**: Precomputes cycle-level state (stored in `CycleState`) and can optionally return a `*PreFilterResult` containing a restricted node name set (`sets.Set[string]`) to prune downstream filter iterations. Returning `Code = Skip` skips this plugin's `Filter` method.
- **`PreFilterExtensions`**: Used during preemption simulation to incrementally update precomputed state when adding/removing prospective victim pods on candidate nodes without re-running full `PreFilter`.

### 5.5. `FilterPlugin`
```go
type FilterPlugin interface {
    Plugin
    Filter(ctx context.Context, state CycleState, pod *v1.Pod, nodeInfo NodeInfo) *Status
}
```
- **Execution**: Evaluated in parallel across worker chunks using `Parallelizer`.
- **Contract**: Returns `Success` if the node can host the pod; `Unschedulable` if currently ineligible; `UnschedulableAndUnresolvable` if unfixable (e.g. architecture/OS mismatch, missing required label); `Error` on internal failure.
- **Cancellation**: Must respect `ctx.Done()` and return `UnschedulableAndUnresolvable` with `context.Cause(ctx)` when parallel searching is short-circuited.

### 5.6. `PostFilterPlugin`
```go
type PostFilterPlugin interface {
    Plugin
    PostFilter(ctx context.Context, state CycleState, pod *v1.Pod, filteredNodeStatusMap NodeToStatusReader) (*PostFilterResult, *Status)
}
```
- **Execution**: Invoked sequentially when 0 nodes pass `Filter`.
- **Contract**: Evaluates preemption (e.g. `DefaultPreemption`) to evict lower-priority pods.
- **Status Returns**: `Success` (preemption candidate found; returns `*PostFilterResult` with `NominatedNodeName`), `Unschedulable` (preemption not feasible), or `Error`.

### 5.7. `PreScorePlugin`
```go
type PreScorePlugin interface {
    Plugin
    PreScore(ctx context.Context, state CycleState, pod *v1.Pod, nodes []NodeInfo) *Status
}
```
- **Execution**: Invoked once after `Filter` passes candidate nodes and before `Score`.
- **Contract**: Precomputes scoring state across the feasible node subset and writes it to `CycleState`. Returning `Code = Skip` skips this plugin's `Score` evaluation.

### 5.8. `ScorePlugin` & `ScoreExtensions`
```go
type ScorePlugin interface {
    Plugin
    Score(ctx context.Context, state CycleState, p *v1.Pod, nodeInfo NodeInfo) (int64, *Status)
    ScoreExtensions() ScoreExtensions
}

type ScoreExtensions interface {
    NormalizeScore(ctx context.Context, state CycleState, p *v1.Pod, scores NodeScoreList) *Status
}
```
- **`Score`**: Assigns an integer score to each feasible node in the range `[0, MaxNodeScore]` (`MaxNodeScore = 100`). Runs in parallel.
- **`NormalizeScore`**: Optional post-scoring normalization to adjust scores across the whole node list (e.g. min-max scaling, inverting preferences).
- **Weighting**: The runtime multiplies each plugin's normalized score by its configured `weight` in the profile and sums them to compute `TotalScore`.

### 5.9. `ReservePlugin`
```go
type ReservePlugin interface {
    Plugin
    Reserve(ctx context.Context, state CycleState, p *v1.Pod, nodeName string) *Status
    Unreserve(ctx context.Context, state CycleState, p *v1.Pod, nodeName string)
}
```
- **`Reserve`**: In-memory state reservation on the selected node before moving to `Permit`/`PreBind` (e.g. reserving volume bindings or DRA claims).
- **`Unreserve`**: **Mandatory Rollback Guarantee**. Must be completely idempotent. The runtime calls `Unreserve` on **all** enabled `ReservePlugin`s if `Reserve` fails on any plugin, if `Permit` rejects or times out, or if `PreBind`/`Bind` fails.

### 5.10. `PermitPlugin`
```go
type PermitPlugin interface {
    Plugin
    Permit(ctx context.Context, state CycleState, p *v1.Pod, nodeName string) (*Status, time.Duration)
}
```
- **Execution**: Coordinates gang scheduling, multi-pod co-scheduling, or external gating.
- **Statuses**:
  - `Success`: Proceed directly to binding.
  - `Wait`: Holds the pod in `WaitingPods` map for up to `time.Duration` (max `15m`).
  - `Unschedulable` / `Reject`: Aborts scheduling and unreserves.
- **Resolution**: External controllers or plugins call `Handle.GetWaitingPod(uid).Allow()` or `.Reject()`.

### 5.11. `PreBindPlugin`
```go
type PreBindPlugin interface {
    Plugin
    PreBindPreFlight(ctx context.Context, state CycleState, p *v1.Pod, nodeName string) (*PreBindPreFlightResult, *Status)
    PreBind(ctx context.Context, state CycleState, p *v1.Pod, nodeName string) *Status
}
```
- **Execution**: Runs in the asynchronous binding goroutine before `Bind`.
- **`PreBindPreFlight`**: Lightweight pre-check returning `Success` (proceed to PreBind), `Skip` (bypass this PreBind plugin), or `Error`.
- **Parallel PreBind**: Plugins designated as safe can execute concurrently (e.g. DRA device claims, volume attachments).

### 5.12. `BindPlugin`
```go
type BindPlugin interface {
    Plugin
    Bind(ctx context.Context, state CycleState, p *v1.Pod, nodeName string) *Status
}
```
- **Execution**: Creates the binding resource on the API server (`/binding` subresource).
- **Sequential Handling**: Plugins are executed in order. A plugin returning `Code = Skip` yields to the next plugin. The first plugin returning `Success` or `Error` terminates the bind chain. Default: `DefaultBinder`.

### 5.13. `PostBindPlugin`
```go
type PostBindPlugin interface {
    Plugin
    PostBind(ctx context.Context, state CycleState, p *v1.Pod, nodeName string)
}
```
- **Execution**: Informational notification after successful binding; used for metrics and resource cleanup.

---

## 6. Framework Handle (`fwk.Handle`)

`fwk.Handle` is passed into every plugin's factory constructor upon initialization (`PluginFactory = func(ctx context.Context, configuration runtime.Object, f fwk.Handle) (fwk.Plugin, error)`).

### Capabilities Provided:
1. **Listers & Snapshots**:
   - `SnapshotSharedLister()`: Read-only node snapshot for the current scheduling cycle. **Rule**: Must only be used during scheduling cycle (up to `Permit`), never during binding or queueing hints.
   - `SharedInformerFactory()`: Access to live Kubernetes informers for event-driven updates.
2. **Cluster Communication**:
   - `ClientSet()`: Typed Kubernetes API client.
   - `KubeConfig()`: Raw REST client configuration.
   - `EventRecorder()`: Kubernetes event publisher for pod event streams.
3. **Queue & Pod Management**:
   - `PodActivator`: Move stashed pods from `unschedulableEntities` or `backoffQ` into `activeQ`.
   - `PodNominator`: Track and query nominated pods on nodes for preemption coordination.
   - `IterateOverWaitingPods`, `GetWaitingPod`, `RejectWaitingPod`: Permit lifecycle management.
4. **Specialized Managers**:
   - `SharedDRAManager()`: Dynamic Resource Allocation (DRA) claim state and in-memory tracking.
   - `SharedCSIManager()`: CSI node storage capacity and volume tracking.
   - `PreemptionManager()`: Preemption victim generation and eviction execution.
   - `APIDispatcher()` / `APICacher()`: Non-blocking asynchronous API request dispatching and cache coordination.
   - `Parallelizer()`: Worker pool for parallel execution of node chunks.

---

## 7. Concurrency & Context Propagation Model

### 7.1. Thread-Safety Rules
1. **`CycleState` Immutability during Parallel Phases**:
   - `CycleState` data structures written during `PreFilter` or `PreScore` **must be thread-safe for concurrent reads** across `Filter` and `Score` goroutines. Never mutate shared `CycleState` values inside `Filter` or `Score` without synchronization.
2. **`NodeInfo` Immutability**:
   - `NodeInfo` instances provided by `SnapshotSharedLister()` are shared across parallel goroutines within a cycle. Plugins **must never mutate** `NodeInfo` fields during `Filter` or `Score`.
3. **Binding Goroutine Handoff**:
   - The scheduling cycle hands off `CycleState` to the asynchronous binding cycle goroutine (`runBindingCycle`). The scheduling cycle ends synchronously, freeing the scheduler loop to dequeue the next pod immediately.

### 7.2. Context & Cancellation
- The framework passes `context.Context` to all plugin methods.
- When sufficient feasible nodes are identified (controlled by `percentageOfNodesToScore`), the framework cancels the filter context (`context.WithCancelCause`). Plugins must check `ctx.Err()` or `ctx.Done()` and exit promptly.

---

## 8. In-Tree Plugin Ecosystem Overview

| Plugin | Extension Points Implemented | Role & Description |
|---|---|---|
| **`NodeResourcesFit`** | PreFilter, Filter, PreScore, Score, EnqueueExtensions | Checks CPU, memory, storage, and scalar resource capacity; scores via LeastAllocated, MostAllocated, or RequestedToCapacityRatio. |
| **`NodeAffinity`** | PreFilter, Filter, PreScore, Score, EnqueueExtensions | Evaluates `nodeSelector`, `nodeAffinity` (required and preferred). |
| **`InterPodAffinity`** | PreFilter, Filter, PreScore, Score, EnqueueExtensions | Evaluates inter-pod affinity and anti-affinity rules across topological domains. |
| **`PodTopologySpread`** | PreFilter, Filter, PreScore, Score, EnqueueExtensions | Enforces even pod distribution across failure domains (zones, nodes) according to `topologySpreadConstraints`. |
| **`TaintToleration`** | Filter, PreScore, Score, EnqueueExtensions | Filters nodes with un-tolerated taints; scores nodes preferring tolerated taints. |
| **`NodePorts`** | PreFilter, Filter, EnqueueExtensions | Validates host port availability (`spec.containers[*].ports[*].hostPort`). |
| **`VolumeBinding`** | PreFilter, Filter, Reserve, PreBind, EnqueueExtensions | Validates and binds PersistentVolumeClaims and dynamically provisioned storage. |
| **`DynamicResources` (DRA)** | PreFilter, Filter, PreScore, Score, Reserve, PreBind, EnqueueExtensions | Allocates dynamic resource claims and hardware devices via DRA drivers. |
| **`DefaultPreemption`** | PostFilter | Evicts lower-priority pods to make room for unschedulable higher-priority pods. |
| **`DefaultBinder`** | Bind | Default implementation issuing HTTP POST to the pod's `/binding` subresource. |

---

## 9. Testing Guide & Verification Strategies

### 9.1. Unit Testing Framework Extensions
- **Mock Framework Handle**: Use `frameworkruntime.NewFramework` or test handles in `pkg/scheduler/framework/runtime` to supply mock listers, clientsets, and recorders.
- **CycleState Fixtures**: Initialize `framework.NewCycleState()` and populate keys directly using `state.Write(key, data)`.
- **NodeToStatus Verification**: Assert status codes and failure reasons using `diff.ObjectReflectDiff` or `status.Code()`.

### 9.2. Executing Package Tests
```bash
# Run framework core unit tests
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/...

# Run framework runtime unit tests
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/runtime/...

# Run staging framework unit tests
GOTOOLCHAIN=auto go test -v -race ./staging/src/k8s.io/kube-scheduler/framework/...
```

---

## 10. Critical Invariants for Developers & AI Agents

1. **Idempotent `Unreserve`**: Always implement `Unreserve` to be completely safe against duplicate calls or calls for reservations that never succeeded.
2. **Snapshot Lifetime Restrictions**: Never call `Handle.SnapshotSharedLister()` outside the scheduling cycle (forbidden in `QueueingHint`, `PreEnqueue`, `PreBind`, `Bind`, `PostBind`, `Unreserve`).
3. **Accurate `EnqueueExtensions`**: Always return precise `ClusterEventWithHint` registrations for plugins that can return `Unschedulable`. Omitting events causes rejected pods to get stuck in `unschedulableQ` until periodic queue flushes.
4. **Distinguish Error vs Unschedulable**: Return `Code = Error` only for transient, unexpected system failures (I/O timeouts, serialization errors). Return `Code = Unschedulable` for policy or capacity rejections.
5. **No Shared Mutation in Filters/Scores**: Never mutate plugin struct fields or `CycleState` values without synchronization during `Filter` or `Score` execution.
