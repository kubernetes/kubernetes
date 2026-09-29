# Agent Guide: Kubernetes Scheduler (`pkg/scheduler`)

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, component map, invariants, and testing guide for the Kubernetes scheduler codebase located under `pkg/scheduler`.

---

## 1. High-Level Overview & Role

The Kubernetes Scheduler (`kube-scheduler`) is the default control-plane component responsible for assigning unscheduled Pods to optimal Nodes in a cluster.

### Core Objectives:
1. **Filtering (Feasibility)**: Eliminate nodes that cannot satisfy a Pod's resource requirements, node selectors, taints, volume constraints, or affinity rules.
2. **Scoring (Prioritization)**: Rank feasible nodes based on configurable scoring strategies (e.g., resource packing/spreading, topology spread, image locality, affinity preference).
3. **Reservation & Binding**: Assume the placement in the in-memory cache to prevent race conditions, execute reserve plugins, and asynchronously bind the pod to the chosen node on the API server.
4. **Preemption & Pod Groups**: Handle preemption of lower-priority pods when no node fits a high-priority pod, and manage atomic gang scheduling for composite pod groups.

---

## 2. Architecture & Subsystems

The `pkg/scheduler` directory is organized into several key modular subsystems:

```
pkg/scheduler/
├── scheduler.go                   # Main Scheduler struct & controller orchestration
├── schedule_one.go                # Single-pod scheduling loop (scheduleOnePod / runBindingCycle)
├── schedule_one_podgroup.go       # Multi-pod / Gang scheduling algorithm for PodGroups
├── algorithm.go                   # Core schedulePod algorithm (filtering, scoring, host selection)
├── eventhandlers.go               # Informer event handlers routing cluster events to the queue & cache
├── extender.go                    # Legacy HTTP extender integration
├── framework_init.go              # Default framework and plugin registry setup
├── types.go                       # Internal scheduler types and revert functions
├── apis/
│   └── config/                    # KubeSchedulerConfiguration (see apis/config/AGENTS.md)
├── backend/
│   ├── cache/                     # Scheduler cache (nodes, assumed/real pods, snapshotting)
│   ├── queue/                     # PriorityQueue (activeQ, backoffQ, unschedulablePods)
│   ├── heap/                      # Generic heap implementation backing scheduling queues
│   ├── api_cache/                 # API cache for async/batched operations
│   └── api_dispatcher/            # API dispatcher for async API calls
├── framework/
│   ├── plugins/                   # In-tree plugins (filtering, scoring, binding, preemption, etc.)
│   ├── runtime/                   # Framework implementation and plugin execution runner
│   ├── preemption/                # Preemption interface & candidate evaluation algorithms
│   ├── parallelize/               # Chunking & parallel evaluation helpers for node iterations
│   ├── autoscaler_contract/       # Contract definitions for cluster autoscaler integration
│   └── api_calls/                 # Helper utilities for framework API interactions
├── metrics/                       # Prometheus metric definitions & resource collector
├── profile/                       # Multi-profile manager (maps schedulerName to Framework instances)
└── util/
    └── assumecache/               # Assume cache data structure
```

---

## 3. Pod Scheduling Lifecycle

The lifecycle of an unscheduled Pod through `kube-scheduler` proceeds through the following phases:

```
[ Informer Event / Queue Ingress ]
             │
             ▼
[ SchedulingQueue (PriorityQueue) ]
   ├── activeQ (Heap ordered by QueueSortPlugin)
   ├── backoffQ (Heap ordered by backoff expiry)
   └── unschedulablePods (Map of blocked pods)
             │
      Dequeue Next Pod
             │
             ▼
[ Scheduling Cycle (Synchronous, single-threaded per pod) ]
   ├── 1. Framework Lookup: Identify profile matching `pod.spec.schedulerName`
   ├── 2. Cache Snapshot: `Cache.UpdateSnapshot(nodeInfoSnapshot)`
   ├── 3. PreFilter: Evaluate prerequisites, compute CycleState caches
   ├── 4. Filter: Parallel evaluation of nodes against Filter plugins
   ├── 5. PostFilter (if 0 nodes fit): Run Preemption (e.g. DefaultPreemption)
   ├── 6. PreScore: Prepare state for scoring plugins
   ├── 7. Score & Normalize: Parallel node scoring + weight aggregation
   ├── 8. Host Selection: Select highest-scoring host (reservoir sampling for ties)
   ├── 9. Reserve: Reserve resources in plugins and Cache.AssumePod()
   └── 10. Permit: Wait/approve/reject on delay plugins (e.g., gang scheduling)
             │
             ▼
[ Binding Cycle (Asynchronous, runs in background goroutine) ]
   ├── 1. WaitOnPermit: Block until Permit approval is granted
   ├── 2. PreBind: Run volume binding, dynamic resource claims, etc.
   ├── 3. Bind: Execute Binder plugin (API server `/binding` subresource call)
   ├── 4. PostBind: Metric reporting, cleanups
   └── On Failure: Cache.ForgetPod() + unreserve in plugins + requeue
```

---

## 4. Scheduling Framework & Extension Points

The scheduling framework (`pkg/scheduler/framework/`) provides an extension point model where plugins intercept different phases:

| Extension Point | Plugin Interface | Purpose & Semantics |
|---|---|---|
| **QueueSort** | `QueueSortPlugin` | Orders pods in `activeQ`. Only one QueueSort plugin can be active per configuration. |
| **PreFilter** | `PreFilterPlugin` | Validates pod requirements before filtering; builds cycle-level cache stored in `CycleState`. |
| **Filter** | `FilterPlugin` | Feasibility check for a given node. Executed in parallel across nodes via `parallelize.Until`. |
| **PostFilter** | `PostFilterPlugin` | Invoked when no node passed Filter. Used primarily for preemption of lower-priority pods. |
| **PreScore** | `PreScorePlugin` | Performs pre-computation needed before scoring nodes. |
| **Score** | `ScorePlugin` | Assigns an integer score (`0` to `100`) to feasible nodes. Supports `ScoreExtensions` (NormalizeScore). |
| **Reserve** | `ReservePlugin` | In-memory allocation of resources/state before binding. Must provide `Unreserve` on failure. |
| **Permit** | `PermitPlugin` | Delays or holds pod binding (Wait/Allow/Reject). Used for coscheduling/gang coordination. |
| **PreBind** | `PreBindPlugin` | Performs stateful external operations (e.g., attaching dynamic resources/volumes) prior to binding. |
| **Bind** | `BindPlugin` | Writes the pod-node assignment to the API server. Exactly one plugin handles binding. |
| **PostBind** | `PostBindPlugin` | Informational notification that binding succeeded; used for metrics and cleanup. |

### CycleState
`CycleState` is an ephemeral, thread-safe key-value store created for each scheduling cycle. Plugins use it to pass data between extension points (e.g. `PreFilter` caches pod requirements, and `Filter` reads them across parallel goroutines).

---

## 5. Backend Subsystem Details

### 5.1 Scheduling Queue (`pkg/scheduler/backend/queue/`)
*For detailed queuing architecture, lock discipline, move request triggers, and testing strategies, see [queue/AGENTS.md](backend/queue/AGENTS.md).*

- **`activeQ`**: Heap storing pods ready to be scheduled, sorted by `QueueSortPlugin.Less`.
- **`backoffQ`**: Heap storing pods that failed scheduling, waiting for their backoff duration (`DefaultPodInitialBackoffDuration` to `DefaultPodMaxBackoffDuration`).
- **`unschedulablePods`**: Map of pods waiting for cluster events (e.g., node added, pod finished, PVC created).
- **Scheduling Gates**: Pods with non-empty `.spec.schedulingGates` are held in `unschedulablePods` until their gates are cleared.
- **Event-Driven Requeueing**: Informer events trigger `SchedulingQueue.MoveAllToActiveOrBackoffQueue` or targeted moves based on event types (`ClusterEvent`).

### 5.2 Cache Subsystem (`pkg/scheduler/backend/cache/`)
- **`schedulerCache`**: In-memory mirror of cluster state. Tracks nodes, node resource usage, assumed pods, and confirmed pods.
- **`AssumePod`**: Speculatively adds a pod to node resource allocations before the API server binding completes.
- **`ForgetPod`**: Reverts an assumed pod on binding failure or timeout.
- **`Snapshot`**: `UpdateSnapshot()` creates an isolated copy of cluster nodes and pod states for a single scheduling cycle, ensuring thread-safe reads without holding cache locks during filtering/scoring.

---

## 6. Multi-Profile Scheduling (`pkg/scheduler/profile/`)

`kube-scheduler` supports multiple scheduling profiles within a single process.
- Each profile specifies a `schedulerName` and custom plugin configurations/weights.
- When dequeuing a pod, the scheduler selects the matching `Framework` instance via `sched.Profiles[pod.Spec.SchedulerName]`.
- If no matching profile exists, the pod is rejected and marked done.

---

## 7. Pod Group & Gang Scheduling (`pkg/scheduler/schedule_one_podgroup.go`)

In addition to individual pod scheduling, the scheduler supports atomic scheduling of Pod Groups:
- Evaluates composite pod groups together to ensure all members find feasible placements.
- Uses `revertFns` to roll back speculative assumptions if any group member fails to schedule.
- Integrates with `gangscheduling` and `podgrouppodscount` plugins.

---

## 8. Development, Testing, and Verification

When developing in `pkg/scheduler`:

### Unit Testing
```bash
# Run unit tests across all scheduler packages
make test WHAT=./pkg/scheduler/... GOFLAGS="-v -race"

# Run tests for a specific plugin (e.g. nodeaffinity)
make test WHAT=./pkg/scheduler/framework/plugins/nodeaffinity GOFLAGS="-v"
```

### Integration Testing
```bash
# Run scheduler integration test suite
make test-integration WHAT=./test/integration/scheduler
```

### Verification & Generators
```bash
# Run all repo verifications
make verify

# Update generated configs/deepcopy/schemes if APIs change
make update
```

---

## 9. Critical Invariants for Agents

1. **Thread Safety in Plugins**:
   - `Filter` and `Score` extension points are executed concurrently across nodes using worker goroutines. Plugins MUST NOT mutate shared internal state without synchronisation. Read-only access to `CycleState` should be used.
2. **CycleState Isolation**:
   - Data stored in `CycleState` exists only for the duration of a single scheduling cycle. Keys must be unique to the plugin.
3. **Reserve / Unreserve Symmetry**:
   - Any plugin implementing `Reserve` MUST implement an idempotent `Unreserve` method to clean up state when subsequent plugins, permits, or binding fail.
4. **Assume / Forget Symmetry**:
   - Whenever `Cache.AssumePod()` is called, any failure path prior to successful binding MUST call `Cache.ForgetPod()`.
5. **No Direct Node Mutation**:
   - Plugins must read node information exclusively from `framework.NodeInfo` snapshots, never mutating `NodeInfo` structs directly during scheduling cycles.
6. **Logging Conventions**:
   - Use contextual logging with `klog/v2` (`logger.V(k).Info(...)` or `logger.Error(...)`). Pass `ctx` and extract `logger := klog.FromContext(ctx)`.
