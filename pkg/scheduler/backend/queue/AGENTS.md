# Agent Guide: Scheduler Backend Queue Subsystem (`pkg/scheduler/backend/queue`)

This guide provides AI agents and human contributors with an in-depth architectural breakdown, state machine specification, concurrency model, requeueing mechanics, and testing strategies for the Kubernetes scheduler queue subsystem located under `pkg/scheduler/backend/queue`.

---

## 1. High-Level Overview & Subsystem Role

The `pkg/scheduler/backend/queue` package implements the core scheduling queue (`SchedulingQueue`) for `kube-scheduler`. It sits between incoming Kubernetes API/informer events and the scheduler's main scheduling loop (`scheduleOne` / `scheduleOnePodGroup`), serving as the central coordination and rate-limiting engine.

```
                  ┌──────────────────────────────────────────────┐
                  │ Informer Events (Nodes, Pods, PVCs, Claims) │
                  └──────────────────────┬───────────────────────┘
                                         │ EventHandlers
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  SchedulingQueue.PriorityQueue       │
                      │                                      │
                      │  ┌──────────────┐  ┌──────────────┐  │
                      │  │   activeQ    │  │   backoffQ   │  │
                      │  └──────┬───────┘  └──────▲───────┘  │
                      │         │ Pop()           │          │
                      │         ▼                 │          │
                      │   [In-Flight]             │          │
                      │         │                 │          │
                      │  ┌──────▼─────────────────┴───────┐  │
                      │  │      unschedulableEntities     │  │
                      │  │  (Unschedulable / Gated Map)   │  │
                      │  └────────────────────────────────┘  │
                      └──────────────────┬───────────────────┘
                                         │ Pop()
                                         ▼
                      ┌──────────────────────────────────────┐
                      │ Single-Pod / Gang Scheduling Cycle   │
                      │ (Filter -> Score -> Reserve -> Bind) │
                      └──────────────────────────────────────┘
```

### Core Responsibilities:
1. **Prioritization & Ordering**: Maintains pending workloads in heap-ordered queues based on configurable `QueueSortPlugin` implementations (e.g., priority, creation timestamp).
2. **Backoff & Rate-Limiting**: Dynamically backs off repeatedly failing pods with exponential backoff windows to prevent CPU starvation and apiserver hammering.
3. **Event-Driven Targeted Requeueing**: Uses fine-grained `QueueingHint` and `PreQueueingHint` functions to selectively wake up blocked pods only when relevant cluster events occur (e.g., node added, resource claim created, pod terminated).
4. **Scheduling Gates & PreEnqueue Control**: Holds pods with declared `.spec.schedulingGates` or blocked by `PreEnqueuePlugin` checks until prerequisite conditions are satisfied.
5. **Workload & Gang Hierarchy (PodGroups)**: Manages atomic group scheduling (`GenericPodGroup`, `CompositePodGroup`) through the `workloadForest` and member tracking caches.
6. **In-Flight Event Recording**: Tracks cluster mutations that occur while a pod is actively in its scheduling or binding cycle, ensuring events are not missed if the cycle ultimately fails.
7. **Preemption Nomination Registry**: Implements `fwk.PodNominator` to track pods nominated to execute on specific nodes.

---

## 2. Core Abstractions & Interfaces

The subsystem defines clean internal and external interfaces to decouple queue operations, locking, and data structures:

```
                                  ┌────────────────────────┐
                                  │    SchedulingQueue     │
                                  └───────────┬────────────┘
                                              │ implements
                                  ┌───────────▼────────────┐
                     ┌───────────┤     PriorityQueue      ├───────────┐
                     │            └───────────┬────────────┘           │
                     │                        │                        │
       ┌─────────────▼────────────┐           │          ┌─────────────▼────────────┐
       │       activeQueuer       │           │          │      backoffQueuer       │
       └─────────────┬────────────┘           │          └─────────────┬────────────┘
                     │ implements             │                        │ implements
       ┌─────────────▼────────────┐           │          ┌─────────────▼────────────┐
       │       activeQueue        │           │          │       backoffQueue       │
       └──────────────────────────┘           │          └──────────────────────────┘
                                              ▼
                                ┌───────────────────────────┐
                                │        PodNominator       │
                                └─────────────┬─────────────┘
                                              │ implements
                                ┌─────────────▼─────────────┐
                                │         nominator         │
                                └───────────────────────────┘
```

### 2.1 `SchedulingQueue` Interface (`scheduling_queue.go`)
The primary public interface consumed by `pkg/scheduler/scheduler.go`:
- `Add(ctx, pod)`: Ingests newly created unscheduled pods.
- `Activate(logger, pods)`: Explicitly forces pods back into `activeQ` (or registers wildcard events if in-flight).
- `AddUnschedulablePodIfNotPresent(logger, pInfo, cycle)`: Requeues a pod that failed scheduling or binding.
- `AddAttemptedPodGroupIfNeeded(logger, pgInfo, cycle, status)`: Requeues an attempted pod group after gang scheduling evaluation.
- `Pop(logger)`: Blocks until the highest-priority entity is available in `activeQ` (or popped from `backoffQ` when enabled), then returns it and increments the scheduling cycle.
- `Done(podUID)`: Notifies the queue that processing of a popped pod has finished (successful binding or termination), pruning associated in-flight tracking.
- `Update(ctx, oldPod, newPod)`: Updates pod attributes across all internal sub-queues.
- `Delete(logger, pod)`: Removes a pod from all internal sub-queues, pod group trackers, and nominator tables.
- `MoveAllToActiveOrBackoffQueue(logger, event, oldObj, newObj, preCheck)`: Evaluates unschedulable entities against incoming cluster events via QueueingHints.
- `AddGenericPodGroup` / `UpdateGenericPodGroup` / `DeleteGenericPodGroup`: Maintains the pod group workload hierarchy.
- `NominatedPodsForNode(logger, nodeName)`: Returns pods nominated for preemption on the target node.

### 2.2 `activeQueuer` & Unlocked Adapters (`active_queue.go`)
Encapsulates `activeQ` operations under `activeQueue.lock`:
- `underLock(func(unlockedActiveQueuer))` / `underRLock(func(unlockedActiveQueueReader))`: Enables compound operations without lock re-acquisition.
- `pop(logger)`: Thread-safe pop with condition variable wait on `cond`.
- `add(logger, entity, event, strategy)`: Inserts into the heap and records incoming entity metrics.

### 2.3 `backoffQueuer` & `backoffQPopper` (`backoff_queue.go`)
Abstracts backoff management across two distinct heaps:
- `isEntityBackingoff(entity)`: Evaluates if an entity's backoff timestamp is still in the future.
- `popAllBackoffCompleted(logger)`: Flushes entities whose backoff timers have elapsed.
- `waitUntilAlignedWithOrderingWindow(fn, stopCh)`: Synchronizes periodic flushes to whole 1-second ordering windows.

---

## 3. Data Structures & Sub-Queues

`PriorityQueue` orchestrates three primary sub-queues and several specialized workload/nomination trackers:

```
PriorityQueue
├── activeQ (activeQueue)
│   ├── queue: *heap.Heap[QueuedEntityInfo] (ordered by QueueSortPlugin)
│   ├── inFlightPods: map[UID]*list.Element
│   ├── inFlightEvents: *list.List (interleaved Pods and clusterEvents)
│   ├── cond: sync.Cond (signaled on entity enqueue)
│   └── schedCycle: int64 (incremented on pop)
│
├── backoffQ (backoffQueue)
│   ├── entityBackoffQ: *heap.Heap[QueuedEntityInfo] (plugin rejection backoff)
│   └── entityErrorBackoffQ: *heap.Heap[QueuedEntityInfo] (infrastructure error backoff)
│
├── unschedulableEntities (*unschedulableEntities)
│   ├── entityInfoMap: map[string]QueuedEntityInfo
│   ├── unschedulableRecorder: metrics.MetricRecorder
│   └── gatedRecorder: metrics.MetricRecorder
│
├── Pod Group Subsystem
│   ├── workloadForest: forest of GenericPodGroup & CompositePodGroup hierarchy
│   ├── incompletePodGroupPods: map[pgKey]map[pKey]*QueuedPodInfo (awaiting CRD/API object)
│   └── pendingPodGroupPods: map[pgKey]map[pKey]*QueuedPodInfo (awaiting in-flight root requeue)
│
└── nominator (*nominator)
    ├── nominatedPods: map[nodeName][]podRef
    └── nominatedPodToNode: map[UID]nodeName
```

### 3.1 `activeQ`
- **Ordering**: Min-heap ordered by `QueueSortPlugin.Less` (e.g., `PrioritySort` orders by priority descending, then timestamp ascending).
- **In-Flight Tracking**: Holds pointers to pods currently dequeued by `Pop()` and undergoing scheduling or asynchronous binding.
- **`Pop()` Behavior**: Blocks on `cond.Wait()`. When `SchedulerPopFromBackoffQ` is enabled, `Pop()` can pop directly from `entityBackoffQ` if `activeQ` is empty and backoff is satisfied.

### 3.2 `backoffQ` (Dual-Heap Architecture)
- **`entityBackoffQ`**: Holds entities that failed scheduling due to plugin filtering/rejections (e.g., `NodeResourcesFit`, `PodTopologySpread`).
- **`entityErrorBackoffQ`**: Holds entities that failed due to infrastructure or unexpected errors (e.g., API server 500, storage RPC failure). These entities have empty `UnschedulablePlugins` and empty `PendingPlugins`.
- **Ordering**: Ordered by backoff expiration time (`GetBackoffExpiration()`). When `SchedulerPopFromBackoffQ` is enabled, ties within the same 1-second window are ordered by `QueueSortPlugin.Less`.

### 3.3 `unschedulableEntities`
- Map keyed by entity key (`namespace/podName` or `pg-type/namespace/groupName`).
- Holds entities that were tried and failed, waiting for cluster events or timeout flushes.
- Tracks scheduling gated entities separately via `gatedRecorder` to distinguish `.spec.schedulingGates` from filter rejections in Prometheus metrics.

### 3.4 Workload Forest & Member Pods
- **`workloadForest`**: Stores `GenericPodGroup` instances and maintains bidirectional parent-child links for hierarchical `CompositePodGroup` trees.
- **`incompletePodGroupPods`**: Holds pods whose group API objects have not yet been observed by informers. Once the `PodGroup` is added via `AddGenericPodGroup`, pods are moved into the group and queued.
- **`pendingPodGroupPods`**: Holds pods that arrive while their parent `PodGroup` is already in-flight. When the group returns via `AddAttemptedPodGroupIfNeeded` or `AddUnschedulablePodIfNotPresent`, pending pods are merged into the group.

### 3.5 `nominator`
- Tracks preemption nominations made during `PostFilter` (e.g., `DefaultPreemption`).
- When a higher-priority pod nominates a node, lower-priority victim pods know they are scheduled for eviction.
- Cleared when the nominated pod is bound, deleted, or its nomination is overwritten.

---

## 4. Pod & Entity Lifecycle State Machine

```
                              [ New Pod Created ]
                                       │
                                       ▼
                             runPreEnqueuePlugins
                                ╱             ╲
                     [All Pass]╱               ╲[Gated / Blocked]
                              ▼                 ▼
                         ┌─────────┐   ┌───────────────────────┐
                         │ activeQ │   │ unschedulableEntities │
                         └────┬────┘   │      (Gated=true)     │
                              │        └───────────▲───────────┘
                        Pop() │                    │
                              ▼                    │ Ungated via
                        [ In-Flight ]              │ Informer Event
                              │                    │
                 ┌────────────┴────────────┐       │
                 │ Scheduling Attempt      │       │
                 └────────────┬────────────┘       │
                              │                    │
              ┌───────────────┴───────────────┐    │
       [ Fits Node ]                   [ Fails / Error ]
              │                               │
              ▼                               ▼
       Reserve & Bind           AddUnschedulablePodIfNotPresent
              │                               │
     ┌────────┴────────┐             ┌────────┴────────┐
[Success]          [Fail]            │ QueueingHint on │
     │                 │             │ In-Flight Events│
     ▼                 ▼             └────────┬────────┘
  Done()            ForgetPod                 │
(UID removed        & Requeue                 │
from InFlight)         │                      │
                       │           ┌──────────┼──────────┐
                       │       [No Match] [QueueAfter] [QueueImmediate]
                       │           │          │          │
                       │           ▼          ▼          ▼
                       └────► ┌───────────────────────┐  │
                              │ unschedulableEntities │  │
                              └───────────┬───────────┘  │
                                          │ ClusterEvent │
                                          │ Matches Hint │
                                          ▼              │
                                    ┌───────────┐        │
                       ┌───────────►│ backoffQ  │        │
                       │ (Backing   └─────┬─────┘        │
                       │   off)           │ Backoff      │
                       │                  │ Expired      │
                       │                  ▼              │
                       └─────────── ┌───────────┐◄───────┘
                                    │  activeQ  │
                                    └───────────┘
```

### State Transition Summary:

| Current Queue / State | Trigger / Event | Action / Transition | Condition / Logic |
|---|---|---|---|
| **Ingress** | `Add(pod)` | `moveToActiveQ` | Runs `PreEnqueue`. If gated -> `unschedulableEntities`; if passes -> `activeQ`. |
| **Ingress (PodGroup)** | `Add(pod)` | `addPodGroupMember` | If group missing -> `incompletePodGroupPods`; if group in-flight -> `pendingPodGroupPods`; else adds to group in `activeQ`/`backoffQ`/`unschedulable`. |
| **`activeQ`** | `Pop()` | `inFlightPods` | Increments `schedCycle`, pushes pod into `inFlightEvents`, decrements unschedulable metrics. |
| **In-Flight** | Bind Success | `Done(podUID)` | Removes UID from `inFlightPods`, prunes dead events from `inFlightEvents`. |
| **In-Flight** | Scheduling / Binding Failure | `AddUnschedulablePodIfNotPresent` | Checks `inFlightEvents` against hints: `queueImmediately` -> `activeQ`; `queueAfterBackoff` -> `activeQ` (if backoff expired) or `backoffQ`; `queueSkip` -> `unschedulableEntities`. |
| **`unschedulableEntities`** | `ClusterEvent` (Informer) | `moveEntitiesToActiveOrBackoffQueue` | Evaluates `QueueingHintFn`. If `Queue` -> moves to `activeQ` or `backoffQ`. If `QueueSkip` -> stays unschedulable. |
| **`unschedulableEntities`** | Flush Leftover (30s) | `moveEntitiesToActiveOrBackoffQueue` | Moves pods staying longer than `podMaxInUnschedulablePodsDuration` to `backoffQ`/`activeQ` (`EventUnschedulableTimeout`). |
| **`backoffQ`** | Backoff Expiration (1s Ticker) | `flushBackoffQCompleted` | Moves pods whose backoff time elapsed to `activeQ` (`BackoffComplete`). |
| **Any Queue** | `Activate(pods)` | `moveToActiveQ` | Explicit activation by plugins (e.g. DRA claim ready, permit unblocked). |

---

## 5. Scheduling Gates & PreEnqueue Plugin Subsystem

### 5.1 Scheduling Gates
Pods may declare scheduling gates via `.spec.schedulingGates`:
```yaml
spec:
  schedulingGates:
  - name: example.com/gate-name
```
- **Ingress Behavior**: When added, `entity.Gated()` returns true. The pod is placed into `unschedulableEntities` with metric `pending_pods{queue="gated"}` incremented.
- **Ungating**: When an external controller removes all gates (`Update` event), the pod transitions from gated to ungated via `updateMetricsOnStateChange` and is moved to `activeQ`.

### 5.2 PreEnqueue Plugins (`runPreEnqueuePlugins`)
Plugins implementing `PreEnqueuePlugin` (e.g., dynamic resource allocation, coscheduling) can gate a pod before it ever reaches `activeQ`:
```go
type PreEnqueuePlugin interface {
    Plugin
    PreEnqueue(ctx context.Context, pod *v1.Pod) *Status
}
```
1. **Execution Order**: If `pInfo.GatingPlugin` is set, that plugin runs first. If it passes, remaining registered `PreEnqueue` plugins run.
2. **Failure Handling**: If a plugin returns non-success, `pInfo.GatingPlugin` is recorded along with its registered `pInfo.GatingPluginEvents`, and the entity is stored in `unschedulableEntities`.
3. **Event-Filtered Wakeup**: Gated entities ignore cluster events that do NOT match `pInfo.GatingPluginEvents`, avoiding unnecessary re-evaluations:
   ```go
   if entity.Gated() && !framework.ClusterEventIsWildCard(event) && !framework.MatchAnyClusterEvent(event, entity.GetGatingPluginEvents()) {
       continue // Skip evaluation
   }
   ```

---

## 6. Event-Driven Requeueing & Queueing Hints Engine

Queueing hints eliminate wasteful scheduling cycles by determining whether a specific cluster mutation can actually fix the rejection reasons of an unschedulable pod.

```
Informer ClusterEvent (e.g., NodeAdded, PodDeleted)
                        │
                        ▼
           isEventOfInterest(event)
             ├── False ──► Early Return (No plugins care)
             └── True
                  │
                  ▼
      runPreQueueingHintPlugins
        (Narrows candidate pods by target keys)
                  │
                  ▼
      For each unschedulable candidate:
        isEntityWorthRequeuing
                  │
                  ▼
        For each rejector plugin in UnschedulablePlugins / PendingPlugins:
          Call QueueingHintFn(logger, pod, oldObj, newObj)
                  │
                  ├── QueueSkip ────────► pod stays in unschedulableEntities
                  ├── Queue (Pending) ──► queueImmediately (bypasses backoff -> activeQ)
                  └── Queue (Unsched) ──► queueAfterBackoff (activeQ if expired, else backoffQ)
```

### 6.1 `ClusterEvent` Model
A `ClusterEvent` describes a cluster mutation:
```go
type ClusterEvent struct {
    Resource   EventResource // Pod, Node, PersistentVolume, ResourceClaim, etc.
    ActionType ActionType    // Add, Update, Delete, WildCard
    Label      string
}
```

### 6.2 `QueueingHintFn` Semantics
Registered per plugin and event:
```go
type QueueingHintFn func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (QueueingHint, error)
```
- **`Queue`**: The event might resolve the pod's unschedulable status. Pod is queued.
- **`QueueSkip`**: The event is irrelevant to this pod (e.g., node added has insufficient CPU for a pod failing `NodeResourcesFit`).
- **Error Fallback**: If `QueueingHintFn` errors, the queue treats it as `Queue` to prevent pods from becoming permanently stuck.

### 6.3 `PreQueueingHint` Narrowing Optimization
When `SchedulerPreQueueingHints` feature is enabled:
- `PreQueueingHintFn` runs once per cluster event to compute the exact set of target `NamespacedName` keys affected by the event.
- `collectEntitiesToEvaluate` directly looks up candidate entities in `unschedulableEntities` by key, avoiding an O(N) linear scan over all unschedulable pods.

### 6.4 `inFlightEvents` Invariant
Cluster events occurring while a pod is being evaluated in `scheduleOne` are appended to `activeQ.inFlightEvents`. When the cycle fails and `AddUnschedulablePodIfNotPresent` is called:
- `determineSchedulingHintForInFlightPod` replays all events logged in `inFlightEvents` after the pod's pop position.
- If any event returns `Queue`, the pod is immediately requeued rather than lost in `unschedulableEntities`.

---

## 7. Backoff Algorithms & Ordering Windows

### 7.1 Exponential Backoff Formula
For an entity with attempt count $N$ (where $N = \text{UnschedulableCount}$ or $\text{ConsecutiveErrorsCount}$):

$$\text{BackoffDuration}(N) = \min\left(\text{InitialBackoff} \times 2^{N-1},\, \text{MaxBackoff} \times \sqrt{\text{EntitySize}}\right)$$

- **Defaults**: `DefaultPodInitialBackoffDuration = 1s`, `DefaultPodMaxBackoffDuration = 10s`.
- **Entity Size Scaling**: For `PodGroup` with $M$ pods, maximum backoff is multiplied by $\sqrt{M}$ to prevent large gang workloads from thrashing the scheduler while still bounding maximum wait time.

### 7.2 1-Second Ordering Window (`SchedulerPopFromBackoffQ`)
Backoff timestamps are aligned to discrete 1-second windows (`backoffQOrderingWindowDuration = 1s`):
```go
func (bq *backoffQueue) alignToWindow(t time.Time) time.Time {
    return t.Truncate(backoffQOrderingWindowDuration)
}
```
- **Tie-Breaking**: Entities in the same window are sorted by `activeQLessFn` (`QueueSortPlugin.Less`).
- **Synchronized Flushing**: `waitUntilAlignedWithOrderingWindow` runs a ticker precisely aligned to window boundaries, flushing all completed entities in bulk.

---

## 8. In-Flight Tracking & Cycle State Synchronization

The in-flight tracking subsystem guarantees that concurrent informer events and scheduling cycle completions never race or drop events.

```
inFlightEvents (Doubly-Linked List in activeQueue)
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Pod A (UID) │ ──► │ ClusterEvent │ ──► │ ClusterEvent │ ──► │  Pod B (UID) │
│  (Head Pod)  │     │  (NodeAdd)   │     │  (PodDelete) │     │  (Tail Pod)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Pop Entry**: `activeQ.pop()` inserts `pInfo.Pod` at the tail of `inFlightEvents` and stores its `*list.Element` in `inFlightPods[pod.UID]`.
2. **Event Ingestion**: When `MoveAllToActiveOrBackoffQueue` runs, `addEventIfAnyInFlight` pushes `clusterEvent` to the tail of `inFlightEvents` if `len(inFlightPods) > 0`.
3. **Replay on Failure**: `clusterEventsForPod(pInfo)` iterates from `inFlightPod.Next()` to the end of `inFlightEvents`, evaluating only events that occurred *after* Pod A started its scheduling cycle.
4. **Pruning on `Done()`**: When Pod A completes:
   - Pod A is deleted from `inFlightPods`.
   - All leading `clusterEvent` elements before the new head pod are pruned from `inFlightEvents`.

---

## 9. Workload Forest & Hierarchical Pod Group Management

```
workloadForest
├── CompositePodGroup: "root-cpg" (Parent)
│   ├── PodGroup: "worker-group-1" (Child)
│   │   ├── Pod: "worker-1-0"
│   │   └── Pod: "worker-1-1"
│   └── PodGroup: "worker-group-2" (Child)
│       ├── Pod: "worker-2-0"
│       └── Pod: "worker-2-1"
```

1. **Workload Invariants**:
   - A member pod is placed in `incompletePodGroupPods` if its root group has not been observed in `workloadForest`.
   - If the root group is currently in-flight, new member pods are buffered in `pendingPodGroupPods`.
   - When the root group is in `activeQ`, `backoffQ`, or `unschedulableEntities`, adding a member pod updates the group in-place in that respective sub-queue.
2. **Cycle Detection**: `workloadForest.buildPodGroupInfo` and `getRootLookupInfoForParentCPG` track visited keys using `sets.Set[EntityKey]` to detect and break cyclic parent-child references.
3. **Requeueing Behavior**: On gang scheduling failure, `AddAttemptedPodGroupIfNeeded` requeues the entire group. If some pods scheduled while others failed, the group preserves its timestamp to allow preemption in the immediate next cycle.

---

## 10. Concurrency Model & Lock Discipline

### 10.1 Strict Lock Hierarchy
To prevent deadlocks across concurrent informer events, scheduling cycles, and background flush tickers, locks MUST always be acquired in the following strict order:

$$\mathbf{PriorityQueue.lock} \;\longrightarrow\; \mathbf{activeQueue.lock} \;\longrightarrow\; \mathbf{backoffQueue.lock} \;\longrightarrow\; \mathbf{nominator.nLock}$$

```
                          ┌───────────────────────┐
                          │   PriorityQueue.lock  │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │   activeQueue.lock    │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │   backoffQueue.lock   │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │    nominator.nLock    │
                          └───────────────────────┘
```

### 10.2 Lock Rules & Invariants:
1. **Never Invert Lock Order**: Never acquire `PriorityQueue.lock` while holding `activeQueue.lock`, `backoffQueue.lock`, or `nominator.nLock`.
2. **Non-Blocking `Pop()`**: `PriorityQueue.Pop()` MUST NOT hold `PriorityQueue.lock`. It acquires only `activeQueue.lock` and waits on `activeQueue.cond`. This prevents blocking queue modifications while the single-threaded scheduling loop waits for work.
3. **Unlocked Queuer Patterns**: Functions operating across multiple sub-queues use `activeQueue.underLock` or `activeQueue.underRLock` with `unlockedActiveQueuer` to run internal manipulations safely.
4. **Done Symmetry**: Every call to `Pop()` that returns a pod MUST be paired with a corresponding call to `Done(podUID)` or `AddUnschedulablePodIfNotPresent(pInfo)` on failure. `Done()` must be executed before requeueing to prevent the pod from colliding with its own in-flight entry.

---

## 11. Move Request Triggers & Informer Dispatch Matrix

The table below outlines cluster events registered in `eventhandlers.go` and their corresponding queue wake-up triggers:

| Resource Event | `ClusterEvent` Definition | Target Plugins & Subsystems | Typical Requeue Impact |
|---|---|---|---|
| **Node Added** | `ClusterEvent{Resource: Node, ActionType: Add}` | `NodeResourcesFit`, `NodeAffinity`, `NodeName`, `TaintToleration`, `VolumeBinding` | Moves pods failing node filters to `activeQ`/`backoffQ`. |
| **Node Updated** | `ClusterEvent{Resource: Node, ActionType: Update}` | `NodeResourcesFit` (allocatable changed), `NodeAffinity` (labels), `TaintToleration` (taints) | Selectively wakes pods if node capacity or labels/taints match. |
| **Node Deleted** | `ClusterEvent{Resource: Node, ActionType: Delete}` | `PodTopologySpread`, `InterPodAffinity` | Wakes pods whose anti-affinity/spread constraints were blocked. |
| **Pod Added / Scheduled** | `ClusterEvent{Resource: Pod, ActionType: Add/Update}` | `InterPodAffinity`, `PodTopologySpread` | Wakes pods waiting for matching affinity workloads. |
| **Pod Deleted (Finished)** | `ClusterEvent{Resource: Pod, ActionType: Delete}` | `NodeResourcesFit`, `PodTopologySpread`, `InterPodAffinity`, `Preemption` | Frees node capacity; triggers immediate requeue of blocked high-priority pods. |
| **PVC / PV Added/Updated** | `ClusterEvent{Resource: PVC/PV, ActionType: Add/Update}` | `VolumeBinding`, `VolumeZone`, `NodeUnschedulable` | Wakes pods blocked on unbound claims or storage capacity. |
| **ResourceClaim Updated** | `ClusterEvent{Resource: ResourceClaim, ActionType: Update}` | `DynamicResources` (`DRA`), `PreEnqueue` | Wakes pods whose dynamic device allocations became ready. |
| **PodGroup Added/Updated** | `ClusterEvent{Resource: PodGroup, ActionType: Add/Update}` | `Coscheduling`, `GangScheduling`, `WorkloadForest` | Flushes `incompletePodGroupPods` into `activeQ`. |
| **Wildcard Flush** | `ClusterEvent{Resource: WildCard, ActionType: All}` | All Plugins | Periodic flush fallback (`flushUnschedulableEntitiesLeftover`). |
| **Backoff Complete** | `framework.BackoffComplete` | Internal Queue Ticker | Moves entities from `backoffQ` to `activeQ`. |

---

## 12. Testing Strategies & Verification Patterns

The `pkg/scheduler/backend/queue` test suite (`*_test.go`) contains over 11,000 lines of comprehensive unit and concurrency tests.

### 12.1 Mocking Time & Deterministic Tickers
Never use real-world `time.Sleep` in queue unit tests. Inject `clock.NewFakeClock(now)` or `clock.WithTicker`:
```go
fakeClock := clock.NewFakeClock(time.Now())
pq := NewTestQueue(ctx, priorityLessFunc, WithClock(fakeClock))
// Advance fake clock deterministically
fakeClock.Step(2 * time.Second)
```

### 12.2 Standard Test Construction (`testing.go`)
Use `NewTestQueue` or `NewTestQueueWithObjects` to initialize a test queue with fake client informers and async metric recorders:
```go
pq := NewTestQueueWithObjects(ctx, priorityLessFunc, []runtime.Object{node, pod})
```

### 12.3 Key Verification Patterns:
1. **Queue Sort Invariants**: Add multiple pods with differing priorities, timestamps, and scheduling groups; verify `Pop()` sequence matches expected `LessFunc` ordering.
2. **Backoff Timing**: Verify pods added via `AddUnschedulablePodIfNotPresent` cannot be popped before backoff duration expires, and verify exponential growth across attempts.
3. **QueueingHint Simulation**: Mock plugin `QueueingHintFn` returning `Queue` or `QueueSkip`; trigger `MoveAllToActiveOrBackoffQueue` and verify only matching pods move to `activeQ`.
4. **In-Flight Mutation Race**: Simulate concurrent informer updates during an active scheduling cycle; verify events are captured in `inFlightEvents` and replayed on failure.
5. **Race Detector Validation**: Run tests with `-race` to ensure strict adherence to lock discipline:
   ```bash
   GOTOOLCHAIN=auto go test -race -v ./pkg/scheduler/backend/queue/...
   ```

---

## 13. Critical Invariants for Developers & AI Agents

1. **Lock Hierarchy Enforcement**: Always maintain `PriorityQueue.lock > activeQueue.lock > backoffQueue.lock > nominator.nLock`. Never acquire a higher lock while holding a lower lock.
2. **`Pop()` Non-Locking**: Never wrap `p.activeQ.pop()` in `PriorityQueue.lock`.
3. **`Done()` Symmetry**: Always call `Done(podUID)` when a popped pod completes binding or is discarded. When re-adding an unschedulable pod, ensure `p.Done()` is invoked before re-inserting into `activeQ`/`backoffQ` to prevent self-collision discards.
4. **PreEnqueue Idempotency**: `PreEnqueue` plugins must be side-effect-free and safe to re-execute on every queue move transition.
5. **Metric Consistency**: When moving entities between `activeQ`, `backoffQ`, and `unschedulableEntities`, always pass the previous `strategy` to prevent double-counting in `scheduler_incoming_entities_total` and `scheduler_pending_pods`.
6. **Contextual Logging**: Use contextual loggers (`logger.V(k).Info(...)`). Include entity types and names via `klog.KObj(entity)`.
