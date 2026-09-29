# Scheduler Cache (`pkg/scheduler/backend/cache`)

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, concurrency model, snapshotting mechanics, and debugging guide for the Kubernetes Scheduler cache subsystem located under `pkg/scheduler/backend/cache`.

---

## 1. Overview & Core Responsibilities

The scheduler cache (`pkg/scheduler/backend/cache`) is the central in-memory state store for `kube-scheduler`. It maintains an aggregated, node-centric view of cluster resources, running/assumed pods, image distributions, volume claim references, and workload/pod-group hierarchies.

### Core Objectives:
1. **Node-Level Resource Aggregation**: Maintain per-node summaries (`framework.NodeInfo`) of requested/allocatable compute resources (CPU, Memory, Ephemeral Storage, Extended Resources), running pods, affinity specifications, and PVC consumption.
2. **Optimistic Pod Placement (Assume/Forget)**: Support instantaneous, lock-isolated reservations (`AssumePod`) to prevent race conditions during parallel scheduling pipelines while asynchronous API server bindings (`BindPod` / Binder plugins) are in-flight.
3. **Decoupling from Informer Delays**: Bridge the gap between fast scheduling decisions and potentially slow, out-of-order, or batched API server Informer watch events (`AddPod`, `UpdatePod`, `RemovePod`).
4. **Snapshot Isolation & Generation Tracking**: Supply immutable, point-in-time snapshots (`Snapshot`) for each scheduling cycle with incremental, generation-based cloning to eliminate lock contention during parallel node filtering and scoring.
5. **Topology-Aware Node Ordering**: Balance node iteration across failure domains (regions/zones) via `nodeTree` round-robin traversal.
6. **Workload Hierarchy & Gang Scheduling**: Track multi-pod gang states (`GenericWorkload` / `CompositePodGroup`) across scheduling cycles.
7. **Cache Integrity Verification & Debugging**: Provide runtime inspection (`debugger`) to detect cache drift against client-go informer listers via signal handlers.

---

## 2. Directory & Subpackage Architecture

```
pkg/scheduler/backend/cache/
├── interface.go              # Cache and Dump interface contracts
├── cache.go                  # cacheImpl implementation & thread-safe operations
├── snapshot.go               # Snapshot struct, SharedLister implementation, mutation session & index tracking
├── node_tree.go              # Zone-aware tree maintaining round-robin node ordering
├── podgroupstate.go          # Runtime state tracking for PodGroups and CompositePodGroups
├── cache_test.go             # Unit tests for cache operations, assume/forget, and concurrency
├── snapshot_test.go          # Tests for snapshot updates, mutations, and isolation
├── node_tree_test.go         # Tests for zone-aware nodeTree mechanics
├── podgroupstate_test.go     # Tests for pod group state transitions
├── debugger/                 # Cache debugging and comparison utilities
│   ├── debugger.go           # CacheDebugger entrypoint & OS signal listener
│   ├── comparer.go           # CacheComparer (reconciles Cache against Informer listers)
│   ├── dumper.go             # CacheDumper (formats NodeInfo and queue state to logs)
│   ├── signal.go             # POSIX signal binding (SIGUSR2)
│   ├── signal_windows.go     # Windows signal binding (SIGINT)
│   └── comparer_test.go      # Tests for cache comparison logic
└── fake/                     # Test mocks
    └── fake_cache.go         # Fake Cache implementation with injectable handler hooks
```

---

## 3. Core Data Structures & State Management

```
                       ┌──────────────────────────────────────────────┐
                       │                  cacheImpl                   │
                       │  - mu sync.RWMutex                           │
                       │  - assumedPods: sets.Set[string]             │
                       │  - podStates: map[string]*podState           │
                       │  - imageStates: map[string]*ImageStateSummary│
                       │  - pvcRefCountsDelta: map[string]int         │
                       └───────┬──────────────────────────────┬───────┘
                               │                              │
             Doubly Linked List│                              │Zone-Aware
             Ordered by Recency│                              │Round-Robin
                               ▼                              ▼
             ┌───────────────────────────────────┐    ┌───────────────┐
headNode ───►│ nodeInfoListItem (Generation: 10) │    │   nodeTree    │
             ├───────────────────────────────────┤    ├───────────────┤
             │ nodeInfoListItem (Generation: 8)  │    │ zone-1: [n1]  │
             ├───────────────────────────────────┤    │ zone-2: [n2]  │
             │ nodeInfoListItem (Generation: 4)  │    └───────────────┘
             └───────────────────────────────────┘
```

### 3.1 `cacheImpl` (`cache.go`)
- **`mu sync.RWMutex`**: Guards all fields within `cacheImpl`. Read locks (`RLock`) are held for inspection methods (`Dump`, `GetPod`, `IsAssumedPod`, `NodeCount`), while write locks (`Lock`) guard all cache mutations (`AddPod`, `AssumePod`, `ForgetPod`, `UpdateSnapshot`, etc.).
- **`nodes map[string]*nodeInfoListItem`**: Map from node name to doubly linked list nodes wrapping `*framework.NodeInfo`.
- **`headNode *nodeInfoListItem`**: Head of the doubly linked list pointing to the most recently updated `NodeInfo`. Whenever a node's pods or properties change, its item is moved to `headNode` and its `Generation` counter is bumped.
- **`assumedPods sets.Set[string]`**: Set of pod keys (`<namespace>/<name>`) that have been optimistically assumed on a node but not yet confirmed via an Informer `AddPod` event.
- **`podStates map[string]*podState`**: Lookup map from pod key to cached `*v1.Pod` object.
- **`nodeTree *nodeTree`**: Zone/region hierarchy organizing node names to produce balanced round-robin iteration slices.
- **`imageStates map[string]*fwk.ImageStateSummary`**: Cluster-wide cache of container image sizes and the set of nodes hosting them (used by `ImageLocality` plugin).
- **`pvcRefCountsDelta map[string]int`**: Delta map accumulating PVC reference count changes between snapshot runs, avoiding costly full-cluster rescans.

### 3.2 `nodeTree` (`node_tree.go`)
- Maintains a mapping from zone identifier (`utilnode.GetZoneKey(node)`) to a slice of node names: `tree map[string][]string`.
- **`list()`**: Iterates through zones in round-robin fashion (taking index 0 from zone A, index 0 from zone B, index 1 from zone A, etc.) to construct `nodeInfoList`.
- **Zone Exhaustion**: Ensures nodes are evenly interleaved across failure domains, enabling `PercentageOfNodesToScore` to evaluate a representative cross-section of the cluster before early termination.

### 3.3 Ghost Nodes & Node Deletion
When `RemoveNode` is invoked:
1. The node is removed from `nodeTree` immediately (`cache.nodeTree.removeNode(node)`), meaning it will no longer be included in new snapshots.
2. The `*v1.Node` pointer in `NodeInfo` is cleared (`n.info.RemoveNode()`).
3. If there are still pods assigned to the node whose deletion events have not yet arrived, `NodeInfo` is retained as a **ghost node** in `cache.nodes`.
4. Once all remaining pods on the ghost node are deleted (`len(n.info.Pods) == 0`), the `nodeInfoListItem` is unlinked and pruned from `cache.nodes`.

---

## 4. Pod Lifecycle & State Machine

The scheduler cache is pod-centric. Pod events are driven by both scheduler decisions and asynchronous Informer watch streams.

```
                  +-------------------------------------------+  +----+
                  |                            Add            |  |    |
                  |                                           |  |    | Update
                  +      Assume                Add            v  v    |
    Initial +--------> Assumed +-----------> Added <--+
                  ^                    +                        +
                  |                    |                        |
                  |                    |                        | Remove
                  |                    |                        |
                  |                    |                        v
                  +--------------------+                     Deleted
                        Forget
```

### 4.1 Pod State Transitions

| Transition | Invocation Trigger | Cache Actions | Invariants & Error Handling |
|---|---|---|---|
| **AssumePod** | Scheduler selects node & Reserve phase passes (`scheduleOne`) | - Adds pod to `NodeInfo.AddPod(pod)`<br>- Moves node item to `headNode`<br>- Adds key to `assumedPods` & `podStates`<br>- Accumulates PVC delta | Fails if pod is already in `podStates`. Node is created on demand if missing. |
| **ForgetPod** | Binding fails, Permit plugin rejects, or Reserve plugin aborts | - Removes pod from `NodeInfo.RemovePod(pod)`<br>- Removes key from `assumedPods` & `podStates`<br>- Reverts PVC delta<br>- Cleans ghost node if empty | Fails if pod is not in `assumedPods` or assigned to a different node. Reverts pod group member to unscheduled. |
| **RemoveAssumedPod** | Assumed pod is deleted from cluster before confirmation | - Same as `ForgetPod`, but permanently removes pod from pod group state rather than reverting to unscheduled | Used during cluster pod deletion events. |
| **AddPod** | Informer detects scheduled pod (`pod.spec.nodeName != ""`) | - If assumed: confirms pod by removing from `assumedPods`, updates `podState`<br>- If not assumed: invokes `addPod` | Logs mismatch if added node differs from assumed node. Errors if pod was already in added state. |
| **UpdatePod** | Informer detects pod update event | - Calls `removePod(oldPod)` followed by `addPod(newPod)` | Assumed pods **must not** be updated before `AddPod`. If `nodeName` changes, triggers `klog.FlushAndExit(1)` (cache corruption). |
| **RemovePod** | Informer detects pod deletion | - Calls `removePod(pod)`<br>- Cleans up node list item if node was deleted (ghost node) | Errors if pod is not present in cache. If `nodeName` changed on deletion, triggers `klog.FlushAndExit(1)`. |

---

## 5. Snapshotting Mechanics & Generation Tracking

To achieve high scheduling throughput, the scheduler decouples read-heavy scheduling algorithms from cache write operations.

```
       cacheImpl (Live Cache)                             Snapshot (Cycle-Isolated)
┌─────────────────────────────────────┐            ┌──────────────────────────────────────┐
│ headNode (Gen: 15)                  │            │ generation: 15                       │
│   ├── NodeA (Gen 15) > SnapshotGen  ├─ Clone ───►│ nodeInfoMap["NodeA"] = clonedNodeA  │
│   ├── NodeB (Gen 12) > SnapshotGen  ├─ Clone ───►│ nodeInfoMap["NodeB"] = clonedNodeB  │
│   └── NodeC (Gen 8) <= SnapshotGen  │ (Skip)     │ nodeInfoMap["NodeC"] (Preserved ptr) │
│ nodeTree (Topology order)           ├─ Order ───►│ nodeInfoList = [NodeA, NodeB, ...]   │
└─────────────────────────────────────┘            └──────────────────────────────────────┘
```

### 5.1 `UpdateSnapshot(nodeSnapshot)` Lifecycle
At the start of every scheduling cycle (`scheduleOne` / `scheduleOnePodGroup`), the scheduler synchronizes its private `Snapshot` instance:
1. **Clear Assumed State**: Clears any leftover assumed pods or placements from previous cycles via `forgetAllAssumedPods()`.
2. **Generation-Based Incremental Sync**:
   - Reads `snapshotGeneration = nodeSnapshot.generation`.
   - Traverses the live cache's doubly linked list starting at `cache.headNode`.
   - Halts traversal the moment `node.info.Generation <= snapshotGeneration` is encountered (guaranteeing that all subsequent nodes in the list are already up to date).
   - Clones changed nodes using `node.info.SnapshotConcrete()`.
   - In-place mutation `*existing = *clone` preserves existing pointers in `nodeInfoList` when the set of nodes has not changed.
3. **Affinity & Anti-Affinity Fast Paths**:
   - Tracks whether nodes transitioned between having/not having pods with affinity (`PodsWithAffinity`), required anti-affinity (`PodsWithRequiredAntiAffinity`), or non-host-scoped anti-affinity (`PodsWithRequiredNonHostScopedAntiAffinity`).
   - Reconstructs filtered slices (`havePodsWithAffinityNodeInfoList`, etc.) only when these boolean transitions occur or when nodes are added/deleted.
4. **PVC RefCount Delta Merge**:
   - Applies accumulated `pvcRefCountsDelta` to `snapshot.usedPVCRefCounts` without scanning every pod in the cluster.
5. **Node Tree Consistency Validation**:
   - Compares `len(nodeSnapshot.nodeInfoList)` against `cache.nodeTree.numNodes`.
   - Rebuilds lists if inconsistency is detected and returns an error to abort the corrupted cycle safely.

### 5.2 Snapshot Mutations for Gang / PodGroup Scheduling
During pod group scheduling (`schedule_one_podgroup.go`), multiple pods within a gang are evaluated together within the same cycle. The snapshot supports intra-cycle mutations:
- **`StartMutations()`**: Takes a shallow backup (`snapshotBackupData`) of snapshot maps/lists and performs deep copies of `nodeInfoMap` and `podGroupStates`.
- **`AddPod(podInfo, nodeName)` / `RemovePod(pod, nodeName)`**: Directly modifies the isolated snapshot during placement exploration.
- **`EndMutations()`**: Restores the snapshot to its exact state prior to `StartMutations()`.
- **`AssumePlacement(placement)` / `ForgetPlacement()`**: Restricts candidate nodes in the snapshot to a defined topology placement slice.

### 5.3 Snapshot Assume/Forget LIFO Invariant
When `Snapshot.AssumePod(podInfo)` is called directly on a snapshot:
- Updates `NodeInfo` and appends to snapshot-wide affinity lists if the node did not previously have affinity pods.
- Records mutations in `assumedPodState` and appends the key to `assumedPodKeys`.
- **LIFO Contract**: `Snapshot.ForgetPod()` strictly requires that pods are forgotten in the reverse order of assumption. This allows `removeAssumedNodeInfo()` to pop entries from the end of affinity slices in $O(1)$ time without searching or re-allocating.

---

## 6. Workload & PodGroup Hierarchy Management

When the `GenericWorkload` and `CompositePodGroup` feature gates are active, the cache maintains hierarchical state structures:

```
                  ┌──────────────────────────────┐
                  │ CompositePodGroup (Root CPG) │
                  └──────────────┬───────────────┘
                                 │ children
                                 ▼
                  ┌──────────────────────────────┐
                  │     PodGroup (Child PG)      │
                  └──────────────┬───────────────┘
                                 │ members
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
     unscheduledPods        assumedPods        assignedPods
     (Queued members)    (Reserved in gang)   (Bound to node)
```

- **`podGroupStateData` (`podgroupstate.go`)**:
  - `allPods`: Map of all member pods by UID.
  - `unscheduledPods`: UIDs of pods awaiting scheduling.
  - `assumedPods`: Map of pods currently in Reserve/Permit waiting for gang completion.
  - `assignedPods`: UIDs of pods bound to nodes.
  - `generation`: Monotonically incremented via atomic global counter (`nextPodGroupGeneration()`) to detect changes during snapshot updates.
- **Hierarchy Traversal**:
  - `BuildHierarchySnapshotFromPod(pod)`: Ascends from pod's `SchedulingGroup` to the root `CompositePodGroup` (capped at `WorkloadMaxTreeDepth = 5`), then descends recursively to build a cycle-free subtree snapshot.
  - `GetRootKeyForGroup(key)`: Identifies the root workload key for scheduling queue synchronization.

---

## 7. Cache Debugger Subsystem (`debugger/`)

The `debugger` package provides diagnostic facilities to detect drift between the scheduler cache, scheduling queue, and the API server:

```
                            OS Signal
                      (SIGUSR2 / SIGINT)
                              │
                              ▼
                   ┌─────────────────────┐
                   │    CacheDebugger    │
                   └──────────┬──────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
      ┌─────────────────┐           ┌─────────────────┐
      │  CacheComparer  │           │   CacheDumper   │
      └────────┬────────┘           └────────┬────────┘
               │                             │
    Compares Cache.Dump()          Dumps NodeInfo & Queue
    with client-go listers         details to klog
```

### 7.1 Components:
1. **`CacheComparer` (`comparer.go`)**:
   - Queries `corelisters.NodeLister` and `corelisters.PodLister`.
   - Calls `cache.Dump()` to obtain an isolated copy of all cached `NodeInfo` structs and `assumedPods`.
   - Calls `podQueue.PendingPods()` to account for queued pods.
   - Computes set differences (`CompareNodes`, `ComparePods`) and logs discrepancies under `missedNodes`, `redundantNodes`, `missedPods`, and `redundantPods`.
2. **`CacheDumper` (`dumper.go`)**:
   - Serializes all cached nodes, requested/allocatable resources, running pods, and nominated pods (`NominatedPodsForNode`) to structured logs.
3. **`ListenForSignal(ctx)` (`debugger.go`)**:
   - Listens for `SIGUSR2` (POSIX) or `SIGINT` (Windows).
   - Executes `Comparer.Compare(logger)` followed by `Dumper.DumpAll(logger)`.

---

## 8. Concurrency & Thread-Safety Model

| Component / Struct | Thread-Safety Guarantee | Synchronization Mechanism |
|---|---|---|
| **`cacheImpl`** | Safe for concurrent access | Internal `sync.RWMutex` (`mu`). Public methods manage locks internally. |
| **`Snapshot`** | **Not** thread-safe for writes; safe for parallel reads | Designed for single goroutine mutation or parallel read-only access (e.g. `parallelize.Until` across nodes during Filter/Score). |
| **`nodeTree`** | **Not** thread-safe | Internal to `cacheImpl`; callers must hold `cache.mu`. |
| **`podGroupState`** | **Not** thread-safe | Internal to `cacheImpl`; guarded by `cache.mu`. |
| **`CacheDebugger`** | Safe for concurrent invocation | Uses read-only `cache.Dump()` snapshot copies and Informer lister indexes. |

---

## 9. Testing Guide & Verification Patterns

When extending or verifying `pkg/scheduler/backend/cache`:

### 9.1 Unit Testing Conventions
- Use `fake.Cache` (`fake/fake_cache.go`) when testing outer scheduler loops or framework plugins without spinning up a full cache.
- For snapshot tests, use `NewEmptySnapshot()`, `NewSnapshot(pods, nodes)`, or `NewTestSnapshotWithPodGroups(...)`.
- Verify generation increments and doubly linked list movement with tests modeled after `TestUpdateSnapshot` in `snapshot_test.go`.

### 9.2 Running Package Tests
Execute unit tests for `cache` and all subpackages:

```bash
# Run cache unit tests
go test -v -race k8s.io/kubernetes/pkg/scheduler/backend/cache/...

# Run cache debugger tests
go test -v -race k8s.io/kubernetes/pkg/scheduler/backend/cache/debugger/...
```
