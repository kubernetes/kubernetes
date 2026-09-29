# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/preemption`

This guide provides AI agents and human contributors with an in-depth architectural overview, interface specifications, execution lifecycles, candidate finding and victim selection algorithms, reprieve mechanics, synchronization semantics, and developer invariants for the Kubernetes scheduler preemption subsystem located under `pkg/scheduler/framework/preemption`.

---

## 1. High-Level Overview & Subsystem Role

The preemption subsystem (`pkg/scheduler/framework/preemption`) provides the foundational algorithms, candidate evaluation engines, and actuation pipelines for evicting lower-priority pods when an incoming high-priority pod (or gang-scheduled pod group) cannot find sufficient resources on any cluster node during the filtering phase.

### Core Responsibilities:
1. **Preemption Framework Abstraction (`Interface`)**: Defines a pluggable interface implemented by preemption plugins (such as `DefaultPreemption`) to customize candidate limits, eligibility rules, victim selection logic, and node scoring.
2. **Candidate Discovery Engine (`Evaluator`, `findCandidates`, `DryRunPreemption`)**: Concurrently simulates preemption across candidate nodes in parallel chunks, evaluating feasibility without mutating live cluster state.
3. **Victim Selection & Reprieve Engine (`SelectVictimsOnNode`, `MoreImportantVictim`)**: Identifies the minimal set of lower-priority victim pods required to fit the preemptor while minimizing Pod Disruption Budget (PDB) violations and respecting workload importance hierarchies.
4. **Workload & Gang Preemption (`PodGroupEvaluator`, `defaultPreemptionManager`)**: Orchestrates atomic preemption for multi-pod workloads (PodGroups and CompositePodGroups) across multiple nodes with disruption mode propagation (`DisruptionModeAll`).
5. **Actuation & Eviction Lifecycle (`Executor`, `PreemptionExecutor`)**: Manages both synchronous and asynchronous (KEP-4832) eviction of victim pods, in-memory waiting pod preemption, pre-bind cancellation, `DisruptionTarget` status patching, and nominated node cleanup.
6. **Extender Integration (`callExtenders`)**: Interacts with external HTTP scheduling extenders supporting preemption (`ProcessPreemption`) to refine victim lists.

---

## 2. Directory Architecture & Subsystem Layout

```
pkg/scheduler/framework/preemption/
├── types.go                   # Core types (Victim, DomainVictim, candidate, candidateList, ExecutorPreemptor)
├── types_test.go              # Unit tests for victim sorting, domain victim creation, and candidate lists
├── util.go                    # Helper functions (MoreImportantVictim, PDB filters, hierarchy traversal, priorities)
├── util_test.go               # Unit tests for PDB filtering, victim importance ranking, and tree traversal
├── manager.go                 # defaultPreemptionManager and DomainVictim ordering
├── manager_test.go            # Unit tests for PreemptionManager victim generation and preemption policies
├── executor.go                # PreemptionExecutor (sync & async eviction, waiting pod cancellation, status patch)
├── executor_test.go           # Unit tests for Executor sync/async execution, metrics, and nominations
├── preemption.go              # Evaluator engine, findCandidates, DryRunPreemption, SelectCandidate, node scoring
├── preemption_test.go         # Unit tests for Evaluator lifecycle, extenders, dry-run, and candidate selection
├── podgrouppreemption.go      # PodGroupEvaluator for gang/workload preemption across the cluster snapshot
├── podgrouppreemption_test.go # Comprehensive unit tests for PodGroupEvaluator and reprieve loops
└── AGENTS.md                  # This agent guide
```

---

## 3. Core Types & Domain Models

### 3.1. `Victim` and `DomainVictim` (`types.go`)

```go
type Victim interface {
    Priority() int32
    Pods() []fwk.PodInfo
    EarliestStartTime() *metav1.Time
    Type() fwk.EntityKeyType
}
```

- **`Victim`**: Represents an atomic preemption unit. For standalone pods, it wraps a single `PodInfo`. For gang-scheduled workloads (PodGroups with `DisruptionModeAll`), it bundles all scheduled member pods across the cluster so they are evaluated and evicted as an indivisible unit.
- **`DomainVictim`**: Enriches a `Victim` with the `affectedNodes map[string]fwk.NodeInfo` from the snapshot. This allows the evaluator to assess the multi-node blast radius of preempting a distributed pod group during single-node or cluster-wide dry runs.
- **`ViolatingVictim[T Victim]`**: Wraps a `Victim` with `ViolateCount int`, indicating how many of its constituent pods violate a `PodDisruptionBudget` upon eviction.

### 3.2. `PreemptionCandidate` and `candidateList` (`types.go`)

- **`candidate`**: Implements `fwk.PreemptionCandidate`. Bundles the target node name, the `extenderv1.Victims` (slice of victim pods and PDB violation count), and the count of disrupted pod groups (`numPodGroupDisruptions`).
- **`candidateList`**: Thread-safe bounded container storing up to `candidatesNum` candidates discovered during parallel dry-run passes. Provides `add(*candidate)`, `size()`, and `get()`.

### 3.3. `ExecutorPreemptor` (`executor.go`)

Abstracts the entity triggering preemption (either a single `*v1.Pod` or a `fwk.PodGroupInfo`):
```go
type ExecutorPreemptor interface {
    klog.KMetadata
    UID() types.UID
    SchedulerName() string
    Obj() runtime.Object
    Pods() map[string]*v1.Pod
    Priority() int32
    Type() fwk.EntityKeyType
}
```

---

## 4. Single-Pod Preemption Lifecycle & Algorithm

When all nodes fail `FilterPlugins` during a scheduling cycle, the scheduler enters the `PostFilter` extension point. The `DefaultPreemption` plugin invokes `Evaluator.Preempt()`, which executes the following 6-step workflow:

```
[ PostFilter Triggered (0 Feasible Nodes) ]
                     │
                     ▼
[ Step 0: Fetch Latest Preemptor Pod ]
   └── Query PodLister to prevent operating on stale pod spec/status
                     │
                     ▼
[ Step 1: Preemption Eligibility Check ]
   ├── Pod.Spec.PreemptionPolicy == PreemptNever -> Abort
   └── Nominated node has terminating pods -> Wait (Abort cycle)
                     │
                     ▼
[ Step 2: Concurrently Find Candidates (findCandidates / DryRunPreemption) ]
   ├── Collect Unschedulable nodes from NodeToStatusReader
   ├── Compute random offset & candidate limit (MinCandidateNodesPercentage/Absolute)
   └── Run checkNode() in parallel via Parallelizer.Until:
         ├── Discover DomainVictims on candidate node (GetVictimsOnNode)
         ├── Run SelectVictimsOnNode on cloned CycleState & snapshot copy
         └── Collect non-violating & violating candidates into candidateList
                     │
                     ▼
[ Step 3: Extender Verification (callExtenders) ]
   └── Invoke extender.ProcessPreemption() for preempt-capable HTTP extenders
                     │
                     ▼
[ Step 4: Pick Best Candidate (SelectCandidate / pickOneNodeForPreemption) ]
   └── Execute 6-stage tie-breaking cascade to pick optimal node
                     │
                     ▼
[ Step 5: Actuate Preemption (Executor.ActuatePodPreemption) ]
   ├── Synchronous Path: Delete pods in parallel, clear lower-priority nominations
   └── Asynchronous Path (KEP-4832): Dispatch background goroutine, set PreEnqueue gate
                     │
                     ▼
[ Return PostFilterResult with NominatedNodeName ]
```

### Detailed Algorithm Steps:

#### Step 0: Retrieve Latest Preemptor
Fetches the updated `*v1.Pod` from `ev.PodLister` to avoid operating on stale annotations or scheduling gates.

#### Step 1: Preemption Eligibility (`PodEligibleToPreemptOthers`)
Evaluates whether the preemptor should be permitted to preempt:
1. **`PreemptionPolicy`**: If `Spec.PreemptionPolicy == PreemptNever`, preemption is immediately aborted with status `Unschedulable`.
2. **Victims in Flight**: If the preemptor already has a `Status.NominatedNodeName`, the evaluator checks whether any terminating pods on that node were marked with scheduler preemption (`PodTerminatingByPreemption`). If terminating pods exist, the scheduler returns `Unschedulable` to allow them to complete their graceful termination period without prematurely evicting additional workloads. (Exception: if the node was marked `UnschedulableAndUnresolvable` by filters, re-preemption is allowed).

#### Step 2: Parallel Dry Run (`findCandidates` / `DryRunPreemption`)
1. Filters cluster nodes using `NodeToStatusReader.NodesForStatusCode(..., fwk.Unschedulable)`. Nodes marked `UnschedulableAndUnresolvable` (e.g. node selector / architecture mismatch) are completely bypassed.
2. Calculates candidate search window:
   - `offset = rand(len(potentialNodes))`
   - `candidatesNum = max(MinCandidateNodesAbsolute, (len(potentialNodes) * MinCandidateNodesPercentage) / 100)`
3. Executes `checkNode` across worker goroutines via `fh.Parallelizer().Until`:
   - Evaluates node at `(offset + i) % len(potentialNodes)`.
   - Calls `GetVictimsOnNode(nodeInfo)` to build `[]*DomainVictim`.
   - Clones `CycleState` via `state.Clone()` and snapshots `nodeInfo.Snapshot()`.
   - Invokes `SelectVictimsOnNode`.
   - Stores candidate in `nonViolatingCandidates` (0 PDB violations) or `violatingCandidates`.
   - Short-circuits context cancellation once `nonViolatingCandidates.size() >= candidatesNum`.

#### Step 3: Scheduling Extender Filtering (`callExtenders`)
For clusters with HTTP scheduling extenders:
- Converts candidates to `map[string]*extenderv1.Victims`.
- Passes map to `extender.ProcessPreemption()`.
- Extenders can drop nodes or add/remove victim pods. Placeholder nodes with empty victim lists are preserved for downstream extenders.

#### Step 4: Candidate Selection Cascade (`pickOneNodeForPreemption`)
When multiple candidate nodes pass dry-run preemption, `pickOneNodeForPreemption` executes a deterministic 6-tier scoring cascade:

| Tier | Evaluation Metric | Preference Rule | Rationale |
|---|---|---|---|
| **1** | `minNumPDBViolatingScoreFunc` | Minimum PDB violations | Preserves application availability budgets. |
| **2** | `minHighestPriorityScoreFunc` | Minimum highest victim priority | Avoids preempting higher-tier workloads. |
| **3** | `minSumPrioritiesScoreFunc` | Minimum sum of victim priorities | Minimizes total priority displacement (normalized by `+ MaxInt32 + 1`). |
| **4** | `minNumPodsScoreFunc` | Minimum count of victim pods | Minimizes churn and restart overhead. |
| **5** | `latestStartTimeScoreFunc` | Latest start time of highest priority victims | Enforces "first-come, first-served" by protecting older, longer-running jobs. |
| **6** | Tie-Breaker | First node in list | Deterministic tie-breaker. |

---

## 5. Victim Selection & Reprieve Algorithm (`SelectVictimsOnNode`)

`SelectVictimsOnNode` finds the minimal subset of lower-priority pods on a node to evict in order to fit the preemptor.

```
[ Input: All DomainVictims on Target Node ]
                     │
                     ▼
[ 1. Eligibility Filter ]
   └── Discard victims with Priority >= Preemptor or failing IsEligiblePod
                     │
                     ▼
[ 2. Complete Removal Pass ]
   ├── Mutate mainNode.RemovePod()
   ├── Run PreFilterExtensionRemovePod() for main & remote nodes
   └── RunFilterPluginsWithNominatedPods()
         ├── If FAIL -> Node cannot fit preemptor even if ALL victims evicted (ABORT)
         └── If PASS -> Proceed to Reprieve Loop
                     │
                     ▼
[ 3. Sort Potential Victims by Importance ]
   └── Order descending using MoreImportantVictim()
                     │
                     ▼
[ 4. Phase A: Reprieve PDB-Violating Victims ]
   └── Iterate descending: Add victim back -> RunFilterPluginsWithNominatedPods()
         ├── Fits? -> Keep in node (Reprieved!)
         └── Fails? -> Remove victim again & mark as Definite Victim
                     │
                     ▼
[ 5. Phase B: Reprieve Non-Violating Victims ]
   └── Iterate descending: Add victim back -> RunFilterPluginsWithNominatedPods()
         ├── Fits? -> Keep in node (Reprieved!)
         └── Fails? -> Remove victim again & mark as Definite Victim
                     │
                     ▼
[ 6. Return Final Minimal Victim List & PDB Violation Count ]
```

### NodeInfo Mutation & PreFilterExtension Contract:
- **Main Node vs Remote Nodes**: Within `SelectVictimsOnNode`, only `nodeInfo` for the **main node** (the placement candidate) is mutated via `RemovePod` / `AddPodInfo`.
- **Remote Nodes**: If a victim belongs to a multi-node PodGroup, remote nodes are **not mutated** in `NodeInfo`. Instead, `PreFilterExtensionRemovePod` and `PreFilterExtensionAddPod` are called with the remote `NodeInfo` so plugins maintaining cluster-wide state in `CycleState` (e.g. topology spread skew, pod affinity counters) can incrementally update their cached state.

### Victim Importance Ranking (`MoreImportantVictim`):
When comparing two preemption units `vi1` and `vi2`:
1. **Priority**: Higher priority is more important.
2. **Workload Type**: `CompositePodGroup` (rank 3) > `PodGroup` (rank 2) > `Pod` (rank 1).
3. **Runtime / Start Time (Single Pods)**: Pod with earlier start time (longer runtime) is more important ("first-come, first-served").
4. **Group Size (PodGroups)**: Larger group size is more important (avoids high rescheduling cost of large jobs).
5. **Group Start Time**: Group with older earliest start time is more important.
6. **UID Tie-Breaker**: Deterministic comparison of first pod UID (`podIdentityLess`).

---

## 6. Workload & Gang Preemption (`PodGroupEvaluator`)

When gang scheduling is enabled (`features.GenericWorkload`), `PodGroupEvaluator` handles preemption where the preemptor is an entire multi-pod workload.

### 6.1. Cross-Node Victim Aggregation
- `getWorkloadPreemptionVictims` traverses the cluster snapshot.
- If a pod belongs to a group hierarchy with `DisruptionModeAll` (discovered via `traverseHierarchyUp`), all scheduled pods belonging to leaf pod groups are bundled into a single atomic `Victim`.
- Preemption of one pod in the group implies evicting all member pods across the cluster to maintain gang atomicity.

### 6.2. Whole-Cluster Reprieve Pipeline (`selectVictimsOnDomain`)
1. **Snapshot Mutation Bracket**: Invokes `MutableSnapshotSharedLister().StartMutations()` and defers `EndMutations()`.
2. **Initial Cluster Eviction**: Removes all candidate victims from the mutable snapshot.
3. **Feasibility Check (`podGroupSchedulingFunc`)**: Invokes the full gang scheduling algorithm on the cleared cluster snapshot. If the gang still cannot schedule, preemption fails immediately.
4. **Reverse Reprieve Loop**: Iterates through victims in reverse order (most important first via `slices.Backward(potentialVictims)`).
   - Simulates adding victim pods back to the snapshot and running `PreFilterExtensionAddPod` across all preemptor proposed assignments.
   - For each proposed assignment, executes `RunFilterPluginsWithNominatedPods` on the proposed node.
   - If feasible, simulates assuming the preemptor pod (`mutableLister.AddPod`) and running `RunReservePluginsReserve`.
   - If any filter fails, rolls back additions using `removeVictimPodsWithPreFilter`, unreserves previous assignments, and confirms the victim for eviction.

---

## 7. Actuation Subsystem (`Executor`)

`Executor` implements `fwk.PreemptionExecutor` to perform the actual eviction of chosen victim pods.

### 7.1. Pod Preemption Mechanisms (`PreemptPod`)
When evicting a victim pod:
1. **Waiting Pod Cancellation**: If the victim is currently waiting in the `Permit` phase (`fh.GetWaitingPod(victim.UID)`), invokes `waitingPod.Preempt(pluginName, "preempted")`. The victim is rejected in memory and returns to the backoff queue without issuing an API delete call.
2. **Pre-Bind Cancellation**: If the victim is in the asynchronous pre-bind phase (`fh.GetPodInPreBind(victim.UID)`), invokes `podInPreBind.CancelPod(...)`.
3. **API Eviction**:
   - Patches `PodStatus` with condition `DisruptionTarget` (`Status = True`, `Reason = PodReasonPreemptionByScheduler`, `Message = "...: preempting to accommodate a higher priority ..."`).
   - Issues `DeletePod` API call via client-go.
   - Publishes a `Preempted` Normal event to the event recorder.

### 7.2. Synchronous vs. Asynchronous Preemption

```
                       [ ActuatePodPreemption ]
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   [ Synchronous Execution ]                 [ Asynchronous Execution ]
   (EnableAsyncPreemption = false)           (EnableAsyncPreemption = true)
              │                                         │
   ├── Parallelize Until() for all           ├── Create detached context.Background()
   │   c.Victims().Pods                      ├── Insert preemptor.UID into preempting map
   ├── PreemptPod() for each                 ├── Launch background goroutine:
   ├── Clear NominatedNodeName on            │     ├── Parallelize Until() for N-1 victims
   │   lower-priority pods on target         │     ├── Set lastVictimsPendingPreemption
   └── Return status to scheduling cycle     │     ├── Preempt last victim Pod
                                             │     ├── Clear lower-priority nominations
                                             │     ├── Delete from preempting map
                                             │     └── If in-memory/error: fh.Activate()
                                             └── Return nil immediately to scheduling cycle
```

### 7.3. Asynchronous Preemption Concurrency Invariants (KEP-4832)
1. **Detached Context**: The background goroutine uses `context.Background()` with cancellation, ensuring eviction proceeds even after the scheduling cycle finishes.
2. **PreEnqueue Gating (`IsPodRunningPreemption`)**: While preemption is in-flight, `preempting.Insert(preemptor.UID)` prevents the preemptor from re-entering the scheduling queue prematurely.
3. **Last Victim Handoff (`lastVictimsPendingPreemption`)**:
   - Deleting the last victim is performed after the first $N-1$ victims finish.
   - `lastVictimsPendingPreemption[uid]` tracks the final victim. If the informer receives the pod deletion event before the goroutine finishes cleanup, `PreEnqueue` inspects the informer to verify that the last victim is already terminating and allows queue entry without stalling.
4. **Activation Fallback**: If an error occurs or the last victim was preempted purely in-memory (producing no API deletion event), `fh.Activate(logger, preemptor.Pods())` forces re-queueing of the preemptor to prevent deadlocks in `unschedulablePods`.

---

## 8. Extender & Policy Utilities (`util.go`)

### 8.1. Pod Disruption Budget Evaluation (`FilterVictimsWithPDBViolation`)
- Inspects active `PodDisruptionBudget` resources across the cluster.
- Decrements `pdb.Status.DisruptionsAllowed` as matching pods are evaluated.
- **Victim Atomicity**: If **any single pod** inside a gang `Victim` violates a PDB on eviction, the **entire Victim** is classified as violating. Partial eviction of gang workloads is strictly forbidden.

### 8.2. Hierarchy Traversal (`traverseHierarchyUp`)
- Uses Go 1.23+ range-over-func iterator (`iter.Seq[*fwk.GenericPodGroup]`) to walk upward from a leaf PodGroup through CompositePodGroups up to `WorkloadMaxTreeDepth` (10).
- Used to compute root group priorities (`getPodPriority`) and locate highest ancestors with `DisruptionModeAll`.

---

## 9. Prometheus Metrics & Observability

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `scheduler_preemption_attempts_total` | Counter | - | Total number of preemption attempts initiated in `PostFilter`. |
| `scheduler_preemption_victims` | Histogram | - | Number of victim pods selected per pod preemption attempt. |
| `scheduler_workload_preemption_victims` | Histogram | - | Number of victim pods selected per workload/gang preemption attempt. |
| `scheduler_preemption_evaluation_duration_seconds` | Histogram | `preemptor_type`, `status_code` | Latency of preemption candidate evaluation passes. |
| `scheduler_preemption_execution_duration_seconds` | Histogram | `preemptor_type`, `result` | Duration of synchronous or asynchronous preemption execution. |
| `scheduler_preemption_goroutines_duration_seconds` | Histogram | `result` | Wall-clock execution time of async preemption background goroutines. |
| `scheduler_preemption_goroutines_execution_total` | Counter | `result` | Number of async preemption goroutines completed (`success` / `error`). |
| `scheduler_preemption_pdb_violations_total` | Counter | `preemptor_type` | Total number of PDB violations incurred during preemption. |
| `scheduler_preemption_workload_disruptions` | Histogram | `preemptor_type` | Number of atomic pod groups disrupted during preemption. |

---

## 10. Critical Developer & Agent Invariants

1. **CycleState Isolation during Dry Run**:
   - `checkNode` in `DryRunPreemption` evaluates candidate nodes concurrently across worker goroutines. It **MUST** pass `state.Clone()` into `SelectVictimsOnNode`. Never share mutable `CycleState` between parallel evaluation workers.
2. **No Cross-Worker Coordination in `checkNode`**:
   - In `DryRunPreemption`, each worker discovers victims for its assigned node independently. If a multi-node PodGroup victim appears on multiple candidate nodes, workers evaluate it independently. Downstream `SelectCandidate` ensures at most one candidate node is chosen.
3. **Snapshot Mutation Safety**:
   - In `PodGroupEvaluator`, snapshot modifications **MUST** be enclosed between `MutableSnapshotSharedLister().StartMutations()` and `EndMutations()`. All speculative pod additions/removals must be restored before returning.
4. **Main Node vs Remote Node Mutation Boundary**:
   - Within `SelectVictimsOnNode`, only the main candidate node's `NodeInfo` is mutated via `RemovePod` / `AddPodInfo`. Remote nodes of distributed pod groups are updated exclusively via `PreFilterExtensionRemovePod` / `AddPod` hooks.
5. **Disruption Target Status Patching**:
   - Victim pods MUST have their status patched with condition `DisruptionTarget` before deletion, preserving attribution and pod lifecycle observability.
6. **Context Detachment in Async Eviction**:
   - `prepareCandidateAsync` MUST create a new context (`context.Background()`) detached from the scheduling cycle, as the scheduling cycle context is canceled immediately upon returning from `scheduleOne`.

---

## 11. Testing Patterns & Verification

```bash
# Run all preemption unit tests with race detection
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/preemption/...

# Run DefaultPreemption plugin tests
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/defaultpreemption/...

# Run scheduler integration preemption tests
GOTOOLCHAIN=auto go test -v ./test/integration/scheduler -run TestPreemption
```
