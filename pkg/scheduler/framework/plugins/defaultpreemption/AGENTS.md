# DefaultPreemption Plugin (`pkg/scheduler/framework/plugins/defaultpreemption`)

This guide provides an architectural overview, interface implementations, victim selection algorithms, reprieve mechanics, PodGroup and composite preemption flows, queueing hints, and testing strategies for the `DefaultPreemption` plugin in `pkg/scheduler/framework/plugins/defaultpreemption`.

---

## 1. High-Level Purpose & Scope

The `DefaultPreemption` plugin implements the scheduler's `PostFilter` extension point. When an incoming high-priority pod or workload cannot fit on any node in the cluster during normal filtering, `DefaultPreemption` searches for candidate nodes where evicting a minimal set of lower-priority "victim" pods will allow the preemptor to fit.

### Core Responsibilities:
1. **Preemption Feasibility Assessment**: Evaluates whether a pod is eligible to preempt others (`PodEligibleToPreemptOthers`), checking `spec.preemptionPolicy` and verifying that the nominated node is not waiting on existing terminating pods.
2. **Victim Selection on Candidate Nodes**: Computes the optimal, minimal set of victim pods on a candidate node (`SelectVictimsOnNode`) while minimizing PodDisruptionBudget (PDB) violations and preserving higher-importance workloads.
3. **Multi-Node PodGroup Eviction**: Coordinates preemption of multi-node PodGroups, discovering all affected nodes and ensuring pre-filter extensions (e.g. topology spread skew, inter-pod affinity) update their cycle state accurately.
4. **Asynchronous Preemption Gating**: Implements `PreEnqueue` to gate pods or pod groups that currently have asynchronous preemption operations in progress (`IsPodRunningPreemption` / `IsPodGroupRunningPreemption`).
5. **Workload & Composite PodGroup Preemption**: Supports hierarchical workload preemption (`PodGroupPostFilter`) utilizing mutable snapshot transactions (`StartMutations` / `EndMutations`).

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/defaultpreemption/
├── default_preemption.go       # Plugin definition, PostFilter, PreEnqueue, SelectVictimsOnNode, reprieve logic
├── default_preemption_test.go  # Comprehensive table-driven tests for victim selection, PDB reprieves, and pod groups
└── AGENTS.md                   # This agent documentation
```

---

## 3. Data Structures & Configuration

### 3.1. `DefaultPreemption` Struct

```go
type DefaultPreemption struct {
    fh                fwk.Handle
    fts               feature.Features
    args              config.DefaultPreemptionArgs
    Executor          fwk.PreemptionExecutor
    Evaluator         *preemption.Evaluator
    pgLister          fwk.PodGroupLister
    pgSnapshotLister  fwk.PodGroupLister
    cpgSnapshotLister fwk.CompositePodGroupLister
    podGroupEvaluator podGroupEvaluator

    IsEligiblePod       IsEligiblePodFunc
    MoreImportantVictim MoreImportantVictimFunc
}
```

### 3.2. Pluggable Hook Functions

- **`IsEligiblePodFunc`**: `func(nodeInfo fwk.NodeInfo, victim preemption.Victim, preemptor *v1.Pod) bool`
  - In addition to the strict priority check (`victim.Priority() < preemptor.Priority()`), allows customized filtering.
  - For PodGroups, evaluated across every affected node; the victim is eligible only if it returns `true` for all affected nodes.
- **`MoreImportantVictimFunc`**: `func(victim1, victim2 preemption.Victim) bool`
  - Sorts potential victims in descending order of importance (most important first) to prioritize reprieving them.

### 3.3. Configuration Parameters (`DefaultPreemptionArgs`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `MinCandidateNodesPercentage` | `int32` | `10` | Minimum percentage of nodes to evaluate during dry-run preemption. |
| `MinCandidateNodesAbsolute` | `int32` | `100` | Minimum absolute number of nodes to evaluate during dry-run preemption. |

---

## 4. Extension Point Implementations

`DefaultPreemption` implements `fwk.PostFilterPlugin`, `fwk.PreEnqueuePlugin`, and `fwk.EnqueueExtensions`.

```
                    ┌──────────────────────────────────────────────┐
                    │               PostFilter Phase               │
                    │      (Pod unschedulable across all nodes)    │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │       PodEligibleToPreemptOthers?            │
                    │   - PreemptionPolicy != PreemptNever?        │
                    │   - No terminating pods on nominated node?   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                    [ Eligible ]
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Dry-Run Find Candidates on Nodes        │
                    │    (Random offset, bounded by MinCandidates) │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │            SelectVictimsOnNode               │
                    │  1. Remove all eligible victims              │
                    │  2. Verify preemptor fits                    │
                    │  3. Reprieve PDB-violating victims (desc)    │
                    │  4. Reprieve non-violating victims (desc)    │
                    │  5. Collect remaining victims                │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │         Execute Preemption Evictions         │
                    │       (Synchronous or Asynchronous)          │
                    └──────────────────────────────────────────────┘
```

### 4.1. `PostFilter` (`PostFilterPlugin`)
- **Signature**: `PostFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, m fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status)`
- **Behavior**: Delegates to `pl.Evaluator.Preempt(ctx, state, pod, m)`, incrementing `metrics.PreemptionAttempts`. Returns nominated node results or unschedulable status.

### 4.2. `PreEnqueue` (`PreEnqueuePlugin`)
- **Signature**: `PreEnqueue(ctx context.Context, p *v1.Pod) *fwk.Status`
- **Behavior**:
  - When `EnableAsyncPreemption` is active:
    - For standalone pods: Checks `pl.Executor.IsPodRunningPreemption(p.GetUID())`. If true, returns `fwk.UnschedulableAndUnresolvable` (`"waiting for the preemption for this pod to be finished"`).
    - For PodGroups: Resolves the root group UID (accounting for hierarchical composite pod groups) and checks `pl.Executor.IsPodGroupRunningPreemption(rootUID)`.

### 4.3. `EventsToRegister` (`EnqueueExtensions`)
- Registers `{Resource: fwk.AssignedPod, ActionType: fwk.Delete}` with `QueueingHintFn` returning `fwk.QueueSkip`.

---

## 5. Victim Selection Algorithm (`SelectVictimsOnNode`)

`SelectVictimsOnNode` finds the minimal set of victim pods to evict on a target node to accommodate the preemptor:

```
[ All Possible Victims ]
        │
        ▼
[ Filter by isPreemptionAllowedAcrossAllVictimNodes ]
        │
        ▼
[ Step 1: Remove ALL Victims ]
  - Main Node: Mutate NodeInfo (nodeInfo.RemovePod) + RunPreFilterExtensionRemovePod
  - Remote Nodes (PodGroup members): RunPreFilterExtensionRemovePod ONLY
        │
        ▼
[ Step 2: Test Fit (RunFilterPluginsWithNominatedPods) ]
        │
  ├── Does NOT Fit ──> Return failure (Node not suitable for preemption)
  └── Fits
        │
        ▼
[ Step 3: Sort Potential Victims by MoreImportantVictim (Descending) ]
        │
        ▼
[ Step 4: Partition into PDB-Violating vs Non-Violating Sets ]
        │
        ▼
[ Step 5: Reprieve Pass - PDB-Violating First ]
  - Add victim back (NodeInfo.AddPodInfo + RunPreFilterExtensionAddPod)
  - Test Fit:
      - Fits: Kept reprieved (not evicted)
      - Fails: Remove again, add to final victims list, count PDB violations
        │
        ▼
[ Step 6: Reprieve Pass - Non-Violating Next ]
  - Add victim back
  - Test Fit:
      - Fits: Kept reprieved
      - Fails: Remove again, add to final victims list
        │
        ▼
[ Return Final Victim Pods & PDB Violation Count ]
```

### 5.1. Main Node vs. Remote Node Mutation Contract
- **Main Node**: Mutates both `NodeInfo` (via `RemovePod` / `AddPodInfo`) and framework `CycleState` (via `RunPreFilterExtensionRemovePod` / `RunPreFilterExtensionAddPod`). This ensures subsequent `RunFilterPluginsWithNominatedPods` passes see the updated node capacity and port bindings.
- **Remote Nodes**: Mutates *only* `CycleState` via `PreFilterExtension` hooks (e.g. topology spread skew counters, pod affinity state). Remote `NodeInfo` instances are not mutated directly during single-node victim selection.

### 5.2. Victim Importance Hierarchy (`MoreImportantVictim`)

When sorting victims for reprieve, higher importance workloads are reprieved first:
1. **Higher Priority**: Victims with higher `Priority()` are more important.
2. **Workload Hierarchy**: `CompositePodGroup` > `PodGroup` > Standalone `Pod`.
3. **Group Size**: Larger groups are more important than smaller groups.
4. **Runtime / StartTime**: Pods/groups with older `StartTime` (running longer) are more important (final tie-breaker).

---

## 6. Workload & PodGroup Preemption (`PodGroupPostFilter`)

For gang-scheduled workloads and pod groups (`EnableGenericWorkload`):
- Executes `PodGroupPostFilter(ctx, state, pgInfo, pgSchedulingFunc)`.
- Wraps execution in `MutableSnapshotSharedLister` transactions (`StartMutations()` / `EndMutations()`) to simulate cross-node group placement dry runs safely.
- Tracks workload metrics via `metrics.WorkloadPreemptionAttempts`.

---

## 7. Test Fixtures & Unit Testing

`default_preemption_test.go` provides extensive test suites:
- **`TestSelectVictimsOnNode`**: Verifies victim selection under varying resource pressures, multiple priorities, PDB constraints, and pod affinity rules.
- **`TestPodEligibleToPreemptOthers`**: Validates preemption policies (`PreemptNever`, `PreemptLowerPriority`) and nominated node terminating pod checks.
- **`TestOffsetAndNumCandidates`**: Verifies randomized node offset selection and minimum candidate bounds.
- **PodGroup Preemption Fixtures**: Tests multi-node victim removal and reprieve across composite and standard pod groups.
