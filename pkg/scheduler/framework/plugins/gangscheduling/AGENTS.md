# GangScheduling Plugin

This guide provides an architectural overview, interface implementations, gang coordination algorithms, queueing hint mechanics, hierarchical CompositePodGroup handling, and testing strategies for the `GangScheduling` plugin in `pkg/scheduler/framework/plugins/gangscheduling`.

---

## 1. High-Level Purpose & Scope

The `GangScheduling` plugin enforces **all-or-nothing (gang)** scheduling semantics for pods associated with a `PodGroup` (or hierarchical `CompositePodGroup`) configured with a gang scheduling policy (`spec.schedulingPolicy.gang`).

### Core Responsibilities:
1. **Pre-Enqueue Quorum Gating**: Prevents individual gang member pods from entering the scheduling queue (`activeQ` / `backoffQ`) until a quorum of pods belonging to the gang or composite hierarchy has arrived (`AllPodsCount() >= minCount`).
2. **Placement Cycle Feasibility Monitoring (`PlacementFeasible`)**: Evaluates placement progress during the PodGroup scheduling cycle to determine if candidate node allocations meet the gang's minimum member threshold (`minCount` / `minGroupCount`).
3. **Early Placement Cycle Termination**: Returns `fwk.Unschedulable` during placement evaluation if the sum of already scheduled and remaining candidate pods cannot satisfy `minCount`, terminating futile placement searches early.
4. **Hierarchical Gang Quorum Resolution**: Recursively traverses `CompositePodGroup` hierarchy trees to ensure composite gang quotas (`minGroupCount`) and child group constraints (`minCount`) are met across multi-tier workload trees.
5. **Fine-Grained Queueing Hints**: Registers targeted informer event handlers (`UnscheduledPod/Add`, `AssignedPod/Add`, `PodGroup/Add`, `PodGroup/Update`, `CompositePodGroup/Add`, `CompositePodGroup/Update`) to awaken waiting gang pods only when relevant hierarchy additions or policy threshold decreases occur.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/gangscheduling/
├── gangscheduling.go       # Plugin definition, PreEnqueue, PlacementFeasible, Queueing Hints, and Hierarchy validation
├── gangscheduling_test.go  # Comprehensive unit tests for PreEnqueue, PlacementFeasible, and Queueing Hints
└── AGENTS.md               # This agent documentation
```

---

## 3. Data Structures & Plugin Configuration

### 3.1. `GangScheduling` Struct

```go
type GangScheduling struct {
    handle                     fwk.Handle
    podGroupManager            fwk.PodGroupManager
    snapshotLister             fwk.SharedLister
    isCompositePodGroupEnabled bool
}
```

- **`handle`**: Framework handle providing access to shared listers, pod group managers, and cluster state.
- **`podGroupManager`**: Manages PodGroup and CompositePodGroup metadata and pod member tracking.
- **`snapshotLister`**: Shared lister snapshot for querying pod group states within scheduling cycles.
- **`isCompositePodGroupEnabled`**: Flag controlled by feature gate `EnableCompositePodGroup`.

### 3.2. Constants

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.GangScheduling` (`"GangScheduling"`) | Registered plugin name in scheduler profiles. |

---

## 4. Extension Point Implementations

`GangScheduling` implements `fwk.PreEnqueuePlugin`, `fwk.PlacementFeasiblePlugin`, and `fwk.EnqueueExtensions`.

```
                        ┌──────────────────────────────────────────────┐
                        │              Pod Admission Flow              │
                        │           (PreEnqueue Evaluation)            │
                        └──────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
          [ pod.Spec.SchedulingGroup == nil ]             [ SchedulingGroup != nil ]
                       │                                               │
                       ▼                                               ▼
             ┌───────────────────┐                         ┌───────────────────────┐
             │    Return nil     │                         │ Fetch PodGroup Spec   │
             │ (Bypass Plugin)   │                         └───────────┬───────────┘
             └───────────────────┘                                     │
                                                  ┌────────────────────┴────────────────────┐
                                                  │                                         │
                                       [ Non-Gang Policy ]                           [ Gang Policy ]
                                                  │                                         │
                                                  ▼                                         ▼
                                        ┌───────────────────┐                  ┌────────────────────────┐
                                        │    Return nil     │                  │ Check Available Quorum │
                                        │ (Bypass Plugin)   │                  │  AllPodsCount >= Min   │
                                        └───────────────────┘                  └───────────┬────────────┘
                                                                                           │
                                                                   ┌───────────────────────┴───────────────────────┐
                                                                   │                                               │
                                                          [ Quorum Satisfied ]                            [ Quorum Not Met ]
                                                                   │                                               │
                                                                   ▼                                               ▼
                                                         ┌───────────────────┐                      ┌─────────────────────────────┐
                                                         │    Return nil     │                      │ Return                      │
                                                         │ (Enter activeQ)   │                      │ UnschedulableAndUnresolvable│
                                                         └───────────────────┘                      └─────────────────────────────┘
```

### 4.1. `PreEnqueue` (`PreEnqueuePlugin`)

- **Signature**: `PreEnqueue(ctx context.Context, pod *v1.Pod) *fwk.Status`
- **Behavior**:
  - If `pod.Spec.SchedulingGroup` is nil, returns `nil` (non-grouped pod).
  - When `isCompositePodGroupEnabled` is false (`preEnqueueHierarchiesDisabled`):
    - Retrieves the `PodGroup` from `podGroupManager`. If not found, returns `UnschedulableAndUnresolvable` waiting for the PodGroup object.
    - If `policy.Gang == nil`, returns `nil`.
    - Retrieves `PodGroupState` and evaluates `podGroupState.AllPodsCount() >= policy.Gang.MinCount`.
    - If quorum is met, returns `nil`; otherwise returns `fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue")`.
  - When `isCompositePodGroupEnabled` is true (`preEnqueueWithHierarchies`):
    - Builds hierarchy snapshot from pod via `handle.PodGroupManager().BuildHierarchySnapshotFromPod(pod)`.
    - If the PodGroup has no parent CompositePodGroup, verifies `isPGReady(snapshot, namespace, podGroup.Name, ...)`.
    - If part of a CompositePodGroup tree, executes `checkCPGHierarchyReadiness` recursively starting from the root composite group.

### 4.2. `PlacementFeasible` (`PlacementFeasiblePlugin`)

- **Signature**: `PlacementFeasible(ctx context.Context, placementCycleState fwk.PlacementCycleState, podGroupInfo fwk.PodGroupInfo, args fwk.PlacementProgress) *fwk.Status`
- **Arguments**:
  - `args.Scheduled`: Count of pods successfully scheduled/assigned in current placement iteration.
  - `args.Remaining`: Count of pods remaining to be evaluated in current placement iteration.
  - `minCount`: Minimum member threshold obtained via `getMinCount(podGroupInfo)` (returns `1` for basic/non-gang policies).
- **Decision Logic**:
  1. **Impossible Quorum**: `if remaining + scheduled < minCount` -> Returns `fwk.NewStatus(fwk.Unschedulable, ...)`. Terminates placement search early, saving CPU cycles across remaining unviable placements.
  2. **Quorum In Progress**: `if scheduled < minCount` -> Returns `fwk.NewStatus(fwk.Wait, ...)`. Instructs placement engine that gang quorum is not yet reached and more candidate pods must be processed.
  3. **Quorum Satisfied**: `if scheduled >= minCount` -> Returns `nil` (`Success`). Placement candidate has met all-or-nothing member capacity.

```
                  ┌───────────────────────────────────────────────────────────┐
                  │                 PlacementFeasible Check                   │
                  │   scheduled = args.Scheduled, remaining = args.Remaining  │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         │                                      │                                      │
 [ remaining + scheduled < minCount ]  [ scheduled < minCount ]              [ scheduled >= minCount ]
         │                                      │                                      │
         ▼                                      ▼                                      ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐       ┌─────────────────┐
│ Return fwk.Unschedulable     │       │ Return fwk.Wait              │       │ Return nil      │
│ (Terminate placement search) │       │ (Continue evaluating pods)   │       │ (Quorum met)    │
└──────────────────────────────┘       └──────────────────────────────┘       └─────────────────┘
```

### 4.3. `EventsToRegister` & Queueing Hints (`EnqueueExtensions`)

`GangScheduling` registers fine-grained event handlers to prevent unneeded active queue churn:

| Cluster Event | Handler Function | Queueing Hint Logic |
| :--- | :--- | :--- |
| `UnscheduledPod/Add` | `isSchedulableAfterPodAdded` | Returns `fwk.Queue` if added pod belongs to same PodGroup/hierarchy (`areSameHierarchy`). Otherwise `QueueSkip`. |
| `AssignedPod/Add` | `isSchedulableAfterPodAdded` | Returns `fwk.Queue` if pre-bound/assigned pod belongs to same hierarchy. |
| `PodGroup/Add` | `isSchedulableAfterPodGroupAdded` | Returns `fwk.Queue` if newly added `PodGroup` is in the waiting pod's hierarchy. |
| `PodGroup/Update` | `isSchedulableAfterPodGroupUpdated` | Returns `fwk.Queue` only if `newPolicy.Gang.MinCount < oldPolicy.Gang.MinCount` and belongs to same hierarchy. |
| `CompositePodGroup/Add` | `isSchedulableAfterCompositePodGroupAdded` | When composite groups enabled, returns `fwk.Queue` if added CPG belongs to same hierarchy. |
| `CompositePodGroup/Update` | `isSchedulableAfterCompositePodGroupUpdated` | When composite groups enabled, returns `fwk.Queue` only if `newPolicy.Gang.MinGroupCount < oldPolicy.Gang.MinGroupCount` in same hierarchy. |

---

## 5. CompositePodGroup Hierarchy Traversal

When `EnableCompositePodGroup` is active, the plugin evaluates readiness across a tree structure:

```
                            ┌────────────────────────────┐
                            │    Root CompositePodGroup  │
                            │   (MinGroupCount: M >= 1)  │
                            └─────────────┬──────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
                   ▼                                             ▼
        ┌─────────────────────┐                       ┌─────────────────────┐
        │ Child CompositeGroup│                       │     PodGroup A      │
        │(MinGroupCount: >= 1)│                       │  (MinCount: N >= 1) │
        └──────────┬──────────┘                       └─────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
  ┌─────────────┐     ┌─────────────┐
  │ PodGroup B  │     │ PodGroup C  │
  └─────────────┘     └─────────────┘
```

- **`checkCPGHierarchyReadiness`**: Locates the root entity key via `snapshot.GetRootKeyForGroup(cpgKey)` and invokes `isCPGTreeReady`.
- **`isCPGTreeReady`**: Counts children satisfying their respective readiness conditions (either sub-tree `isCPGTreeReady` for composite children or `isPGReady` for basic pod group children). Returns `true` iff `successfulChildren >= minGroupCount`.
- **`isPGReady`**: Asserts `readinessCountFn(pgState) >= minCount` (where `readinessCountFn` counts all known pods via `s.AllPodsCount()`).

---

## 6. Gang Coordination Lifecycle & Scheduler Integration

1. **Submission Phase**:
   - Workload controller creates `PodGroup` and member pods with `.spec.schedulingGroup.podGroupName`.
   - As pods arrive, `PreEnqueue` checks `AllPodsCount()`. Pods arriving before quorum is reached receive `UnschedulableAndUnresolvable` and rest in `unschedulableEntities`.
2. **Quorum Arrival**:
   - The arrival of the final pod triggers `isSchedulableAfterPodAdded`, which emits `fwk.Queue` for waiting hierarchy members, moving all gang pods to `activeQ`.
3. **Placement Phase**:
   - PodGroup scheduling engine batches member pods and evaluates candidate placements across nodes.
   - `PlacementFeasible` monitors scheduling progress. If node filtering failures reduce remaining candidates below `minCount`, the placement cycle aborts immediately.
   - Once all `minCount` pods are placed onto candidate nodes, the placement passes to reserve and permit phases.

---

## 7. Testing Strategy & Test Coverage

The unit test suite in `gangscheduling_test.go` validates:

1. **Queueing Hints**:
   - `Test_isSchedulableAfterPodAdded`: Verifies queueing on matching gang pods vs skipping non-matching namespaces/groups.
   - `Test_isSchedulableAfterPodGroupAdded` & `Test_isSchedulableAfterCompositePodGroupAdded`: Validates hierarchy awareness on group additions.
   - `Test_isSchedulableAfterPodGroupUpdated` & `Test_isSchedulableAfterCompositePodGroupUpdated`: Asserts that pods only re-enqueue when `minCount`/`minGroupCount` decreases, skipping no-op updates or count increases.
2. **PreEnqueue Quorum Validation**:
   - `TestPreEnqueue`: Validates immediate pass for non-gang / basic policy pods, rejection when `PodGroup` is missing, rejection when quorum < `minCount`, and success when quorum is satisfied across single and composite hierarchies.
3. **Placement Feasibility**:
   - `TestPlacementFeasible`: Tests progressive state transitions (`fwk.Wait` -> `fwk.Success`), early abortion on first/subsequent pod failures (`fwk.Unschedulable`), and non-gang fallback behavior.
