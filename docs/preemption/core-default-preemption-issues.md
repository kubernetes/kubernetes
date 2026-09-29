# Deep Dive: DefaultPreemption Algorithm, Victim Ordering, PDB Matching, and Scheduler Extender Issues

## Executive Summary

The `DefaultPreemption` plugin serves as Kubernetes' core preemption mechanism within the scheduling framework. Executing at the `PostFilter` extension point, it identifies, ranks, and evicts lower-priority victim pods to accommodate higher-priority pending workloads when cluster nodes lack sufficient allocatable capacity or fail scheduling filters.

While conceptually straightforward, the preemption pipeline encompasses a multi-phase distributed algorithm that interacts with custom scheduler extenders, PodDisruptionBudgets (PDBs), in-memory scheduling lifecycle states (Permit/PreBind), asynchronous API deletions, and scheduling queue wakeups. Over recent Kubernetes release cycles, subtle edge cases, race conditions, and algorithmic flaws were discovered across these subsystems.

This technical report provides an exhaustive, code-level analysis of recent critical bug fixes and architectural enhancements in the DefaultPreemption pipeline:
1. **Extender Processing & Empty Victims (PR #135486 / commit `cb33cf457d0`)**: Resolving default-preemption failures when using filter extenders by preserving placeholder empty-victim nodes for `ProcessPreemption`.
2. **PodDisruptionBudget (PDB) Matching Mechanics (PR #141785 / commit `82dead7c815`)**: Fixing the preemption algorithm ignoring empty PDB label selectors (`{}`) and mishandling unlabeled victim pods.
3. **Victim Ordering Determinism & Timestamp Invariants (PR #140999 / commits `24127a76250`, `8ffd4531cb6`, `df58fb32e60`)**: Enforcing strict weak ordering and determinism in `MoreImportantVictim` for unstarted pods and identical start timestamps.
4. **In-Memory Preemption, Asynchronous Execution & Framework Decoupling (PR #140054, PR #136613, PR #134927)**: Decoupling candidate evaluation from actuation, avoiding duplicate API calls on terminating pods, and eliminating preemptor pod starvation caused by silent in-memory preemption.

---

## 1. DefaultPreemption Architecture and Pipeline Lifecycle

### 1.1 The PostFilter Preemption Pipeline

When all nodes fail the `Filter` phase during a scheduling cycle, the scheduler triggers registered `PostFilter` plugins. `DefaultPreemption` executes across five distinct algorithmic phases:

```
+----------------------------------------------------------------------------------------------------+
|                                    DefaultPreemption Pipeline                                      |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 1: Preemption Eligibility Evaluation               |
                     | - PodEligibleToPreemptOthers (Priority, Policy != Never) |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 2: Candidate Node Discovery & Dry-Run Simulation   |
                     | - FindNodesThatFit / selectVictimsOnNode (Parallelized)  |
                     | - Minimal victim selection via simulateRemoval           |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 3: Scheduler Extender Preemption Processing        |
                     | - Evaluator.callExtenders -> HTTPExtender.ProcessPreempt |
                     | - Extenders add, trim, or filter candidate nodes/victims |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 4: Candidate Selection & Tie-Breaking              |
                     | - SelectCandidate / moreImportantVictimList              |
                     | - Pick node with lowest PDB violations & least disruption|
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 5: Preemption Actuation (Sync / Async)             |
                     | - Executor.prepareCandidate / prepareCandidateAsync      |
                     | - In-memory cancellation (Permit/PreBind) or API delete  |
                     | - Set NominatedNodeName & clear conflicting nominations  |
                     +----------------------------------------------------------+
```

### 1.2 Core Data Structures

* `preemption.Candidate`: Interface representing a nominated node along with its selected victims (`extenderv1.Victims`).
* `extenderv1.Victims`: Contains `Pods []*v1.Pod` to evict and `NumPDBViolations int64`.
* `preemption.Evaluator`: Stateless simulator responsible for candidate node discovery, dry-run filtering, victim minimalization, extender delegation, and candidate selection.
* `preemption.Executor`: State machine responsible for actuating preemption, managing asynchronous goroutines, patching `DisruptionTarget` status, issuing API `DELETE` calls, managing in-memory cancellations, and reactivating preemptors.

---

## 2. Issue 1: Extender Processing & Empty Victims (PR #135486 / commit `cb33cf457d0`)

### 2.1 The Problem & Root Cause

Scheduler extenders allow external webhooks to participate in scheduling decisions (e.g., custom resource managers for hardware accelerators, licenses, or network topologies). Extenders can implement:
1. **Filter Predicates**: Rejecting nodes that lack extender-managed resources.
2. **Preemption Processing (`ProcessPreemption`)**: Identifying specific victim pods that hold extender-managed resources on candidate nodes.

Prior to PR #135486, a critical algorithmic bug occurred when a node satisfied all **in-tree** filter plugins (such as `NodeResourcesFit`, `NodeAffinity`, `PodTopologySpread`) without needing to preempt any in-tree pods, but was rejected by an **extender filter predicate**:

```
Scheduling Cycle:
1. Filter Phase: In-tree plugins PASS on Node-1, but Extender-A FAILS Node-1 (out of GPU licenses).
2. PostFilter Preemption:
   - Evaluator runs in-tree DryRunPreemption on Node-1.
   - In-tree filters fit immediately with 0 victims (len(pods) == 0).
   - BUG: DryRunPreemption returned an error:
     "expected at least one victim pod on node Node-1"
     OR dropped Node-1 from the candidate list entirely!
   - RESULT: Node-1 was never passed to Extender-A's ProcessPreemption endpoint.
   - Preemption failed even though evicting a pod on Node-1 would free the GPU license!
```

Furthermore, in `callExtenders`, if a candidate node with an empty victim list was passed to an extender, returning an empty list was treated as an error (`"expected at least one victim pod on node %q"`) or stripped out, preventing downstream extenders in a chained configuration from processing the node.

### 2.2 The Algorithmic Fix

PR #135486 fundamentally updated the contract between the in-tree `Evaluator` and preemption-capable extenders:

#### A. Creating Placeholder Empty-Victim Candidates
In `pkg/scheduler/framework/preemption/preemption.go`, `DryRunPreemption` was modified so that if in-tree filters fit with zero victims (`len(pods) == 0`), but an extender that supports preemption and is interested in the pod exists (`ev.hasInterestedPreemptExtender(pod)`), the node is preserved as a **placeholder candidate**:

```go
// DryRunPreemption in pkg/scheduler/framework/preemption/preemption.go
if status.IsSuccess() && (len(pods) != 0 || ev.hasInterestedPreemptExtender(pod)) {
    victims := extenderv1.Victims{
        Pods:             pods,
        NumPDBViolations: int64(numPDBViolations),
    }
    c := &candidate{
        victims: &victims,
        name:    nodeInfo.Node().Name,
    }
    nodeList.add(c)
    return
}
if status.IsSuccess() && len(pods) == 0 {
    status = fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "No preemption victims found for incoming pod")
}
```

#### B. Placeholder Preservation in `callExtenders`
In `callExtenders`, the preemption engine now tracks which candidate nodes entered the extender call as empty placeholders using `emptyInputNodes := sets.New[string]()`.

```go
func (ev *Evaluator) callExtenders(logger klog.Logger, pod *v1.Pod, candidates []Candidate) ([]Candidate, *fwk.Status) {
    // ...
    for _, extender := range extenders {
        if !extender.SupportsPreemption() || !extender.IsInterested(pod) {
            continue
        }
        emptyInputNodes := sets.New[string]()
        for nodeName, victims := range victimsMap {
            if victims == nil || len(victims.Pods) == 0 {
                emptyInputNodes.Insert(nodeName)
            }
        }
        nodeNameToVictims, err := extender.ProcessPreemption(pod, victimsMap, nodeLister)
        if err != nil {
            if extender.IsIgnorable() {
                continue
            }
            return nil, fwk.AsStatus(err)
        }
        for nodeName, victims := range nodeNameToVictims {
            if victims == nil || len(victims.Pods) == 0 {
                if emptyInputNodes.Has(nodeName) {
                    // Keep placeholders for subsequent extenders. To reject a node,
                    // an extender should omit it from the returned map.
                    continue
                }
                if extender.IsIgnorable() {
                    delete(nodeNameToVictims, nodeName)
                    continue
                }
                return nil, fwk.AsStatus(fmt.Errorf("expected at least one victim pod on node %q", nodeName))
            }
        }
        victimsMap = nodeNameToVictims
    }
    
    // Prune candidates that still have no victims after all extenders have run
    var newCandidates []Candidate
    for nodeName, victims := range victimsMap {
        if victims == nil || len(victims.Pods) == 0 {
            logger.V(2).Info("Dropped node because no preemption extender reported victims", "node", klog.KRef("", nodeName))
            continue
        }
        newCandidates = append(newCandidates, &candidate{
            victims: victims,
            name:    nodeName,
        })
    }
    return newCandidates, nil
}
```

### 2.3 Formal Extender Contract

The API contract in `k8s.io/kube-scheduler/extender/v1` and `k8s.io/kube-scheduler/framework` was formalized:
1. **Empty Candidate Semantics**: A node mapping to `&extenderv1.Victims{Pods: []}` signifies that in-tree filter plugins fit without preemption, but the node requires extender victim selection for extender-managed resources.
2. **Extender Actions**:
   - **Add Victims**: The extender populates `Victims.Pods` with victim pods holding extender resources.
   - **Pass-Through**: The extender leaves the candidate empty so subsequent extenders in the chain can process it.
   - **Reject Node**: The extender omits the node key entirely from the returned `nodeNameToVictims` map.
3. **Post-Extender Cleanup**: Any candidate node that remains empty after all extenders have run is pruned from the candidate set.

---

## 3. Issue 2: PodDisruptionBudget (PDB) Matching Mechanics (PR #141785 / commit `82dead7c815`)

### 3.1 The Problem & Root Cause

PodDisruptionBudgets (PDBs) limit the number of concurrent disruptions for critical workloads. When selecting preemption victims, `DefaultPreemption` categorizes potential victims into PDB-violating and non-violating victims via `FilterVictimsWithPDBViolation`. The scheduler prioritizes candidates with zero or minimal PDB violations to protect high-availability services.

Prior to PR #141785, `FilterVictimsWithPDBViolation` contained two severe selector processing bugs in `pkg/scheduler/framework/preemption/util.go`:

```go
// BUGGY IMPLEMENTATION prior to PR #141785
func FilterVictimsWithPDBViolation[T Victim](victims []T, pdbs []*policy.PodDisruptionBudget) (...) {
    // ...
    podIsViolating := func(pod *v1.Pod) bool {
        if len(pod.Labels) == 0 { // BUG #1: Unlabeled pods immediately skipped!
            return false
        }
        for i, pdb := range pdbs {
            if pdb.Namespace != pod.Namespace {
                continue
            }
            selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
            if err != nil {
                continue
            }
            // BUG #2: selector.Empty() evaluated to true for empty selector {}, causing continue!
            if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
                continue
            }
            // ...
        }
    }
}
```

#### Bug Mechanism 1: Unconditional Skip of Unlabeled Pods
`if len(pod.Labels) == 0 { return false }` caused the scheduler to assume that any pod without labels could never violate a PDB. However, a PDB with an empty selector `{}` or a negative expression (e.g. `app DoesNotExist`) matches unlabeled pods. As a result, unlabeled pods covered by namespace-wide PDBs were evicted without tracking PDB violations.

#### Bug Mechanism 2: Misinterpreting `selector.Empty()`
In Kubernetes label selector mechanics:
- `metav1.LabelSelectorAsSelector(&metav1.LabelSelector{})` parses an empty label selector `{}` into `labels.Everything()`.
- For `labels.Everything()`, `selector.Empty()` returns `true`.
- In standard Kubernetes semantics, an empty selector selects **all objects** in the namespace.
- Conversely, a `nil` selector (`Selector: nil`) parses into `labels.Nothing()`, which matches no objects.

Because the code checked `if selector.Empty() || ... { continue }`, any PDB configured with `spec.selector: {}` (the standard pattern for matching all pods in a namespace) was skipped! The scheduler treated the PDB as matching zero pods, completely disabling PDB protection during preemption.

### 3.2 The Algorithmic Fix

PR #141785 removed both erroneous checks:

```go
// FIXED IMPLEMENTATION (PR #141785)
func FilterVictimsWithPDBViolation[T Victim](victims []T, pdbs []*policy.PodDisruptionBudget) (violatingVictims []ViolatingVictim[T], nonViolatingVictims []T) {
    pdbsAllowed := make([]int32, len(pdbs))
    podIsViolating := func(pod *v1.Pod) bool {
        for i, pdb := range pdbs {
            if pdb.Namespace != pod.Namespace {
                continue
            }
            selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
            if err != nil {
                // Invalid selector does not match the pod
                continue
            }
            // Correct matching: selector.Matches handles labels.Everything (empty selector)
            // and labels.Nothing (nil selector) correctly.
            if !selector.Matches(labels.Set(pod.Labels)) {
                continue
            }
            if pdbsAllowed[i] <= 0 {
                return true
            }
            pdbsAllowed[i]--
        }
        return false
    }
    // ...
}
```

### 3.3 Semantic Verification Matrix

| PDB Selector Specification | Parsed Selector Type | Pod Labels | `selector.Matches()` | Old Behavior | Fixed Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `spec.selector: {}` (Empty) | `labels.Everything()` | `app: frontend` | `true` | Skipped (No PDB violation) | **Violates PDB** if `DisruptionsAllowed == 0` |
| `spec.selector: {}` (Empty) | `labels.Everything()` | *(None / Unlabeled)* | `true` | Skipped (No PDB violation) | **Violates PDB** if `DisruptionsAllowed == 0` |
| `spec.selector: nil` (Nil) | `labels.Nothing()` | `app: frontend` | `false` | Skipped (No match) | Correctly ignored (No match) |
| `app DoesNotExist` | `Requirement(DoesNotExist)`| *(None / Unlabeled)* | `true` | Skipped (No PDB violation) | **Violates PDB** if `DisruptionsAllowed == 0` |

---

## 4. Issue 3: Victim Ordering Determinism & Timestamp Invariants (PR #140999 / commits `24127a76250`, `8ffd4531cb6`)

### 4.1 The Victim Sorting Hierarchy

During dry-run preemption on a candidate node, all lower-priority pods on the node are sorted by importance via `MoreImportantVictim(vi1, vi2)`. The scheduler evicts pods in reverse order of importance (least important pods first) and then reprieves unnecessary victims.

```
+-----------------------------------------------------------------------------+
|                     MoreImportantVictim Decision Hierarchy                  |
+-----------------------------------------------------------------------------+
                                       |
                                       v
                     [ 1. Compare Priority Values ]
                     - Higher Priority is MORE IMPORTANT
                                       | (If Equal)
                                       v
                     [ 2. Compare PDB Violations ]
                     - Pod violating PDB is MORE IMPORTANT (avoid evicting)
                                       | (If Equal)
                                       v
                     [ 3. Compare Number of Pods ]
                     - Larger group size is MORE IMPORTANT
                                       | (If Equal)
                                       v
                     [ 4. Compare Earliest Start Time ]
                     - Earlier StartTime is MORE IMPORTANT (preserve running)
                                       | (If Equal or Unstarted)
                                       v
                     [ 5. Tie-Breaker: Pod UID Identity ]
                     - Lower UID string is MORE IMPORTANT (Strict Determinism)
```

Two distinct bugs corrupted steps 4 and 5 of this hierarchy.

### 4.2 Problem A: Unstarted Pods & `CreationTimestamp` Inversion (commit `24127a76250`)

#### Root Cause
In `pkg/scheduler/util/utils.go`, `GetPodStartTime` calculates the start time of a pod:
```go
// PREVIOUS IMPLEMENTATION:
func GetPodStartTime(pod *v1.Pod) *metav1.Time {
    if pod.Status.StartTime != nil {
        return pod.Status.StartTime
    }
    return &pod.CreationTimestamp
}
```
When pods are assumed or bound but have not yet started on a node, `pod.Status.StartTime` is `nil`. Falling back to `&pod.CreationTimestamp` created a severe inversion:
1. An unstarted pod `P_unstarted` created 10 minutes ago has `CreationTimestamp = T-10m`.
2. A running pod `P_running` created 2 minutes ago and started immediately has `StartTime = T-2m`.
3. The comparator judged `P_unstarted` as older (`T-10m < T-2m`) and therefore **more important** than `P_running`!
4. The scheduler evicted the actively running workload `P_running` instead of the pending/unstarted pod `P_unstarted`.

Furthermore, fallback to `time.Now()` dynamically during sorting violated the mathematical definition of strict weak ordering ($A < B \land B < A$ across comparisons), leading to non-deterministic quicksort behavior.

#### The Fix
Commit `24127a76250` introduced a static sentinel timestamp:
```go
var maxPodStartTime = metav1.NewTime(time.Date(9999, time.December, 31, 23, 59, 59, 0, time.UTC))

func GetPodStartTime(pod *v1.Pod) *metav1.Time {
    if pod.Status.StartTime != nil {
        return pod.Status.StartTime
    }
    // Treat unstarted pods as newer than all started pods without generating
    // a dynamic timestamp during sorting.
    return &maxPodStartTime
}
```
This guarantees that any running pod with an actual `StartTime` is strictly older and preserved over unstarted pods.

### 4.3 Problem B: Non-Deterministic Tie-Breaking on Equal Start Times (commit `8ffd4531cb6` / `df58fb32e60`)

#### Root Cause
When multiple unstarted pods (all sharing `maxPodStartTime`) or pods started at the exact same second were compared, `t1.Equal(t2)` was true. In Go's `sort.Slice` / `sort.SliceStable`:
```go
// OLD CODE in util.go
func MoreImportantVictim(vi1, vi2 Victim) bool {
    // ...
    return vi1.EarliestStartTime().Before(vi2.EarliestStartTime())
}
```
When `t1.Equal(t2)`, `t1.Before(t2)` returned `false`. If `vi1` and `vi2` had equal priority, equal PDB status, and equal start times, the comparison returned `false` in both directions. Consequently, victim selection ordering depended entirely on slice order in cache iteration, leading to non-deterministic preemption across scheduler restarts or cluster architectures.

#### The Fix
PR #140999 introduced a deterministic fallback by comparing the unique identifier (UID) of the representative pod:

```go
// FIXED IMPLEMENTATION in pkg/scheduler/framework/preemption/util.go
func MoreImportantVictim(vi1, vi2 Victim) bool {
    if vi1.Priority() != vi2.Priority() {
        return vi1.Priority() > vi2.Priority()
    }
    if vi1.ViolatingCount() != vi2.ViolatingCount() {
        return vi1.ViolatingCount() > vi2.ViolatingCount()
    }
    if len(vi1.Pods()) != len(vi2.Pods()) {
        return len(vi1.Pods()) > len(vi2.Pods())
    }

    t1, t2 := vi1.EarliestStartTime(), vi2.EarliestStartTime()
    if t1 != nil && t2 != nil && !t1.Equal(t2) {
        return t1.Before(t2)
    }

    return podIdentityLess(vi1, vi2)
}

func podIdentityLess(vi1, vi2 Victim) bool {
    return vi1.Pods()[0].GetPod().UID < vi2.Pods()[0].GetPod().UID
}
```

---

## 5. Issue 4: In-Memory Preemption, Asynchronous Execution & Framework Decoupling

### 5.1 Architectural Decoupling: Evaluator vs. Executor (PR #136613)

Prior to PR #136613, `pkg/scheduler/framework/preemption/preemption.go` combined simulation logic (finding nodes, dry-running plugins, ranking candidates) with execution logic (API calls, status patching, goroutines).

PR #136613 restructured the subsystem into a clean separation of concerns:
- **`Candidate` & `candidateList` (`candidate.go`)**: Encapsulates candidate nodes and provides a lock-free, atomic collector (`newCandidateList`) for concurrent node evaluations in `findCandidates`.
- **`Evaluator` (`preemption.go`)**: Pure evaluation engine that identifies candidate nodes without mutating cluster state.
- **`Executor` (`executor.go`)**: State machine that orchestrates synchronous and asynchronous preemption actuation.

```
                    +--------------------------------+
                    |        PreemptionPlugin        |
                    +--------------------------------+
                               /          \
                              /            \
                             v              v
               +-------------------+  +-------------------+
               |     Evaluator     |  |     Executor      |
               | (Dry-Run, Extender|  | (Actuation, Async |
               |  Victim Selection)|  |  Status, In-Memory|
               +-------------------+  +-------------------+
```

### 5.2 Skipping Pods with `DeletionTimestamp` (PR #134927)

When a node contains pods already undergoing graceful termination (`DeletionTimestamp != nil`), the scheduler's dry-run simulation accounts for their impending removal.

However, during execution (`prepareCandidate` and `prepareCandidateAsync`), attempting to call `DeletePod` or patch `DisruptionTarget` status on already-terminating pods produced superfluous API calls, `404 Not Found` or `409 Conflict` errors, and delayed preemption actuation.

PR #134927 added explicit checks in `prepareCandidate` and `prepareCandidateAsync`:
```go
victimPods := make([]*v1.Pod, 0, len(c.Victims().Pods))
for _, victim := range c.Victims().Pods {
    if victim.DeletionTimestamp != nil {
        logger.V(2).Info("Victim Pod is already being deleted, skipping the API call for it",
            "preemptor", klog.KObj(pod), "node", c.Name(), "victim", klog.KObj(victim))
        continue
    }
    victimPods = append(victimPods, victim)
}
if len(victimPods) == 0 {
    cancel()
    return
}
```

### 5.3 Asynchronous In-Memory Preemption & Preemptor Starvation (PR #140054)

#### The Asynchronous Preemption State Machine
Under `EnableAsyncPreemption: true`, the scheduler performs victim deletions in a background goroutine to avoid blocking the scheduling queue:
1. The preemptor pod is added to the `executor.preempting` set to prevent it from re-entering scheduling cycles while preemption is in-flight.
2. All victims except the last one are deleted in parallel.
3. The preemptor records the last victim in `lastVictimsPendingPreemption[preemptor.UID]`.
4. The preemptor pod is moved to the `unschedulable` queue.
5. When the API server processes the last victim's deletion, the scheduler's informer receives the Pod `DELETE` event. The `PreEnqueue` plugin verifies that the deleted pod matches `lastVictimsPendingPreemption`, removes the entry, and moves the preemptor pod to `activeQ`.

```
                    +-------------------------------------------------------------+
                    |                Async Preemption Goroutine                   |
                    +-------------------------------------------------------------+
                                                   |
                                                   v
                                 [ Evict Victims 0 .. N-2 in Parallel ]
                                                   |
                                                   v
                             [ Set lastVictimsPendingPreemption = Victim N-1 ]
                                                   |
                                                   v
                                [ Evict Last Victim (Victim N-1) ]
                                  /                             \
                (API Server Deletion)                 (In-Memory Preemption)
                         |                                      |
                         v                                      v
            API Server emits DELETE Watch event     NO API DELETE event produced!
                         |                                      |
                         v                                      v
          Informer event triggers PreEnqueue      Preemptor permanently stuck in
                         |                        unschedulable queue until flush!
                         v                                      |
          Preemptor moved to ActiveQ                            v
                                                  [ FIX: Explicit fh.Activate() ]
```

#### The In-Memory Preemption Bug
Kubernetes supports in-memory preemption for pods in intermediate scheduling lifecycle states:
- **`WaitingPod` (Permit Phase)**: A victim waiting in the permit phase is cancelled via `waitingPod.Preempt(pluginName, "preempted")` without an API call.
- **`PodInPreBind` (PreBind Phase)**: A victim binding context is cancelled via `podInPreBind.CancelPod(...)` without an API call.

**The Failure Mode**: When the **last victim** (`Victim N-1`) was preempted in-memory, `skipAPICall = true` was set. No `DELETE` request was sent to the API server, and no watch event was ever generated. As a result, the `PreEnqueue` event handler was never triggered, leaving the preemptor pod indefinitely stranded in the `unschedulable` queue until a periodic queue flush occurred.

#### The Fix (commits `8d5731c016d`, `de33b423915`, `bd6cae4a3fe`)
1. **Standardized In-Memory Return**: `PreemptPod` was updated to return `(preemptedInMemory bool, err error)`:
   ```go
   PreemptPod func(ctx context.Context, c Candidate, preemptor ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error)
   ```
2. **Immediate Preemptor Reactivation**: In `prepareCandidateAsync`, if the last victim was preempted in-memory (`preemptedLastVictimInMemory == true`), the completion `defer` block explicitly activates the preemptor pods:
   ```go
   defer func() {
       if result == metrics.GoroutineResultError || preemptedLastVictimInMemory {
           e.fh.Activate(logger, preemptor.Pods())
       }
   }()
   ```
3. **Race Condition Prevention**: `e.preempting` set is unlocked and cleared *before* `e.fh.Activate` runs, preventing a race condition where the activated pod is rejected because its UID is still listed in `preempting`.

---

## 6. Comprehensive Summary & Cross-Cutting Impact

### 6.1 Issue Comparison Matrix

| Area | Pull Request / Commits | Component | Root Cause | Failure Mode | Architectural Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scheduler Extenders** | PR #135486 (`cb33cf457d0`) | `Evaluator.callExtenders`, `DryRunPreemption` | Nodes passing in-tree filters without in-tree victims were dropped or errored before extenders ran. | Preemption failed on nodes managed by custom resource extenders (e.g. GPUs, licenses). | Pass placeholder empty-victim candidates to preemption-capable extenders; prune empty candidates post-extender. |
| **PDB Selectors** | PR #141785 (`82dead7c815`) | `util.FilterVictimsWithPDBViolation` | `if len(pod.Labels) == 0` skipped unlabeled pods; `selector.Empty()` skipped empty `{}` selectors. | Wildcard PDBs (`{}`) and negative selectors ignored; PDBs violated during preemption. | Removed erroneous checks; rely strictly on `selector.Matches()` across all pods and selectors. |
| **Victim Ordering (Unstarted)** | PR #140999 (`24127a76250`) | `util.GetPodStartTime` | Unstarted pods fell back to `CreationTimestamp`, appearing older than running pods. | Running workloads preempted before unstarted/pending pods. | Unstarted pods assigned static `maxPodStartTime` (year 9999), prioritizing running workloads. |
| **Victim Ordering (Determinism)** | PR #140999 (`8ffd4531cb6`, `df58fb32e60`) | `util.MoreImportantVictim` | Incomplete strict weak ordering when start times were equal (`t1.Equal(t2)`). | Non-deterministic victim eviction ordering across runs and architectures. | Added deterministic tie-breaker comparing representative pod UID (`podIdentityLess`). |
| **Framework Decoupling** | PR #136613 (`677c8ec05fb`) | `preemption.Evaluator`, `preemption.Executor` | Monolithic preemption codebase mixing simulation and API mutation. | Architectural rigidity; inability to cleanly test and extend async actuation. | Split into `Evaluator` (simulation), `Executor` (actuation), and lock-free `candidateList`. |
| **Terminating Victims** | PR #134927 (`4e5220939e5`) | `Executor.prepareCandidate*` | Redundant API delete calls on pods already having `DeletionTimestamp`. | API server request amplification and `404/409` errors during preemption. | Filter out pods with `DeletionTimestamp != nil` prior to issuing deletion calls. |
| **In-Memory Preemption** | PR #140054 (`bd6cae4a3fe`, `de33b423915`, `8d5731c016d`) | `Executor.prepareCandidateAsync`, `PreemptPod` | In-memory preemption of last victim generated no API delete event for `PreEnqueue`. | Preemptor pod permanently stuck in `unschedulable` queue. | Track `preemptedInMemory` in `PreemptPod` and trigger immediate `fh.Activate()` on completion. |

---

## 7. Verification and Testing Guidelines

When developing or modifying `DefaultPreemption` plugins and extensions, ensure the following test matrices are satisfied:

1. **Extender Integration Tests**:
   - Verify extenders receive empty placeholder candidates when in-tree plugins fit without preemption.
   - Verify chained extenders preserve or omit empty candidates appropriately.
   - Verify error status `UnschedulableAndUnresolvable` is returned when no extenders exist and no victims fit.

2. **PDB Selection Unit Tests**:
   - Test empty selector `{}` (`labels.Everything()`) against both labeled and unlabeled pods.
   - Test nil selector (`labels.Nothing()`) against both labeled and unlabeled pods.
   - Test negative requirement operators (`LabelSelectorOpDoesNotExist`) against unlabeled pods.

3. **Victim Sorting Property Tests**:
   - Verify strict weak ordering: `MoreImportantVictim(A, B)` and `MoreImportantVictim(B, A)` are never both `true`.
   - Verify determinism across slice permutations for unstarted pods and pods with identical timestamps.
   - Verify that running pods with non-nil `StartTime` are strictly sorted before unstarted pods regardless of `CreationTimestamp`.

4. **Async & In-Memory Preemption Tests**:
   - Verify that preemptors are immediately activated when the last victim is a `WaitingPod` (Permit phase) or `PodInPreBind` (PreBind phase).
   - Verify that `executor.preempting` set is completely cleared prior to calling `fh.Activate()`.
   - Verify that victim pods with `DeletionTimestamp != nil` do not trigger duplicate API deletion calls.
