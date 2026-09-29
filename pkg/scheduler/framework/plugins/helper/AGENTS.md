# Scheduler Plugin Helpers (`pkg/scheduler/framework/plugins/helper`)

This guide provides an architectural overview, mathematical algorithms, tree traversal mechanics, controller label inference, and testing strategies for the shared utility functions in `pkg/scheduler/framework/plugins/helper`.

---

## 1. High-Level Purpose & Scope

The `helper` package contains common utility functions, mathematical modeling routines, and domain helpers used across multiple scheduler framework plugins. It unifies recurring patterns such as score normalization, broken-linear scoring curves, workload hierarchy traversal, controller label selector deduction, and taint filtering.

### Core Modules:
1. **Score Normalization (`normalize_score.go`)**: Scales raw scores linearly to a target range `[0, maxPriority]` with support for inverted/reverse scoring.
2. **Broken Linear Scoring Curves (`shape_score.go`)**: Evaluates piecewise continuous linear functions defined by utilization-to-score coordinate pairs.
3. **Workload & PodGroup Hierarchy (`podgroup.go`)**: Implements group identity matching and recursive iterator-based traversal of CompositePodGroup trees.
4. **Controller Label Selector Deduction (`spread.go`)**: Infers merged label selectors from owner controllers (ReplicationControllers, ReplicaSets, StatefulSets) and matching Services for spreading algorithms.
5. **Taint Effect Filtering (`taint.go`)**: Filters node taints to isolate hard scheduling rejections (`NoSchedule`, `NoExecute`).

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/helper/
├── normalize_score.go       # DefaultNormalizeScore implementation
├── normalize_score_test.go  # Unit tests for score normalization and reversal
├── shape_score.go           # FunctionShape and BuildBrokenLinearFunction
├── podgroup.go              # MatchingSchedulingGroup and GetPodGroupStates iterator
├── podgroup_test.go         # Unit tests for hierarchy traversal and cycle bounds
├── spread.go                # DefaultSelector and GetPodServices
├── spread_test.go           # Unit tests for controller and service selector resolution
├── taint.go                 # DoNotScheduleTaintsFilterFunc
├── taint_test.go            # Unit tests for taint effect filtering
└── AGENTS.md                # This agent documentation
```

---

## 3. Detailed Component Implementations

### 3.1. Score Normalization (`DefaultNormalizeScore`)

Scales raw integer scores from $[0, \max(\text{scores})]$ to $[0, \text{maxPriority}]$:

```go
func DefaultNormalizeScore(maxPriority int64, reverse bool, scores fwk.NodeScoreList) *fwk.Status
```

- **Algorithm**:
  1. Finds $\text{maxCount} = \max_{i}(\text{scores}[i].\text{Score})$.
  2. If $\text{maxCount} == 0$:
     - If `reverse == true`, assigns $\text{scores}[i].\text{Score} = \text{maxPriority}$ for all nodes.
     - Returns immediately.
  3. For each node score:
     $$\text{score} = \left\lfloor \frac{\text{maxPriority} \times \text{score}}{\text{maxCount}} \right\rfloor$$
     $$\text{if reverse}: \quad \text{score} = \text{maxPriority} - \text{score}$$
- **Use Cases**: Used by plugins like `NodeResourcesBalancedAllocation`, `InterPodAffinity`, and `NodeAffinity` during `NormalizeScore`.

---

### 3.2. Broken Linear Function Shaping (`shape_score.go`)

Constructs a continuous piecewise linear scoring function from arbitrary coordinate points:

```go
type FunctionShape []FunctionShapePoint
type FunctionShapePoint struct {
    Utilization int64 // X-axis coordinate
    Score       int64 // Y-axis coordinate
}

func BuildBrokenLinearFunction(shape FunctionShape) func(int64) int64
```

- **Piecewise Interpolation Formula**:
  For an input utilization $p$:
  - If $p \le \text{shape}[0].\text{Utilization} \implies \text{shape}[0].\text{Score}$
  - If $p \ge \text{shape}[n-1].\text{Utilization} \implies \text{shape}[n-1].\text{Score}$
  - For $p$ between segment $[i-1, i]$:
    $$\text{score} = \text{Score}_{i-1} + \frac{(\text{Score}_i - \text{Score}_{i-1}) \times (p - \text{Utilization}_{i-1})}{\text{Utilization}_i - \text{Utilization}_{i-1}}$$
- **Use Cases**: Used in custom resource allocation curves (`RequestedToCapacityRatio`) for bin packing or custom provisioning thresholds.

---

### 3.3. Workload & PodGroup Hierarchy (`podgroup.go`)

Provides utilities for generic workloads and composite pod group scheduling:

1. **`MatchingSchedulingGroup(pod1, pod2 *v1.Pod) bool`**:
   Verifies whether two pods share identical namespaces and non-nil `Spec.SchedulingGroup.PodGroupName`.

2. **`GetPodGroupStates(sharedLister fwk.SharedLister, rootEntityKey fwk.EntityKey) iter.Seq2[fwk.PodGroupState, error]`**:
   - Uses Go's `iter.Seq2` generator sequence to yield all leaf `PodGroupState` objects under a composite tree.
   - **Recursion Guard**: Guarantees recursion terminates by enforcing `depth < schedulingapi.WorkloadMaxTreeDepth` (yielding an error if cyclic or excessively deep hierarchies are detected).

---

### 3.4. Controller Selector Deduction (`spread.go`)

Deduces the combined label selector representing all pods in the same logical application group:

```go
func DefaultSelector(
    pod *v1.Pod,
    sl corelisters.ServiceLister,
    cl corelisters.ReplicationControllerLister,
    rsl appslisters.ReplicaSetLister,
    ssl appslisters.StatefulSetLister,
) labels.Selector
```

- **Resolution Order**:
  1. Finds all Services matching `pod.Labels` via `GetPodServices` and merges their `service.Spec.Selector`.
  2. Inspects `metav1.GetControllerOfNoCopy(pod)`.
  3. If owned by a `ReplicationController`, `ReplicaSet`, or `StatefulSet`, retrieves the owner controller and merges its `Spec.Selector`.
  4. Returns the unified `labels.Selector`.

---

### 3.5. Taint Filtering (`taint.go`)

```go
func DoNotScheduleTaintsFilterFunc() func(t *v1.Taint) bool
```

- Returns a predicate that selects taints with `Effect == v1.TaintEffectNoSchedule` or `Effect == v1.TaintEffectNoExecute`, used when validating node schedulability against pod tolerations.

---

## 4. Test Fixtures & Unit Testing

- **`normalize_score_test.go`**: Tests standard scaling, zero-max edge cases, and reverse normalization.
- **`podgroup_test.go`**: Tests single-level pod groups, multi-level composite trees, missing entity error handling, and `WorkloadMaxTreeDepth` recursion limits.
- **`spread_test.go`**: Tests label selector deduction across pods owned by ReplicaSets, StatefulSets, ReplicationControllers, and multiple overlapping Services.
- **`taint_test.go`**: Validates filtering of `NoSchedule`, `NoExecute`, and `PreferNoSchedule` taints.
