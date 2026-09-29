# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/podtopologyspread`

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, state management mechanics, performance optimizations, and testing strategies for the `PodTopologySpread` scheduler plugin located at `pkg/scheduler/framework/plugins/podtopologyspread`.

---

## 1. High-Level Overview & Core Role

The `PodTopologySpread` plugin (`names.PodTopologySpread`) controls the distribution of pods across failure domains (e.g. regions, zones, nodes, racks) according to declared `topologySpreadConstraints` to ensure high availability and prevent single-point failures.

### Key Objectives:
1. **Hard Spreading (`WhenUnsatisfiable: DoNotSchedule`)**: Filters out nodes that would cause the skew between topology domains to exceed `maxSkew`.
2. **Soft Spreading (`WhenUnsatisfiable: ScheduleAnyway`)**: Prioritizes nodes in under-populated topology domains to balance distribution gradually.
3. **System Default Constraints**: Applies default spread rules across `kubernetes.io/hostname` (maxSkew=3) and `topology.kubernetes.io/zone` (maxSkew=5) when pod-level constraints are omitted.
4. **Node Inclusion Policies**: Dynamically respects or ignores node affinities (`NodeAffinityPolicy`) and node taints (`NodeTaintsPolicy`) when evaluating topology domain capacities.
5. **Minimum Domain Protection (`MinDomains`)**: Treats the global minimum pod count as `0` if the count of available topology domains is below `minDomains`.

---

## 2. Implemented Extension Points

`PodTopologySpread` implements the following framework interfaces:

| Extension Point | Interface | Phase & Execution Mode | Description |
|---|---|---|---|
| **PreFilter** | `fwk.PreFilterPlugin` | Scheduling cycle (synchronous) | Gathers hard spread constraints, scans nodes in parallel, builds `TpValueToMatchNum`, and tracks `CriticalPaths`. |
| **PreFilterExtensions** | `fwk.PreFilterExtensions` | Preemption simulation | Implements `AddPod` and `RemovePod` to incrementally update topology counts and critical paths during victim eviction. |
| **Filter** | `fwk.FilterPlugin` | Scheduling cycle (parallelized) | Evaluates node skew against `maxSkew` using precomputed global minimums and domain pod counts. |
| **PreScore** | `fwk.PreScorePlugin` | Scheduling cycle (parallelized over nodes) | Computes soft spread constraint pod distributions and calculates logarithmic topology normalizing weights. |
| **Score** | `fwk.ScorePlugin` | Scheduling cycle (parallelized) | Computes raw score per candidate node based on domain pod counts, `maxSkew`, and topology normalizing weights. |
| **NormalizeScore** | `fwk.ScoreExtensions` | Scheduling cycle (synchronous) | Inverts and normalizes scores to `[0, MaxNodeScore]` so under-utilized domains receive the highest score. |
| **SignPlugin** | `fwk.SignPlugin` | Queueing / Batching | Marks pods with topology spread constraints as unsignable to preserve domain skew accuracy. |
| **EnqueueExtensions** | `fwk.EnqueueExtensions` | Queueing / Informers | Registers cluster events for assigned pods, target pod changes, and node label/taint modifications. |

---

## 3. Topology Spread Calculations & Filtering Algorithm

```
                             [ PreFilter Phase ]
                                      │
               Count matching pods per topology domain across all nodes
                                      │
                    Populate TpValueToMatchNum & CriticalPaths
                                      │
                                      ▼
                             [ Filter Evaluation ]
                                      │
          For each candidate node & constraint:
          ├─ matchNum     = TpValueToMatchNum[constraint][node.TopologyValue]
          ├─ selfMatchNum = 1 if pod matches constraint.Selector, else 0
          ├─ minMatchNum  = CriticalPaths[constraint][0].MatchNum (or 0 if domains < minDomains)
          │
          └─ skew = (matchNum + selfMatchNum) - minMatchNum
             IF skew > maxSkew:
                 REJECT NODE (Unschedulable)
```

### 3.1. Critical Paths Optimization (`criticalPaths`)
Instead of recalculating the global minimum pod count across all topology domains for every node evaluation or during preemption simulations, the plugin maintains a fixed 2-element structure:

```go
type criticalPaths [2]struct {
    TopologyValue string
    MatchNum      int
}
```
- `CriticalPaths[i][0].MatchNum`: Always holds the minimum matching pod count for constraint `i`.
- `CriticalPaths[i][1].MatchNum`: Holds the second smallest (or equal) matching count.
- **Why 2 paths suffice**: Preemption evaluates evicting pods on a single node at a time. If evicting pods from the current minimum domain reduces its count or changes the minimum, `[2]criticalPath` guarantees that the true global minimum can be maintained with $O(1)$ operations without re-scanning the entire cluster.

### 3.2. Minimum Domains (`minDomains`)
When `minDomains` is configured (e.g. `minDomains: 3`) and the cluster currently has fewer than 3 active domains matching the topology key, the effective global minimum is clamped to `0`:
$$\text{minMatchNum} = \begin{cases} 0 & \text{if } |\text{domains}| < \text{minDomains} \\ \text{CriticalPaths}[i][0].\text{MatchNum} & \text{otherwise} \end{cases}$$

### 3.3. PreFilterExtensions (`AddPod` / `RemovePod`)
When the preemption engine simulates pod removals or additions:
1. `RemovePod`: Decrements `TpValueToMatchNum[i][v]` by 1 and calls `CriticalPaths[i].update(v, newCount)`.
2. `AddPod`: Increments `TpValueToMatchNum[i][v]` by 1 and updates `CriticalPaths`.

---

## 4. Scoring Architecture & Inverted Normalization

Soft constraints (`WhenUnsatisfiable: ScheduleAnyway`) prioritize nodes in domains that currently host fewer matching pods.

### 4.1. PreScore & Logarithmic Topology Normalizing Weight
1. `initPreScoreState`: Determines domain sizes and sets up `TopologyValueToPodCounts`.
2. **Logarithmic Normalizing Weight**:
   $$\text{TopologyNormalizingWeight} = \ln(\text{domainCount} + 2)$$
   This prevents larger topology groupings (e.g. zones with 100 nodes) from dominating smaller topology groupings (e.g. racks with 4 nodes) when summing across multiple constraints.
3. **Parallel Pod Counting**: Counts matching pods across all cluster nodes using `atomic.AddInt64`.

### 4.2. Score Calculation
For each soft constraint matching candidate node $n$:
$$\text{rawScore}(n) = \sum_{c} \left( \text{cnt}(c, n) \times \text{TopologyNormalizingWeight}_c + (\text{maxSkew}_c - 1) \right)$$
*(For hostname-scoped constraints, $\text{cnt}$ is counted directly on candidate node $n$ during `Score`)*.

### 4.3. Inverted NormalizeScore
Because lower pod counts represent better placement targets, `NormalizeScore` inverts the scores:
$$\text{Score}(n) = \text{MaxNodeScore} \times \frac{\text{maxScore} + \text{minScore} - \text{rawScore}(n)}{\text{maxScore}}$$
Nodes with the lowest matching pod counts receive `MaxNodeScore` (100), while nodes with the highest counts receive the lowest score.

---

## 5. State Management & `CycleState` Usage

### Data Structures:

#### `preFilterState` (`preFilterStateKey = "PreFilterPodTopologySpread"`)
```go
type preFilterState struct {
    Constraints       []topologySpreadConstraint
    CriticalPaths     []*criticalPaths           // Index: constraintID
    TpValueToMatchNum []map[string]int           // Index: constraintID, Key: topologyValue
}
```
- **Cloning Behavior**: Clones each `criticalPaths` array and deep-copies each `TpValueToMatchNum` map via `maps.Clone(tpMap)` to allow isolated preemption branch simulations.

#### `preScoreState` (`preScoreStateKey = "PreScorePodTopologySpread"`)
```go
type preScoreState struct {
    Constraints               []topologySpreadConstraint
    IgnoredNodes              sets.Set[string]
    TopologyValueToPodCounts  []map[string]*int64
    TopologyNormalizingWeight []float64
}
```
- **Cloning Behavior**: `Clone()` returns `s` as score state is read-only.

---

## 6. Advanced Features & Inclusion Policies

1. **`MatchLabelKeys` (`EnableMatchLabelKeysInPodTopologySpread`)**:
   - Allows Pods to dynamically incorporate their own label values (e.g. `pod-template-hash`) into the spread selector without hardcoding them into pod specs.
2. **`NodeInclusionPolicy` (`EnableNodeInclusionPolicyInPodTopologySpread`)**:
   - `NodeAffinityPolicy`: `Honor` (default) considers only nodes matching pod's node affinity/selectors when counting domain distribution. `Ignore` counts all nodes.
   - `NodeTaintsPolicy`: `Honor` excludes nodes with untolerated taints. `Ignore` (default) includes tainted nodes in topology domain counts.
3. **Smart Event Queueing Hints (`EventsToRegister`)**:
   - `AssignedPod` (Add, UpdatePodLabel, Delete): Requeues pod if an assigned pod in the same namespace matching the spread selector is modified or deleted.
   - `TargetPod` (UpdatePodLabel, UpdatePodToleration): Requeues if pod acquires new tolerations (for `NodeTaintsPolicy: Honor`) or labels.
   - `Node` (Add, Delete, UpdateNodeLabel, UpdateNodeTaint): Requeues if topology keys or taints on matching nodes change.

---

## 7. Testing & Verification Guide

### Unit Tests
Execute unit tests for `podtopologyspread`:
```bash
cd /home/debian/work
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/podtopologyspread/...
```

### Key Test Scenarios:
- `TestPodTopologySpreadPreFilter`: Verifies critical path computation, `minDomains` evaluation, and map initialization.
- `TestPodTopologySpreadFilter`: Tests hard skew boundaries, missing topology labels, and node inclusion policies.
- `TestPodTopologySpreadScore`: Asserts logarithmic weight normalization, hostname fast-path scoring, and inverted score scaling.
- `TestPreFilterExtensions`: Validates incremental `AddPod` / `RemovePod` state transitions during preemption simulation.
