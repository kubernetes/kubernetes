# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/nodeaffinity`

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, state management mechanics, performance optimizations, and testing strategies for the `NodeAffinity` scheduler plugin located at `pkg/scheduler/framework/plugins/nodeaffinity`.

---

## 1. High-Level Overview & Core Role

The `NodeAffinity` plugin (`names.NodeAffinity`) is responsible for selecting candidate nodes that satisfy a Pod's node selector and node affinity requirements, as well as scoring feasible nodes according to preferred node affinity terms and scheduler-enforced node affinity.

### Key Objectives:
1. **Node Selector Matching**: Enforces `.spec.nodeSelector` key-value pairs against candidate node labels.
2. **Hard Node Affinity (`requiredDuringSchedulingIgnoredDuringExecution`)**: Evaluates `NodeSelectorTerms` expressions (`In`, `NotIn`, `Exists`, `DoesNotExist`, `Gt`, `Lt`) against node labels and node fields (`metadata.name`).
3. **Soft Node Affinity (`preferredDuringSchedulingIgnoredDuringExecution`)**: Prioritizes nodes based on matching weights assigned to `PreferredSchedulingTerm` entries.
4. **Scheduler-Enforced Affinity (`AddedAffinity`)**: Allows cluster administrators or profiles to inject default mandatory selectors and preferred scheduling terms via `config.NodeAffinityArgs`.

---

## 2. Implemented Extension Points

`NodeAffinity` implements the following framework interfaces:

| Extension Point | Interface | Phase & Execution Mode | Description |
|---|---|---|---|
| **PreFilter** | `fwk.PreFilterPlugin` | Scheduling cycle (synchronous) | Pre-parses required affinity terms and extracts specific node names for candidate narrowing. Skips filter if no rules apply. |
| **Filter** | `fwk.FilterPlugin` | Scheduling cycle (parallelized) | Evaluates candidate nodes against scheduler-enforced selectors and the Pod's required selector/affinity terms. |
| **PreScore** | `fwk.PreScorePlugin` | Scheduling cycle (synchronous) | Pre-parses preferred affinity terms into cached structures in `CycleState`. Skips scoring if no preferred terms exist. |
| **Score** | `fwk.ScorePlugin` | Scheduling cycle (parallelized) | Computes the sum of weights of matched preferred terms from the Pod and scheduler configuration. |
| **NormalizeScore** | `fwk.ScoreExtensions` | Scheduling cycle (synchronous) | Normalizes scores across candidate nodes to the standard `[0, MaxNodeScore]` range using `helper.DefaultNormalizeScore`. |
| **SignPlugin** | `fwk.SignPlugin` | Queueing / Batching | Produces `SignFragment` entries for node affinity and node selectors for opportunistic batching. |
| **EnqueueExtensions** | `fwk.EnqueueExtensions` | Queueing / Informers | Registers cluster events with targeted `QueueingHint` functions to re-enqueue pods when relevant node labels change. |

---

## 3. Algorithm & Lifecycle Walkthrough

### 3.1. PreFilter (`PreFilter`)
1. **Deferred Pod Resize Check**: If `EnableInPlacePodVerticalScalingSchedulerPreemption` is enabled and `resource.IsPodResizeDeferred(pod)` is true, returns `framework.NewStatus(framework.Skip)`.
2. **Eligibility Check**: If the Pod declares no node selector, no required node affinity, and no `addedNodeSelector` is configured in plugin args, returns `framework.NewStatus(framework.Skip)` to bypass the `Filter` phase entirely across all nodes.
3. **Parse & Cache State**: Parses required node affinity using `nodeaffinity.GetRequiredNodeAffinity(pod)` and stores `preFilterState` into `CycleState` with key `"PreFilterNodeAffinity"`.
4. **Target Node Pre-filtering**:
   - Inspects `NodeSelectorTerms.MatchFields` for `metav1.ObjectNameField` with operator `NodeSelectorOpIn`.
   - Computes intersections of node names within terms (ANDed constraints) and unions across terms (ORed constraints).
   - If conflicting node names result in an empty set, immediately returns `framework.NewStatus(framework.UnschedulableAndUnresolvable, errReasonConflict)`.
   - If a specific subset of node names is identified, returns `&fwk.PreFilterResult{NodeNames: nodeNames}` to restrict the scheduler's node evaluation set.

### 3.2. Filter (`Filter`)
1. **Enforced Node Selector**: Verifies `pl.addedNodeSelector.Match(node)`. If false, fails with `framework.UnschedulableAndUnresolvable` (`"node(s) didn't match scheduler-enforced node affinity"`).
2. **Pod Required Affinity**: Reads `preFilterState` from `CycleState` (falling back to on-the-fly parsing if PreFilter was omitted).
3. **Evaluation**: Checks `s.requiredNodeSelectorAndAffinity.Match(node)`. Returns `nil` on match, or `framework.UnschedulableAndUnresolvable` (`"node(s) didn't match Pod's node affinity/selector"`) on failure.

### 3.3. PreScore (`PreScore`)
1. **Parse Preferred Terms**: Extracts `PreferredSchedulingTerms` from `.spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution`.
2. **Fast Skip**: If both the Pod and plugin args lack preferred terms, returns `framework.NewStatus(framework.Skip)` to bypass `Score` across all nodes.
3. **CycleState Registration**: Stores parsed `preScoreState` into `CycleState` with key `"PreScoreNodeAffinity"`.

### 3.4. Score & NormalizeScore
1. **Score Calculation**:
   - Scores `pl.addedPrefSchedTerms.Score(node)` if scheduler-enforced terms exist.
   - Reads `preScoreState` from `CycleState` and scores `s.preferredNodeAffinity.Score(node)`.
   - Returns the cumulative raw score.
2. **Normalization**: Invokes `helper.DefaultNormalizeScore(fwk.MaxNodeScore, false, scores)` to linearly scale raw scores into `[0, 100]`.

---

## 4. State Management & `CycleState` Usage

`NodeAffinity` leverages `CycleState` to pass parsed, thread-safe affinity structures between single-threaded pre-phases and multi-threaded parallel execution phases:

```
[ PreFilter ] ─── Write("PreFilterNodeAffinity") ───► [ Filter (Parallel) ]
                                                            │
                                                     Reads preFilterState

[ PreScore ]  ─── Write("PreScoreNodeAffinity")  ───► [ Score (Parallel) ]
                                                            │
                                                     Reads preScoreState
```

### Data Structures:

#### `preFilterState` (`preFilterStateKey = "PreFilterNodeAffinity"`)
```go
type preFilterState struct {
    requiredNodeSelectorAndAffinity nodeaffinity.RequiredNodeAffinity
}
```
- **Cloning Behavior**: `Clone()` returns the same instance (`s`) because parsed affinity terms are read-only and immutable across pod additions/removals during preemption cycles.

#### `preScoreState` (`preScoreStateKey = "PreScoreNodeAffinity"`)
```go
type preScoreState struct {
    preferredNodeAffinity *nodeaffinity.PreferredSchedulingTerms
}
```
- **Cloning Behavior**: `Clone()` returns `s` as preferred terms are static during scoring.

---

## 5. Caching Strategies & Performance Optimizations

1. **Pre-Parsed Expression Trees**:
   - `nodeaffinity.GetRequiredNodeAffinity` and `nodeaffinity.NewPreferredSchedulingTerms` compile selector expressions into internal matcher structures once per scheduling cycle during `PreFilter`/`PreScore`, eliminating parsing overhead during concurrent node evaluations in `Filter` and `Score`.
2. **Selective Filter & Score Skipping**:
   - When a Pod lacks node selectors or affinity terms, `PreFilter` and `PreScore` return `framework.Skip`, instructing the framework runtime to avoid invoking `Filter` and `Score` across thousands of nodes.
3. **Node Pre-filtering via `PreFilterResult.NodeNames`**:
   - When required affinity specifies exact node names in `matchFields`, `PreFilter` computes the feasible node subset upfront, pruning non-matching nodes before the `Filter` phase even begins.
4. **Smart Queueing Hints (`isSchedulableAfterNodeChange`)**:
   - Evaluates node addition and label modification events against the Pod's required affinity terms.
   - For node updates, compares match status on `originalNode` vs `modifiedNode`: only returns `Queue` if the node changed from **unmatched to matched** (`wasMatched == false && isMatched == true`), skipping queue churn on irrelevant label updates.
5. **Pod Signing (`SignPod`)**:
   - Computes deterministic signature fragments (`NodeAffinitySignerName` and `NodeSelectorSignerName`) to facilitate batch equivalence scheduling.

---

## 6. Plugin Configuration & Arguments

Configured via `config.NodeAffinityArgs`:

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: NodeAffinityArgs
addedAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
      - matchExpressions:
          - key: topology.kubernetes.io/zone
            operator: In
            values: ["zone-a", "zone-b"]
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 50
      preference:
        matchExpressions:
          - key: instance-type
            operator: In
            values: ["c5.large"]
```

---

## 7. Testing & Verification Guide

### Unit Tests
Execute unit tests for `nodeaffinity`:
```bash
cd /home/debian/work
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/nodeaffinity/...
```

### Key Test Scenarios:
- `TestNodeAffinityPreFilter`: Validates node name extraction, conflicting term handling, and skip conditions.
- `TestNodeAffinityFilter`: Verifies required affinity matching with various operators (`In`, `NotIn`, `Exists`, `Gt`, `Lt`).
- `TestNodeAffinityScore`: Asserts correct weighted term aggregation and score normalization.
- `TestQueueingHint`: Validates queue hint decisions across node additions and label modifications.
