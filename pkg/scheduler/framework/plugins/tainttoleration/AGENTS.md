# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/tainttoleration`

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, state management mechanics, performance optimizations, and testing strategies for the `TaintToleration` scheduler plugin located at `pkg/scheduler/framework/plugins/tainttoleration`.

---

## 1. High-Level Overview & Core Role

The `TaintToleration` plugin (`names.TaintToleration`) ensures that Pods only schedule onto Nodes whose taints they tolerate, and prioritizes Nodes with the fewest untolerated `PreferNoSchedule` taints.

### Key Objectives:
1. **Hard Filtering (`NoSchedule` & `NoExecute`)**: Filters out any candidate node that possesses at least one `NoSchedule` or `NoExecute` taint not covered by the Pod's tolerations.
2. **Soft Prioritization (`PreferNoSchedule`)**: Scores feasible nodes inversely to the number of untolerated `PreferNoSchedule` taints present on the node.
3. **Toleration Comparison Operators**: Supports advanced numeric/string comparison operators when matching taint values (under `EnableTaintTolerationComparisonOperators`).

---

## 2. Implemented Extension Points

`TaintToleration` implements the following framework interfaces:

| Extension Point | Interface | Phase & Execution Mode | Description |
|---|---|---|---|
| **PreFilter** | `fwk.PreFilterPlugin` | Scheduling cycle (synchronous) | Handles deferred pod resize checks. |
| **Filter** | `fwk.FilterPlugin` | Scheduling cycle (parallelized) | Evaluates candidate nodes for untolerated `NoSchedule` / `NoExecute` taints. |
| **PreScore** | `fwk.PreScorePlugin` | Scheduling cycle (synchronous) | Pre-filters the Pod's tolerations to extract only those applicable to `PreferNoSchedule`. |
| **Score** | `fwk.ScorePlugin` | Scheduling cycle (parallelized) | Counts intolerable `PreferNoSchedule` taints on candidate nodes. |
| **NormalizeScore** | `fwk.ScoreExtensions` | Scheduling cycle (synchronous) | Uses reverse default normalization so nodes with zero intolerable taints receive maximum score (`100`). |
| **SignPlugin** | `fwk.SignPlugin` | Queueing / Batching | Produces `TolerationsSignerName` signature fragment for equivalence batching. |
| **EnqueueExtensions** | `fwk.EnqueueExtensions` | Queueing / Informers | Registers events for node taints and target pod toleration changes. |

---

## 3. Algorithm & Lifecycle Walkthrough

### 3.1. Filter Phase (`Filter`)
1. **Deferred Resize Check**: Skips filter if `resource.IsPodResizeDeferred(pod)` is true.
2. **Untolerated Taint Detection**:
   - Calls `v1helper.FindMatchingUntoleratedTaint(logger, node.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc(), pl.enableTaintTolerationComparisonOperators)`.
   - `DoNotScheduleTaintsFilterFunc()` filters candidate taints to those with `Effect == NoSchedule` or `Effect == NoExecute`.
3. **Rejection Verdict**: If an untolerated taint is found, immediately returns `framework.NewStatus(framework.UnschedulableAndUnresolvable, "node(s) had untolerated taint(s)")`.

### 3.2. PreScore Phase (`PreScore`)
1. **Extract `PreferNoSchedule` Tolerations**:
   - Gathers all tolerations with `Effect == PreferNoSchedule` or with empty effect (which acts as a wildcard matching all taint effects).
2. **CycleState Registration**: Writes `preScoreState` into `CycleState` with key `"PreScoreTaintToleration"`.

### 3.3. Score & NormalizeScore
1. **Score (`countIntolerableTaintsPreferNoSchedule`)**:
   - Iterates over node taints where `taint.Effect == v1.TaintEffectPreferNoSchedule`.
   - Checks if `v1helper.TolerationsTolerateTaint(...)` is false.
   - Returns the count of intolerable `PreferNoSchedule` taints as raw score.
2. **NormalizeScore**:
   - Calls `helper.DefaultNormalizeScore(fwk.MaxNodeScore, true, scores)` with `reverse = true`.
   - A raw count of `0` untolerated taints maps to `MaxNodeScore` (100).
   - Higher counts of untolerated taints scale proportionally down toward `0`.

---

## 4. State Management & `CycleState` Usage

`TaintToleration` uses `CycleState` during scoring to avoid repeatedly filtering the Pod's toleration slice across every candidate node:

### Data Structure:

#### `preScoreState` (`preScoreStateKey = "PreScoreTaintToleration"`)
```go
type preScoreState struct {
    tolerationsPreferNoSchedule []v1.Toleration
}
```
- **Cloning Behavior**: `Clone()` returns `s` because the extracted slice of tolerations is read-only.

---

## 5. Caching Strategies & Performance Optimizations

1. **Pre-Filtering Toleration Slices**:
   - `getAllTolerationPreferNoSchedule` filters the Pod's `.spec.tolerations` once during `PreScore`. Concurrent worker goroutines in `Score` evaluate only against this pre-filtered slice, reducing comparison overhead.
2. **Fast Status Rejections**:
   - `Filter` exits immediately upon encountering the first untolerated taint, rather than checking the rest of the node's taints.
3. **Targeted Event Queueing Hints (`EventsToRegister`)**:
   - `Node` (`Add | UpdateNodeTaint`): `isSchedulableAfterNodeChange` tests if the node was previously untolerated and is now tolerated. Only state transitions from untolerated to tolerated trigger `fwk.Queue`, avoiding spurious wakeups when irrelevant taints are added or updated.
   - `TargetPod` (`UpdatePodToleration`): `isSchedulableAfterTargetPodTolerationChange` unblocks the unschedulable pod when new tolerations are added to its spec.
4. **Pod Equivalence Signing (`SignPod`)**:
   - Computes deterministic hash signatures of pod tolerations (`TolerationsSigner`) to support opportunistic batch scheduling.

---

## 6. Testing & Verification Guide

### Unit Tests
Execute unit tests for `tainttoleration`:
```bash
cd /home/debian/work
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/tainttoleration/...
```

### Key Test Scenarios:
- `TestTaintTolerationFilter`: Tests filtering across combinations of `NoSchedule`, `NoExecute`, wildcard keys/effects, and toleration operators (`Exists`, `Equal`).
- `TestTaintTolerationScore`: Asserts correct counting of `PreferNoSchedule` taints and reverse score normalization.
- `TestQueueingHint`: Verifies accurate queueing hints on node taint mutations and target pod toleration updates.
