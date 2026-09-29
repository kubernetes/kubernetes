# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/interpodaffinity`

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, state management mechanics, performance optimizations, and testing strategies for the `InterPodAffinity` scheduler plugin located at `pkg/scheduler/framework/plugins/interpodaffinity`.

---

## 1. High-Level Overview & Core Role

The `InterPodAffinity` plugin (`names.InterPodAffinity`) enforces hard and soft inter-pod affinity and anti-affinity rules across topological domains (e.g., node hostnames, availability zones, failure domains).

### Key Objectives:
1. **Hard Pod Affinity (`podAffinity.requiredDuringSchedulingIgnoredDuringExecution`)**: Ensures incoming pods land in topology domains that already host matching pods.
2. **Hard Pod Anti-Affinity (`podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution`)**: Prevents incoming pods from landing in topology domains hosting pods that violate anti-affinity rules.
3. **Bi-directional Rule Enforcement**:
   - **Incoming Pod Rules**: The incoming pod must satisfy its own affinity/anti-affinity requirements against existing pods.
   - **Existing Pod Anti-Affinity**: The incoming pod must not violate the anti-affinity rules of already scheduled pods.
4. **Soft Affinity / Anti-Affinity Scoring (`preferredDuringSchedulingIgnoredDuringExecution`)**: Prioritizes nodes situated in topology domains that satisfy weighted affinity preferences and penalizes topology domains matching anti-affinity terms.

---

## 2. Implemented Extension Points

`InterPodAffinity` implements the following framework interfaces:

| Extension Point | Interface | Phase & Execution Mode | Description |
|---|---|---|---|
| **PreFilter** | `fwk.PreFilterPlugin` | Scheduling cycle (synchronous) | Precomputes cluster-wide topology match maps, unrolls namespaces, and classifies terms by scope. |
| **PreFilterExtensions** | `fwk.PreFilterExtensions` | Preemption simulation | Implements `AddPod` and `RemovePod` to incrementally mutate topology counts during victim evaluation. |
| **Filter** | `fwk.FilterPlugin` | Scheduling cycle (parallelized) | Evaluates candidate nodes against cluster-wide topology maps and executes node-local checks for host-scoped rules. |
| **PreScore** | `fwk.PreScorePlugin` | Scheduling cycle (parallelized over nodes) | Aggregates soft affinity/anti-affinity weights into a per-topology score map (`topologyScore`). |
| **Score** | `fwk.ScorePlugin` | Scheduling cycle (parallelized) | Sums score weights for topology labels present on the candidate node. |
| **NormalizeScore** | `fwk.ScoreExtensions` | Scheduling cycle (synchronous) | Scales raw score ranges across candidate nodes to `[0, MaxNodeScore]` via Min-Max normalization. |
| **SignPlugin** | `fwk.SignPlugin` | Queueing / Batching | Marks pods with inter-pod affinity constraints as unsignable or extracts label signers. |
| **EnqueueExtensions** | `fwk.EnqueueExtensions` | Queueing / Informers | Registers cluster events for assigned pods, target pod label changes, and node topology changes. |

---

## 3. Filtering Architecture & Algorithms

Inter-pod affinity filtering requires validating three distinct constraints:
1. **Incoming Pod's Affinity**: The candidate node's topology domain must contain an existing pod matching all affinity terms (or allow self-affinity if no pods exist yet).
2. **Incoming Pod's Anti-Affinity**: The candidate node's topology domain must not contain any existing pod matching any anti-affinity terms.
3. **Existing Pods' Anti-Affinity**: The incoming pod must not match anti-affinity terms of any existing pod in the candidate node's topology domain.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 Incoming Pod Evaluation                 │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼                                                     ▼
    ┌───────────────────────────────┐                     ┌───────────────────────────────┐
    │ HostnameFastPath Enabled      │                     │ HostnameFastPath Disabled     │
    │ (Hostname-scoped rules)       │                     │ (Standard Global Scan)        │
    ├───────────────────────────────┤                     ├───────────────────────────────┤
    │ - Host-scoped anti-affinity   │                     │ - Builds cluster-wide maps    │
    │   evaluated locally on node.  │                     │   for all topology keys in    │
    │ - Host-scoped affinity checks │                     │   PreFilter across all nodes. │
    │   candidate node's pods.      │                     │ - Evaluates candidate node    │
    │ - Global scan skipped when    │                     │   labels against global maps  │
    │   all rules are host-scoped.  │                     │   during Filter.              │
    └───────────────────────────────┘                     └───────────────────────────────┘
```

### 3.1. The `InterPodAffinityHostnameFastPath` Optimization
When rules use `topologyKey: kubernetes.io/hostname`, they depend strictly on pods on that exact node. The `InterPodAffinityHostnameFastPath` feature gate optimizes this:

1. **Term Scope Classification (`classifyTermsBasedOnScope`)**:
   - `hostScopedAffinityTerms`: Populated only when **all** affinity terms on the incoming pod are hostname-scoped.
   - `hostScopedAntiAffinityTerms`: Contains all hostname-scoped anti-affinity terms of the incoming pod.
   - `clusterWideAffinityTerms` & `clusterWideAntiAffinityTerms`: Terms requiring global topology scans (e.g. `topology.kubernetes.io/zone`).
2. **Local Filter Fast Path**:
   - Hostname-scoped anti-affinity (from both existing pods and incoming pod) is evaluated directly against the candidate node's local pods (`nodeInfo.GetPodsWithRequiredAntiAffinity()` and `nodeInfo.GetPods()`), bypassing global topology map lookups.
3. **Self-Affinity Counter (`matchingHostScopedAffinityPodsCount`)**:
   - For incoming pods with exclusively host-scoped affinity, building a cluster-wide map is skipped. Instead, a lightweight global counter tracks if *any* matching pods exist in the cluster.
   - If `matchingHostScopedAffinityPodsCount == 0`, the first pod in a self-affinity group is allowed to bootstrap placement on any node.

### 3.2. PreFilterExtensions (`AddPod` / `RemovePod`)
During preemption candidate evaluation, `DefaultPreemption` simulates removing victim pods and adding the preemptor pod. Rather than rebuilding the cluster-wide topology maps from scratch:
- `RemovePod`: Decrements counts in `clusterWideAffinityCounts`, `clusterWideAntiAffinityCounts`, `existingClusterWideAntiAffinityCounts`, and `matchingHostScopedAffinityPodsCount` using a multiplier of `-1`.
- `AddPod`: Increments counts with a multiplier of `+1`.

---

## 4. Scoring Architecture & Algorithms

### 4.1. PreScore Parallel Aggregation
`PreScore` computes a unified `scoreMap` (`map[topologyKey]map[topologyValue]int64`):
1. **Node Candidate Selection**:
   - If the incoming pod has preferred terms, scans all cluster nodes (`sharedLister.NodeInfos().List()`).
   - If the incoming pod has no preferred terms, scans only nodes hosting pods with affinity (`sharedLister.NodeInfos().HavePodsWithAffinityList()`).
   - If `IgnorePreferredTermsOfExistingPods` is enabled and incoming pod has no preferences, `PreScore` returns `framework.Skip`.
2. **Parallel Node Processing**:
   - For each existing pod on candidate nodes, processes:
     - **Incoming pod's preferred affinity**: `+weight` on matching existing pod's topology value.
     - **Incoming pod's preferred anti-affinity**: `-weight` on matching existing pod's topology value.
     - **Existing pods' hard affinity**: `+HardPodAffinityWeight` on incoming pod match.
     - **Existing pods' preferred affinity**: `+weight` on incoming pod match.
     - **Existing pods' preferred anti-affinity**: `-weight` on incoming pod match.
3. **CycleState Registration**: Merges worker thread score maps into `state.topologyScore` and writes `preScoreStateKey`.

### 4.2. Score & NormalizeScore
1. **Score**: Looks up candidate node's topology label values in `state.topologyScore` and accumulates scores:
   $$\text{rawScore} = \sum_{\text{tpKey}} \text{topologyScore}[\text{tpKey}][\text{node.Labels}[\text{tpKey}]]$$
2. **NormalizeScore**: Normalizes across all filtered nodes using Min-Max scaling:
   $$\text{Score}(n) = \text{MaxNodeScore} \times \frac{\text{rawScore}(n) - \text{minCount}}{\text{maxCount} - \text{minCount}}$$

---

## 5. State Management & `CycleState` Usage

### Data Structures:

#### `preFilterState` (`preFilterStateKey = "PreFilterInterPodAffinity"`)
```go
type preFilterState struct {
    existingClusterWideAntiAffinityCounts topologyToMatchedTermCount
    clusterWideAffinityCounts             topologyToMatchedTermCount
    clusterWideAntiAffinityCounts         topologyToMatchedTermCount
    podInfo                               fwk.PodInfo
    namespaceLabels                       labels.Set
    hostScopedAffinityTerms               []fwk.AffinityTerm
    hostScopedAntiAffinityTerms           []fwk.AffinityTerm
    clusterWideAffinityTerms              []fwk.AffinityTerm
    clusterWideAntiAffinityTerms          []fwk.AffinityTerm
    matchingHostScopedAffinityPodsCount   int64
    enableInterPodAffinityHostnameFastPath bool
}
```
- **Cloning Behavior**: Performs deep copies of `topologyToMatchedTermCount` maps via `sync.Map` / map clones, allowing preemption simulations on isolated cloned states without corrupting parent scheduling cycle state.

#### `preScoreState` (`preScoreStateKey = "PreScoreInterPodAffinity"`)
```go
type preScoreState struct {
    topologyScore   scoreMap // map[string]map[string]int64
    podInfo         fwk.PodInfo
    namespaceLabels labels.Set
}
```
- **Cloning Behavior**: `Clone()` returns `s` (read-only during scoring).

---

## 6. Caching Strategies & Performance Optimizations

1. **Namespace Unrolling & Selector Merging**:
   - `mergeAffinityTermNamespacesIfNotEmpty`: Resolves `NamespaceSelector` via `nsLister.List` during `PreFilter` and merges matched namespaces directly into `AffinityTerm.Namespaces`, setting `NamespaceSelector = labels.Nothing()`. This avoids expensive namespace label queries in inner loops.
2. **Namespace Label Snapshotting**:
   - Caches the incoming pod's namespace labels upfront using `GetNamespaceLabelsSnapshot(logger, pod.Namespace, pl.nsLister)`.
3. **`HavePodsWithAffinityList` Pruning**:
   - In `PreScore`, when the incoming pod has no soft affinity terms, skips scanning empty nodes by evaluating only nodes tracked in `NodeInfoSnapshot.HavePodsWithAffinityList()`.
4. **Fast-Path Hostname Isolation**:
   - Avoids quadratic cluster scans for single-host topologies by handling hostname anti-affinity within node-local loops.
5. **Smart Event Queueing Hints (`EventsToRegister`)**:
   - `AssignedPod` (Add, UpdatePodLabel, Delete): Checks whether the pod update/addition creates or destroys affinity/anti-affinity matches.
   - `TargetPod` (UpdatePodLabel): Requeues pod when its own labels change.
   - `Node` (Add, UpdateNodeLabel): Requeues when topology keys matching the pod's affinity/anti-affinity are added or modified.

---

## 7. Configuration & Arguments

Configured via `config.InterPodAffinityArgs`:

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: InterPodAffinityArgs
hardPodAffinityWeight: 1 # Weight added to nodes satisfying existing pods' hard affinity
ignorePreferredTermsOfExistingPods: false # When true, skips existing pods' soft preferences if incoming pod has none
```

---

## 8. Testing & Verification Guide

### Unit Tests
Execute unit tests for `interpodaffinity`:
```bash
cd /home/debian/work
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/interpodaffinity/...
```

### Key Test Scenarios:
- `TestInterPodAffinityPreFilter`: Tests term classification, namespace resolution, and cluster-wide map initialization.
- `TestInterPodAffinityFilter`: Verifies bi-directional hard affinity and anti-affinity across node and zone topologies.
- `TestInterPodAffinityPreScoreAndScore`: Validates weight accumulations, soft affinity preferences, and Min-Max score normalization.
- `TestInterPodAffinityHostnameFastPath`: Validates fast-path execution and self-affinity handling when only hostname topology keys are used.
