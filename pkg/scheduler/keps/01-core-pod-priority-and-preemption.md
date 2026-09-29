# Domain 1: Core Pod Priority & Single-Pod Preemption

**Governing KEPs:**
- **KEP-562**: *Pod Priority and Preemption*
- **KEP-3838**: *Respect PodDisruptionBudget in Preemption*

**Primary Packages:**
- `pkg/scheduler/framework/preemption/`
- `pkg/scheduler/framework/plugins/defaultpreemption/`
- `pkg/scheduler/algorithm.go`
- `pkg/apis/scheduling/`

---

## 1. Executive Summary & Problem Statement

In standard multi-tenant Kubernetes clusters, workloads have differing criticalities—ranging from essential control plane and system services (e.g., CoreDNS, CNI plugins) to production databases, API endpoints, and best-effort background batch tasks. When cluster compute or storage capacity is exhausted, newly submitted critical pods cannot schedule if the scheduler operates under strict FIFO or un-prioritized logic.

**KEP-562** introduced first-class **Pod Priority and Preemption** into Kubernetes, allowing cluster operators to assign 32-bit integer priority values to pods via `PriorityClass` resources. When a higher-priority pod is unschedulable because all matching nodes are fully utilized, `kube-scheduler` simulates node states, identifies lower-priority victim pods whose removal would allow the preemptor to fit, nominates a candidate node, and triggers graceful eviction of the chosen victims.

**KEP-3838** augmented this algorithm to respect `PodDisruptionBudget` (PDB) constraints during victim selection, ensuring that preemption minimizes voluntary disruptions to replicated services.

---

## 2. API Specifications & Data Structures

### 2.1 PriorityClass API (`scheduling.k8s.io/v1`)

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-apps
value: 1000000
globalDefault: false
preemptionPolicy: PreemptLowerPriority # Options: PreemptLowerPriority | Never
description: "Mission-critical user-facing applications."
```

- **`value`**: A 32-bit signed integer (`int32`, range `-2147483648` to `1000000000`). Values greater than 1 billion are reserved for built-in critical system priorities (e.g., `system-cluster-critical` = 2000000000, `system-node-critical` = 2000001000).
- **`preemptionPolicy`**:
  - `PreemptLowerPriority` (default): The pod can preempt lower-priority pods when unschedulable.
  - `Never`: Non-preempting priority. The pod receives high scheduling priority in the queue, but will never preempt other pods if resources are lacking.
- **`globalDefault`**: Specifies whether this priority class applies to pods created without a `priorityClassName`.

### 2.2 Pod Spec & Status Fields

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: preemptor-workload
spec:
  priorityClassName: high-priority-apps
  priority: 1000000
  preemptionPolicy: PreemptLowerPriority
  ...
status:
  nominatedNodeName: "node-worker-04"
```

- **`spec.priority`**: Resolved by the `Priority` admission controller at pod creation time.
- **`status.nominatedNodeName`**: Written by the scheduler's preemption engine. Informs the scheduler that the pod is waiting for victims on this specific node to finish graceful termination.

---

## 3. Core Preemption Algorithm (DefaultPreemption)

The core preemption logic is invoked during the scheduler's `PostFilter` extension point when no node satisfies the pod's `Filter` predicates.

```
                  +-------------------------------+
                  |  PostFilter: Select Victims   |
                  +-------------------------------+
                                  |
                                  v
                  +-------------------------------+
                  | 1. Find Candidate Nodes       |
                  |    - Run Filter with victims  |
                  |      hypothetically removed   |
                  +-------------------------------+
                                  |
                                  v
                  +-------------------------------+
                  | 2. Per-Node Victim Selection  |
                  |    - Sort victims by priority |
                  |    - Partition: Non-PDB vs PDB|
                  |    - Remove lowest priority   |
                  |      until preemptor fits     |
                  +-------------------------------+
                                  |
                                  v
                  +-------------------------------+
                  | 3. Pick Best Candidate Node   |
                  |    - Lowest highest victim pri|
                  |    - Fewest PDB disruptions   |
                  |    - Fewest total victims     |
                  |    - Highest node score       |
                  +-------------------------------+
                                  |
                                  v
                  +-------------------------------+
                  | 4. Actuation: Evict Victims   |
                  |    - Dispatch Evictions/Deletes|
                  |    - Set nominatedNodeName    |
                  +-------------------------------+
```

### Step 1: Candidate Node Discovery
The scheduler iterates across all cluster nodes in parallel (chunked via `parallelize.Until`). For each node:
1. Verify whether the node has pods with priority lower than the preemptor. If not, the node is disqualified immediately.
2. Construct a hypothetical node copy where all lower-priority pods are removed.
3. Run all `Filter` plugins (e.g., `NodeResourcesFit`, `PodTopologySpread`, `NodeAffinity`, `VolumeRestrictions`). If the preemptor still does not fit on the empty node, the node cannot resolve the deficit and is discarded.

### Step 2: Per-Node Minimal Victim Selection & Reprieval
To minimize cluster disruption, the scheduler must not evict all lower-priority pods. Instead, it finds the minimal subset:
1. **Sort Lower-Priority Pods**: Pods on the node with `priority < preemptor.priority` are sorted primarily by priority (lowest priority first), then by PDB violation status (pods without PDB protection first), then by start time (newest first).
2. **Greedy Removal**: The scheduler removes sorted pods one by one from the node snapshot until the preemptor passes all `Filter` checks.
3. **Reprieval (Backtracking)**: The algorithm attempts to add back (reprieve) evicted pods in reverse order (highest priority first) while ensuring the preemptor remains schedulable. This prevents over-eviction.

### Step 3: Candidate Node Selection
When multiple nodes can host the preemptor via preemption, the scheduler scores the candidate nodes using a lexicographical comparator:
1. **Minimize Highest Victim Priority**: Select the node where the highest-priority victim is lower than on other nodes.
2. **Minimize PDB Violations (KEP-3838)**: Select the node with the fewest victim pods violating a `PodDisruptionBudget`.
3. **Minimize Total Number of Victims**: Select the node requiring the fewest total evictions.
4. **Tie-Breaking via Node Scoring**: If all victim metrics are tied, run standard `Score` plugins on the candidate nodes and choose the highest-scoring node.

### Step 4: Actuation & NominatedNodeName
1. Issue graceful eviction / deletion API calls for each victim in the chosen set:
   - For regular victims: Issue `Delete` or `Evict` with grace period.
   - For `WaitingPods` (pods holding in-memory reserve permits): Cancel the permit in the local framework cache without issuing remote API deletions.
2. Set `preemptor.Status.NominatedNodeName = candidateNodeName`.
3. Re-queue the preemptor in the scheduler's `activeQ` / `backoffQ`.

---

## 4. PodDisruptionBudget (PDB) Semantics & Invariants

A `PodDisruptionBudget` limits the number of concurrent voluntary disruptions for a set of pods:

- **Partitioning Strategy**: Candidate victims on a node are partitioned into:
  - Set $V_{respect}$: Pods whose disruption does not violate their matching PDB (`pdb.Status.DisruptionsAllowed > 0`).
  - Set $V_{violate}$: Pods whose disruption violates their matching PDB (`pdb.Status.DisruptionsAllowed == 0`).
- **Preemption Rule**: The algorithm always exhausts $V_{respect}$ before evicting any pod from $V_{violate}$.
- **Fallback**: If the preemptor cannot fit without evicting pods from $V_{violate}$, and the preemptor has higher priority, the scheduler is permitted to violate PDBs to ensure cluster-critical workloads make progress.

---

## 5. Failure Modes & Edge Cases

1. **Starvation via Repeated Preemption**: A lower-priority pod is repeatedly scheduled, preempted, and restarted. Solved by backoff queuing and priority-based throttling.
2. **Nomination Deadlock / Shadowing**: A nominated preemptor reserves node capacity in subsequent cycles. If the victims take a long time to terminate, lower-priority pods that could fit on the remaining capacity might be blocked. Solved by checking if a pod can fit on the node *with* the nominated preemptor accounted for.
3. **PDB Selector Mismatches**: Unlabeled pods or universal selectors (`{}`) causing unexpected PDB violation accounting across namespaces.

---

## 6. Verification & Test Matrix

- **Unit Tests**: `pkg/scheduler/framework/plugins/defaultpreemption/default_preemption_test.go`
- **Integration Tests**: `test/integration/scheduler/preemption/`
  - `TestPreemption`: Basic preemption across resource types.
  - `TestDeterministicEqualTimestampVictimSelection`: Determinism in victim ordering.
  - `TestEmptyPDBSelectorMultiNamespacePreemption`: Namespace boundary isolation for universal PDBs.
