# Domain 3: Workload-Aware & Gang Preemption

**Governing KEPs:**
- **KEP-5710**: *Workload-Aware Preemption (WAP) / Gang Scheduling*
- **KEP-6012**: *Composite Pod Groups & Hierarchical Workload Preemption*

**Primary Packages:**
- `pkg/scheduler/schedule_one_podgroup.go`
- `pkg/scheduler/framework/preemption/`
- `pkg/scheduler/framework/plugins/coscheduling/`
- `pkg/apis/scheduling/`

---

## 1. Executive Summary & Problem Statement

Distributed computing frameworks—such as PyTorch distributed training (DDP), TensorFlow multi-node workers, Ray, MPI, and big-data processing pipelines—require **all-or-nothing (gang) scheduling**. If a job requires 64 worker pods, scheduling only 63 pods while 1 pod remains blocked wastes 63 nodes of compute capacity while the partial gang idles waiting for the final pod.

Under legacy single-pod preemption (KEP-562), gang workloads suffered from catastrophic starvation and deadlocks:
1. **Partial Preemption Waste**: The scheduler preempted victims on Node A to schedule Pod 1 of a gang, but could not find capacity for Pod 2 on any other node. The victims on Node A were needlessly killed while Pod 1 eventually timed out and was rejected.
2. **Inter-Gang Deadlocks**: Gang $A$ (high priority) and Gang $B$ (high priority) could concurrently preempt disjoint subsets of nodes, preventing either gang from reaching its `minMember` quorum.
3. **Hierarchical Workload Disconnect**: Modern multi-tier workloads (e.g., a parameter server tier + worker pool tier + driver) could not coordinate preemption priorities across composite trees.

**KEP-5710** and **KEP-6012** introduced **Workload-Aware Preemption (WAP)** and **CompositePodGroups**, extending the scheduler from single-pod evaluation to atomic, multi-node, all-or-nothing gang preemption.

---

## 2. API Specifications & Data Structures

### 2.1 PodGroup Specification (`scheduling.x-k8s.io/v1alpha1`)

```yaml
apiVersion: scheduling.x-k8s.io/v1alpha1
kind: PodGroup
metadata:
  name: distributed-training-job
  namespace: ml-workloads
spec:
  minMember: 16
  minResources:
    cpu: "128"
    memory: "512Gi"
    nvidia.com/gpu: "64"
  scheduleTimeoutSeconds: 300
  disruptionMode: all # Options: all | single
  priorityClassName: high-priority-ml
```

- **`minMember`**: The minimum number of pods that must be schedulable concurrently for the gang to be admitted.
- **`scheduleTimeoutSeconds`**: Maximum time members may wait in the `Permit` phase before the entire gang is rejected and rolled back.
- **`disruptionMode`**:
  - `all`: If any member of this group is chosen as a preemption victim, all other running members of the group are evicted simultaneously to free cohesive cluster capacity.
  - `single`: Individual member pods may be preempted independently without evicting the entire group.

### 2.2 CompositePodGroup Specification (KEP-6012)

```yaml
apiVersion: scheduling.x-k8s.io/v1alpha1
kind: CompositePodGroup
metadata:
  name: hierarchical-training-cluster
spec:
  parent: "root-training-job"
  groups:
    - name: "ps-tier"
      minMember: 4
      priority: 1000000
    - name: "worker-tier"
      minMember: 32
      priority: 900000
  strategy: StrictOrdering # Options: StrictOrdering | Simultaneous
```

---

## 3. Workload-Aware Gang Preemption Algorithm

The gang preemption pipeline coordinates multi-pod feasibility across the entire cluster:

```
                  +----------------------------------------------+
                  |         scheduleOnePodGroup Pipeline         |
                  +----------------------------------------------+
                                         |
                                [Evaluate Feasibility]
                                         |
                        Are >= minMember Pods Feasible?
                                  /            \
                           (Yes) /              \ (No)
                                /                \
                     [Bind All Members]   [PostFilter: Gang Preemption]
                                                 |
                                  +--------------v--------------+
                                  | 1. Cluster-Wide Simulation  |
                                  |    - Snapshot all nodes     |
                                  |    - Track global victims   |
                                  +-----------------------------+
                                                 |
                                  +--------------v--------------+
                                  | 2. Multi-Node Fitting       |
                                  |    - Place members greedily |
                                  |    - Select minimal victims |
                                  +-----------------------------+
                                                 |
                                  +--------------v--------------+
                                  | 3. Quorum Verification      |
                                  |    Can all minMembers fit?  |
                                  +-----------------------------+
                                         /              \
                                  (Yes) /                \ (No)
                                       /                  \
                        +--------------v-----+     +------v-------------+
                        | Actuate All Victims|     | Abort Preemption   |
                        | Across All Nodes   |     | Zero Victims Killed|
                        +--------------------+     +--------------------+
```

### 3.1 Step-by-Step Pipeline

1. **Global Cluster State Snapshotting**:
   - The scheduler creates an isolated working copy of the entire cluster cache (`NodeTree` and `NodeInfoMap`).
2. **Iterative Multi-Pod Candidate Fitting**:
   - For each unscheduled member pod in the `PodGroup`:
     - Run `Filter` plugins against the simulated cluster.
     - If the member cannot fit, find victim pods on candidate nodes whose priority is strictly less than `min(member.Priority, podGroup.Priority)`.
     - Record candidate victims in the global `VictimAccumulator` set.
     - Hypothetically remove victims and place the member pod on the chosen node.
3. **Quorum Verification (The "All-or-Nothing" Gate)**:
   - If fewer than `minMember` pods can be accommodated even after preemption across all nodes, the entire preemption cycle is **aborted**. **Zero victims are touched**.
4. **Global Cross-Node Victim Reprieval**:
   - If quorum is met, the scheduler executes a global reprieval pass. It iterates through candidate victims in descending order of priority, attempting to re-add them to their respective nodes. If all `minMember` pods still fit, the victim is reprieved.
5. **Atomic Multi-Node Actuation**:
   - The scheduler issues asynchronous evictions for all final victims across all involved nodes.
   - Sets `nominatedNodeName` for each respective member pod.
   - Puts all gang members into the `Permit` waiting phase.

---

## 4. Invariants & Guardrails

1. **Zero-Victim-Leakage Invariant**:
   - Under no circumstances may a single victim pod be evicted if the entire `PodGroup` cannot satisfy its `minMember` requirement.
2. **Composite Hierarchy Tree Priority**:
   - In a `CompositePodGroup`, priority evaluation respects tree inheritance. A victim cannot be evicted by a child pod if the victim has higher priority than the child's effective priority root.
3. **DisruptionMode Cascades**:
   - When an active victim belongs to a `PodGroup` configured with `DisruptionMode=all`, evicting that victim triggers a coordinated eviction of all peer pods in that group, preventing zombie resource holding.

---

## 5. Failure Modes & Edge Cases

1. **Gang Preemption Timeouts**: If some members take longer than `scheduleTimeoutSeconds` to bind due to slow victim terminations, the `WaitingPod` timeout fires, rejecting all gang members and clearing nominations.
2. **Multi-Gang Preemption Collisions**: Multiple gangs vying for overlapping nodes. Handled via queue sorting where older gangs with higher priority are evaluated first.

---

## 6. Verification & Test Matrix

- **Unit Tests**: `pkg/scheduler/schedule_one_podgroup_test.go`
- **Integration Tests**: `test/integration/scheduler/preemption/`
  - `TestGangPreemptionAllOrNothing`: Validates zero victims are evicted when quorum cannot be satisfied.
  - `TestCompositePodGroupHierarchicalPreemption`: Validates parent-child priority cascades.
  - `TestDisruptionModeAllCascadingEviction`: Validates group-wide eviction when a member is preempted.
