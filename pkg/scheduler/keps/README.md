# Kubernetes Scheduler Preemption & Workload Enhancement Proposals (KEPs)

This directory maintains comprehensive imported specifications, architectural summaries, and integration analyses for all Kubernetes Enhancement Proposals (KEPs) governing the **Preemption**, **Priority**, **Gang Scheduling**, **Asynchronous Actuation**, **In-Place Resizing**, and **Resource Topology** subsystems within `pkg/scheduler`.

---

## 1. Overview & Architectural Evolution

Pod Preemption in Kubernetes enables high-priority workloads to obtain immediate placement on compute nodes by evicting or reprieving lower-priority pods when cluster resources are saturated. Over recent Kubernetes release cycles (from Kubernetes v1.14 through v1.37+), preemption has evolved from a synchronous, single-pod, greedy heuristic into a distributed, multi-paradigm scheduling engine supporting:

1. **Deterministic Single-Pod Preemption** with strict PodDisruptionBudget (PDB) protection.
2. **Asynchronous Non-Blocking Actuation** removing API server evictions from the critical scheduling cycle.
3. **All-or-Nothing Workload & Gang Preemption** for distributed AI/ML training and composite multi-job topologies.
4. **In-Place Vertical Scaling Preemption** allowing running pods to dynamically expand CPU/memory allocations on their assigned nodes.
5. **Hardware-Aware & Batched Preemption** integrating Dynamic Resource Allocation (DRA), NUMA boundaries, pod-level resource specs, and cryptographic pod signature batching.

---

## 2. KEP Inventory & Domain Organization

The preemption-related KEPs are categorized into five distinct architectural domains:

| Domain | Document | Primary KEPs | Feature Gates | Subsystems & Packages |
| :--- | :--- | :--- | :--- | :--- |
| **1. Core Priority & Preemption** | [`01-core-pod-priority-and-preemption.md`](01-core-pod-priority-and-preemption.md) | **KEP-562** (Pod Priority & Preemption)<br>**KEP-3838** (PDB Respect in Preemption) | `PodPriority` (GA)<br>`RespectPodDisruptionBudget` | `pkg/scheduler/framework/preemption/`<br>`pkg/scheduler/algorithm.go`<br>`pkg/apis/scheduling/` |
| **2. Asynchronous Preemption & Queuing** | [`02-asynchronous-preemption-and-queueing.md`](02-asynchronous-preemption-and-queueing.md) | **KEP-4832** (Async Preemption & API Calls)<br>**KEP-5142** (Scheduling Queue Backoff) | `AsyncPreemption`<br>`SchedulerAsyncAPICalls` | `pkg/scheduler/backend/queue/`<br>`pkg/scheduler/framework/preemption/`<br>`pkg/scheduler/schedule_one.go` |
| **3. Workload-Aware & Gang Preemption** | [`03-workload-aware-and-gang-preemption.md`](03-workload-aware-and-gang-preemption.md) | **KEP-5710** (Workload-Aware Preemption)<br>**KEP-6012** (Composite Pod Groups) | `GenericWorkload`<br>`CompositePodGroup` | `pkg/scheduler/schedule_one_podgroup.go`<br>`pkg/scheduler/framework/preemption/`<br>`pkg/apis/scheduling/` |
| **4. In-Place Vertical Scaling Preemption** | [`04-inplace-vertical-scaling-preemption.md`](04-inplace-vertical-scaling-preemption.md) | **KEP-1287** (In-Place Pod Vertical Scaling)<br>**KEP-5836** (Scheduler Resize Preemption) | `InPlacePodVerticalScaling`<br>`InPlacePodVerticalScalingSchedulerPreemption` | `pkg/scheduler/framework/plugins/deferredpodscheduling/`<br>`pkg/scheduler/backend/cache/`<br>`pkg/kubelet/` |
| **5. Advanced Resource & Topology Preemption** | [`05-advanced-resource-topology-batching-preemption.md`](05-advanced-resource-topology-batching-preemption.md) | **KEP-2837** (Pod-Level Resources)<br>**KEP-5598** (Opportunistic Batching)<br>**KEP-4818** (Node Declared Features)<br>**KEP-6072** (Structured DRA NUMA) | `PodLevelResources`<br>`OpportunisticBatching`<br>`NodeDeclaredFeatures`<br>`DynamicResourceAllocation` | `pkg/scheduler/framework/plugins/noderesources/`<br>`pkg/scheduler/backend/queue/`<br>`staging/src/k8s.io/kube-scheduler/` |

---

## 3. Subsystem Cross-Reference Matrix

```
                      +------------------------------------------+
                      |         Scheduling Queue (activeQ)       |
                      +------------------------------------------+
                                           |
                                [Filtering / Feasibility]
                                           |
                                  Nodes Feasible?
                                  /            \
                           (Yes) /              \ (No)
                                /                \
                     [Scoring Pipeline]    [PostFilter / Preemption]
                                |                |
                                |         +---------------+---------------+
                                |         |               |               |
                                |    [Core KEP-562] [Gang KEP-5710] [Resize KEP-5836]
                                |    (Single Pod)   (PodGroups)    (In-Place Expansion)
                                |         |               |               |
                                |         +---------------+---------------+
                                |                         |
                                |                 [Victim Selection]
                                |                 - Minimize Priority
                                |                 - Minimize PDB Violations (KEP-3838)
                                |                 - Reprieve Unneeded Victims
                                |                         |
                                |                 [Actuation Engine]
                                |                 - Async Goroutines (KEP-4832)
                                |                 - Set NominatedNodeName
                                |                 - Pre-Enqueue Gating (KEP-5142)
                                |                         |
                     [Reserve / AssumeCache] <------------+
                                |
                     [Permit / WaitingPods]
                                |
                     [PreBind / BindingCycle]
```

---

## 4. Key Invariants for Preemption Implementations

1. **Priority Monotonicity**: A higher-priority pod (`Pod A`, priority $P_A$) may only preempt a victim pod (`Pod V`, priority $P_V$) if $P_A > P_V$. Under no circumstances may equal-priority pods preempt one another.
2. **PDB Violation Minimization**: Preemption algorithms must partition candidate victim sets into PDB-respecting and PDB-violating subsets, strictly selecting victims from the PDB-respecting subset whenever feasible.
3. **Starvation Prevention & NominatedNode Tracking**: When a pod is nominated on a node (`spec.nominatedNodeName`), lower-priority pods are prevented from consuming resources required by the nominated preemptor on subsequent scheduling cycles.
4. **All-or-Nothing Gang Invariant**: For workloads requiring gang scheduling (`GenericWorkload` / `PodGroup`), preemption actuation must only trigger if all $M$ required gang members (`minMember`) can successfully find feasible placements and victims across the cluster.
5. **Async Fast-Fail Rollback**: In asynchronous preemption (KEP-4832), if an API deletion call fails with a non-404 error, all pending nominations and in-memory reservations for the cycle must be rolled back immediately.

---

## 5. Related Reading & Reference Guides

- [Scheduler Architecture Guide (`../AGENTS.md`)](../AGENTS.md)
- [Preemption Framework Internals (`../framework/preemption/AGENTS.md`)](../framework/preemption/AGENTS.md)
- [Comprehensive Guide to Pod Preemption (`../../docs/guide-to-kubernetes-pod-preemption.md`)](../../docs/guide-to-kubernetes-pod-preemption.md)
- [Preemption Issues & Root Cause Summary (`../../docs/preemption-issues-summary.md`)](../../docs/preemption-issues-summary.md)
