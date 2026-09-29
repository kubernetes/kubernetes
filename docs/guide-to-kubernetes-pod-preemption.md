# Unified Architecture Guide and Troubleshooting Manual for Kubernetes Pod Preemption

## Table of Contents
1. [Executive Summary & Architectural Foundations](#1-executive-summary--architectural-foundations)
   - [1.1 The Multi-Paradigm Preemption Ecosystem](#11-the-multi-paradigm-preemption-ecosystem)
   - [1.2 Scope, Objectives, and Architectural Tenets](#12-scope-objectives-and-architectural-tenets)
2. [Core Primitives, APIs & Configuration Specifications](#2-core-primitives-apis--configuration-specifications)
   - [2.1 PriorityClass API & Preemption Policies](#21-priorityclass-api--preemption-policies)
   - [2.2 Pod Priority, NominatedNodeName & Scheduling Invariants](#22-pod-priority-nominatednodename--scheduling-invariants)
   - [2.3 PodDisruptionBudgets (PDBs) and In-Tree Evaluation Semantics](#23-poddisruptionbudgets-pdbs-and-in-tree-evaluation-semantics)
   - [2.4 Workload & Gang Scheduling APIs (PodGroup, CompositePodGroup, GenericPodGroup)](#24-workload--gang-scheduling-apis-podgroup-compositepodgroup-genericpodgroup)
   - [2.5 In-Place Pod Vertical Scaling (IPPVS) Preemption API (KEP-1287 / KEP-5836)](#25-in-place-pod-vertical-scaling-ippvs-preemption-api-kep-1287--kep-5836)
   - [2.6 Feature Gates & Capabilities Matrix](#26-feature-gates--capabilities-matrix)
3. [Component Architecture & Multi-Component Coordination](#3-component-architecture--multi-component-coordination)
   - [3.1 kube-apiserver: Admission, Validation & Watch Pipelines](#31-kube-apiserver-admission-validation--watch-pipelines)
   - [3.2 kube-scheduler Framework Architecture & Extension Points](#32-kube-scheduler-framework-architecture--extension-points)
   - [3.3 kubelet: Node-Pressure Eviction Subsystem vs. Scheduler Preemption](#33-kubelet-node-pressure-eviction-subsystem-vs-scheduler-preemption)
   - [3.4 Workload Controllers, Gang Operators & Cluster Autoscaler Coordination](#34-workload-controllers-gang-operators--cluster-autoscaler-coordination)
4. [In-Depth Preemption Algorithms & State Machines](#4-in-depth-preemption-algorithms--state-machines)
   - [4.1 Default Single-Pod Preemption Pipeline (DefaultPreemption)](#41-default-single-pod-preemption-pipeline-defaultpreemption)
   - [4.2 Workload-Aware Gang Preemption Pipeline (GenericWorkload / KEP-5710 / KEP-6012)](#42-workload-aware-gang-preemption-pipeline-genericworkload--kep-5710--kep-6012)
   - [4.3 Asynchronous Preemption Pipeline & Queue Invariants (KEP-4832)](#43-asynchronous-preemption-pipeline--queue-invariants-kep-4832)
   - [4.4 In-Place Pod Vertical Scaling (IPPVS) Preemption Pipeline](#44-in-place-pod-vertical-scaling-ippvs-preemption-pipeline)
   - [4.5 End-to-End State Machine Diagrams](#45-end-to-end-state-machine-diagrams)
5. [Unified Taxonomy of Preemption Failure Modes, Race Conditions & Root Causes](#5-unified-taxonomy-of-preemption-failure-modes-race-conditions--root-causes)
   - [5.1 Failure Domain 1: Scheduling Queue Starvation & Gating Invariants](#51-failure-domain-1-scheduling-queue-starvation--gating-invariants)
   - [5.2 Failure Domain 2: Victim Ordering, Determinism & PDB Edge Cases](#52-failure-domain-2-victim-ordering-determinism--pdb-edge-cases)
   - [5.3 Failure Domain 3: Concurrency Hazards & Multi-Actor Race Conditions](#53-failure-domain-3-concurrency-hazards--multi-actor-race-conditions)
   - [5.4 Failure Domain 4: Workload-Aware & Gang Preemption Hazards](#54-failure-domain-4-workload-aware--gang-preemption-hazards)
   - [5.5 Failure Domain 5: In-Place Resize & Node Coordination Hazards](#55-failure-domain-5-in-place-resize--node-coordination-hazards)
   - [5.6 Failure Domain 6: Topology, Storage & Cluster Constraint Interlocks](#56-failure-domain-6-topology-storage--cluster-constraint-interlocks)
6. [Developer Architecture, Test Infrastructure & Benchmarking Guidelines](#6-developer-architecture-test-infrastructure--benchmarking-guidelines)
   - [6.1 Test Suite Partitioning & Timeout Exhaustion Prevention](#61-test-suite-partitioning--timeout-exhaustion-prevention)
   - [6.2 Test Performance Optimization: Shared API Server Lifecycle](#62-test-performance-optimization-shared-api-server-lifecycle)
   - [6.3 Test Concurrency, Isolation & Flake Prevention Patterns](#63-test-concurrency-isolation--flake-prevention-patterns)
   - [6.4 Performance Benchmarking Methodologies & Ratio Calibrations](#64-performance-benchmarking-methodologies--ratio-calibrations)
7. [Operational Runbook, Troubleshooting & Observability Manual](#7-operational-runbook-troubleshooting--observability-manual)
   - [7.1 Diagnostic Workflows with kubectl, Events & Scheduler Logs](#71-diagnostic-workflows-with-kubectl-events--scheduler-logs)
   - [7.2 Prometheus Metrics Reference & Monitoring Guide](#72-prometheus-metrics-reference--monitoring-guide)
   - [7.3 Production Alerting Rules & SLI/SLO Guidelines](#73-production-alerting-rules--slislo-guidelines)
   - [7.4 Operational Best Practices & Tuning Guide](#74-operational-best-practices--tuning-guide)
8. [Cross-Domain Comparative Reference Matrix](#8-cross-domain-comparative-reference-matrix)

---

## 1. Executive Summary & Architectural Foundations

### 1.1 The Multi-Paradigm Preemption Ecosystem

In modern Kubernetes clusters hosting heterogeneous workloads—ranging from latency-sensitive web services and microservices to large-scale distributed AI/ML training jobs, stateful databases, and dynamically autoscaling enterprise applications—resource contention is inevitable. **Pod Preemption** is the foundational mechanism ensuring that high-priority, mission-critical workloads obtain immediate compute, memory, storage, and topology resources even when the cluster is fully saturated.

Historically, Kubernetes preemption operated strictly as an in-band, synchronous, per-pod, per-node mechanism within `kube-scheduler`. Over recent development cycles (Kubernetes v1.30 through v1.37+ / 2025–2026), the preemption engine has evolved into a sophisticated, multi-paradigm subsystem spanning five distinct architectural domains:

1. **Default Single-Pod Preemption (`DefaultPreemption`)**: The core in-tree scheduling framework plugin executing at the `PostFilter` extension point. It discovers candidate nodes, calculates minimal victim sets, enforces strict PodDisruptionBudget (PDB) constraints, and coordinates with external scheduler extenders.
2. **Workload-Aware Gang Preemption (WAP / `GenericWorkload` / KEP-5710 & KEP-6012)**: Extends preemption from single pods to cohesive workload units (`PodGroup` and `CompositePodGroup`). It guarantees "all-or-nothing" scheduling semantics, preventing partial gang starvation, enforcing tree-level priority invariants, and coordinating multi-node victim reprieval and `NominatedNodeName` tracking across distributed topologies.
3. **Asynchronous Preemption (KEP-4832 & Scheduler Async API Calls)**: Decouples preemption evaluation from API deletion actuation. By executing victim evictions in asynchronous goroutines, it eliminates head-of-line (HoL) blocking on the active scheduling path while preserving strict queue invariants, avoiding inter-preemptor race collisions, and supporting non-destructive in-memory preemption during `Permit` and `PreBind` phases.
4. **In-Place Pod Vertical Scaling Preemption (IPPVS / KEP-1287 & KEP-5836)**: Enables running pods to dynamically resize their CPU and memory requests in place without restarting containers. When target nodes lack headroom for resizing, `kube-scheduler` orchestrates preemption of lower-priority pods on the specific node while coordinating status handshakes (`Deferred` conditions) with `kubelet`.
5. **Node-Pressure Local Eviction (`kubelet`)**: The node-level physical safety system. Executing independently of scheduling priorities, it reclaims hardware stability under memory, disk, or PID exhaustion based on real-time usage and Quality of Service (QoS) classes.

```
+========================================================================================================+
|                                    KUBERNETES PREEMPTION SUBSYSTEM                                     |
+========================================================================================================+
|                                                                                                        |
|  +--------------------------------------------------------------------------------------------------+  |
|  |                                  CONTROL PLANE (kube-scheduler)                                  |  |
|  |                                                                                                  |  |
|  |   [ Scheduling Framework Pipeline ]                                                              |  |
|  |         |                                                                                        |  |
|  |         +---> PreFilter / Filter (NodeResourcesFit, VolumeBinding, TopologySpread, InterPodAffinity) |
|  |         |                                                                                        |  |
|  |         +---> PostFilter Preemption Extensions:                                                  |  |
|  |                 |                                                                                |  |
|  |                 +-- [ DefaultPreemption ] ---------> Single-Pod Minimal Victim Selection         |  |
|  |                 |                                    (Extender hooks, PDBs, Deterministic Sort)  |  |
|  |                 |                                                                                |  |
|  |                 +-- [ PodGroupPostFilter ] --------> Workload-Aware Gang Preemption (WAP)       |  |
|  |                 |                                    (CompositePodGroups, Tree Hierarchy,        |  |
|  |                 |                                     Atomic Disruption, Monotonic Reprieval)    |  |
|  |                 |                                                                                |  |
|  |                 +-- [ InPlaceResizePreemption ] ---> In-Place Pod Vertical Scaling (IPPVS)       |  |
|  |                                                      (Delta resource fit, Node-pinned preemption)|  |
|  |                                                                                                  |  |
|  |   [ Preemption Execution Engines ]                                                               |  |
|  |         |                                                                                        |  |
|  |         +---> In-Memory Evaluator (Dry-run node cloning, victim candidate synthesis)             |  |
|  |         +---> Async Preemption Executor (KEP-4832 goroutines, fast-fail rollback, 404 toleration)|  |
|  |         +---> Non-Destructive In-Memory Engine (PreBind context cancellation, Permit routing)   |  |
|  |         +---> SchedulingQueue Engine (activeQ, backoffQ, unschedulablePods, gatedPods)           |  |
|  +-------------------------------------------------+------------------------------------------------+  |
|                                                    |                                                   |
|                                    API Server Watch Events / Patches                                   |
|                                                    |                                                   |
|  +-------------------------------------------------v------------------------------------------------+  |
|  |                                  DATA PLANE (Worker Nodes / kubelet)                             |  |
|  |                                                                                                  |  |
|  |   [ Kubelet Node Subsystems ]                                                                    |  |
|  |         +---> Pod Lifecycle & Graceful Termination (preStop hooks, SIGTERM, SIGKILL)             |  |
|  |         +---> In-Place Resize Actuator (Cgroups v2 dynamic reallocation, Local admission bypass) |  |
|  |         +---> Node-Pressure Eviction Manager (MemoryPressure, DiskPressure, PIDPressure)       |  |
|  |               (Strict hardware safeguard based on usage vs request & QoS; ignores PDBs)          |  |
|  +--------------------------------------------------------------------------------------------------+  |
|                                                                                                        |
+========================================================================================================+
```

### 1.2 Scope, Objectives, and Architectural Tenets

This manual provides an exhaustive, code-level architectural reference and operational runbook consolidating research, root-cause analyses, state machine definitions, and failure mode taxonomies across all preemption domains.

#### Core Architectural Invariants:
1. **Schedulability Monotonicity**: Adding or reprieving candidate nodes and victims must never cause a previously valid scheduling plan to regress into an unschedulable state.
2. **Deterministic Victim Ordering**: Given identical cluster state and timestamp precision, victim selection across nodes must yield mathematically identical, deterministic outcomes to prevent scheduler cache thrashing.
3. **Queue Liveness & Anti-Starvation**: Gated, nominated, and unschedulable preemptor pods must receive timely event notifications and equal queue flush frequencies, preventing indefinite residency in unschedulable queues.
4. **Hierarchical Consistency**: Composite workload hierarchies must maintain uniform priority and preemption policies across all ancestor, sibling, and leaf entities to prevent self-preemption and scheduling deadlocks.
5. **Non-Destructive Lifecycle Preemption**: When a preemptor can be accommodated by canceling an in-flight binding cycle (`PodsInPreBind`) or unblocking a waiting permit (`WaitOnPermit`), the scheduler must actuate in-memory cancellation without issuing destructive API deletion calls.

---

## 2. Core Primitives, APIs & Configuration Specifications

### 2.1 PriorityClass API & Preemption Policies

The `PriorityClass` (`scheduling.k8s.io/v1`) resource defines priority values and preemption behaviors for workloads:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-apps
value: 1000000
globalDefault: false
preemptionPolicy: PreemptLowerPriority
description: "Mission-critical production applications requiring immediate placement."
```

#### Field Specifications & Invariants:
* `value` (`int32`): Priority integer ranging from `-2,147,483,648` to `1,000,000,000`. Values exceeding `1,000,000,000` are reserved for internal Kubernetes control plane components:
  - `system-cluster-critical` (`2,000,000,000`): Cluster-wide infrastructure add-ons (CoreDNS, CNI plugins, CSI node drivers).
  - `system-node-critical` (`2,000,001,000`): Node-level daemons that must never be terminated (e.g., node-problem-detector).
* `globalDefault` (`bool`): When `true`, pods without an explicit `priorityClassName` inherit this priority value. Exactly one `PriorityClass` in the cluster may specify `globalDefault: true`. If none exists, unassigned pods default to priority `0`.
* `preemptionPolicy` (`v1.PreemptionPolicy`):
  - `PreemptLowerPriority` (Default): Pods can evaluate and evict lower-priority pods when unschedulable.
  - `Never` (Non-Preempting Priority): Pods jump to the head of the `SchedulingQueue` ahead of lower-priority workloads based on `value`, but **never** trigger preemption of active running pods.

### 2.2 Pod Priority, NominatedNodeName & Scheduling Invariants

When a pod is submitted to `kube-apiserver`:
1. The `Priority` admission controller maps `spec.priorityClassName` to its resolved integer `spec.priority` and sets `spec.preemptionPolicy`. Both fields are immutable once admitted.
2. If `kube-scheduler` fails to find a fitting node and initiates preemption, it selects a candidate node and writes `status.nominatedNodeName` to the pod via an API status patch.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: production-inference-worker
  namespace: ml-workloads
spec:
  priorityClassName: high-priority-apps
  priority: 1000000
  preemptionPolicy: PreemptLowerPriority
  containers:
  - name: engine
    image: nvcr.io/nvidia/triton:25.01
    resources:
      requests:
        cpu: "8"
        memory: "32Gi"
        nvidia.com/gpu: "2"
status:
  phase: Pending
  nominatedNodeName: worker-gpu-node-04
```

#### NominatedNodeName Invariants:
* `status.nominatedNodeName` is an informational hint tracking where the preemptor is expected to fit once victims terminate.
* **Non-Reservation Semantics**: Setting `nominatedNodeName` does **not** create an exclusive lock on the node. In subsequent scheduling cycles, if a different node becomes available sooner (e.g., via voluntary pod exits or scale-up), the preemptor will bind to the new node and clear `nominatedNodeName`.
* **Cross-Pod Stealing Guard**: During scheduling cycles for other pods, the scheduler treats nominated pods as if they were already running on their nominated nodes when evaluating lower-priority pods. Higher-priority pods, however, can preempt the nominated node or steal the released capacity.

### 2.3 PodDisruptionBudgets (PDBs) and In-Tree Evaluation Semantics

A `PodDisruptionBudget` (`policy/v1`) enforces availability bounds during voluntary disruptions:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: payment-service-pdb
  namespace: finance
spec:
  minAvailable: 80%
  selector:
    matchLabels:
      app: payment-service
```

#### In-Tree Preemption Evaluation Contract:
* The preemption engine partitions candidate victim pods into two sets: **PDB-Violating Victims** and **Non-PDB-Violating Victims**.
* **Empty Selector (`{}`) Semantics (PR #141785 / commit `82dead7c815`)**: A PDB with an empty selector (`spec.selector: {}`) targets **all pods in the namespace**. The scheduler preemption algorithm strictly respects this invariant and handles unlabeled victim pods correctly without skipping them.
* **Violation Fallback**: PDBs protect against voluntary operational drains, not hard resource deadlocks. If no candidate node can accommodate the preemptor without violating a PDB, the scheduler falls back to selecting nodes with the minimum number of PDB violations.

### 2.4 Workload & Gang Scheduling APIs (PodGroup, CompositePodGroup, GenericPodGroup)

Under the `GenericWorkload` feature gate (KEP-5710 & KEP-6012), workloads can be grouped into hierarchical structures:

```yaml
apiVersion: scheduling.k8s.io/v1alpha3
kind: CompositePodGroup
metadata:
  name: distributed-training-run-01
  namespace: ml-workloads
spec:
  minMember: 9
  priority: 500000
  preemptionPolicy: PreemptLowerPriority
  disruptionMode: DisruptionModeAll
  childPodGroups:
  - name: master-coordinator
    minMember: 1
    priority: 500000
    preemptionPolicy: PreemptLowerPriority
    disruptionMode: DisruptionModeAll
  - name: worker-nodes
    minMember: 8
    priority: 500000
    preemptionPolicy: PreemptLowerPriority
    disruptionMode: DisruptionModeAll
```

#### Workload API Primitives:
* **`GenericPodGroup` (`fwk.PodGroupInfo`)**: Unified framework interface abstracting single `PodGroup` and `CompositePodGroup` entities. Exposes `GetKey()`, `GetPriority()`, `GetPreemptionPolicy()`, `GetDisruptionMode()`, and `GetAllUnscheduledPods()`.
* **`DisruptionMode`**:
  - `DisruptionModeAll`: When any pod in the composite hierarchy is selected as a victim, the entire composite subtree is evicted atomically.
  - `DisruptionModePodGroup`: Evictions are scoped to individual child `PodGroup` boundaries.
  - `DisruptionModePod`: Individual pods may be evicted independently.
* **`WorkloadForest`**: Hierarchical tree maintained in the scheduler cache tracking ancestor-descendant relationships across composite workloads.

### 2.5 In-Place Pod Vertical Scaling (IPPVS) Preemption API (KEP-1287 / KEP-5836)

Under `InPlacePodVerticalScalingSchedulerPreemption` (Alpha in v1.37), pods can resize resources dynamically:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dynamic-cache-node
  namespace: caching
spec:
  resizePolicy:
  - resourceName: cpu
    restartPolicy: NotRequired
  - resourceName: memory
    restartPolicy: NotRequired
  containers:
  - name: redis
    image: redis:7.2
    resources:
      requests:
        cpu: "4"       # Resized in-place from 2 to 4
        memory: "16Gi"  # Resized in-place from 8Gi to 16Gi
status:
  resize: InProgress
  allocatedResources:
    cpu: "2"
    memory: "8Gi"
  conditions:
  - type: PodResizeDeferred
    status: "True"
    reason: "NodeCapacityDeficit"
    message: "Insufficient capacity on node worker-01; scheduler preemption triggered"
```

#### API Lifecycle Fields:
* `spec.resizePolicy`: Defines whether resizing requires container restart (`NotRequired` vs `RestartContainer`).
* `status.allocatedResources`: The node capacity currently allocated to the pod by the node runtime.
* `status.resize`: State indicator (`Proposed`, `InProgress`, `Deferred`, `Infeasible`).
* `PodResizeDeferred` Condition: Set by `kubelet` when local node headroom is insufficient; watched by `kube-scheduler` to initiate targeted node preemption.

### 2.6 Feature Gates & Capabilities Matrix

| Feature Gate | Default Stage | Primary Purpose & Architectural Impact |
| :--- | :--- | :--- |
| `GenericWorkload` | Beta (v1.33+) | Consolidates `GangScheduling` and `WorkloadAwarePreemption` (PR #139520); enables `GenericPodGroup`, `CompositePodGroup`, and `PodGroupPostFilter`. |
| `InPlacePodVerticalScaling` | Beta (v1.33+) | Enables Kubelet in-place container resizing without restarting pods. |
| `InPlacePodVerticalScalingSchedulerPreemption` | Alpha (v1.37+) | Extends `kube-scheduler` to execute preemption on target nodes when in-place pod resize requests are deferred due to capacity deficits (KEP-5836). |
| `SchedulerAsyncAPICalls` / `AsyncPreemption` | Beta (v1.32+) | Asynchronous victim deletion goroutines offloaded from scheduling cycle (KEP-4832). |
| `TopologySpreadConstraints` | GA (v1.19+) | Evaluates topology skew during preemption dry-run filter checks. |
| `ReadWriteOncePod` | GA (v1.30+) | Single-pod volume access mode requiring serial preemption evaluation. |

---

## 3. Component Architecture & Multi-Component Coordination

```
                                  +---------------------------------------+
                                  |            kube-apiserver             |
                                  |  - Priority Admission Controller      |
                                  |  - CompositePodGroup Validation       |
                                  |  - In-Place Resize Schema Engine      |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         | (Watch Pods, Nodes, PDBs, CPGs)                         | (Watch Resize Status & Pods)
                         v                                                         v
  +-----------------------------------------------+             +-----------------------------------------------+
  |                kube-scheduler                 |             |                    kubelet                    |
  |                                               |             |                                               |
  |  +-----------------------------------------+  |             |  +-----------------------------------------+  |
  |  | SchedulingQueue                         |  |             |  | Pod Lifecycle Handler                   |  |
  |  | - activeQ (PriorityQueue)               |  |             |  | - Graceful Termination (preStop/SIGTERM)|  |
  |  | - backoffQ (Backoff Rate-Limiter)       |  |             |  | - Pod Eviction Finalization             |  |
  |  | - unschedulablePods (Event Flush Map)   |  |             |  +-----------------------------------------+  |
  |  | - gatedPods (Wildcard Event Watchers)   |  |             |  +-----------------------------------------+  |
  |  +-----------------------------------------+  |             |  | In-Place Resize Subsystem               |  |
  |  +-----------------------------------------+  |             |  | - Headroom check & Deferred trigger     |  |
  |  | Framework PostFilter Extensions         |  |             |  | - Cgroup dynamic resource update        |  |
  |  | - DefaultPreemption (Single-Pod)        |  |             |  | - Bypass local admission preemption     |  |
  |  | - PodGroupPostFilter (WAP Gang)         |  |             |  +-----------------------------------------+  |
  |  | - InPlaceResizePreemption (IPPVS)       |  |             |  +-----------------------------------------+  |
  |  +-----------------------------------------+  |             |  | Node-Pressure Eviction Subsystem        |  |
  |  +-----------------------------------------+  |             |  | - Evaluates Real-Time Usage vs Capacity |  |
  |  | Execution & In-Memory Engines           |  |             |  | - Evicts Low QoS / High Usage Pods      |  |
  |  | - Async Preemption Worker Pool          |  |             |  | - Ignores PDBs (Node hardware safeguard)|  |
  |  | - PodsInPreBind Cancellation Registry   |  |             |  +-----------------------------------------+  |
  |  | - WaitingPod (Permit) Registry          |  |             +-----------------------------------------------+
  |  +-----------------------------------------+  |
  +-----------------------------------------------+
```

### 3.1 kube-apiserver: Admission, Validation & Watch Pipelines

1. **Admission Enforcement**: The `Priority` admission plugin validates that pod priority values match declared `PriorityClass` limits and sets `spec.preemptionPolicy`.
2. **Workload Hierarchy Validation (PR #141930)**: Validates that child pod groups in a `CompositePodGroup` hierarchy declare identical `priority`, `preemptionPolicy`, and `schedulerName` values.
3. **Resize Validation (PR #140000 / commits `9f663aa01a8`, `efb871ca6bb`)**: Enforces immutability of `spec.resizePolicy` after creation and ensures resource resize requests do not violate namespace `LimitRange` or `ResourceQuota` policies.
4. **Optimistic Concurrency Control**: Handles resource versioning for `nominatedNodeName` patches and pod deletions. When concurrent preemption occurs, stale update attempts return `409 Conflict` or `404 Not Found`, which are handled by the scheduler's cache reconciliation loop.

### 3.2 kube-scheduler Framework Architecture & Extension Points

The scheduling cycle executes synchronously on a single goroutine per cycle, while the binding cycle runs concurrently in separate goroutines:

* **`PreFilter` / `Filter`**: Evaluates whether nodes have sufficient resources (`NodeResourcesFit`), volume attachment limits (`VolumeBinding`), and satisfy affinity and topology rules (`InterPodAffinity`, `PodTopologySpread`). For in-place resizing pods, `NodeName` PreFilter pins evaluation to the pod's existing node.
* **`PostFilter`**: Executed when all nodes fail the `Filter` phase. Evaluates `DefaultPreemption`, `PodGroupPostFilter`, and `InPlaceResizePreemption` plugins.
* **`Permit`**: Allows plugins to delay binding (e.g., gang barrier synchronization). Pods waiting on permits reside in the `WaitingPod` registry.
* **`PreBind`**: Executes final binding preconditions (e.g., volume provisioning, network attachment). In-flight prebind pods are tracked in `PodsInPreBind`.
* **`Reserve` & `Unreserve`**: Updates in-memory node state cache when pods are assigned or canceled.

### 3.3 kubelet: Node-Pressure Eviction Subsystem vs. Scheduler Preemption

`kubelet` hosts two distinct preemption-related subsystems:
1. **Node-Pressure Eviction Subsystem**:
   - Evaluates physical node metrics (memory available, disk inodes/space, PID capacity).
   - When hard eviction thresholds are breached (e.g., `memory.available < 100Mi`), `kubelet` terminates pods based on **QoS Class** and **Usage exceeding Requests**.
   - **Critical Distinction**: Kubelet eviction **completely ignores PDBs** and scheduling `PriorityClass` integers when hard thresholds trigger.
2. **In-Place Resize Actuator & Local Admission Bypass (PR #140000 / commit `36e85e715eb`)**:
   - Detects when container resizing exceeds node allocatable capacity.
   - Sets `PodResizeDeferred=True` condition to notify `kube-scheduler`.
   - Bypasses local Kubelet admission eviction to prevent conflicting local terminations while scheduler preemption coordinates victim removal.

### 3.4 Workload Controllers, Gang Operators & Cluster Autoscaler Coordination

* **Cluster Autoscaler (CA)**: When a high-priority pod triggers preemption, CA evaluates whether preemption or node group scale-up is faster. CA respects `status.nominatedNodeName` and pauses scale-up on nodes where preemption is actively clearing capacity.
* **Gang Operators (Kubeflow, Volcano, Ray)**: Interface with `CompositePodGroup` and `PodGroup` objects. If gang preemption fails to find cluster-wide capacity within a configurable timeout, operators handle gang backoff or checkpointing.

---

## 4. In-Depth Preemption Algorithms & State Machines

### 4.1 Default Single-Pod Preemption Pipeline (DefaultPreemption)

When a single pod fails scheduling, `DefaultPreemption` executes a deterministic five-phase pipeline:

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
                     | - Parallel evaluation across all schedulable nodes       |
                     | - Filter extenders receive placeholder empty candidates  |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 3: Per-Node Minimal Victim Selection               |
                     | - Sort victims using deterministic MoreImportantVictim   |
                     | - Partition victims into Non-PDB and PDB sets            |
                     | - Reprieve victims iteratively while preserving fit      |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 4: Best Candidate Node Selection (Tie-Breaking)    |
                     | 1. Minimum PDB Violations (0 preferred)                  |
                     | 2. Lowest Highest Victim Priority                        |
                     | 3. Minimal Total Victim Priority Sum                     |
                     | 4. Minimal Number of Victims                             |
                     | 5. Highest Preemptor Score from Score Plugins            |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | Phase 5: Preemption Actuation & Node Nomination          |
                     | - Evict victims (Async deletion or In-Memory cancellation)|
                     | - Patch status.nominatedNodeName on Preemptor            |
                     +----------------------------------------------------------+
```

#### Key Algorithmic Fixes and Enhancements:
1. **Extender Processing with Empty Victims (PR #135486 / commit `cb33cf457d0`)**:
   - *Problem*: Nodes where the preemptor fit without evicting in-tree pods were discarded prior to invoking filter extenders, preventing extenders with preemption logic from nominating extender-managed victims.
   - *Fix*: Synthesized placeholder empty-victim candidates (`Candidate{Victims: &extenderv1.Victims{}}`) to ensure all potentially schedulable nodes reach `ProcessPreemption`.
2. **PDB Empty Selector `{}` Matching (PR #141785 / commit `82dead7c815`)**:
   - *Problem*: `DefaultPreemption` skipped checking PDBs for unlabeled victim pods and treated `selector.Empty()` as matching no pods instead of matching all pods in the namespace.
   - *Fix*: Refactored `podMatchesPDB` to evaluate `selector.Empty()` as universal namespace match and evaluate unlabeled pods against empty selectors.
3. **Deterministic Victim Ordering (`MoreImportantVictim` PR #140999)**:
   - *Problem*: Comparing unstarted pods (`StartTime == nil`) against started pods used `CreationTimestamp`, causing temporal inversions; pods with equal `StartTime` had non-deterministic sort order.
   - *Fix*: Enforced strict ordering: unstarted pods are always less important than started pods (evicted first). Identical timestamps are tie-broken using string UID lexicographical comparison:
     ```go
     func MoreImportantVictim(pod1, pod2 *v1.Pod) bool {
         if pod1.Spec.Priority != pod2.Spec.Priority {
             return pod1.Spec.Priority > pod2.Spec.Priority
         }
         p1Started, p2Started := pod1.Status.StartTime != nil, pod2.Status.StartTime != nil
         if p1Started != p2Started {
             return p1Started // Started pods are more important than unstarted pods
         }
         if p1Started && p2Started && !pod1.Status.StartTime.Equal(pod2.Status.StartTime) {
             return pod1.Status.StartTime.After(pod2.Status.StartTime.Time)
         }
         if !pod1.CreationTimestamp.Equal(&pod2.CreationTimestamp) {
             return pod1.CreationTimestamp.After(pod2.CreationTimestamp.Time)
         }
         return pod1.UID > pod2.UID // Deterministic tie-breaker
     }
     ```
4. **Decoupling Evaluation from Actuation (PR #136613 & PR #134927)**:
   - Evaluator runs purely in-memory on cloned node snapshots.
   - Pods with `DeletionTimestamp != nil` are skipped during actuation to prevent redundant API deletion requests.

---

### 4.2 Workload-Aware Gang Preemption Pipeline (GenericWorkload / KEP-5710 / KEP-6012)

Gang preemption coordinates multi-pod workload placement across the cluster:

```
                  +-----------------------------------------------------+
                  | Gang Scheduling Fails (rootStatus.Code() == Unsched)|
                  +--------------------------+--------------------------+
                                             |
                                             v
                  +-----------------------------------------------------+
                  | PodGroupPostFilter Preemption Evaluation            |
                  | 1. Build WorkloadForest and traverse hierarchy      |
                  | 2. Validate uniform Priority and PreemptionPolicy   |
                  +--------------------------+--------------------------+
                                             |
                                             v
                  +-----------------------------------------------------+
                  | Cluster-Wide Multi-Node Victim Selection            |
                  | - Atomic eviction for DisruptionModeAll subtrees    |
                  | - Multi-Node Candidate Placement Simulation         |
                  +--------------------------+--------------------------+
                                             |
                                             v
                  +-----------------------------------------------------+
                  | Monotonic Victim Reprieval Algorithm                |
                  | - Iteratively test victim preservation              |
                  | - Guarantee maxScheduledCount never decreases       |
                  +--------------------------+--------------------------+
                                             |
                                             v
                  +-----------------------------------------------------+
                  | Atomic Status & NNN Coordination                    |
                  | - Assign NominatedNodeName across all gang pods     |
                  | - Clear PodGroupCycleState to prevent pollution     |
                  | - Propagate conditions to PodGroupStatus            |
                  +-----------------------------------------------------+
```

#### Key Architectural Features:
1. **Monotonic Victim Reprieval Guarantee (PR #138757 & PR #138886)**:
   - *Problem*: In gang preemption, reprieving a victim on Node $A$ could cause the gang solver to place fewer pods overall if the reprieved pod created topology or affinity conflicts on Node $B$.
   - *Fix*: The reprieval algorithm tracks `maxScheduledCount` monotonically. If reprieving a victim causes the scheduled gang count to drop below `minMember`, the reprieval is immediately rolled back.
2. **Composite Workload Hierarchy Traversal (PR #140634)**:
   - `traverseHierarchyUp` walks from leaf pods to composite root. If any ancestor specifies `DisruptionModeAll`, the entire composite workload is grouped as a single atomic victim.
3. **Cycle State Isolation (PR #140871)**:
   - Calls `cycleState.Clear()` for pod group cycle caches to prevent state leakage between successive gang scheduling attempts.
4. **Snapshot Lister Consistency (PR #140745)**:
   - `PodEligibleToPreemptOthers` accesses `PodGroup` through the frozen snapshot lister rather than raw mutable cache to prevent race conditions during concurrent informers updates.
5. **Authoritative PodGroup Priority (PR #139030)**:
   - `PodGroup.Spec.Priority` overrides individual pod priority fields during gang preemption decisions.

---

### 4.3 Asynchronous Preemption Pipeline & Queue Invariants (KEP-4832)

Asynchronous preemption offloads API deletion calls to background worker goroutines:

```
  [ Scheduling Cycle: Thread 1 ]                 [ Async Worker Pool: Thread 2 ]
  ==============================                 ===============================
  1. Select Victims (Node N)
  2. Synthesize Preemption Plan
  3. Mark IsPodRunningPreemption = true
  4. Dispatch Async Eviction Task -------------> 5. Execute API Deletions (Parallel)
  5. Nominate Node N on Preemptor                   - If 404: Tolerated (Already gone)
  6. Requeue Preemptor to backoffQ/unschedQ         - If Error: Abort subsequent deletions
                                                    - If Success: Await termination
                                                 6. Clear IsPodRunningPreemption
                                                 7. Emit Cluster Event (VictimsDeleted)
                                                          |
                                                          v
                                                 [ SchedulingQueue Engine ]
                                                 - Matches Cluster Event
                                                 - Moves Preemptor -> activeQ
```

#### Queue Invariants and Fixes:
1. **Unschedulable Queue Starvation & Gating (PR #139162, PR #139330, PR #139331)**:
   - *Problem*: Preemptor pods got permanently trapped in `unschedulablePods` when cluster events did not match registered event hints or when gated pods flushed at unequal intervals.
   - *Fix*: Enabled wildcard cluster event re-evaluation for gated pods, enforced proper lifecycle resets of `WasFlushedFromUnschedulable`, and synchronized flush intervals via `FlushTimestamp`.
2. **In-Memory Non-Destructive Preemption (PR #135502 & PR #135719)**:
   - **PreBind Phase Preemption (`PodsInPreBind`)**: When a higher-priority preemptor selects a node occupied by a pod currently in its `PreBind` phase, the scheduler **cancels the prebind context**. The prebind pod immediately unreserves its node capacity and requeues to `backoffQ` without issuing an API Delete call.
   - **Permit Phase Preemption (`WaitOnPermit`)**: If a victim pod is waiting on a permit, its permit channel is rejected, and it is routed cleanly to `backoffQ`.
3. **Multi-Victim Error Abortion (PR #135495)**:
   - If deleting victim $V_1$ fails with an API error (e.g., unauthorized or webhook failure), the async worker immediately halts and does not delete subsequent victims $V_2 \dots V_k$, preventing partial disruption.
4. **Inter-Preemptor Collision Verification (PR #134730)**:
   - `IsPodRunningPreemption` guards against race conditions where a second higher-priority preemptor attempts to preempt the same victims while async eviction is in progress.

---

### 4.4 In-Place Pod Vertical Scaling (IPPVS) Preemption Pipeline

When a running pod requests additional CPU/memory on a saturated node:

```
  +-----------------------------------------------------------------------------------+
  | 1. User updates pod.spec.containers[*].resources.requests                         |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | 2. Kubelet detects capacity deficit; sets status.resize = Deferred                 |
  |    and condition PodResizeDeferred = True; bypasses local preemption              |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | 3. kube-scheduler Enqueues Deferred Pod via Resize Event Handler                  |
  |    - NodeName PreFilter pins evaluation to pod's assigned node                    |
  |    - NodeResourcesFit calculates Delta = ResizeAllocated - AllocatedResources     |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | 4. PostFilter Preemption executes on Target Node                                  |
  |    - Evicts lower-priority victim pods on the same node to free Delta capacity    |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | 5. Victims Terminate -> Headroom Released -> Kubelet updates Cgroups & Allocations|
  |    - status.resize transitions: Deferred -> InProgress -> Completed               |
  +-----------------------------------------------------------------------------------+
```

#### Core Components:
* **Delta Fit Calculation**: Evaluates $\Delta = R_{target} - R_{current}$. Only the delta capacity is checked during `Filter` and reclaimed during `PostFilter`.
* **Target Node Pinning**: Unlike new pod scheduling which scans all nodes, IPPVS preemption strictly targets the node where the resizing pod is already bound.
* **Dynamic Node Preemption Policy (`2fa5a2eda68`)**: Supports runtime policy updates for in-place resize preemption.

---

### 4.5 End-to-End State Machine Diagrams

#### Pod Preemption and Queue Lifecycle State Machine

```
                                +-------------------+
                                |    Pod Created    |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |      activeQ      |
                                +---------+---------+
                                          |
                                    ScheduleOne()
                                          |
                        +-----------------+-----------------+
                        | (Fit Found)                       | (All Nodes Fail Filter)
                        v                                   v
              +-------------------+               +-------------------+
              | Reserve / Permit  |               | PostFilter Phase  |
              +---------+---------+               +---------+---------+
                        |                                   |
                  (Permit Pass)                     Preemption Evaluator
                        |                                   |
                        v                         +---------+---------+
              +-------------------+               | (Victims Found)   | (No Candidates)
              |     PreBind       |               v                   v
              +---------+---------+     +-------------------+   +-------------------+
                        |               | Nominate Node     |   | unschedulablePods |
                        v               | Trigger Evictions |   +-------------------+
              +-------------------+     +---------+---------+             ^
              |    Bound / Run    |               |                       |
              +-------------------+               v                 (Queue Flush)
                                        +-------------------+             |
                                        |     backoffQ      |-------------+
                                        +-------------------+
```

#### Gang Preemption & Composite Hierarchy State Machine

```
         +-------------------------------------------------------------+
         |              CompositePodGroup Submission                   |
         +------------------------------+------------------------------+
                                        |
                                        v
         +-------------------------------------------------------------+
         | Hierarchical Consistency Validation                         |
         | (Enforce equal Priority, PreemptionPolicy, SchedulerName)   |
         +------------------------------+------------------------------+
                                        |
                         +--------------+--------------+
                         | (Valid)                     | (Validation Error)
                         v                             v
         +------------------------------+    +------------------------------+
         | GenericWorkload Scheduling   |    | Reject PodGroup / Mark Failed|
         +--------------+---------------+    +------------------------------+
                        |
            +-----------+-----------+
            | (All Gang Fits)       | (Gang Unschedulable)
            v                       v
  +--------------------+  +----------------------------------------------------+
  | Bind All Gang Pods |  | PodGroupPostFilter Preemption                      |
  +--------------------+  | - Traverse hierarchy up to highest DisruptionAll   |
                          | - Evaluate multi-node atomic victim sets           |
                          | - Monotonic reprieval (maxScheduledCount check)    |
                          +-------------------------+--------------------------+
                                                    |
                                     +--------------+--------------+
                                     | (Victims Found)             | (No Fit)
                                     v                             v
                          +--------------------+         +--------------------+
                          | Assign NNN to Gang |         | Requeue to backoff |
                          | Evict Victims      |         +--------------------+
                          +--------------------+
```

---

## 5. Unified Taxonomy of Preemption Failure Modes, Race Conditions & Root Causes

```
+========================================================================================================+
|                                    PREEMPTION FAILURE MODE TAXONOMY                                    |
+========================================================================================================+
|                                                                                                        |
|  [ Category 1: Scheduling Queue & Starvation Invariants ]                                             |
|  * FM-101: Gated Pod Event Starvation (Missing wildcard triggers for gated preemptors)                |
|  * FM-102: Unschedulable Flush Frequency Skew (Unequal flush rates causing starvation)                 |
|  * FM-103: WasFlushedFromUnschedulable Leak (Permanent stuck in backoffQ)                              |
|  * FM-104: Silent In-Memory Deactivation Starvation (Failure to reactivate preemptors after dry-run)  |
|                                                                                                        |
|  [ Category 2: Victim Ordering, Determinism & PDB Mechanics ]                                         |
|  * FM-201: Unstarted Pod Timestamp Inversion (CreationTimestamp vs StartTime inversion)               |
|  * FM-202: Non-Deterministic Equal Timestamp Sort (Missing UID tie-breaker leading to cache thrash)   |
|  * FM-203: Empty PDB Selector {} Inversion (Unlabeled pods skipped; universal selector ignored)        |
|  * FM-204: Extender Empty-Victim Node Drop (Premature dropping of candidate nodes for extenders)      |
|                                                                                                        |
|  [ Category 3: Concurrency Hazards & Multi-Actor Races ]                                               |
|  * FM-301: Inter-Preemptor Async Eviction Collision (Higher priority preemptor colliding with ongoing) |
|  * FM-302: Multi-Victim Async Deletion Failure Cascade (Partial victim eviction leaving unusable node) |
|  * FM-303: Terminating Pod Duplicate API Delete Floods (Redundant deletes on terminating pods)         |
|  * FM-304: Storage Slot Hijacking in RWOP Volumes (Parallel e2e/workloads stealing volume slots)       |
|                                                                                                        |
|  [ Category 4: Workload-Aware & Gang Preemption Hazards ]                                              |
|  * FM-401: Monotonicity Schedulability Regression (Reprieval causing scheduled count to drop)         |
|  * FM-402: CycleState Pollution Across Retries (Leaked state across gang scheduling attempts)          |
|  * FM-403: Stale Cache Lister Inconsistency (Reading mutable cache instead of frozen snapshot)        |
|  * FM-404: Child Priority & Policy Divergence (Split priority in CompositePodGroups causing deadlock) |
|  * FM-405: NNN Overwrite by SuggestedHost (Preemption nomination wiped by fallback suggestion)        |
|  * FM-406: Premature Ongoing Preemption Rejection (Prematurely aborting gang preemption)               |
|                                                                                                        |
|  [ Category 5: In-Place Resize & Node Coordination Hazards ]                                          |
|  * FM-501: Deferred Condition Deadlock (Scheduler missing resize event handler enqueue)               |
|  * FM-502: Kubelet Local Admission Collision (Kubelet evicting resizing pod instead of waiting)       |
|  * FM-503: Delta Fit Calculation Oversubscription (Full request evaluated instead of delta)           |
|                                                                                                        |
|  [ Category 6: Topology, Storage & Cluster Constraint Interlocks ]                                     |
|  * FM-601: Topology Spread Skew Inversion (Preemption breaking topology constraints for other pods)   |
|  * FM-602: Affinity & Anti-Affinity Cascade Disruption (Evicting affinity anchor breaking dependents)  |
|  * FM-603: Storage Locality Interlock (Preempting on nodes where volume topology cannot attach)       |
|  * FM-604: Stalled Finalizer / PreStop Hang (Victims stuck in Terminating blocking preemptor)          |
+========================================================================================================+
```

### 5.1 Failure Domain 1: Scheduling Queue Starvation & Gating Invariants

#### Failure Mode FM-101: Gated Pod Event Starvation
* **Mechanism**: Pods with scheduling gates (`spec.schedulingGates`) or unschedulable pods waiting for specific cluster conditions were not re-evaluated upon generic cluster events.
* **Root Cause**: Event handlers in `SchedulingQueue` used narrow event filters that discarded wildcard cluster state transitions for gated pods.
* **Mitigation (PR #139162)**: Implemented wildcard event re-evaluation for all gated and unschedulable workloads.

#### Failure Mode FM-102: Unschedulable Flush Frequency Skew
* **Mechanism**: Pods in `unschedulablePods` were flushed at irregular frequencies depending on queue load, causing high-priority preemptors to wait longer than lower-priority pods.
* **Mitigation (PR #139331)**: Standardized queue flushing using synchronized `FlushTimestamp` markers, ensuring all unschedulable pods flush at equal frequencies.

#### Failure Mode FM-103: WasFlushedFromUnschedulable State Leak
* **Mechanism**: If `WasFlushedFromUnschedulable` remained set after a pod transitioned through `activeQ` to `backoffQ`, subsequent backoff calculations were corrupted.
* **Mitigation (PR #139330)**: Enforced strict lifecycle resets of `WasFlushedFromUnschedulable` whenever a pod enters `activeQ`.

#### Failure Mode FM-104: Silent In-Memory Preemption Deactivation
* **Mechanism**: When in-memory preemption completed dry-run evaluation, preemptor pods were not properly reactivated in the queue if asynchronous deletion was pending.
* **Mitigation (PR #140054 / commit `bd6cae4a3fe`)**: Standardized in-memory preemption return codes and guaranteed preemptor reactivation.

---

### 5.2 Failure Domain 2: Victim Ordering, Determinism & PDB Edge Cases

#### Failure Mode FM-201: Unstarted Pod Timestamp Inversion
* **Mechanism**: Pods that were scheduled but whose containers had not started (`StartTime == nil`) were sorted against started pods using `CreationTimestamp`. A newly created unstarted pod could be judged "more important" than an older running pod.
* **Mitigation (PR #140999 / commit `24127a76250`)**: Formally established that any unstarted pod is less important than any started pod, regardless of creation timestamp.

#### Failure Mode FM-202: Non-Deterministic Equal Timestamp Tie-Breaking
* **Mechanism**: When multiple victim pods had identical `StartTime` values, Go's `sort.Slice` produced non-deterministic victim selection across scheduling cycles, causing oscillating preemption nominations.
* **Mitigation (PR #140999 / commit `8ffd4531cb6`)**: Added lexicographical comparison of pod UIDs (`pod1.UID > pod2.UID`) as the definitive, stable tie-breaker.

#### Failure Mode FM-203: Empty PDB Selector `{}` Inversion
* **Mechanism**: `DefaultPreemption` skipped checking PDBs for unlabeled victim pods and treated `selector.Empty()` as matching no pods.
* **Mitigation (PR #141785 / commit `82dead7c815`)**: Fixed selector matching so `{}` matches all pods in the namespace and protects unlabeled pods under universal PDBs.

#### Failure Mode FM-204: Extender Empty-Victim Node Drop
* **Mechanism**: Nodes where the preemptor fit without evicting in-tree pods were discarded before invoking filter extenders, preventing extenders from identifying extender-managed victims.
* **Mitigation (PR #135486 / commit `cb33cf457d0`)**: Maintained placeholder empty-victim candidates for all schedulable nodes passed to extenders.

---

### 5.3 Failure Domain 3: Concurrency Hazards & Multi-Actor Race Conditions

#### Failure Mode FM-301: Inter-Preemptor Async Eviction Collision
* **Mechanism**: While async preemption was evicting victims on Node $N$ for Preemptor $P_1$, a higher-priority Preemptor $P_2$ arrived and attempted to preempt the same victims, causing double-eviction races.
* **Mitigation (PR #134730)**: Added `IsPodRunningPreemption` checks to serialize preemption attempts against active victim sets.

#### Failure Mode FM-302: Multi-Victim Async Deletion Failure Cascade
* **Mechanism**: When evicting a set of victims $\{V_1, V_2, V_3\}$, if $V_1$ deletion failed with an API error, continuing to delete $V_2$ and $V_3$ caused useless disruption since the preemptor could still not fit.
* **Mitigation (PR #135495)**: Fast-fail eviction: any API error during multi-victim deletion immediately halts subsequent deletions.

#### Failure Mode FM-303: Terminating Pod Duplicate API Delete Floods
* **Mechanism**: The scheduler issued API Delete requests for pods already in `Terminating` state (`DeletionTimestamp != nil`).
* **Mitigation (PR #134927)**: Explicitly filtered out pods with `DeletionTimestamp != nil` from preemption candidate sets.

#### Failure Mode FM-304: Storage Slot Hijacking in RWOP Volumes
* **Mechanism**: In parallel test suites or dense clusters, when a `ReadWriteOncePod` (RWOP) volume was freed by preemption, an unrelated pending pod stole the volume attachment slot.
* **Mitigation (PR #135623)**: Applied serial execution guards and strict volume locking during preemption flows.

---

### 5.4 Failure Domain 4: Workload-Aware & Gang Preemption Hazards

#### Failure Mode FM-401: Monotonicity Schedulability Regression
* **Mechanism**: During victim reprieval in gang preemption, reprieving a victim on Node $A$ caused total scheduled gang members to drop below `minMember` due to cluster-wide constraints.
* **Mitigation (PR #138757 & PR #138886)**: Monitored `maxScheduledCount` monotonically during reprieval; rolled back any victim reprieval that caused scheduled count to regress.

#### Failure Mode FM-402: CycleState Pollution Across Retries
* **Mechanism**: Internal plugin data stored in `CycleState` during an initial gang preemption attempt leaked into subsequent retry attempts within the same cycle.
* **Mitigation (PR #140871)**: Added explicit `cycleState.Clear()` calls between gang preemption passes.

#### Failure Mode FM-403: Stale Cache Lister Inconsistency
* **Mechanism**: `PodEligibleToPreemptOthers` queried the live informer cache directly while the scheduler was operating on a frozen snapshot.
* **Mitigation (PR #140745)**: Refactored eligibility checks to strictly consume snapshot listers (`fwk.PodGroupLister`).

#### Failure Mode FM-404: Child Priority & Policy Divergence in Composite Hierarchies
* **Mechanism**: A `CompositePodGroup` with priority `1000` containing a child `PodGroup` with priority `100` caused self-preemption or inverted scheduling order.
* **Mitigation (PR #141930)**: Enforced strict validation requiring uniform `priority`, `preemptionPolicy`, and `schedulerName` across all levels of composite hierarchies.

#### Failure Mode FM-405: NNN Overwrite by SuggestedHost
* **Mechanism**: In gang scheduling, when a pod received a `nominatedNodeName` from preemption, a subsequent fallback pass overwrote `nominatedNodeName` with an invalid `suggestedHost`.
* **Mitigation (PR #140590)**: Preserved valid preemption nominations against unintended fallback overrides.

#### Failure Mode FM-406: Premature Ongoing Preemption Rejection
* **Mechanism**: Gang scheduling cycles treated in-flight preemption on member nodes as permanent failures, rejecting the entire gang prematurely.
* **Mitigation (PR #140641)**: Added detection for ongoing preemption, allowing the gang to remain in backoff until victim evictions complete.

---

### 5.5 Failure Domain 5: In-Place Resize & Node Coordination Hazards

#### Failure Mode FM-501: Deferred Condition Deadlock
* **Mechanism**: When `kubelet` marked a pod resize as `Deferred`, the scheduler failed to enqueue the pod if it only watched standard pod add/update events.
* **Mitigation (PR #140000)**: Implemented dedicated resize status change event handlers to enqueue deferred pods into `activeQ`.

#### Failure Mode FM-502: Kubelet Local Admission Collision
* **Mechanism**: Kubelet local admission attempted to evict the resizing pod or reject the resize while scheduler preemption was actively freeing capacity.
* **Mitigation (PR #140000 / commit `36e85e715eb`)**: Instructed Kubelet to bypass local admission preemption when handling scheduler-coordinated resizes.

#### Failure Mode FM-503: Delta Fit Calculation Oversubscription
* **Mechanism**: The scheduler evaluated total requested capacity ($R_{target}$) rather than the incremental delta ($\Delta = R_{target} - R_{current}$) on the node where the pod was already running.
* **Mitigation (PR #140000)**: Implemented delta-based resource accounting in `NodeResourcesFit` PreFilter/Filter plugins for in-place resizing pods.

---

### 5.6 Failure Domain 6: Topology, Storage & Cluster Constraint Interlocks

#### Failure Mode FM-601: Topology Spread Skew Inversion
* **Mechanism**: Evicting a victim in Zone $A$ altered the zone skew calculation, causing other workloads or the preemptor itself to violate `DoNotSchedule` topology spread constraints.
* **Mitigation**: Dry-run filter phase executes `PodTopologySpread` simulation against the cloned cluster snapshot including all affected zones.

#### Failure Mode FM-602: Affinity & Anti-Affinity Cascade Disruption
* **Mechanism**: Evicting a victim pod that served as an affinity anchor for co-located pods caused secondary probe failures or application disruption.
* **Mitigation**: Preemption evaluation calculates `InterPodAffinity` impact; operational guidance recommends soft affinity (`preferredDuringSchedulingIgnoredDuringExecution`) for non-critical co-location.

#### Failure Mode FM-603: Storage Locality Interlock
* **Mechanism**: Preempting pods on Node 1 when the preemptor's `PersistentVolumeClaim` was bound to local storage on Node 2.
* **Mitigation**: `VolumeBinding` plugin rejects non-storage-compatible nodes during Phase 2 dry-run filter checks before victim selection occurs.

#### Failure Mode FM-604: Stalled Finalizers and Hanging PreStop Hooks
* **Mechanism**: A victim pod selected for preemption had a hanging `preStop` hook or an unmanaged `metadata.finalizer`, remaining in `Terminating` indefinitely and blocking the preemptor.
* **Mitigation**: Bound `terminationGracePeriodSeconds` on batch pods and configure alerting on pods with `nominatedNodeName` in `Pending` beyond grace thresholds.

---

## 6. Developer Architecture, Test Infrastructure & Benchmarking Guidelines

### 6.1 Test Suite Partitioning & Timeout Exhaustion Prevention

#### Context & Root Cause (PR #141048)
Monolithic integration test packages evaluating multi-dimensional feature gate permutations repeatedly exceeded the 600-second integration test deadline (`KUBE_TIMEOUT`). When timeouts occurred, context cancellation tore down API server instances mid-test, producing cascading "connection refused" flakes across subsequent tests.

#### Resolution Architecture:
* Partitioned monolithic test packages into domain-specific packages:
  - `test/integration/scheduler/preemption`: Standard `DefaultPreemption` and async preemption integration tests.
  - `test/integration/scheduler/preemption/podgroup`: Dedicated `PodGroup` and `CompositePodGroup` gang preemption test suite.
* Isolated heavy feature combinations into independent CI execution lanes.

### 6.2 Test Performance Optimization: Shared API Server Lifecycle

#### Context & Optimization (PR #140737)
Prior to PR #140737, the preemption test harness instantiated fresh `kube-apiserver` and `etcd` processes for every individual subtest (72 startups across feature permutations), requiring ~360 seconds of execution time and regularly failing on slower CI architectures (e.g., ppc64le).

#### Implementation Pattern:
```go
// Shared test context across subtests using WithNewNamespace helper
func TestPreemptionWithSharedAPIServer(t *testing.T) {
    testCtx := testutils.InitTestAPIServer(t, "preemption-suite", nil)
    defer testutils.CleanupTestAPIServer(testCtx)

    for _, tc := range testCases {
        t.Run(tc.name, func(t *testing.T) {
            // Allocate an isolated namespace per subtest on the shared API server
            ns := testutils.WithNewNamespace(testCtx, t, tc.name)
            runPreemptionScenario(testCtx, ns, tc)
        })
    }
}
```
* **Impact**: Reduced API server startups from 72 to 8, cutting test execution from ~360s to ~70s (~80% runtime reduction) while maintaining complete inter-test namespace isolation.

### 6.3 Test Concurrency, Isolation & Flake Prevention Patterns

1. **Namespace-Scoped Extended Resources (PR #140872)**:
   - *Anti-Pattern*: Using hardcoded cluster-wide extended resource names (e.g., `example.com/gpu`) across parallel e2e tests.
   - *Best Practice*: Dynamically scope extended resources to the test namespace: `example.com/<namespace>`.
2. **Deterministic State Synchronization & Mutex Protection (PR #138017)**:
   - *Anti-Pattern*: Relying on arbitrary `time.Sleep()` calls to wait for preemption actuation.
   - *Best Practice*: Initialize plugin dispatch maps prior to scheduler startup, protect shared test registries with `sync.Mutex`, and explicitly poll for `PodScheduled=True` conditions.
3. **Serial Execution Guards for Storage Preemption (PR #135623)**:
   - Annotate tests involving single-pod volume bindings (`ReadWriteOncePod`) with `ginkgo.Serial` / `f.WithSerial()` to prevent parallel tests from hijacking volume slots.
4. **Sequential Staging for Async Preemption Tests (PR #135372)**:
   - Stage pod creation sequentially: create low-priority workloads, verify placement, create medium-priority workloads, verify preemption, and then create high-priority workloads.

### 6.4 Performance Benchmarking Methodologies & Ratio Calibrations

The `test/integration/scheduler_perf` suite provides standardized benchmarks for evaluating preemption scalability:

1. **Topology Spreading Gang Preemption Benchmark (PR #140408)**:
   - Evaluates gang preemption throughput across multi-zone topologies under varying skew constraints (`maxSkew: 1` vs `maxSkew: 3`).
   - Normalizes priority values across all benchmark templates to a uniform standard (`10000`).
2. **PodGroup Disruption Mode Benchmarks (PR #140651)**:
   - Measures scheduler throughput comparing single-pod preemption against `NoPodGroup`, `DisruptionModePodGroup`, and `DisruptionModeAll`.
   - **Calibrated 10:1 Ratio**: Employs a standardized ratio of 10 victim pods per 1 preemptor pod across 1,000-node simulated cluster topologies to generate realistic victim reprieval and PDB evaluation pressure.

---

## 7. Operational Runbook, Troubleshooting & Observability Manual

### 7.1 Diagnostic Workflows with kubectl, Events & Scheduler Logs

#### Diagnostic Workflow: Troubleshooting a Stuck Preemptor Pod

```
                          +---------------------------------------+
                          | Preemptor Pod Stuck in Pending Status |
                          +-------------------+-------------------+
                                              |
                                              v
                          +---------------------------------------+
                          | 1. Inspect Pod Status & Events        |
                          |    kubectl describe pod <pod-name>    |
                          +-------------------+-------------------+
                                              |
                     +------------------------+------------------------+
                     | (nominatedNodeName is set)                      | (nominatedNodeName is empty)
                     v                                                 v
  +---------------------------------------+         +---------------------------------------+
  | 2. Check Nominated Node State         |         | 2. Check Preemption Eligibility       |
  |    - Are victim pods Terminating?     |         |    - Is preemptionPolicy: Never?      |
  |    - Check for stuck finalizers       |         |    - Are there lower-priority pods?   |
  |    - Check preStop hook execution     |         |    - Do PDBs block all candidate nodes?
  +-------------------+-------------------+         +---------------------------------------+
                      |
        +-------------+-------------+
        | (Victims stuck)           | (Victims exited)
        v                           v
  +-----------------------+   +---------------------------------------+
  | Fix hanging finalizer |   | 3. Check for Nominated Node Stealing  |
  | or force delete victim|   |    - Did a higher priority pod bind?  |
  +-----------------------+   |    - Check scheduler logs for races   |
                              +---------------------------------------+
```

#### Key Diagnostic Commands:
```bash
# 1. Check preemptor pod nomination and scheduling events
kubectl get pod <pod-name> -o jsonpath='{.status.nominatedNodeName}'
kubectl get events --field-selector involvedObject.name=<pod-name> --sort-by='.metadata.creationTimestamp'

# 2. Inspect victim pods on the nominated node
kubectl get pods --field-selector spec.nodeName=<nominated-node> -o custom-columns=NAME:.metadata.name,PRIORITY:.spec.priority,PHASE:.status.phase,DELETION:.metadata.deletionTimestamp

# 3. Check for PodDisruptionBudgets in the namespace
kubectl get pdb -n <namespace> -o custom-columns=NAME:.metadata.name,MIN_AVAIL:.spec.minAvailable,ALLOWED:.status.disruptionsAllowed,CURRENT:.status.currentHealthy

# 4. Filter scheduler logs for preemption decisions
kubectl logs -n kube-system -l component=kube-scheduler --tail=500 | grep -E "preempt|victim|nominatedNode"
```

### 7.2 Prometheus Metrics Reference & Monitoring Guide

| Metric Name | Type | Description & Diagnostic Utility |
| :--- | :--- | :--- |
| `scheduler_preemption_attempts_total` | Counter | Total number of preemption evaluation passes initiated by `PostFilter` plugins. |
| `scheduler_preemption_victims` | Counter | Number of victim pods selected for preemption (labeled by extension plugin: `DefaultPreemption`, `PodGroupPostFilter`). |
| `scheduler_preemption_evaluation_duration_seconds` | Histogram | Latency distribution of preemption candidate discovery and victim selection. |
| `scheduler_pod_scheduling_attempts` | Histogram | Number of scheduling cycles a pod undergoes before binding. Elevated values on nominated pods indicate nomination stealing or slow victim exit. |
| `scheduler_pod_group_preemption_attempts_total` | Counter | Total gang preemption passes evaluated under `GenericWorkload`. |
| `scheduler_pod_group_preemption_victims_total` | Counter | Total pods evicted as part of gang or composite disruption boundaries. |
| `scheduler_inplace_resize_preemption_attempts_total` | Counter | Number of targeted node preemption cycles initiated for in-place pod resizing. |
| `scheduler_inplace_resize_preemption_duration_seconds`| Histogram | Latency of in-place resize preemption from `Deferred` condition to victim termination. |
| `kubelet_evictions_total` | Counter | Number of pods evicted locally by Kubelet node-pressure manager (labeled by `eviction_signal`: `memory`, `nodefs`, `pid`). |

### 7.3 Production Alerting Rules & SLI/SLO Guidelines

```yaml
# Prometheus Alerting Rules for Kubernetes Preemption Subsystem
groups:
- name: kubernetes-preemption-alerts
  rules:
  - alert: HighPreemptionRate
    expr: sum(rate(scheduler_preemption_victims[5m])) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High cluster-wide preemption rate"
      description: "Scheduler is preempting > 10 pods/sec over 5m. Indicates severe resource exhaustion or priority misconfiguration."

  - alert: PreemptorPodStarvation
    expr: kube_pod_status_phase{phase="Pending"} == 1 and kube_pod_status_nominated_node != ""
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Preemptor pod stuck with nominatedNodeName"
      description: "Pod {{ $labels.pod }} has had a nominated node for > 10m without binding. Check for stuck finalizers or slow preStop hooks on victim pods."

  - alert: PreemptionEvaluationLatencyHigh
    expr: histogram_quantile(0.99, sum(rate(scheduler_preemption_evaluation_duration_seconds_bucket[5m])) by (le)) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High preemption evaluation latency (p99 > 500ms)"
      description: "Scheduler PostFilter preemption evaluation is taking longer than 500ms at p99. Check cluster node count, PDB density, and extender response latency."

  - alert: KubeletHardNodeEvictionsSpike
    expr: sum(rate(kubelet_evictions_total[5m])) > 2
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Kubelet node-pressure evictions detected"
      description: "Nodes are executing local hardware-pressure evictions (ignoring PDBs). Check node memory/disk utilization."
```

### 7.4 Operational Best Practices & Tuning Guide

1. **Adopt a Multi-Tier Priority Hierarchy**:
   ```
   [2,000,000,000+] System Critical (CoreDNS, CNI DaemonSets, Storage Node Plugins)
         ^
   [1,000,000]     Production Tier 0 (Core APIs, Payment Gateways, Primary Ingress)
         ^
   [500,000]       Production Tier 1 (Background Async Workers, Message Consumers)
         ^
   [100,000]       Non-Production / Staging Workloads
         ^
   [1,000]         Batch / AI Training Jobs (preemptionPolicy: Never or Gang Scheduled)
         ^
   [0]             BestEffort / Scavenger Workloads
   ```
2. **PDB Configuration Safeguards**:
   - Ensure PDBs allow at least 1 disruption whenever possible (`maxUnavailable: 1` or `minAvailable: 90%`). Avoid `maxUnavailable: 0` or `minAvailable: 100%` on non-critical workloads, as this forces the scheduler into PDB violation fallbacks.
   - Do not use empty label selectors (`spec.selector: {}`) unless the intention is explicitly to apply the budget to every pod in the namespace.
3. **Grace Period Management**:
   - Keep `terminationGracePeriodSeconds` on lower-priority batch and worker pods reasonably bounded (e.g., 30s–60s). Excessively long grace periods (e.g., > 300s) directly delay the binding and startup of preemptor pods.
4. **Queue Flush and Gating Tuning**:
   - Maintain equal queue flush intervals across scheduler profiles to prevent gating starvation.
   - Avoid creating pods with unsatisfied scheduling gates unless an external controller is actively managing gate removal.

---

## 8. Cross-Domain Comparative Reference Matrix

| Feature / Dimension | Default Single-Pod (`DefaultPreemption`) | Workload-Aware Gang (`GenericWorkload`) | Asynchronous Preemption (`AsyncPreemption`) | In-Place Resize Preemption (`IPPVS`) | Kubelet Node-Pressure Eviction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Architectural Scope** | Single Pod on Single Node | Multi-Pod Gang on Multi-Node Domain | Decoupled Background API Eviction | Single Resizing Pod on Target Node | Local Node Hardware Safety Subsystem |
| **Trigger Condition** | Unschedulable pod fails all node filters | Gang fails all-or-nothing placement | Scheduled preemptor has chosen victims | Running pod resize marked `Deferred` | Node hardware threshold breach (Mem/Disk/PID) |
| **Governing Entity** | `kube-scheduler` (`PostFilter`) | `kube-scheduler` (`PodGroupPostFilter`) | `kube-scheduler` Background Worker Pool | `kube-scheduler` (`InPlaceResizePreemption`) | `kubelet` Local Eviction Manager |
| **Primary Criterion** | `spec.priority` integer | `PodGroup.Spec.Priority` (Authoritative) | `spec.priority` integer | `spec.priority` & Resize Delta ($R_{\Delta}$) | Actual Resource Usage vs Request & QoS Class |
| **PDB Respect** | Enforces PDBs; violates only as fallback | Enforces PDBs across gang disruption units | Enforces PDBs during candidate selection | Enforces PDBs on target node | **Completely ignores PDBs** |
| **Termination Mechanism** | API Delete or In-Memory Cancellation | API Delete across composite disruption unit | Async Goroutines with fast-fail rollback | API Delete of lower-priority node pods | Local container runtime kill (SIGKILL on hard) |
| **NominatedNode Tracking** | Single pod `status.nominatedNodeName` | Multi-pod NNN across all gang members | Preserves NNN during async eviction | Retains existing node binding | N/A (Pod terminated immediately) |
| **Non-Destructive Preemption** | Cancels `PodsInPreBind` / Rejects Permit | Requeues unstarted gang members | Cancels context; routes to `backoffQ` | N/A (Victims evicted from target node) | N/A (Always destructive) |
| **Relevant Feature Gates** | In-Tree GA | `GenericWorkload` | `SchedulerAsyncAPICalls` | `InPlacePodVerticalScalingSchedulerPreemption` | In-Tree GA |

---
