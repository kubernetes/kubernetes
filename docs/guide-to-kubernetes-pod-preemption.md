# Comprehensive Guide to Kubernetes Pod Preemption

## Table of Contents
1. [Introduction and Executive Summary](#1-introduction-and-executive-summary)
2. [Core Concepts and Primitives](#2-core-concepts-and-primitives)
   - [PriorityClass API](#priorityclass-api)
   - [Pod Priority and Preemption Policy Fields](#pod-priority-and-preemption-policy-fields)
   - [Pod Disruption Budgets (PDBs)](#pod-disruption-budgets-pdbs)
   - [Quality of Service (QoS) Classes vs. Scheduling Priority](#quality-of-service-qos-classes-vs-scheduling-priority)
3. [Component Architecture and Roles](#3-component-architecture-and-roles)
   - [kube-apiserver & Priority Admission Controller](#kube-apiserver--priority-admission-controller)
   - [kube-scheduler & Scheduling Framework](#kube-scheduler--scheduling-framework)
   - [kubelet & Node-Pressure Eviction Subsystem](#kubelet--node-pressure-eviction-subsystem)
   - [Workload Controllers & Autoscalers](#workload-controllers--autoscalers)
4. [In-Depth Scheduling Preemption Algorithm](#4-in-depth-scheduling-preemption-algorithm)
   - [The Scheduling Cycle and PostFilter Extension Point](#the-scheduling-cycle-and-postfilter-extension-point)
   - [Phase 1: Preemption Eligibility Evaluation](#phase-1-preemption-eligibility-evaluation)
   - [Phase 2: Candidate Node Discovery and Sampling](#phase-2-candidate-node-discovery-and-sampling)
   - [Phase 3: Per-Node Minimal Victim Selection](#phase-3-per-node-minimal-victim-selection)
   - [Phase 4: Best Candidate Node Selection (Tie-Breaking)](#phase-4-best-candidate-node-selection-tie-breaking)
   - [Phase 5: Preemption Actuation and Node Nomination](#phase-5-preemption-actuation-and-node-nomination)
   - [Phase 6: The Nominated Pod Cycle and Final Binding](#phase-6-the-nominated-pod-cycle-and-final-binding)
5. [Preemption Flows and Sequence Diagrams](#5-preemption-flows-and-sequence-diagrams)
   - [Flow A: Standard Scheduler Preemption](#flow-a-standard-scheduler-preemption)
   - [Flow B: Non-Preempting Priority Pods (`preemptionPolicy: Never`)](#flow-b-non-preempting-priority-pods-preemptionpolicy-never)
   - [Flow C: Kubelet Node-Pressure Eviction](#flow-c-kubelet-node-pressure-eviction)
   - [Flow D: Gang / PodGroup Preemption](#flow-d-gang--podgroup-preemption)
6. [Detailed Analysis of Corner Cases & Failure Modes](#6-detailed-analysis-of-corner-cases--failure-modes)
   - [1. Preemption Thrashing & Ping-Pong Loops](#1-preemption-thrashing--ping-pong-loops)
   - [2. Nominated Node Stealing & Starvation](#2-nominated-node-stealing--starvation)
   - [3. Affinity and Anti-Affinity Cascades](#3-affinity-and-anti-affinity-cascades)
   - [4. Topology Spread Constraint Interactions](#4-topology-spread-constraint-interactions)
   - [5. PDB Deadlocks and Violation Fallbacks](#5-pdb-deadlocks-and-violation-fallbacks)
   - [6. Volume Binding and Storage Locality Interlocks](#6-volume-binding-and-storage-locality-interlocks)
   - [7. Termination Delays, Stalled Finalizers & PreStop Hooks](#7-termination-delays-stalled-finalizers--prestop-hooks)
   - [8. ResourceQuota & PriorityClass Scope Races](#8-resourcequota--priorityclass-scope-races)
   - [9. Scheduler Cache vs. Etcd State Inconsistencies](#9-scheduler-cache-vs-etcd-state-inconsistencies)
   - [10. DaemonSet Preemption Nuances](#10-daemonset-preemption-nuances)
7. [Scheduler Preemption vs. Kubelet Eviction Comparison](#7-scheduler-preemption-vs-kubelet-eviction-comparison)
8. [Production Best Practices, Tuning & Observability](#8-production-best-practices-tuning--observability)
   - [Designing a Robust Priority Hierarchy](#designing-a-robust-priority-hierarchy)
   - [PDB and Grace Period Tuning](#pdb-and-grace-period-tuning)
   - [Key Metrics and Alerting](#key-metrics-and-alerting)

---

## 1. Introduction and Executive Summary

In a multi-tenant or resource-constrained Kubernetes cluster, high-priority workloads (such as mission-critical services, control-plane agents, or production APIs) must be scheduled reliably even when all nodes are fully saturated. **Pod Preemption** is the mechanism by which Kubernetes makes room for higher-priority pending pods by evicting lower-priority pods already running on cluster nodes.

Preemption in Kubernetes operates across two distinct planes:
1. **Cluster-Wide Placement (Scheduler Preemption)**: Driven by `kube-scheduler`, where unschedulable high-priority pods trigger the graceful termination of lower-priority pods on a target node to release required CPU, memory, storage, ports, or topology domains.
2. **Node-Level Pressure Relief (Kubelet Eviction)**: Driven locally by `kubelet` on an individual worker node when node-level physical limits (memory, disk, PIDs) breach safety thresholds, reclaiming node stability by evicting pods based on resource consumption and QoS ranking.

Understanding the interaction between these components, the intricate multi-stage preemption algorithms, the asynchronous deletion lifecycles, and subtle corner cases (like nominated node stealing and affinity cascades) is vital for architecting resilient Kubernetes infrastructure.

---

## 2. Core Concepts and Primitives

### PriorityClass API

The `PriorityClass` (`scheduling.k8s.io/v1`) is a cluster-scoped API resource that establishes a mapping between a human-readable priority name and a 32-bit signed integer.

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-apps
value: 1000000
globalDefault: false
preemptionPolicy: PreemptLowerPriority
description: "Mission critical production applications requiring immediate placement."
```

#### Field Details:
- `value`: An `int32` ranging from `-2,147,483,648` to `1,000,000,000`. Values greater than `1 billion` are reserved for built-in Kubernetes system components:
  - `system-cluster-critical` (`value: 2000000000`): Critical cluster-wide add-ons (e.g., CoreDNS, Calico/Cilium agents).
  - `system-node-critical` (`value: 2000001000`): Essential node-level daemons that must never be terminated (e.g., node-problem-detector).
- `globalDefault`: When set to `true`, pods without an explicit `priorityClassName` receive this priority value. Only one `PriorityClass` in the cluster may have `globalDefault: true`. If no default exists, pods without a priority class default to priority `0`.
- `preemptionPolicy`:
  - `PreemptLowerPriority` (Default): The pod can preempt lower-priority pods if unschedulable.
  - `Never`: The pod is placed ahead of lower-priority pods in the scheduling queue, but it will **never** trigger preemption of running pods.

### Pod Priority and Preemption Policy Fields

When a pod is submitted:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: production-api
spec:
  priorityClassName: high-priority-apps
  # Automatically populated by Priority Admission Controller:
  # priority: 1000000
  # preemptionPolicy: PreemptLowerPriority
  containers:
  - name: api
    image: nginx
    resources:
      requests:
        cpu: "4"
        memory: "8Gi"
```
- `spec.priority`: Resolved and populated by the API server's admission controller. It is immutable once set.
- `status.nominatedNodeName`: Set by `kube-scheduler` when the pod preempts victims on a specific node, tracking where the preemptor intends to land once victims exit.

### Pod Disruption Budgets (PDBs)

A `PodDisruptionBudget` (`policy/v1`) specifies the minimum number or percentage of replicas that must remain available during voluntary disruptions (such as drains, updates, and scheduler preemption).

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: batch-worker-pdb
spec:
  minAvailable: 80%
  selector:
    matchLabels:
      app: batch-worker
```

During victim selection, `kube-scheduler` attempts to choose candidate nodes and victim pods that **do not violate PDBs**. However, PDBs protect against *voluntary* disruptions—if no node can be found without violating a PDB, the scheduler will, as a last resort, preempt pods even if it breaches their PDBs (unless configured otherwise).

### Quality of Service (QoS) Classes vs. Scheduling Priority

It is crucial not to confuse **Scheduling Priority** with **QoS Class**:
- **Scheduling Priority** (Integer `spec.priority`): Governs `kube-scheduler` queue ordering and scheduler-level preemption decisions across the cluster.
- **QoS Class** (`Guaranteed`, `Burstable`, `BestEffort`): Inferred from `resources.requests` and `resources.limits`. Primarily dictates `kubelet` node-pressure eviction ranking and Linux kernel `oom_score_adj`.

| Attribute | Scheduling Priority | QoS Class |
| :--- | :--- | :--- |
| **Determined By** | `spec.priorityClassName` / `spec.priority` | Container requests and limits configuration |
| **Evaluated By** | `kube-scheduler` (during scheduling cycle) | `kubelet` (during node resource exhaustion) & Linux OOM |
| **Scope** | Cluster-wide pod placement | Local worker node physical resource protection |
| **Preemption Metric**| Numerical integer comparison ($Priority_A > Priority_B$) | Resource usage over request, then QoS tier |

---

## 3. Component Architecture and Roles

```
 +-----------------------------------------------------------------------------------+
 |                                   CONTROL PLANE                                   |
 |                                                                                   |
 |  +--------------------+        +---------------------+        +----------------+  |
 |  |    kube-apiserver  |        |    kube-scheduler   |        |   Workload     |  |
 |  |                    |        |                     |        |  Controllers   |  |
 |  |  +---------------+ |        | +-----------------+ |        | (Deployments,  |  |
 |  |  | Priority      | |        | | PriorityQueue   | |        |  ReplicaSets,  |  |
 |  |  | Admission Ctrl| |        | | (Active/Backoff)| |        |  StatefulSets) |  |
 |  |  +---------------+ |        | +-----------------+ |        +-------+--------+  |
 |  |         |          |        |         |           |                |           |
 |  |         v          |        |         v           |                |           |
 |  |  +---------------+ | Updates| +-----------------+ | Recreates Pods |           |
 |  |  | etcd Database | |<-------| | PostFilter      | |<---------------+           |
 |  |  | (Pods, Nodes, | | Pod    | | (Default        | |                            |
 |  |  |  PDBs)        | | Status | |  Preemption)    | |                            |
 |  |  +---------------+ |        | +-----------------+ |                            |
 |  +---------+----------+        +----------+----------+                            |
 +------------|------------------------------|---------------------------------------+
              |                              |
              | Watches Deletions            | Sends Eviction / Delete Calls
              v                              v
 +-----------------------------------------------------------------------------------+
 |                                   WORKER NODE                                     |
 |                                                                                   |
 |  +-----------------------------------------------------------------------------+  |
 |  | kubelet                                                                     |  |
 |  |                                                                             |  |
 |  |  +------------------------+             +--------------------------------+  |  |
 |  |  | EvictionManager        |             | Pod Lifecycle Manager (PLEG)   |  |  |
 |  |  | (Hard/Soft Thresholds) |             | Graceful Termination (SIGTERM) |  |  |
 |  |  +------------------------+             +---------------+----------------+  |  |
 |  |               |                                         |                   |  |
 |  |               v                                         v                   |  |
 |  |  +-----------------------------------------------------------------------+  |  |
 |  |  | Container Runtime (CRI / containerd / CRI-O)                          |  |  |
 |  |  |                                                                       |  |  |
 |  |  |   [ Victim Pod (Terminating) ]      ----->      [ Preemptor Pod ]     |  |  |
 |  |  |   SIGTERM -> Wait -> SIGKILL                    (Scheduled & Started) |  |  |
 |  |  +-----------------------------------------------------------------------+  |  |
 |  +-----------------------------------------------------------------------------+  |
 +-----------------------------------------------------------------------------------+
```

### kube-apiserver & Priority Admission Controller
1. **Mutating Phase**: Intercepts Pod creation requests. Looks up the `PriorityClass` referenced in `spec.priorityClassName`. Resolves and injects the integer `spec.priority` and string `spec.preemptionPolicy`.
2. **Validating Phase**: Ensures non-system service accounts cannot assign protected system priorities (`system-cluster-critical` or `system-node-critical`).
3. **Immutability Enforcement**: Prevents modifications to `spec.priority` and `spec.priorityClassName` on existing pods.

### kube-scheduler & Scheduling Framework
The scheduler coordinates all cluster-level preemption via the **Scheduling Framework**:
- **Scheduling Queue (`PriorityQueue`)**: Prioritizes pods by `spec.priority`. Higher-priority pods are popped before lower-priority pods.
- **PreFilter / Filter Plugins**: Checks if a node can accommodate the pod (evaluating CPU, Memory, NodeAffinity, Taints/Tolerations, VolumeBinding, TopologySpread, etc.).
- **PostFilter Extension Point**: If a pod cannot fit on **any** node in the cluster, the scheduler enters the `PostFilter` phase. The built-in `DefaultPreemption` plugin runs here to evaluate victim pods across candidate nodes.
- **Scheduler Cache**: Maintains in-memory representations of nodes (`NodeInfo`), running pods, assumed pods, and **nominated pods** (`status.nominatedNodeName`).

### kubelet & Node-Pressure Eviction Subsystem
While the scheduler handles cluster placement, `kubelet` protects individual worker nodes against out-of-resource (OOR) conditions:
- Periodically polls local node metrics against configured thresholds (e.g., `memory.available < 250Mi`, `nodefs.available < 10%`).
- When thresholds are breached, `kubelet` initiates local eviction without consulting `kube-scheduler`.
- Sends `SIGTERM` signals to containers, allows `terminationGracePeriodSeconds`, and forcefully sends `SIGKILL` upon expiration.

---

## 4. In-Depth Scheduling Preemption Algorithm

The preemption algorithm in `pkg/scheduler/framework/plugins/defaultpreemption` follows a structured 6-phase pipeline.

```
+-----------------------------------------------------------------------------------+
|                        PREEMPTION ALGORITHM PIPELINE                              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
                   +-----------------------------------------------+
                   | Phase 1: Preemption Eligibility Evaluation    |
                   | - Check preemptionPolicy != Never             |
                   | - Check if already nominated and still fitting|
                   +-----------------------------------------------+
                                          |
                                          v
                   +-----------------------------------------------+
                   | Phase 2: Candidate Node Discovery & Sampling  |
                   | - Identify nodes where lower-priority pods run|
                   | - Sample nodes based on percentageOfNodes     |
                   +-----------------------------------------------+
                                          |
                                          v
                   +-----------------------------------------------+
                   | Phase 3: Per-Node Minimal Victim Selection    |
                   | - Dry-run: remove all lower-priority pods     |
                   | - Re-run Filter plugins                       |
                   | - Reprioritize & add back non-essential pods  |
                   | - Calculate minimal victims & PDB violations  |
                   +-----------------------------------------------+
                                          |
                                          v
                   +-----------------------------------------------+
                   | Phase 4: Best Candidate Node Selection        |
                   | - Ordered Score Functions:                    |
                   |   1. Min PDB violations                       |
                   |   2. Min highest-priority victim              |
                   |   3. Min sum of victim priorities             |
                   |   4. Min total victim count                   |
                   +-----------------------------------------------+
                                          |
                                          v
                   +-----------------------------------------------+
                   | Phase 5: Preemption Actuation & Nomination    |
                   | - Delete/Evict victim pods via API server     |
                   | - Annotate preemptor with nominatedNodeName   |
                   +-----------------------------------------------+
                                          |
                                          v
                   +-----------------------------------------------+
                   | Phase 6: Requeueing & Nominated Pod Cycle     |
                   | - Requeue preemptor in PriorityQueue          |
                   | - Account for preemptor in subsequent cycles  |
                   | - Bind once victims terminate cleanly         |
                   +-----------------------------------------------+
```

### The Scheduling Cycle and PostFilter Extension Point

When `scheduleOne()` runs for a pending pod:
1. `PreFilter` and `Filter` plugins run across all nodes.
2. If at least one node passes `Filter`, the scheduler proceeds to `PreScore`, `Score`, `Reserve`, `Permit`, and `Bind`.
3. If **zero** nodes pass `Filter`, the scheduler invokes `RunPostFilterPlugins()`.
4. `DefaultPreemption.PostFilter()` is executed.

### Phase 1: Preemption Eligibility Evaluation

Before scanning nodes, the scheduler performs checks on the preemptor pod (`PodEligibleToPreemptOthers`):
1. **Policy Check**: If `pod.Spec.PreemptionPolicy` is set to `PreemptLowerPriority` (or nil, defaulting to preempt), the pod is eligible. If `Never`, preemption is aborted immediately.
2. **Nominated Node Evaluation**: If the pod already has `status.nominatedNodeName` set from an earlier preemption attempt, the scheduler inspects the nominated node's current status:
   - If the nominated node is currently experiencing victim termination and is on track to fit the preemptor, the scheduler **skips** preempting victims on other nodes to prevent cluster-wide thrashing.
   - If the nominated node is no longer viable (e.g., node unschedulable, tainted, or failed), the nomination is cleared, allowing a new node search.

### Phase 2: Candidate Node Discovery and Sampling

Evaluating preemption across tens of thousands of nodes would incur high CPU and latency overhead. The scheduler employs smart sampling:
1. **Node Filtering**: Identifies nodes that have at least one lower-priority pod running. Nodes containing only equal or higher-priority pods are immediately discarded.
2. **Sampling Ratio**: Uses `percentageOfNodesToFind` (configurable, default scaling dynamically between 5% and 50% depending on cluster size) starting from a pseudo-random offset. Once the required number of potential candidate nodes is shortlisted, the search finishes.

### Phase 3: Per-Node Minimal Victim Selection

For each candidate node, the scheduler executes `SelectVictimsOnNode`:
1. **Simulated Removal**: The scheduler creates a snapshot of the node's `NodeInfo` and simulates the removal of **all** lower-priority pods.
2. **Dry-Run Filtering**: The scheduler runs the complete suite of `Filter` plugins against the empty node snapshot. If the preemptor pod *still* does not fit (e.g., due to unschedulable taints, node architecture mismatch, or unresolvable persistent volume topology), the node is marked ineligible.
3. **Iterative Reprioritization (Adding Back Victims)**:
   - The scheduler sorts the removed victims by importance in descending order:
     - Higher priority pods first.
     - Pods protected by PDBs (disrupting them violates PDB) are considered more important than unprotected pods.
     - Newer pods vs older pods (tie-breaker).
   - The scheduler iteratively adds pods back to the node snapshot one by one, checking if the preemptor continues to pass `Filter` checks.
   - Any pod whose presence breaks the preemptor's fit is retained in the **final victim list**. Pods that can coexist with the preemptor are added back and spared.
4. **PDB Violation Count**: Records the number of victims in the final list whose eviction would breach their respective `PodDisruptionBudget`.

### Phase 4: Best Candidate Node Selection (Tie-Breaking)

If multiple nodes yield valid victim sets, the scheduler selects the single best target node using a sequence of deterministic score functions (`OrderedScoreFuncs`):

$$\text{Best Node} = \arg\min_{N \in \text{Candidates}} \Big( \text{Score}_1(N), \text{Score}_2(N), \text{Score}_3(N), \text{Score}_4(N) \Big)$$

1. **Score 1: Minimum PDB Violations**: Nodes requiring 0 PDB violations are preferred over nodes causing 1 or more violations.
2. **Score 2: Lowest Highest-Priority Victim**: Prefers the node whose highest-priority victim is lower than other nodes' highest-priority victims.
3. **Score 3: Lowest Sum of Victim Priorities**: Minimizes the total cumulative priority score of all victims being terminated.
4. **Score 4: Fewest Total Victims**: Prefers evicting 1 large pod over evicting 10 small pods if resource recovery is equivalent.
5. **Score 5: Earliest Start Time / Random Tie-Break**: Breaks identical scores deterministically.

### Phase 5: Preemption Actuation and Node Nomination

Once the winning node and its victim set are chosen:
1. **Eviction / Deletion API Calls**: The scheduler's `PreemptionExecutor` issues API deletion calls for all victim pods:
   - Sets `GracePeriodSeconds` to the victim's configured `spec.terminationGracePeriodSeconds`.
   - Pods enter the `Terminating` state (`metadata.deletionTimestamp` is populated).
2. **Setting `NominatedNodeName`**: The scheduler patches the preemptor's `status.nominatedNodeName` to the chosen node name.
3. **Emitting Events**: Emits a `Preemption` event on both the preemptor and victim pods.

### Phase 6: The Nominated Pod Cycle and Final Binding

Crucially, **the scheduler does not immediately bind the preemptor pod to the node**. It cannot bind until the victim pods have fully terminated and released physical memory, CPU, and device allocations.

1. **Requeueing**: The preemptor is returned to the scheduler's `PriorityQueue` (in `BackoffQ` or `ActiveQ`).
2. **Nominated Pod Accounting**: In subsequent scheduling cycles, when the scheduler evaluates the nominated node for *other* pods, it invokes `addNominatedPods()`:
   - The scheduler treats the preemptor as if it were already running on that node when checking filters for other pods.
   - This prevents lower-priority or equal-priority pods from "stealing" the resources being cleared for the preemptor.
3. **Clean Binding**: Once the victim pods disappear from the API server (etcd) and the scheduler cache, the preemptor is evaluated again. It passes `Filter` unconditionally, clears `status.nominatedNodeName`, and undergoes `Reserve`, `Permit`, and `Bind` to the node.

---

## 5. Preemption Flows and Sequence Diagrams

### Flow A: Standard Scheduler Preemption

```
Preemptor Pod          kube-scheduler              kube-apiserver              kubelet (Node A)          Victim Pod (Node A)
     |                       |                           |                            |                          |
     |--- (1) Submit Pod --->|                           |                            |                          |
     |   (Priority: 1000)    |                           |                            |                          |
     |                       |--- (2) Run Filters ------>|                            |                          |
     |                       |    (All Nodes Ineligible) |                            |                          |
     |                       |                           |                            |                          |
     |                       |--- (3) Run PostFilter --->|                            |                          |
     |                       |    - Select Node A        |                            |                          |
     |                       |    - Select Victim V1     |                            |                          |
     |                       |                           |                            |                          |
     |                       |--- (4) Delete Pod V1 ---->|                            |                          |
     |                       |    (GracePeriod: 30s)     |--- (5) Watch Deletion ---->|                          |
     |                       |                           |                            |--- (6) SIGTERM --------->|
     |                       |--- (7) Set NominatedNode->|                            |                          |
     |                       |    Node A on Preemptor    |                            |                          |
     |                       |                           |                            |                          |
     |                       |                           |                            |<-- (8) App Clean Exits --|
     |                       |                           |<-- (9) Pod V1 Deleted -----|                          |
     |                       |                           |    (Released from etcd)    |                          |
     |<-- (10) Next Cycle ---|                           |                            |                          |
     |    Preemptor Pops     |--- (11) Run Filters ----->|                            |                          |
     |    from PriorityQueue |    (Node A now fits!)     |                            |                          |
     |                       |                           |                            |                          |
     |                       |--- (12) Bind Pod -------->|                            |                          |
     |                       |     to Node A             |--- (13) Spawn Container -->|                          |
     |                       |                           |    (Preemptor Running)     |                          |
```

### Flow B: Non-Preempting Priority Pods (`preemptionPolicy: Never`)

```
   Pending Pod P (Priority: 500, Policy: Never)
                       |
                       v
             Scheduler PriorityQueue
      (Pops ahead of Priority 0-499 pods)
                       |
                       v
                 Filter Phase
                 /          \
       (Nodes Available)   (No Nodes Fit)
              /                \
             v                  v
         Bind Pod          PostFilter Phase
                       (Preemption Skipped!)
                                |
                                v
                       Requeue in BackoffQ
                  (Waits for natural capacity)
```

### Flow C: Kubelet Node-Pressure Eviction

```
Kubelet EvictionManager               Node Condition & cgroups               Victim Pods on Node
           |                                     |                                   |
           |--- (1) Poll Node Metrics ---------->|                                   |
           |    (Memory Available: 150Mi)        |                                   |
           |    (Threshold: < 250Mi breached!)   |                                   |
           |                                     |                                   |
           |--- (2) Rank Pods for Eviction ----------------------------------------->|
           |    Order:                                                               |
           |    1. BestEffort exceeding requests                                     |
           |    2. Burstable exceeding requests (sorted by priority & usage)         |
           |    3. Guaranteed / Below request pods                                   |
           |                                                                         |
           |--- (3) Select Lowest Ranked Pod (Pod-Low) ----------------------------->|
           |                                                                         |
           |--- (4) Mark Pod Status: Failed (Reason: Evicted) ---------------------->|
           |                                                                         |
           |--- (5) Send SIGTERM (Wait GracePeriod) -------------------------------->|
           |--- (6) Send SIGKILL (if grace period expires) ------------------------->|
           |                                                                         |
           |<-- (7) Memory reclaimed; threshold restored ----------------------------|
```

### Flow D: Gang / PodGroup Preemption

In advanced AI/ML and batch environments utilizing Workload-Aware Preemption (WAP) or Coscheduling plugins (e.g., Volcano, Kubernetes Scheduling Framework PodGroup extensions):
1. **Atomic Evaluation**: A PodGroup consists of $M$ pods requiring simultaneous placement.
2. **Cross-Node Victim Selection**: If a PodGroup cannot be scheduled, the preemption evaluator evaluates potential victims across multiple nodes concurrently.
3. **All-or-Nothing Preemption**: Preemption is executed if and only if **all** $M$ pods in the incoming PodGroup can be accommodated across the cluster. If even one member cannot find space, no victim pods anywhere in the cluster are preempted.

---

## 6. Detailed Analysis of Corner Cases & Failure Modes

### 1. Preemption Thrashing & Ping-Pong Loops

#### The Scenario:
Workload A (Priority 100) is managed by a Deployment of 5 replicas. Workload B (Priority 100) is also managed by a Deployment of 5 replicas. The cluster only has capacity for 5 pods total.
If both workloads are assigned Priority 100, they cannot preempt each other. However, if an operator creates a cyclical priority rule or dynamic scaling loop:
- Workload A (Priority 200) preempts Workload B (Priority 100).
- An autoscaler or automated operator immediately elevates Workload B's priority or schedules a higher-priority task B' (Priority 300) which preempts Workload A.
- Workload A's controller recreates Workload A pods, triggering a continuous cycle of eviction, startup, teardown, and wasted compute.

#### Mitigation:
- Maintain strict, deterministic PriorityClass hierarchies.
- Implement rate limiting and exponential backoff on workload controllers.
- Use `preemptionPolicy: Never` for batch workloads where restarting carries high overhead.

### 2. Nominated Node Stealing & Starvation

#### The Scenario:
1. High-priority Pod $H$ (Priority 1000, Request: 8 CPU) preempts low-priority Pod $L$ (Priority 100, 8 CPU) on Node 1.
2. Pod $L$ receives a 60-second termination grace period. Node 1 is marked as `nominatedNodeName` for Pod $H$.
3. While Pod $L$ is terminating (say, 5 seconds in), a medium-priority Pod $M$ (Priority 500, Request: 2 CPU) enters the queue.
4. Node 1 currently has 2 CPU free (unrelated to Pod $L$).
5. Does Pod $M$ "steal" Node 1 and prevent Pod $H$ from scheduling when Pod $L$ finally exits?

#### The Mechanism & Solution:
`kube-scheduler` executes a sophisticated dual-check algorithm via `addNominatedPods()`:
- When checking if Pod $M$ fits on Node 1, the scheduler **assumes Pod $H$ is already scheduled on Node 1**.
- If Node 1 has enough total capacity for **both** $H$ (8 CPU) and $M$ (2 CPU), Pod $M$ is allowed to schedule immediately.
- If accommodating Pod $M$ would leave less than 8 CPU for Pod $H$, Pod $M$'s Filter check on Node 1 **fails**, preserving the reserved capacity for $H$.

### 3. Affinity and Anti-Affinity Cascades

#### The Scenario:
Pod $P_1$ requires `podAntiAffinity` with any pod carrying label `app: analytics`.
- Node 1 runs Pod $V_1$ (`app: analytics`, Priority: 100) and Pod $V_2$ (`app: cache`, Priority: 100).
- High-priority Pod $P_1$ (Priority 1000) arrives.
- The scheduler identifies Node 1 as a candidate. To place $P_1$ on Node 1, it **must** evict $V_1$ to satisfy $P_1$'s anti-affinity rule, even if Node 1 had plenty of raw CPU and memory.

#### The Risk:
If another running pod $R$ on Node 1 required `podAffinity` to $V_1$, evicting $V_1$ might cause Pod $R$ to become unhealthy or fail internal probes, causing secondary application disruption across the cluster.

### 4. Topology Spread Constraint Interactions

`TopologySpreadConstraints` with `WhenUnsatisfiable: DoNotSchedule` can create complex preemption interactions:
- Preempting a pod in Zone `us-east-1a` changes the skew calculation for the entire workload topology.
- During dry-run evaluation, `DefaultPreemption` must run the `PodTopologySpread` plugin against the whole cluster snapshot to verify that evicting a victim does not violate topology rules for the preemptor or cause illegal topology skew.

### 5. PDB Deadlocks and Violation Fallbacks

#### The Scenario:
Every node with lower-priority pods contains pods protected by a `PodDisruptionBudget` where `allowedDisruptions == 0`.
- The scheduler first scans all nodes attempting to find a victim set with `PDBViolations == 0`.
- If zero candidate nodes satisfy this condition, the scheduler falls back to nodes with the minimal number of PDB violations.
- **Result**: The scheduler will proceed with preemption, intentionally violating the PDB, to satisfy the higher-priority pod's scheduling requirement. PDBs are designed to protect against voluntary operational disruptions (e.g., `kubectl drain`), not to permanently deadlock higher-priority pods.

### 6. Volume Binding and Storage Locality Interlocks

Consider a stateful pod $P_{stateful}$ with a PersistentVolumeClaim bound to a local NVMe disk (`VolumeBindingMode: WaitForFirstConsumer`) on Node 2:
- If Node 1 has lower-priority pods that could be preempted, but $P_{stateful}$'s storage can only exist on Node 2, the scheduler's `VolumeBinding` plugin will reject Node 1 during the dry-run Filter phase.
- Preemption will **only** target victims on Node 2, preventing useless victim evictions on nodes where the preemptor could never run.

### 7. Termination Delays, Stalled Finalizers & PreStop Hooks

When a victim pod is chosen for preemption:
1. `kube-apiserver` sets `deletionTimestamp`.
2. `kubelet` invokes container `preStop` hooks and sends `SIGTERM`.
3. If the victim pod has a poorly written `preStop` hook that hangs, or a Kubernetes Finalizer attached to `metadata.finalizers` that is never removed by an external controller:
   - The victim remains in `Terminating` indefinitely.
   - The preemptor pod remains stuck in `Pending` with `status.nominatedNodeName` set.
   - The scheduler will continue to wait and will not automatically re-preempt other nodes while the nominated node is considered viable.

### 8. ResourceQuota & PriorityClass Scope Races

When `ResourceQuota` is configured with `scopeSelector` targeting specific `PriorityClass` tiers:
- High-priority quotas and low-priority quotas are tracked independently.
- If a namespace has reached its low-priority memory quota, creating a low-priority pod is rejected at admission time by the API server, before the scheduler ever evaluates preemption.
- Conversely, a high-priority pod can consume cluster resources even if low-priority quotas are full, provided high-priority quotas have available headroom.

### 9. Scheduler Cache vs. Etcd State Inconsistencies

Under extreme control-plane load:
- The scheduler makes preemption decisions based on its local in-memory informer cache.
- If watch events from `kube-apiserver` are delayed, the scheduler might attempt to preempt a victim pod that was already deleted or moved.
- When the scheduler issues the Delete API call, it receives a `404 Not Found` or `409 Conflict`. The scheduler handles this gracefully by invalidating its cache entry for that node and retrying in the next scheduling cycle.

### 10. DaemonSet Preemption Nuances

- DaemonSet pods are created by the `DaemonSetController` and scheduled by the default scheduler with a default priority.
- If a node is fully utilized and a new DaemonSet pod must run on every node, the DaemonSet pod must be assigned a sufficiently high `PriorityClass` (e.g., `system-node-critical` or a dedicated infra priority) to preempt user workloads and secure its spot on the node.

---

## 7. Scheduler Preemption vs. Kubelet Eviction Comparison

A side-by-side comparison of the two distinct preemption mechanisms in Kubernetes:

| Dimension | `kube-scheduler` Preemption | `kubelet` Node-Pressure Eviction |
| :--- | :--- | :--- |
| **Trigger Condition** | Unschedulable high-priority pod in queue | Physical node resource exhaustion (Memory/Disk/PID) |
| **Decision Authority** | Central Control Plane (`kube-scheduler`) | Local Node Agent (`kubelet`) |
| **Primary Criterion** | `spec.priority` (Integer value) | Resource Usage exceeding Requests, then QoS Class |
| **PDB Respect** | Tries to preserve PDBs; violates only if no choice | **Ignores PDBs completely** (Hard thresholds) |
| **Grace Period** | Honors `terminationGracePeriodSeconds` | Honors grace period on Soft; 0s on Hard eviction |
| **Victim Pod Status** | Deleted / Terminated | Marked `Phase: Failed`, `Reason: Evicted` |
| **Target Selection** | Minimal victim set across the best cluster node | Lowest-ranked local pod consuming excess resources |
| **Preemption Policy** | Can be disabled via `preemptionPolicy: Never` | Cannot be disabled (node hardware safeguard) |

---

## 8. Production Best Practices, Tuning & Observability

### Designing a Robust Priority Hierarchy

Adopt a standardized 4-to-5 tier priority architecture across the enterprise:

```
[Priority: 2,000,000,000+]  System Critical (CoreDNS, CNI, Node Daemons)
            ^
[Priority: 1,000,000]       Production Tier 0 (Core APIs, Payment Gateways)
            ^
[Priority: 500,000]         Production Tier 1 (Background Processing, Workers)
            ^
[Priority: 100,000]         Non-Production / Staging Environments
            ^
[Priority: 1,000]           Batch / CI/CD Jobs (preemptionPolicy: Never)
            ^
[Priority: 0]               BestEffort / Scavenger Workloads
```

#### Example PriorityClass Manifests:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: prod-tier0
value: 1000000
preemptionPolicy: PreemptLowerPriority
description: "Tier 0 production services that must preempt batch and dev workloads."
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: batch-ci
value: 1000
preemptionPolicy: Never
description: "CI/CD batch jobs that queue ahead of low priority but never preempt active pods."
```

### PDB and Grace Period Tuning

1. **Keep Grace Periods Reasonable**: Avoid setting `terminationGracePeriodSeconds` to excessive durations (e.g., > 120s) on lower-priority batch workloads unless necessary, as this directly delays the startup of preemptor pods.
2. **Do Not Over-Constrain PDBs**: Ensure PDBs allow at least 1 disruption whenever possible (`maxUnavailable: 1` or `minAvailable: 90%`). PDBs with `maxUnavailable: 0` or `minAvailable: 100%` create scheduling friction and force the scheduler into violation fallbacks.

### Key Metrics and Alerting

Monitor the following Prometheus metrics exposed by `kube-scheduler` and `kubelet`:

#### kube-scheduler Metrics:
- `scheduler_preemption_attempts_total`: Total number of preemption evaluation cycles initiated.
- `scheduler_preemption_victims`: Number of victim pods selected for preemption (partitioned by extension plugin).
- `scheduler_preemption_evaluation_duration_seconds`: Latency of the preemption evaluation algorithm.
- `scheduler_pod_scheduling_attempts`: Tracks how many attempts a nominated pod requires before binding.

#### kubelet Metrics:
- `kubelet_evictions_total`: Number of pods evicted due to node resource pressure (labeled by eviction signal: `memory`, `nodefs`, etc.).

#### Alerting Recommendations:
- **High Preemption Rate Alert**: Trigger when `rate(scheduler_preemption_victims[5m]) > threshold`, indicating cluster under-provisioning or priority misconfigurations.
- **Preemption Starvation Alert**: Trigger when a pod with `status.nominatedNodeName` remains in `Pending` state for longer than $2 \times \text{Average Grace Period}$.
