# Domain 4: In-Place Vertical Scaling Preemption

**Governing KEPs:**
- **KEP-1287**: *In-Place Update of Pod Resources (In-Place Pod Vertical Scaling / IPPVS)*
- **KEP-5836**: *Scheduler Preemption for In-Place Pod Vertical Scaling*

**Primary Packages:**
- `pkg/scheduler/framework/plugins/deferredpodscheduling/`
- `pkg/scheduler/backend/cache/`
- `pkg/kubelet/cm/`
- `pkg/apis/core/`

---

## 1. Executive Summary & Problem Statement

Traditionally in Kubernetes, updating container CPU or memory requests required recreating the pod, causing application restart downtime, state loss, and cache invalidation. **KEP-1287** introduced **In-Place Pod Vertical Scaling (IPPVS)**, allowing users and autoscalers (such as VPA) to modify `pod.spec.containers[*].resources` on running pods without container restarts.

However, an architectural gap emerged when high-priority running pods requested resource expansions on nodes with insufficient unallocated capacity:
1. **Deferred Resize Deadlock**: When a node lacked immediate capacity, Kubelet placed the resize into a `Deferred` state (`PodResizePending=True`, `Reason=Deferred`).
2. **Missing Preemption Path**: Because the pod was already assigned and running on the node, the scheduler's standard `scheduleOne` loop never evaluated it, leaving the high-priority pod permanently starved of resources while lower-priority pods continued running on the same node.

**KEP-5836** established the scheduler as the authoritative controller for resizing preemption under the feature gate `InPlacePodVerticalScalingSchedulerPreemption`. When a deferred resize is detected, the scheduler's `DeferredPodScheduling` plugin triggers preemption specifically on the pod's assigned node to free required resources.

---

## 2. In-Place Resize State Machine & Preemption Flow

```
                  +--------------------------------------------------+
                  |  Pod Spec Resource Update (e.g. CPU 2 -> 8 cores)|
                  +--------------------------------------------------+
                                           |
                               [Kubelet Node Evaluation]
                                           |
                              Is Node Capacity Available?
                                    /            \
                             (Yes) /              \ (No)
                                  /                \
                       [Kubelet Resizes Cgroup]  [Status: PodResizePending=True]
                       [Status: Allocated]       [Reason: Deferred]
                                                           |
                                            [kube-scheduler Informer Watch]
                                                           |
                                            [DeferredPodScheduling Plugin]
                                                           |
                                            +--------------v---------------+
                                            | Preemption Evaluation on Node|
                                            | - Target Node ONLY           |
                                            | - Find Lower-Priority Victims|
                                            |   on Target Node             |
                                            +------------------------------+
                                                           |
                                              Can Capacity Be Reclaimed?
                                                    /            \
                                             (Yes) /              \ (No)
                                                  /                \
                                    +-------------v------+   +-----v--------------+
                                    | Evict Lower-Pri    |   | Keep Deferred      |
                                    | Victims on Node    |   | Wait for Natural   |
                                    | (Async Preemption) |   | Pod Completion     |
                                    +--------------------+   +--------------------+
                                           |
                                    [Victims Terminate]
                                           |
                                    [Kubelet Sees Free Room]
                                           |
                                    [Cgroups Resized & PodResizePending Cleared]
```

---

## 3. Detailed Architecture & Synchronization Protocol

### 3.1 The `DeferredPodScheduling` Plugin Lifecycle
The `DeferredPodScheduling` plugin listens for pod update events where:
1. `pod.Spec.NodeName` is non-empty (pod is running).
2. `pod.Status.Conditions` contains `PodResizePending=True` with `Reason=Deferred`.
3. The pod's desired requests exceed allocated requests (`spec.containers[*].resources.requests > status.containerStatuses[*].allocatedResources`).

### 3.2 Single-Node Target Constrained Preemption
Unlike standard preemption which searches across all nodes in the cluster, resize preemption is strictly constrained:
- **Node Invariance**: The candidate node is fixed to `pod.Spec.NodeName`. The pod cannot be moved to another node.
- **Resource Delta Calculation**:
  $$\Delta R = \sum_{\text{containers}} (\text{DesiredRequest} - \text{AllocatedRequest})$$
- **Victim Selection**: The scheduler evaluates only lower-priority pods running on `pod.Spec.NodeName`.
  - It sorts local lower-priority pods by priority and PDB compliance.
  - Simulates removing victims until $\Delta R$ is fully satisfied.
  - Reprieves unneeded victims.
- **Actuation**: Dispatches asynchronous evictions for the selected victims while tracking in-memory reservations to prevent other incoming pods from consuming the freed capacity.

### 3.3 Kubelet Handshake & Race Prevention
1. **Capacity Reservation**: The scheduler cache marks the delta resources $\Delta R$ as "reserved for resize" on the target node.
2. **Kubelet Cgroup Update**: When victim pods exit, Kubelet's runtime update loop detects that allocatable capacity now accommodates $\Delta R$. Kubelet updates the container cgroups and writes `status.containerStatuses[*].allocatedResources = desiredResources`, clearing `PodResizePending`.
3. **Cache Synchronization**: The scheduler receives the updated pod status and releases the temporary reservation.

---

## 4. Invariants & Guardrails

1. **Fixed-Node Invariant**: Resize preemption must never attempt to nominate or evict pods on any node other than `pod.Spec.NodeName`.
2. **Self-Preemption Prevention**: A resizing pod must never select itself or its co-located gang members as victims.
3. **No Over-Eviction**: Only the minimal delta capacity $\Delta R$ required for the resize may be reclaimed from lower-priority pods.

---

## 5. Verification & Test Matrix

- **Unit Tests**: `pkg/scheduler/framework/plugins/deferredpodscheduling/deferred_pod_scheduling_test.go`
- **Integration Tests**: `test/integration/scheduler/preemption/`
  - `TestInPlaceResizePreemption`: Validates lower-priority victim eviction on the specific node.
  - `TestInPlaceResizeRespectPDB`: Ensures PDB-respecting victims are prioritized during resize preemption.
  - `TestInPlaceResizeMultiResourceDelta`: Validates simultaneous CPU and memory expansion preemption.
