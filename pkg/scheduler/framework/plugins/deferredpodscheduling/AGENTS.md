# DeferredPodScheduling Plugin

This guide provides an architectural overview, interface implementations, in-place vertical scaling resize preemption mechanics, permit rejection rationale, Kubelet actuation handoff, and testing strategies for the `DeferredPodScheduling` plugin in `pkg/scheduler/framework/plugins/deferredpodscheduling`.

---

## 1. High-Level Purpose & Scope

The `DeferredPodScheduling` plugin coordinates **In-Place Pod Vertical Scaling (KEP-1287)** when a pod's in-place resource resize cannot be immediately satisfied on its assigned node due to resource contention, causing the resize to become **Deferred** (`PodResizePending` condition with reason `Deferred`).

### Core Responsibilities:
1. **Targeted In-Place Preemption**: Enables the scheduler to simulate and trigger preemption on the pod's currently assigned node (`pod.Spec.NodeName`) to evict lower-priority victim pods and free capacity for the resize delta.
2. **Node Pinning Enforcement**: Ensures that deferred resize pods are evaluated exclusively against their currently assigned node, rejecting all other cluster nodes in `Filter`.
3. **Node Preemption Policy Gating**: Honors node-level policies (`node.Spec.PodPreemptionPolicy.DisableResizePreemption`), preventing resize preemption on nodes where it is explicitly disabled.
4. **Permit-Stage Kubelet Handoff**: Intercepts the scheduling cycle at the **Permit** extension point. When the pod fits (or space is cleared via preemption), `Permit` deliberately **rejects** the pod with `UnschedulableAndUnresolvable`. This prevents binding (since the pod is already bound and running) and leaves the pod in the unschedulable pool while Kubelet actuates the resize.
5. **Dynamic Re-enqueueing**: Registers queueing hints for node preemption policy transitions to re-evaluate deferred pods when nodes re-enable resize preemption.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/deferredpodscheduling/
├── deferred_pod_scheduling.go       # Plugin definition, PreFilter, Filter, Permit, and EnqueueExtensions
├── deferred_pod_scheduling_test.go  # Comprehensive unit tests for PreFilter, Filter, Permit, and Queueing Hints
└── AGENTS.md                        # This agent documentation
```

---

## 3. Data Structures & Plugin Configuration

### 3.1. `DeferredPodScheduling` Struct

```go
type DeferredPodScheduling struct {
    enableInPlacePodVerticalScalingSchedulerPreemption bool
}
```

- **`enableInPlacePodVerticalScalingSchedulerPreemption`**: Controls whether in-place resize preemption logic is active (backed by the corresponding feature gate).

### 3.2. Constants

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.DeferredPodScheduling` (`"DeferredPodScheduling"`) | Registered plugin name in scheduler profiles. |
| `ErrReasonNodeDisablesResizePreemption` | `"node had resize preemption disabled"` | Filter failure reason returned when node policy disables resize preemption. |

---

## 4. Extension Point Implementations

`DeferredPodScheduling` implements `fwk.PreFilterPlugin`, `fwk.FilterPlugin`, `fwk.PermitPlugin`, and `fwk.EnqueueExtensions`.

```
               ┌─────────────────────────────────────────────────────────────┐
               │                    PreFilter Evaluation                     │
               │   Gate On && resource.IsPodResizeDeferred(pod) == true      │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │                                                 │
              [ Not Deferred ]                                   [ Is Deferred ]
                     │                                                 │
                     ▼                                                 ▼
          ┌─────────────────────┐                            ┌───────────────────┐
          │ Return fwk.Skip     │                            │ Return nil, nil   │
          │ (Bypass this plugin)│                            │ (Proceed to Filter│
          └─────────────────────┘                            └─────────┬─────────┘
                                                                       │
                                                                       ▼
                                                     ┌───────────────────────────────────┐
                                                     │         Filter Evaluation         │
                                                     │ node.Name == pod.Spec.NodeName && │
                                                     │ !node.DisablesResizePreemption    │
                                                     └─────────────────┬─────────────────┘
                                                                       │
                                                                       ▼
                                                     ┌───────────────────────────────────┐
                                                     │         Permit Evaluation         │
                                                     │  (Pod fits on node / after preemp)│
                                                     └─────────────────┬─────────────────┘
                                                                       │
                                                                       ▼
                                                     ┌───────────────────────────────────┐
                                                     │ Return:                           │
                                                     │ fwk.UnschedulableAndUnresolvable  │
                                                     │ "pod resize fits, waiting for     │
                                                     │  Kubelet actuation" (Timeout: 0)  │
                                                     └───────────────────────────────────┘
```

### 4.1. `PreFilter` (`PreFilterPlugin`)

- **Signature**: `PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status)`
- **Evaluation**:
  - Checks `if !pl.enableInPlacePodVerticalScalingSchedulerPreemption || !resource.IsPodResizeDeferred(pod)`.
  - Returns `fwk.NewStatus(fwk.Skip)` for non-deferred pods or when the feature gate is disabled.
  - Returns `nil, nil` for active deferred resize pods.

### 4.2. `Filter` (`FilterPlugin`)

- **Signature**: `Filter(ctx context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status`
- **Validation Rules**:
  1. **Node Identity**: If `pod.Spec.NodeName != "" && pod.Spec.NodeName != node.Name`, returns `fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "pod assigned to different node")`.
  2. **Preemption Policy**: If `node.Spec.PodPreemptionPolicy != nil && len(node.Spec.PodPreemptionPolicy.DisableResizePreemption) > 0`, returns `fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonNodeDisablesResizePreemption)`.
  3. Returns `nil` (`Success`) if the candidate is the pod's assigned node and resize preemption is permitted.

### 4.3. `Permit` (`PermitPlugin`)

- **Signature**: `Permit(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) (*fwk.Status, time.Duration)`
- **Behavior**:
  - For non-deferred pods, returns `nil, 0` (`Success`).
  - For deferred resize pods:
    ```go
    return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "pod resize fits, waiting for Kubelet actuation"), 0
    ```

---

## 5. Architectural Deep Dive: Why Permit Rejects Deferred Pods

The use of `Permit` returning `UnschedulableAndUnresolvable` is a deliberate architectural pattern for in-place vertical scaling:

```
┌───────────────────────────────┐
│ 1. Workload Resize Request:   │
│    spec.containers[*].res     │
│    updated on running pod     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 2. Kubelet Check:             │
│    Insufficient node headroom │
│    Sets status: ResizeDeferred│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 3. Scheduler Enqueue:         │
│    Pod enters scheduler queue │
│    PreFilter activates plugin │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 4. Preemption Simulation:     │
│    Scheduler identifies lower-│
│    priority victims on node   │
│    and issues evictions       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 5. Permit Rejection:          │
│    Permit returns             │
│    UnschedulableAnd...        │
│    (Prevents binding loop)    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 6. Kubelet Actuation:         │
│    Victims terminate; Kubelet │
│    allocates resources and    │
│    clears ResizeDeferred      │
└───────────────────────────────┘
```

### Key Invariants:
1. **Pods are Already Bound**: An in-place resize pod is already assigned to a node (`spec.nodeName` is set and containers are running). The scheduler cannot and must not execute the `Bind` phase.
2. **Preemption Execution**: The scheduler preemption engine evicts victims on the node to create resource headroom for the delta between requested and allocated resources.
3. **Clean Handoff**: By rejecting in `Permit`, the scheduler cleanly terminates its scheduling cycle, releases in-memory cycle states, and places the pod back into the unschedulable queue until Kubelet updates `pod.Status.AllocatedResources` or removes the `Deferred` condition.

---

## 6. Queueing Hints & Event Lifecycle

`DeferredPodScheduling` registers event handlers in `EventsToRegister`:

| Cluster Event | Handler | Trigger Condition |
| :--- | :--- | :--- |
| `Node / UpdateNodePreemptionPolicy` | `isSchedulableAfterNodeChange` | Requeues pod (`fwk.Queue`) when `oldNode` had resize preemption disabled and `newNode` has it enabled on `pod.Spec.NodeName`. |
| `Node / Add` | `isSchedulableAfterNodeAdd` | Requeues pod (`fwk.Queue`) when the added node matches `pod.Spec.NodeName` and does not disable resize preemption. |

---

## 7. Testing Strategy & Test Coverage

The unit test suite in `deferred_pod_scheduling_test.go` validates:

1. **`TestDeferredPodScheduling_PreFilter`**:
   - Feature gate disabled -> `Skip`.
   - Feature gate enabled, non-deferred pod -> `Skip`.
   - Feature gate enabled, deferred pod -> `nil` (`Success`).
2. **`TestDeferredPodScheduling_Filter`**:
   - Matching vs non-matching node names.
   - Node preemption policy disabled -> `UnschedulableAndUnresolvable` with `ErrReasonNodeDisablesResizePreemption`.
   - Missing node info error handling.
3. **`TestDeferredPodScheduling_Permit`**:
   - Non-deferred pods return `nil, 0`.
   - Deferred pods return `UnschedulableAndUnresolvable` with `"pod resize fits, waiting for Kubelet actuation"`.
4. **`TestDeferredPodScheduling_isSchedulableAfterNodeChange` & `TestDeferredPodScheduling_isSchedulableAfterNodeAdd`**:
   - Validates queueing hint state transitions when nodes enable or add preemption policies.
