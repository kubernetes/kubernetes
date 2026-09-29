# SchedulingGates Plugin

This guide provides an architectural overview, interface implementations, declarative gating mechanics, queueing hint algorithms, and testing strategies for the `SchedulingGates` plugin in `pkg/scheduler/framework/plugins/schedulinggates`.

---

## 1. High-Level Purpose & Scope

The `SchedulingGates` plugin implements declarative, pre-enqueue admission gating for Kubernetes pods that define `.spec.schedulingGates`.

### Core Responsibilities:
1. **Pre-Enqueue Gating**: Blocks pods from entering the scheduler's active scheduling queue (`activeQ` / `backoffQ`) as long as one or more scheduling gates remain defined in `pod.Spec.SchedulingGates`.
2. **Zero-Overhead Waiting**: Keeps gated pods idle in `unschedulableEntities` without running node filtering, scoring, or reservation cycles until external controllers clear all gates.
3. **Preemption Protection (`UnschedulableAndUnresolvable`)**: Rejects gated pods with `fwk.UnschedulableAndUnresolvable`, preventing the preemption engine from attempting eviction for workloads blocked by external dependencies.
4. **Immediate Awakening on Gate Removal**: Utilizes the `UpdatePodSchedulingGatesEliminated` cluster event and an optimized queueing hint to immediately move the pod from `unschedulableEntities` directly into `activeQ` when its final scheduling gate is removed.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/schedulinggates/
├── scheduling_gates.go       # Plugin definition, PreEnqueue, and EnqueueExtensions implementations
├── scheduling_gates_test.go  # Unit tests for PreEnqueue gating and queueing hint evaluation
└── AGENTS.md                 # This agent documentation
```

---

## 3. Data Structures & Plugin Configuration

### 3.1. `SchedulingGates` Struct

```go
type SchedulingGates struct{}
```

- Stateless plugin instantiated via `New(_ context.Context, _ runtime.Object, _ fwk.Handle, _ feature.Features) (fwk.Plugin, error)`.

### 3.2. Constants

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.SchedulingGates` (`"SchedulingGates"`) | Registered plugin name in scheduler profiles. |

---

## 4. Extension Point Implementations

`SchedulingGates` implements `fwk.PreEnqueuePlugin` and `fwk.EnqueueExtensions`.

```
                        ┌──────────────────────────────────────────────┐
                        │             Pod Admission Event              │
                        │           (PreEnqueue Evaluation)            │
                        └──────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
        [ len(pod.Spec.SchedulingGates) == 0 ]       [ len(pod.Spec.SchedulingGates) > 0 ]
                       │                                               │
                       ▼                                               ▼
             ┌───────────────────┐                         ┌───────────────────────────────────┐
             │    Return nil     │                         │ Extract Gate Names:               │
             │ (Admit to activeQ)│                         │ ["gate-a", "gate-b", ...]         │
             └───────────────────┘                         └─────────────────┬─────────────────┘
                                                                             │
                                                                             ▼
                                                           ┌───────────────────────────────────┐
                                                           │ Return:                           │
                                                           │ UnschedulableAndUnresolvable      │
                                                           │ "waiting for scheduling gates: .."│
                                                           │ (Held in unschedulableEntities)   │
                                                           └───────────────────────────────────┘
```

### 4.1. `PreEnqueue` (`PreEnqueuePlugin`)

- **Signature**: `PreEnqueue(ctx context.Context, p *v1.Pod) *fwk.Status`
- **Behavior**:
  - Checks `len(p.Spec.SchedulingGates)`.
  - If empty, returns `nil` (`Success`), allowing the pod to be enqueued into `activeQ`.
  - If non-empty, extracts all gate names (`gate.Name`) and returns:
    ```go
    return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for scheduling gates: %v", gates))
    ```
  - The status code `UnschedulableAndUnresolvable` marks the pod as permanently blocked until external object mutation occurs, keeping it out of scheduling cycles.

### 4.2. `EventsToRegister` & Queueing Hints (`EnqueueExtensions`)

- **Signature**: `EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error)`
- **Registered Event**:
  ```go
  {
      Event: fwk.ClusterEvent{
          Resource:   fwk.TargetPod,
          ActionType: fwk.UpdatePodSchedulingGatesEliminated,
      },
      QueueingHintFn: pl.isSchedulableAfterUpdateTargetPodSchedulingGatesEliminated,
  }
  ```
- **Queueing Hint (`isSchedulableAfterUpdateTargetPodSchedulingGatesEliminated`)**:
  - Invoked by the scheduler informer when a pod update removes its last scheduling gate.
  - Returns `fwk.Queue`, immediately waking the pod and moving it to `activeQ`.

---

## 5. Architectural Comparison: SchedulingGates vs. Permit Plugins

| Dimension | `SchedulingGates` (PreEnqueue) | `PermitPlugin` (Permit Phase) |
| :--- | :--- | :--- |
| **Execution Point** | Before queue admission (`PreEnqueue`). | At the end of the scheduling cycle (after `Filter` and `Reserve`). |
| **Cluster Resource Cost** | **Zero**. No node filtering or scoring is executed while gates exist. | **High**. Consumes CPU evaluating all nodes and reserves node resources. |
| **Node Reservation** | None. Node is not yet selected. | Node is selected and temporary in-memory allocations are reserved. |
| **Wait Duration / Timeout** | Indefinite (controlled by external controller lifecycle). | Bounded timeout (max `15m`), auto-rejected on expiry. |
| **Use Cases** | External resource provisioning (IPAM, dynamic network attachment, quota verification, pre-provisioning). | Co-scheduling / gang synchronization, multi-pod atomic binding coordination. |

---

## 6. End-to-End Scheduling Gate Lifecycle

```
┌────────────────────────┐
│  Workload Creation:    │
│  Pod created with      │
│  spec.schedulingGates  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  PreEnqueue Rejection: │
│  SchedulingGates plugin│
│  returns               │
│  UnschedulableAnd...   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Unschedulable Pool:   │
│  Pod rests in          │
│  unschedulableEntities │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  External Controller:  │
│  Provisions external   │
│  dependencies, then    │
│  patches pod to remove │
│  schedulingGates       │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Informer Event:       │
│  UpdatePodScheduling-  │
│  GatesEliminated fires │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Queueing Hint:        │
│  isSchedulableAfter... │
│  returns fwk.Queue     │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Active Scheduling:    │
│  Pod moved to activeQ; │
│  PreEnqueue returns nil│
└────────────────────────┘
```

---

## 7. Testing Strategy & Test Coverage

The unit tests in `scheduling_gates_test.go` validate:

1. **`TestPreEnqueue`**:
   - Pods without `.spec.schedulingGates` return `nil` (`Success`).
   - Pods with single or multiple gates return `UnschedulableAndUnresolvable` with formatted gate names.
2. **`TestIsSchedulableAfterUpdateTargetPodSchedulingGatesEliminated`**:
   - Asserts that the queueing hint handler unconditionally returns `fwk.Queue` to awaken un-gated pods.
