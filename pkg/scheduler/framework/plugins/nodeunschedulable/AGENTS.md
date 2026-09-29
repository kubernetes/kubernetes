# NodeUnschedulable Plugin (`pkg/scheduler/framework/plugins/nodeunschedulable`)

This guide provides an architectural overview, interface implementations, condition and taint filtering mechanics, queueing hint algorithms, and testing strategies for the `NodeUnschedulable` plugin in `pkg/scheduler/framework/plugins/nodeunschedulable`.

---

## 1. High-Level Purpose & Scope

The `NodeUnschedulable` plugin filters out nodes marked as unschedulable (e.g., via `kubectl cordon` or node draining where `node.Spec.Unschedulable = true`), unless the pod explicitly carries a toleration for the unschedulable taint (`node.kubernetes.io/unschedulable:NoSchedule`).

### Core Responsibilities:
1. **Node Cordon Filtering**: Blocks pods from scheduling onto cordoned nodes (`node.Spec.Unschedulable == true`).
2. **Taint Toleration Bypassing**: Permits scheduling onto unschedulable nodes if the pod specifies a matching toleration for `node.kubernetes.io/unschedulable` with effect `NoSchedule`.
3. **Preemption Protection (`UnschedulableAndUnresolvable`)**: Rejection returns `framework.UnschedulableAndUnresolvable`, preventing preemption engines from evicting workloads on cordoned nodes.
4. **Deferred Resize Awareness**: Bypasses filtering for already-placed pods undergoing in-place vertical scaling resize preemption.
5. **Fine-Grained Queueing Hints**: Registers targeted informer event handlers to re-evaluate unschedulable pods only when a node is uncordoned or when the pod gains the required toleration.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/nodeunschedulable/
├── node_unschedulable.go       # Plugin definition, PreFilter, Filter, EnqueueExtensions, and Queueing Hints
├── node_unschedulable_test.go  # Unit tests for Filter, node change hints, and toleration change hints
└── AGENTS.md                   # This agent documentation
```

---

## 3. Data Structures & Plugin Configuration

### 3.1. `NodeUnschedulable` Struct

```go
type NodeUnschedulable struct {
    enableInPlacePodVerticalScalingSchedulerPreemption bool
    enableTaintTolerationComparisonOperators           bool
}
```

- **`enableInPlacePodVerticalScalingSchedulerPreemption`**: Controls whether deferred-resize pods bypass `PreFilter` / `Filter`.
- **`enableTaintTolerationComparisonOperators`**: Controls whether comparison operators (e.g. `Gt`, `Lt`) are enabled during toleration matching.

### 3.2. Constants

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.NodeUnschedulable` (`"NodeUnschedulable"`) | Registered plugin name. |
| `ErrReasonUnschedulable` | `"node(s) were unschedulable"` | Filter failure reason. |
| `ErrReasonUnknownCondition` | `"node(s) had unknown conditions"` | Legacy predicate error constant. |

---

## 4. Extension Point Implementations

`NodeUnschedulable` implements `fwk.PreFilterPlugin`, `fwk.FilterPlugin`, `fwk.EnqueueExtensions`, and `fwk.SignPlugin`.

```
                        ┌───────────────────────────────┐
                        │      PreFilter Evaluation     │
                        └───────────────┬───────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           │                                                         │
[ IsPodResizeDeferred == true ]                           [ Standard Pod ]
           │                                                         │
           ▼                                                         ▼
┌─────────────────────────────────────┐                    ┌─────────────────┐
│ Return framework.Skip               │                    │ Return nil, nil │
│ (Bypasses Filter on all nodes)      │                    └────────┬────────┘
└─────────────────────────────────────┘                             │
                                                                    ▼
                                                   ┌─────────────────────────────────┐
                                                   │        Filter Evaluation        │
                                                   └────────────────┬────────────────┘
                                                                    │
                                         ┌──────────────────────────┴──────────────────────────┐
                                         │                                                     │
                             [ node.Spec.Unschedulable == false ]             [ node.Spec.Unschedulable == true ]
                                         │                                                     │
                                         ▼                                                     ▼
                                 ┌───────────────┐                             ┌───────────────────────────────┐
                                 │  Return nil   │                             │ Check TolerationsTolerateTaint│
                                 │   (Success)   │                             │ for node.kubernetes.io/       │
                                 └───────────────┘                             │ unschedulable:NoSchedule      │
                                                                               └───────────────┬───────────────┘
                                                                                               │
                                                                       ┌───────────────────────┴───────────────────────┐
                                                                       │                                               │
                                                                   [ Tolerated ]                                [ Not Tolerated ]
                                                                       │                                               │
                                                                       ▼                                               ▼
                                                                ┌───────────────┐                      ┌───────────────────────────────┐
                                                                │  Return nil   │                      │ Return                        │
                                                                │   (Success)   │                      │ UnschedulableAndUnresolvable  │
                                                                └───────────────┘                      └───────────────────────────────┘
```

### 4.1. `PreFilter` (`PreFilterPlugin`)
- **Signature**: `PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status)`
- **Behavior**:
  - If `enableInPlacePodVerticalScalingSchedulerPreemption` is enabled and `resource.IsPodResizeDeferred(pod)` is true, returns `fwk.NewStatus(fwk.Skip)`. This skips the filter phase entirely because the pod is already running on the node and is only seeking resource preemption for resizing.
  - Otherwise, returns `nil, nil`.
- **Extensions**: `PreFilterExtensions()` returns `nil`.

### 4.2. `Filter` (`FilterPlugin`)
- **Signature**: `Filter(ctx context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status`
- **Evaluation Logic**:
  1. Checks if `resource.IsPodResizeDeferred(pod)` is true (returns `nil` immediately).
  2. Inspects `nodeInfo.Node().Spec.Unschedulable`. If `false`, returns `nil` (node is schedulable).
  3. If `node.Spec.Unschedulable` is `true`, evaluates `v1helper.TolerationsTolerateTaint` for:
     ```go
     &v1.Taint{
         Key:    v1.TaintNodeUnschedulable, // "node.kubernetes.io/unschedulable"
         Effect: v1.TaintEffectNoSchedule,
     }
     ```
  4. If untolerated, returns `fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonUnschedulable)`.

### 4.3. `EventsToRegister` & Queueing Hints (`EnqueueExtensions`)
- **Signature**: `EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error)`
- **Registered Events**:
  1. **`Node (Add | UpdateNodeTaint)` with `isSchedulableAfterNodeChange`**:
     - Returns `fwk.Queue` if:
       - A node is added with `Spec.Unschedulable == false`.
       - An existing node transitions from `Unschedulable: true` to `Unschedulable: false` (uncordoned).
     - Returns `fwk.QueueSkip` for unrelated node updates or nodes added/remaining in cordoned state.
  2. **`TargetPod (UpdatePodToleration)` with `isSchedulableAfterTargetPodTolerationChange`**:
     - Compares new pod tolerations against `node.kubernetes.io/unschedulable:NoSchedule`.
     - Returns `fwk.Queue` if the updated pod now tolerates the unschedulable taint.
     - Returns `fwk.QueueSkip` if the added toleration is unrelated.

### 4.4. `SignPod` (`SignPlugin`)
- **Signature**: `SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status)`
- **Signature Fragment**:
  - Key: `fwk.TolerationsSignerName`
  - Value: `fwk.TolerationsSigner(pod)`
- **Purpose**: Groups pods by their toleration characteristics for opportunistic scheduling cache reuse.

---

## 5. Testing Strategy & Test Coverage

The unit tests in `node_unschedulable_test.go` provide full coverage of filter decisions and event queueing hints:

### Test Suites:
1. **`TestNodeUnschedulable` (Filter Phase)**:
   - Verifies rejection (`UnschedulableAndUnresolvable`) when `node.Spec.Unschedulable == true` and pod has no toleration.
   - Verifies success (`nil`) when `node.Spec.Unschedulable == false`.
   - Verifies success (`nil`) when `node.Spec.Unschedulable == true` and pod specifies a matching `node.kubernetes.io/unschedulable:NoSchedule` toleration.
2. **`TestIsSchedulableAfterNodeChange` (Node Queueing Hints)**:
   - Validates `fwk.Queue` upon node addition (schedulable) and node uncordon (`true -> false`).
   - Validates `fwk.QueueSkip` for unschedulable node additions and unrelated taint modifications on already-cordoned or already-schedulable nodes.
   - Handles type assertion error scenarios returning `fwk.Queue, err`.
3. **`TestIsSchedulableAfterTargetPodTolerationChange` (Pod Toleration Queueing Hints)**:
   - Validates `fwk.Queue` when the target pod receives an update adding a matching toleration.
   - Validates `fwk.QueueSkip` when the target pod's toleration update does not match `node.kubernetes.io/unschedulable`.
