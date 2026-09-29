# NodeName Plugin (`pkg/scheduler/framework/plugins/nodename`)

This guide provides an architectural overview, interface implementations, data structures, scheduling invariants, and testing strategies for the `NodeName` plugin in `pkg/scheduler/framework/plugins/nodename`.

---

## 1. High-Level Purpose & Scope

The `NodeName` plugin implements direct, deterministic node targeting for pods that explicitly declare a target host in `pod.Spec.NodeName`.

### Core Responsibilities:
1. **Explicit Node Targeting**: Verifies whether a pod requesting a specific node (`pod.Spec.NodeName`) matches the candidate node under evaluation.
2. **Pre-Filter Candidate Pruning**: When `pod.Spec.NodeName` is set, `PreFilter` restricts the scheduling framework's node evaluation set exclusively to the named node via `PreFilterResult.NodeNames`, skipping feasibility checks on all other nodes in the cluster.
3. **Preemption Unresolvability Guarantee**: If a node's name does not match the pod's requested `Spec.NodeName`, the plugin rejects the node with `framework.UnschedulableAndUnresolvable`. This informs the preemption engine (`DefaultPreemption`) that evicting pods on other nodes can never satisfy this constraint, preventing futile preemption attempts across the cluster.
4. **Opportunistic Batch Signatures**: Implements `SignPlugin` to fingerprint pods by their target node name for opportunistic batching.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/nodename/
├── node_name.go       # Plugin definition, PreFilter, Filter, EventsToRegister, and SignPod implementations
├── node_name_test.go  # Table-driven unit tests covering Filter and PreFilter node pruning
└── AGENTS.md          # This agent documentation
```

---

## 3. Extension Point Implementations

`NodeName` implements `fwk.PreFilterPlugin`, `fwk.FilterPlugin`, `fwk.EnqueueExtensions`, and `fwk.SignPlugin`.

```
                  ┌───────────────────────────────┐
                  │           Pod Spec            │
                  │     (pod.Spec.NodeName)       │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
  [ NodeName != "" ]                                [ NodeName == "" ]
         │                                                 │
         ▼                                                 ▼
┌─────────────────────────────────┐               ┌─────────────────┐
│ PreFilter:                      │               │ PreFilter:      │
│ Return PreFilterResult with     │               │ Return nil, nil │
│ NodeNames: sets.New(nodeName)   │               └────────┬────────┘
└────────────────┬────────────────┘                        │
                 │                                         │
                 ▼                                         ▼
┌───────────────────────────────────────────────────────────────────┐
│ Filter:                                                           │
│ Fits(pod, nodeInfo) => len(NodeName)==0 || NodeName==nodeInfo.Name│
│ - Match:    return nil (Success)                                  │
│ - Mismatch: return UnschedulableAndUnresolvable                   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.1. `PreFilter` (`PreFilterPlugin`)
- **Signature**: `PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status)`
- **Behavior**:
  - If `pod.Spec.NodeName` is non-empty, constructs and returns `&fwk.PreFilterResult{NodeNames: sets.New(pod.Spec.NodeName)}`. The scheduler runtime uses this set to bypass filter evaluation for all nodes not present in the set.
  - If `pod.Spec.NodeName` is empty (unassigned pod standard scheduling flow), returns `nil, nil`, allowing all cluster nodes to proceed to subsequent filter plugins.
- **Extensions**: `PreFilterExtensions()` returns `nil` (no add/remove cycle state adjustments needed).

### 3.2. `Filter` (`FilterPlugin`)
- **Signature**: `Filter(ctx context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status`
- **Evaluation Logic**:
  ```go
  func Fits(pod *v1.Pod, nodeInfo fwk.NodeInfo) bool {
      return len(pod.Spec.NodeName) == 0 || pod.Spec.NodeName == nodeInfo.Node().Name
  }
  ```
- **Return Status**:
  - Returns `nil` when `Fits` evaluates to `true`.
  - Returns `fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReason)` on mismatch (`"node(s) didn't match the requested node name"`).

### 3.3. `EventsToRegister` (`EnqueueExtensions`)
- **Signature**: `EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error)`
- **Registered Events**:
  - `fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add}`: Requeues rejected pods when new nodes join the cluster. No queueing hint function is attached (`QueueingHintFn: nil`) because pod scheduling retries with standard backoff when a new node is added.

### 3.4. `SignPod` (`SignPlugin`)
- **Signature**: `SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status)`
- **Signature Fragment**:
  - Key: `fwk.NodeNameSignerName`
  - Value: `pod.Spec.NodeName`
- **Purpose**: Groups pods targeting the same node for opportunistic scheduling batching.

---

## 4. Key Constants & Status Invariants

| Identifier | Value / Type | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.NodeName` (`"NodeName"`) | Registered plugin name in scheduler profile. |
| `ErrReason` | `"node(s) didn't match the requested node name"` | Diagnostic failure reason returned on filter mismatch. |
| Status Code | `fwk.UnschedulableAndUnresolvable` | Marks the failure as impossible to resolve via preemption/eviction. |

---

## 5. Interaction with Preemption & Other Plugins

1. **Preemption Bypassing (`UnschedulableAndUnresolvable`)**:
   - Marking mismatched nodes as `UnschedulableAndUnresolvable` ensures `pkg/scheduler/framework/preemption` excludes them from candidate victim searches, avoiding wasteful eviction of running workloads on nodes that can never satisfy `pod.Spec.NodeName`.
2. **In-Place Pod Resize & Deferred Pod Scheduling**:
   - For assigned pods undergoing deferred resize or scheduler-directed preemption, `pod.Spec.NodeName` is already set. `PreFilter` ensures the scheduler exclusively evaluates and performs preemption simulation on the exact node where the pod is currently placed.

---

## 6. Testing Strategy & Test Coverage

The unit tests in `node_name_test.go` validate correctness across both extension points:

### Test Suites:
1. **`TestNodeName` (Filter Evaluation)**:
   - Evaluates combinations of:
     - Unset `pod.Spec.NodeName` -> `Success` (`nil`).
     - Matching `pod.Spec.NodeName == node.Name` -> `Success` (`nil`).
     - Mismatched `pod.Spec.NodeName != node.Name` -> Returns `fwk.UnschedulableAndUnresolvable` with `ErrReason`.
2. **`TestNodeName_PreFilter` (Node Set Pruning)**:
   - Validates that assigned pods with `pod.Spec.NodeName` return a `PreFilterResult` containing exactly `{pod.Spec.NodeName}`.
   - Validates that unassigned pods return `nil` result without error.
