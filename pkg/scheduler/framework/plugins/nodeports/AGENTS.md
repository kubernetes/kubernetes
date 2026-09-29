# NodePorts Plugin (`pkg/scheduler/framework/plugins/nodeports`)

This guide provides an architectural overview, interface implementations, port conflict algorithms, preemption semantics, queueing hints, and testing strategies for the `NodePorts` plugin in `pkg/scheduler/framework/plugins/nodeports`.

---

## 1. High-Level Purpose & Scope

The `NodePorts` plugin ensures that container host ports (`spec.containers[*].ports[*].hostPort`) requested by a pod do not conflict with host ports already allocated to existing pods running on candidate nodes.

### Core Responsibilities:
1. **Host Port Conflict Detection**: Compares incoming pod host port requirements against `NodeInfo.GetUsedPorts()` (`fwk.HostPortInfo`) to prevent port collisions on the host network.
2. **IP & Protocol Scoping**: Supports TCP, UDP, and SCTP protocols, and handles both specific host IP bindings and wildcard (`0.0.0.0` / `::`) bindings.
3. **PreFilter State Extraction**: Collects all host ports declared across the pod's regular and init containers once in `PreFilter`, storing them in `CycleState` to avoid redundant allocations during parallel node filtering.
4. **Preemption Resolvability (`Unschedulable`)**: Rejections use `fwk.Unschedulable` (not unresolvable), signaling to `DefaultPreemption` that evicting a pod currently occupying the conflicting host port could make the node schedulable.
5. **Port-Aware Queueing Hints**: Re-evaluates unscheduled pods only when a deleted pod was bound to a host and released overlapping host ports.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/nodeports/
├── node_ports.go       # Plugin definition, PreFilter, Filter, EnqueueExtensions, conflict checking
├── node_ports_test.go  # Unit tests for port conflicts, wildcard IPs, protocols, and queueing hints
└── AGENTS.md           # This agent documentation
```

---

## 3. Data Structures & State Management

### 3.1. `NodePorts` Struct & Key Constants

```go
type NodePorts struct {
    enableInPlacePodVerticalScalingSchedulerPreemption bool
}
```

| Identifier | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.NodePorts` (`"NodePorts"`) | Registered plugin name. |
| `preFilterStateKey` | `"PreFilterNodePorts"` | Key in `CycleState` storing precomputed port slice. |
| `ErrReason` | `"node(s) didn't have free ports for the requested pod ports"` | Filter failure diagnostic message. |

### 3.2. `preFilterState` (`CycleState`)

```go
type preFilterState []v1.ContainerPort

func (s preFilterState) Clone() fwk.StateData {
    return s // Immutable across nodes; shallow copy is safe
}
```

- Extracted via `util.GetHostPorts(pod)` during `PreFilter`.
- Aggregates container ports where `HostPort > 0`. If `HostIP` is unspecified, it defaults to `0.0.0.0`.

### 3.3. `fwk.HostPortInfo` (`pkg/scheduler/framework`)

`NodeInfo` maintains `UsedPorts` as a `HostPortInfo` map structure:
- **Structure**: `map[string]map[ProtocolPort]struct{}` where keys are Host IPs (or `"0.0.0.0"`) mapping to sets of `(Protocol, Port)`.
- **Conflict Evaluation (`CheckConflict`)**:
  - Checks exact IP + Protocol + Port match.
  - Checks wildcard IP (`0.0.0.0`) + Protocol + Port conflict with specific IPs and vice versa.

---

## 4. Extension Point Implementations

`NodePorts` implements `fwk.PreFilterPlugin`, `fwk.FilterPlugin`, `fwk.EnqueueExtensions`, and `fwk.SignPlugin`.

```
                        ┌───────────────────────────────┐
                        │     PreFilter: GetHostPorts   │
                        └───────────────┬───────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           │                                                         │
[ No HostPorts declared ]                                 [ HostPorts present ]
           │                                                         │
           ▼                                                         ▼
┌─────────────────────────────────────┐               ┌───────────────────────────────┐
│ Return framework.Skip               │               │ Write preFilterState to       │
│ (Bypasses Filter on all nodes)      │               │ CycleState; return nil, nil   │
└─────────────────────────────────────┘               └───────────────┬───────────────┘
                                                                      │
                                                                      ▼
                                                      ┌───────────────────────────────┐
                                                      │ Filter: Read preFilterState   │
                                                      │ Check nodeInfo.GetUsedPorts() │
                                                      └───────────────┬───────────────┘
                                                                      │
                                        ┌─────────────────────────────┴─────────────────────────────┐
                                        │                                                           │
                               [ No Port Conflict ]                                        [ Port Conflict Exists ]
                                        │                                                           │
                                        ▼                                                           ▼
                                ┌───────────────┐                                          ┌────────────────────────┐
                                │  Return nil   │                                          │ Return Unschedulable   │
                                │   (Success)   │                                          │ (Preemption Candidate) │
                                └───────────────┘                                          └────────────────────────┘
```

### 4.1. `PreFilter` (`PreFilterPlugin`)
- **Signature**: `PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status)`
- **Behavior**:
  1. Checks if `enableInPlacePodVerticalScalingSchedulerPreemption` is on and `resource.IsPodResizeDeferred(pod)` is true. If so, returns `fwk.NewStatus(fwk.Skip)` (port allocations do not change during in-place resize).
  2. Extracts host ports with `s := util.GetHostPorts(pod)`.
  3. If `len(s) == 0`, returns `fwk.NewStatus(fwk.Skip)` to bypass the filter phase on all cluster nodes.
  4. Writes `preFilterState(s)` to `cycleState` under `preFilterStateKey`.
- **Extensions**: `PreFilterExtensions()` returns `nil`.

### 4.2. `Filter` (`FilterPlugin`)
- **Signature**: `Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status`
- **Evaluation Logic**:
  1. Retrieves `wantPorts` from `cycleState`.
  2. Checks each container port against `nodeInfo.GetUsedPorts()`:
     ```go
     func fitsPorts(wantPorts []v1.ContainerPort, portsInUse fwk.HostPortInfo) bool {
         for _, cp := range wantPorts {
             if portsInUse.CheckConflict(cp.HostIP, string(cp.Protocol), cp.HostPort) {
                 return false
             }
         }
         return true
     }
     ```
  3. If any conflict exists, returns `fwk.NewStatus(fwk.Unschedulable, ErrReason)`.

### 4.3. `EventsToRegister` & Queueing Hints (`EnqueueExtensions`)
- **Signature**: `EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error)`
- **Registered Events**:
  1. **`AssignedPod (Delete)` with `isSchedulableAfterAssignedPodDeleted`**:
     - Ignores unscheduled or unassigned deleted pods (`NodeName == "" && NominatedNodeName == ""`).
     - Ignores deleted pods that do not declare any host ports.
     - Builds a temporary `HostPortInfo` from the deleted pod's ports and checks if it overlaps with `pod`'s requested ports (`!fitsPorts(...)`).
     - Returns `fwk.Queue` if the deleted pod held any overlapping host ports; otherwise returns `fwk.QueueSkip`.
  2. **`Node (Add)`**:
     - Requeues unscheduled pods when new nodes are added to the cluster.

### 4.4. `SignPod` (`SignPlugin`)
- **Signature**: `SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status)`
- **Signature Fragment**:
  - Key: `fwk.HostPortsSignerName`
  - Value: `fwk.HostPortsSigner(pod)`
- **Purpose**: Creates deterministic signatures of requested `(HostIP, Protocol, HostPort)` tuples to enable opportunistic batching.

---

## 5. Interaction with Preemption (`DefaultPreemption`)

Unlike plugins that return `UnschedulableAndUnresolvable`, `NodePorts` returns `fwk.Unschedulable`.
- During preemption simulations in `pkg/scheduler/framework/preemption`, victim pods are iteratively removed from a candidate node's `NodeInfo`.
- As victim pods are removed, their host ports are freed from `nodeInfo.UsedPorts`.
- If evicting lower-priority victim pods clears the conflicting host port, the candidate node becomes feasible for preemption.

---

## 6. Testing Strategy & Test Coverage

Unit tests in `node_ports_test.go` validate conflict checking, wildcard IP rules, and event queueing:

### Test Suites:
1. **`TestNodePorts` (Filter Conflict Matrix)**:
   - Tests conflicting and non-conflicting scenarios across TCP, UDP, and SCTP.
   - Tests exact IP matches (`1.1.1.1:80` vs `1.1.1.1:80` -> conflict).
   - Tests distinct IP matches (`1.1.1.1:80` vs `1.1.1.2:80` -> fits).
   - Tests wildcard IP bindings (`0.0.0.0:80` vs `1.1.1.1:80` -> conflict).
   - Tests distinct protocols on same port (`TCP:80` vs `UDP:80` -> fits).
2. **`TestPreFilter`**:
   - Verifies `fwk.Skip` status when pod requests no host ports.
   - Verifies correct population of `preFilterState` in `CycleState` when host ports are specified.
3. **`TestIsSchedulableAfterAssignedPodDeleted` (Queueing Hints)**:
   - Validates `fwk.Queue` when a deleted pod released a conflicting port on the same protocol/IP.
   - Validates `fwk.QueueSkip` when the deleted pod used unrelated ports or had no host ports.
   - Validates `fwk.QueueSkip` for unassigned deleted pods.
