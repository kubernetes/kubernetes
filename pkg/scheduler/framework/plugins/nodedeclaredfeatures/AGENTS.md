# NodeDeclaredFeatures Plugin (`pkg/scheduler/framework/plugins/nodedeclaredfeatures`)

This guide provides an architectural overview, interface implementations, feature inference mechanics, subset matching algorithms, queueing hints, and testing strategies for the `NodeDeclaredFeatures` plugin in `pkg/scheduler/framework/plugins/nodedeclaredfeatures`.

---

## 1. High-Level Purpose & Scope

The `NodeDeclaredFeatures` plugin implements feature-based node gating (KEP-4818: *Node Declared Features*). It matches inferred hardware or runtime feature requirements of a pod (e.g., specific CPU architecture features, runtime capabilities, or node capabilities) against features declared in node status (`node.Status.DeclaredFeatures`).

### Core Responsibilities:
1. **Pod Feature Inference**: Uses the Node Declared Features component helper library (`k8s.io/component-helpers/nodedeclaredfeatures`) to infer required features from `pod.Spec` against the cluster's Kubernetes version.
2. **Subset Feasibility Matching**: Validates that the pod's required feature set (`ndf.FeatureSet`) is a strict subset of the candidate node's declared features (`nodeInfo.GetNodeDeclaredFeatures()`).
3. **Preemption Protection (`UnschedulableAndUnresolvable`)**: Returns `fwk.UnschedulableAndUnresolvable` on mismatch because evicting running pods cannot alter the hardware or kernel capabilities declared by a node.
4. **Targeted Event Queueing Hints**: Requeues blocked pods only when a node's declared features are updated or when an update to the target pod alters its inferred feature requirements.
5. **Opportunistic Batch Signatures**: Implements `SignPlugin` to fingerprint pods by their required feature set.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/nodedeclaredfeatures/
├── nodedeclaredfeatures.go       # Plugin definition, PreFilter, Filter, EnqueueExtensions, Queueing Hints
├── nodedeclaredfeatures_test.go  # Unit tests for feature matching, versioning, node updates, and pod updates
└── AGENTS.md                     # This agent documentation
```

---

## 3. Data Structures & State Management

### 3.1. `NodeDeclaredFeatures` Struct & Key Constants

```go
type NodeDeclaredFeatures struct {
    ndfFramework                                       *ndf.Framework
    version                                            *versionutil.Version
    enabled                                            bool
    enableInPlacePodVerticalScalingSchedulerPreemption bool
}
```

| Identifier | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.NodeDeclaredFeatures` (`"NodeDeclaredFeatures"`) | Registered plugin name. |
| `preFilterStateKey` | `"PreFilterNodeDeclaredFeatures"` | Key in `CycleState` storing inferred `ndf.FeatureSet`. |
| `errReasonUnsatisfiedRequirements` | `"node(s) didn't match Pod's required features"` | Filter failure diagnostic message. |

### 3.2. `preFilterState` (`CycleState`)

```go
type preFilterState struct {
    reqs ndf.FeatureSet
}

func (s *preFilterState) Clone() fwk.StateData {
    return s
}
```

- Encapsulates `ndf.FeatureSet` inferred from `pod.Spec`.
- Stored in `CycleState` during `PreFilter` and read concurrently during parallel `Filter` evaluations.

---

## 4. Extension Point Implementations

`NodeDeclaredFeatures` implements `fwk.PreFilterPlugin`, `fwk.FilterPlugin`, `fwk.EnqueueExtensions`, and `fwk.SignPlugin`.

```
                        ┌────────────────────────────────────────┐
                        │ PreFilter: InferForPodScheduling(pod)  │
                        └───────────────────┬────────────────────┘
                                            │
           ┌────────────────────────────────┴────────────────────────────────┐
           │                                                                 │
[ Feature disabled / Reqs empty / Deferred resize ]                 [ Reqs non-empty ]
           │                                                                 │
           ▼                                                                 ▼
┌─────────────────────────────────────┐                    ┌───────────────────────────────────┐
│ Return framework.Skip               │                    │ Write preFilterState to           │
│ (Bypasses Filter on all nodes)      │                    │ CycleState; return nil, nil       │
└─────────────────────────────────────┘                    └─────────────────┬─────────────────┘
                                                                             │
                                                                             ▼
                                                           ┌───────────────────────────────────┐
                                                           │ Filter: Read preFilterState       │
                                                           │ isMatch = reqs.IsSubset(          │
                                                           │   nodeInfo.DeclaredFeatures)      │
                                                           └─────────────────┬─────────────────┘
                                                                             │
                                              ┌──────────────────────────────┴──────────────────────────────┐
                                              │                                                             │
                                      [ isMatch == true ]                                           [ isMatch == false ]
                                              │                                                             │
                                              ▼                                                             ▼
                                      ┌───────────────┐                                     ┌───────────────────────────────┐
                                      │  Return nil   │                                     │ Return                        │
                                      │   (Success)   │                                     │ UnschedulableAndUnresolvable  │
                                      └───────────────┘                                     └───────────────────────────────┘
```

### 4.1. `PreFilter` (`PreFilterPlugin`)
- **Signature**: `PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status)`
- **Behavior**:
  1. Checks if `enableInPlacePodVerticalScalingSchedulerPreemption` is on and `resource.IsPodResizeDeferred(pod)` is true. If so, returns `fwk.NewStatus(fwk.Skip)`.
  2. If plugin is not enabled (`fts.EnableNodeDeclaredFeatures == false`), returns `fwk.NewStatus(fwk.Skip)`.
  3. Infers required features using `pl.ndfFramework.InferForPodScheduling(&ndf.PodInfo{Spec: &pod.Spec}, pl.version)`.
  4. If inferred requirements are empty (`reqs.IsEmpty()`), returns `fwk.NewStatus(fwk.Skip)` to bypass filter evaluation across all nodes.
  5. Writes `preFilterState{reqs: reqs}` to `cycleState`.
- **Extensions**: `PreFilterExtensions()` returns `nil`.

### 4.2. `Filter` (`FilterPlugin`)
- **Signature**: `Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status`
- **Evaluation Logic**:
  1. Retrieves `preFilterState` from `cycleState`.
  2. Evaluates `s.reqs.IsSubset(nodeInfo.GetNodeDeclaredFeatures())`.
     - *Performance Optimization*: Calls `IsSubset` directly rather than `ndf.MatchNodeFeatureSet` to avoid computing expensive feature diff allocations during tight filter loops.
  3. Returns `nil` if the node possesses all required features; otherwise returns `fwk.NewStatus(fwk.UnschedulableAndUnresolvable, errReasonUnsatisfiedRequirements)`.

### 4.3. `EventsToRegister` & Queueing Hints (`EnqueueExtensions`)
- **Signature**: `EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error)`
- **Registered Events**:
  1. **`Node (Add | UpdateNodeDeclaredFeature)` with `isSchedulableAfterNodeChange`**:
     - Compares `oldNode.Status.DeclaredFeatures` with `newNode.Status.DeclaredFeatures` using `slices.Equal`.
     - If declared features changed or a new node is added, returns `fwk.Queue`.
     - If declared features are identical, returns `fwk.QueueSkip`.
  2. **`TargetPod (Update)` with `isSchedulableAfterTargetPodUpdate`**:
     - Infers feature requirements for `oldPod` and `newPod`.
     - If `newReqs.Equal(oldReqs)` is true, returns `fwk.QueueSkip`.
     - If requirements changed, returns `fwk.Queue` to retry scheduling.

### 4.4. `SignPod` (`SignPlugin`)
- **Signature**: `SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status)`
- **Signature Fragment**:
  - Key: `fwk.FeaturesSignerName`
  - Value: `fs.String()` (string representation of the inferred `FeatureSet`)
- **Purpose**: Groups pods requiring identical feature sets for opportunistic batch scheduling.

---

## 5. Testing Strategy & Test Coverage

Unit tests in `nodedeclaredfeatures_test.go` provide full validation of feature inference, matching, and event handling:

### Test Suites:
1. **`TestNodeDeclaredFeatures` (Filter Matching)**:
   - Tests pods with required features against nodes providing matching, superset, and subset features.
   - Verifies rejection with `UnschedulableAndUnresolvable` when required features are absent.
2. **`TestIsSchedulableAfterNodeChange` (Node Event Hints)**:
   - Validates `fwk.Queue` when `Status.DeclaredFeatures` is modified on a node.
   - Validates `fwk.QueueSkip` when unrelated node fields change.
3. **`TestIsSchedulableAfterTargetPodUpdate` (Pod Event Hints)**:
   - Validates `fwk.Queue` when a pod update alters its inferred feature requirements.
   - Validates `fwk.QueueSkip` when a pod update leaves required features identical.
