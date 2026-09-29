# AGENTS.md: Developer & Agent Guide for Scheduler Testing Infrastructure (`pkg/scheduler/testing`)

This guide provides AI agents and human contributors with an architectural overview, catalog of test harnesses and mock implementations, object builder references, and step-by-step methodologies for writing unit, table-driven, and integration tests for Kubernetes scheduler plugins and core subsystems.

---

## 1. High-Level Purpose & Subsystem Role

The `pkg/scheduler/testing` package and its subpackages form the testing backbone for `kube-scheduler`. They decouple unit and integration test suites from the complexity of spinning up live Kubernetes clusters, providing:

1. **Fluent Object Builders (`wrappers.go`)**: Composable, chainable wrappers for creating and mutating Kubernetes domain objects (Pods, Nodes, PodGroups, Workloads, ResourceClaims, StorageClasses, PVCs, PDBs, etc.) with safe defaults.
2. **Mock Framework Plugins (`framework/fake_plugins.go`)**: Configurable mock plugins covering every framework extension point (PreEnqueue, QueueSort, PreFilter, Filter, PostFilter, PreScore, Score, Reserve, Permit, PreBind, Bind) to verify framework orchestration, short-circuiting, and status handling.
3. **In-Memory Fake Listers (`framework/fake_listers.go`)**: Lightweight in-memory implementations of client-go and scheduler lister interfaces (`NodeInfoLister`, `ServiceLister`, `ControllerLister`, `ReplicaSetLister`, `StatefulSetLister`, `PersistentVolumeClaimLister`, `CSINodeLister`, `StorageClassLister`, `VolumeAttachmentLister`).
4. **Mock Scheduler Extenders (`framework/fake_extender.go`)**: Programmable mock HTTP extenders to validate extender filtering, scoring, and preemption flows.
5. **Framework Construction Helpers (`framework/framework_helpers.go`)**: Declarative helpers to assemble custom `fwk.Framework` and `runtime.Registry` instances for targeted extension testing.
6. **Plugin Test Harnesses (`pkg/scheduler/framework/plugins/testing`)**: Helpers (`SetupPlugin`, `SetupPluginWithInformers`) to instantiate plugins with mocked or fake client-go informers.
7. **Queue & Cache Test Utilities (`pkg/scheduler/backend/...`)**: Specialized test fixtures (`queue.NewTestQueue`, `fake.Cache`) for testing queueing logic, backoff, queueing hints, and cache snapshotting.

```
                              ┌────────────────────────────────────────┐
                              │       Scheduler Test Testbed           │
                              └──────────────────┬─────────────────────┘
                                                 │
            ┌────────────────────────────────────┼────────────────────────────────────┐
            ▼                                    ▼                                    ▼
┌──────────────────────┐             ┌──────────────────────┐             ┌──────────────────────┐
│   Fluent Builders    │             │   Framework Fakes    │             │    Harness Helpers   │
│  (testing/wrappers)  │             │ (testing/framework)  │             │   (plugins/testing)  │
├──────────────────────┤             ├──────────────────────┤             ├──────────────────────┤
│ • MakePod()          │             │ • FakeFilterPlugin   │             │ • SetupPlugin()      │
│ • MakeNode()         │             │ • FakeScorePlugin    │             │ • SetupPluginWith-   │
│ • MakePodGroup()     │             │ • FakeReservePlugin  │             │   Informers()        │
│ • MakeResourceClaim()│             │ • FakePermitPlugin   │             │ • NewFramework()     │
│ • MakePDB()          │             │ • NodeInfoLister     │             │ • NewTestQueue()     │
│ • MakeStorageClass() │             │ • FakeExtender       │             │ • fake.Cache         │
└──────────────────────┘             └──────────────────────┘             └──────────────────────┘
            │                                    │                                    │
            └────────────────────────────────────┼────────────────────────────────────┘
                                                 │
                                                 ▼
                              ┌────────────────────────────────────────┐
                              │           Test Target Layers           │
                              ├────────────────────────────────────────┤
                              │ 1. Plugin Unit & Table-Driven Tests    │
                              │ 2. Framework Runtime Pipeline Tests    │
                              │ 3. Queue & Cache Subsystem Tests       │
                              │ 4. Preemption & Gang Evaluation Tests  │
                              │ 5. Full Integration Tests (test/integ) │
                              └────────────────────────────────────────┘
```

---

## 2. Directory Architecture & Subsystem Layout

```
pkg/scheduler/testing/
├── wrappers.go                   # Fluent object builders for Pods, Nodes, Storage, Claims, and PodGroups
├── workload_prep.go              # Synthetic cluster workload generator (MakeNodesAndPodsForEvenPodsSpread)
├── AGENTS.md                     # This agent guide
│
├── framework/                    # Framework mocking, fake listers, and test framework constructors
│   ├── fake_plugins.go           # Mock plugin implementations across all framework extension points
│   ├── fake_listers.go           # In-memory fake lister implementations for nodes, pods, services, storage
│   ├── fake_extender.go          # Mock HTTP extender with predicate, prioritizer, and preemption hooks
│   └── framework_helpers.go      # Framework test builder (NewFramework) and extension registration helpers
│
└── [Related Subsystem Test Packages]
    ├── ../framework/plugins/testing/ # SetupPlugin and SetupPluginWithInformers test harnesses
    ├── ../backend/queue/testing.go   # NewTestQueue and NewTestQueueWithObjects queue test harnesses
    ├── ../backend/cache/fake/        # Mock cache implementation (fake.Cache)
    └── ../../../test/integration/scheduler/ # Full integration test harness (InitTestSchedulerForFrameworkTest)
```

---

## 3. Fluent Object Builders (`pkg/scheduler/testing/wrappers.go`)

The `wrappers.go` file provides builder patterns returning wrapper structs with chaining methods. Call `.Obj()` at the end of the chain to obtain the underlying typed Kubernetes object.

### 3.1. Pod Builder (`MakePod()`)

Constructs `*v1.Pod` objects with fine-grained configuration for resource requests, affinity/anti-affinity, scheduling gates, tolerations, volume mounts, DRA claims, and pod groups.

```go
pod := st.MakePod().
    Namespace("default").
    Name("test-pod").
    UID("test-pod-uid").
    Priority(100).
    Req(map[v1.ResourceName]string{
        v1.ResourceCPU:    "500m",
        v1.ResourceMemory: "512Mi",
    }).
    Lim(map[v1.ResourceName]string{
        v1.ResourceCPU:    "1000m",
        v1.ResourceMemory: "1Gi",
    }).
    NodeSelector(map[string]string{"disk": "ssd"}).
    Toleration("key1").
    SchedulingGates([]string{"gate1"}).
    PodGroupName("gang-1").
    Obj()
```

#### Common `PodWrapper` Methods:
| Category | Method | Description |
| :--- | :--- | :--- |
| **Identity & Meta** | `.Name(string)`, `.Namespace(string)`, `.UID(string)`, `.Label(k, v)`, `.Annotation(k, v)` | Sets metadata fields. |
| **Resources** | `.Req(map)`, `.Lim(map)`, `.Res(map)` | Adds default container with resource requests and limits. |
| **Containers** | `.Container(image)`, `.Containers([]v1.Container)` | Configures custom containers (default image is `"pause"`). |
| **Init & Sidecars** | `.InitReq(map)`, `.SidecarReq(map)`, `.InitContainerPort(...)` | Adds restartable sidecars and init container resources/ports. |
| **Pod-Level Resources** | `.PodLevelResourceRequests(map)` | Configures pod-level requests (`.spec.resources.requests`). |
| **Affinity** | `.NodeAffinityIn(k, vals, t)`, `.NodeAffinityNotIn(k, vals)` | Appends NodeSelectorTerms to `.spec.affinity.nodeAffinity`. |
| **Pod Affinity** | `.PodAffinity(topKey, selector, kind)`, `.PodAntiAffinity(...)` | Configures PodAffinity / PodAntiAffinity terms. |
| **Topology Spread** | `.SpreadConstraint(maxSkew, tpKey, mode, selector, ...)` | Appends a `TopologySpreadConstraint`. |
| **Tolerations** | `.Toleration(key)`, `.Tolerations([]v1.Toleration)` | Configures tolerations for tainted nodes. |
| **Priority & Preemption** | `.Priority(int32)`, `.PreemptionPolicy(v1.PreemptionPolicy)` | Configures priority and preemption behavior. |
| **Volumes & Storage** | `.PVC(pvcName)`, `.Volume(v1.Volume)`, `.Volumes([]v1.Volume)` | Adds volume claims or volume specs. |
| **DRA Claims** | `.PodResourceClaims(...v1.PodResourceClaim)` | Configures DRA ResourceClaims for device allocation. |
| **PodGroups / Gang** | `.PodGroupName(name)` | Sets the `scheduling.x-k8s.io/pod-group` label/reference. |
| **State / Placement** | `.Node(nodeName)`, `.Phase(v1.PodPhase)`, `.Condition(...)` | Simulates scheduled, running, or terminating pod states. |

---

### 3.2. Node Builder (`MakeNode()`)

Constructs `*v1.Node` objects with allocatable capacities, taints, conditions, labels, images, and declared features.

```go
node := st.MakeNode().
    Name("node-1").
    Capacity(map[v1.ResourceName]string{
        v1.ResourceCPU:    "4",
        v1.ResourceMemory: "8Gi",
        v1.ResourcePods:   "110",
    }).
    Label("topology.kubernetes.io/zone", "zone-a").
    Label("node.kubernetes.io/instance-type", "m5.large").
    Taints([]v1.Taint{{Key: "special", Value: "gpu", Effect: v1.TaintEffectNoSchedule}}).
    Condition(v1.NodeReady, v1.ConditionTrue, "Ready", "Node is ready").
    Obj()
```

#### Common `NodeWrapper` Methods:
| Method | Description |
| :--- | :--- |
| `.Name(string)`, `.UID(string)` | Sets node metadata name and UID. |
| `.Capacity(map[v1.ResourceName]string)` | Populates both `.status.capacity` and `.status.allocatable`. |
| `.Label(k, v)`, `.Annotation(k, v)` | Adds labels or annotations. |
| `.Taints([]v1.Taint)` | Configures node taints. |
| `.Unschedulable(bool)` | Sets `.spec.unschedulable`. |
| `.Images(map[string]int64)` | Adds cached container images with byte sizes. |
| `.DeclaredFeatures([]string)` | Adds declared feature tags to node annotations. |
| `.Condition(type, status, message, reason)` | Sets node condition status. |

---

### 3.3. Gang Scheduling & Workload Builders

For testing composite pod groups, workloads, and gang scheduling:

```go
// 1. PodGroup Builder
pg := st.MakePodGroup().
    Namespace("default").
    Name("gang-1").
    MinCount(4).
    BasicPolicy().
    DisruptionModeAll().
    Priority(100).
    TopologyKey("topology.kubernetes.io/zone").
    Obj()

// 2. Workload & CompositePodGroup Builders
workload := st.MakeWorkload().
    Name("batch-workload").
    Namespace("default").
    PodGroupTemplate(st.MakePodGroupTemplate().Name("sub-pg-1").MinCount(2).Obj()).
    Obj()

cpg := st.MakeCompositePodGroup().
    Name("cpg-1").
    Namespace("default").
    MinGroupCount(2).
    DisruptionModeAll().
    Obj()
```

---

### 3.4. Dynamic Resource Allocation (DRA) Builders

For testing device claims, resource slices, and device classes:

```go
claim := st.MakeResourceClaim().
    Namespace("default").
    Name("gpu-claim").
    ReservedForPod("test-pod", "pod-uid").
    Allocation(&resourceapi.AllocationResult{...}).
    Obj()

slice := st.MakeResourceSlice("node-1", "gpu.example.com").
    Devices("gpu-0", "gpu-1").
    Obj()
```

---

### 3.5. Storage & Policy Builders

- **`MakePersistentVolumeClaim()`**: Builds `*v1.PersistentVolumeClaim` with access modes, storage requests, and storage class names.
- **`MakePersistentVolume()`**: Builds `*v1.PersistentVolume` with capacity, node affinity terms, and reclaim policies.
- **`MakeStorageClass()`**: Builds `*storagev1.StorageClass` with volume binding modes (`VolumeBindingWaitForFirstConsumer`, `VolumeBindingImmediate`) and allowed topologies.
- **`MakeCSINode()`**, **`MakeCSIDriver()`**, **`MakeCSIStorageCapacity()`**, **`MakeVolumeAttachment()`**: CSI storage plugin testing wrappers.
- **`MakePDB()`**: Builds `*policy.PodDisruptionBudget` with `MinAvailable`, match labels, and allowed disruptions for preemption testing.

---

### 3.6. Workload Preparation Generator (`workload_prep.go`)

`MakeNodesAndPodsForEvenPodsSpread` generates a synthetic cluster topology for topology spread and balanced allocation tests:

```go
existingPods, allNodes, filteredNodes := st.MakeNodesAndPodsForEvenPodsSpread(
    map[string]string{"app": "foo"}, // Pod labels distributed round-robin
    20,                               // Number of existing pods
    10,                               // Total nodes generated across 10 zones
    8,                                // Subset of filtered nodes
)
```

---

## 4. Framework Test Helpers & Mock Implementations (`pkg/scheduler/testing/framework`)

The `pkg/scheduler/testing/framework` subpackage provides mock plugin implementations, fake listers, and framework bootstrap functions.

### 4.1. Mock Plugin Implementations (`fake_plugins.go`)

Mock plugins implement framework extension points to simulate successes, rejections, errors, and wait timeouts:

```
                            ┌────────────────────────┐
                            │    fwk.Plugin (Mock)   │
                            └───────────┬────────────┘
                                        │
     ┌──────────────────────────────────┼──────────────────────────────────┐
     ▼                                  ▼                                  ▼
[ Filtering Fakes ]            [ Scoring & Permit Fakes ]         [ Binding & Reserve ]
• TrueFilterPlugin             • FakePreScoreAndScorePlugin       • FakeReservePlugin
• FalseFilterPlugin            • FakePermitPlugin (Timeout/Wait)  • FakePreBindPlugin
• FakeFilterPlugin             • NewEqualPrioritizerPlugin        • FakePostFilterPlugin
• FakePreFilterPlugin
```

| Mock Plugin | Extension Points | Key Behavior |
| :--- | :--- | :--- |
| **`TrueFilterPlugin`** | `Filter` | Always returns `Success` (`nil`). |
| **`FalseFilterPlugin`** | `Filter` | Always returns `Unschedulable` with `ErrReasonFake`. |
| **`FakeFilterPlugin`** | `Filter` | Returns configurable `fwk.Code` per node name (via `failedNodeReturnCodeMap`). Counts invocations with `NumFilterCalled`. |
| **`FakePreFilterPlugin`** | `PreFilter` | Returns pre-configured `*fwk.PreFilterResult` and `*fwk.Status`. |
| **`FakePreFilterAndFilterPlugin`** | `PreFilter`, `Filter` | Combines PreFilter and Filter mocking. |
| **`MatchFilterPlugin`** | `Filter` | Matches pod name against node name. |
| **`FakeReservePlugin`** | `Reserve`, `Unreserve` | Returns configured status on Reserve; tracks `UnreserveCalled`. |
| **`FakePermitPlugin`** | `Permit` | Returns configured `fwk.Status` and timeout duration. |
| **`FakePreScoreAndScorePlugin`**| `PreScore`, `Score` | Returns static score and statuses for PreScore/Score phases. |
| **`FakePostFilterPlugin`** | `PostFilter` | Returns pre-configured `*fwk.PostFilterResult` and `*fwk.Status`. |
| **`FakePreBindPlugin`** | `PreBind`, `PreBindPreFlight` | Returns statuses for PreBindPreFlight and PreBind phases. |

---

### 4.2. In-Memory Fake Listers (`fake_listers.go`)

`fake_listers.go` provides slice-backed implementations of client-go and framework listers without needing running informers:

```go
// Create a fake NodeInfoLister from a slice of fwk.NodeInfo
nodeInfoList := tf.BuildNodeInfos([]*v1.Node{node1, node2})
nodeLister := tf.NodeInfoLister(nodeInfoList)

node, err := nodeLister.Get("node1")
allNodes, err := nodeLister.List()

// Fake Service and Controller Listers
serviceLister := tf.ServiceLister([]*v1.Service{svc1, svc2})
controllerLister := tf.ControllerLister([]*v1.ReplicationController{rc1})
pvcLister := tf.PersistentVolumeClaimLister([]v1.PersistentVolumeClaim{pvc1})
```

---

### 4.3. Mock Scheduler Extender (`fake_extender.go`)

`FakeExtender` allows testing external HTTP extender interactions:

```go
extender := &tf.FakeExtender{
    Predicates: []tf.FitPredicate{
        tf.Node1PredicateExtender, // only allows "node1"
    },
    Prioritizers: []tf.PriorityConfig{
        {Function: tf.Node1PrioritizerExtender, Weight: 1},
    },
    Weight: 1,
}
```

Pre-defined extender predicates include:
- `TruePredicateExtender`: Always allows the node.
- `FalsePredicateExtender`: Always marks node unschedulable.
- `ErrorPredicateExtender`: Returns `fwk.Error`.
- `OccupiedNodePredicateExtender`: Allows node only if it has zero scheduled pods.
- `Node1PredicateExtender` / `Node2PredicateExtender`: Allows only `"node1"` / `"node2"`.

---

### 4.4. Framework Construction Helpers (`framework_helpers.go`)

Constructs a fully initialized `framework.Framework` using fluent registration functions:

```go
ctx := context.Background()
fwkHandle, err := tf.NewFramework(
    ctx,
    []tf.RegisterPluginFunc{
        tf.RegisterQueueSortPlugin("PrioritySort", queuesort.New),
        tf.RegisterPreFilterPlugin("FakePreFilter", tf.NewFakePreFilterPlugin("FakePreFilter", nil, nil)),
        tf.RegisterFilterPlugin("FakeFilter", tf.NewFakeFilterPlugin(map[string]fwk.Code{"node2": fwk.Unschedulable})),
        tf.RegisterScorePlugin("FakeScore", tf.NewFakePreScoreAndScorePlugin("FakeScore", 10, nil, nil), 1),
    },
    "test-scheduler-profile",
    runtime.WithSnapshotSharedLister(sharedLister),
)
```

---

## 5. Plugin Test Harnesses (`pkg/scheduler/framework/plugins/testing`)

For testing individual plugin implementations (such as `NodeResourcesFit`, `NodeAffinity`, `PodTopologySpread`), use `SetupPlugin` or `SetupPluginWithInformers`:

### 5.1. `SetupPlugin` vs `SetupPluginWithInformers`

```go
// 1. SetupPlugin (No informer events needed; snapshot shared lister only)
p := plugintesting.SetupPlugin(ctx, t, noderesources.NewFit, &config.NodeResourcesFitArgs{}, sharedLister)

// 2. SetupPluginWithInformers (Requires informers and fake clientset)
p := plugintesting.SetupPluginWithInformers(
    ctx,
    t,
    podtopologyspread.New,
    &config.PodTopologySpreadArgs{},
    sharedLister,
    []apiruntime.Object{ns, svc, pod1, pod2},
)
```

`SetupPluginWithInformers`:
- Initializes an in-memory `fake.Clientset` with `objs`.
- Starts the `SharedInformerFactory` and waits for `WaitForCacheSync`.
- Injects the `SharedInformerFactory` and `SharedLister` into the plugin `fwk.Handle`.

---

## 6. Writing Unit & Table-Driven Tests for Scheduler Plugins

Table-driven testing is the standard pattern across `pkg/scheduler`. Below are canonical test templates for common extension points.

### 6.1. Table-Driven Test for `Filter` & `PreFilter` Plugins

```go
func TestMyFilterPlugin(t *testing.T) {
    tests := []struct {
        name           string
        pod            *v1.Pod
        node           *v1.Node
        existingPods   []*v1.Pod
        expectedStatus *fwk.Status
    }{
        {
            name: "pod fits on empty node",
            pod:  st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
            node: st.MakeNode().Name("n1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
            expectedStatus: nil, // Success
        },
        {
            name: "pod exceeds node allocatable CPU",
            pod:  st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
            node: st.MakeNode().Name("n1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
            expectedStatus: fwk.NewStatus(fwk.Unschedulable, "Insufficient cpu"),
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            tCtx := ktesting.Init(t)

            // 1. Build NodeInfo
            nodeInfo := framework.NewNodeInfo(tt.existingPods...)
            nodeInfo.SetNode(tt.node)

            // 2. Build Snapshot SharedLister
            snapshot := cache.NewEmptySnapshot()
            snapshot.UpdateNodeInfoMap(map[string]fwk.NodeInfo{tt.node.Name: nodeInfo})

            // 3. Initialize Plugin
            p := plugintesting.SetupPlugin(tCtx, t, myplugin.New, &config.MyPluginArgs{}, snapshot)
            filterPlugin := p.(fwk.FilterPlugin)

            // 4. Initialize CycleState
            state := framework.NewCycleState()

            // 5. Run PreFilter if implemented
            if preFilterPlugin, ok := p.(fwk.PreFilterPlugin); ok {
                _, status := preFilterPlugin.PreFilter(tCtx, state, tt.pod, []fwk.NodeInfo{nodeInfo})
                if !status.IsSuccess() {
                    if diff := cmp.Diff(tt.expectedStatus.Code(), status.Code()); diff != "" {
                        t.Fatalf("PreFilter status code mismatch (-want +got):\n%s", diff)
                    }
                    return
                }
            }

            // 6. Run Filter
            gotStatus := filterPlugin.Filter(tCtx, state, tt.pod, nodeInfo)
            if (tt.expectedStatus == nil && !gotStatus.IsSuccess()) ||
                (tt.expectedStatus != nil && gotStatus.Code() != tt.expectedStatus.Code()) {
                t.Errorf("Filter() status = %v, want %v", gotStatus, tt.expectedStatus)
            }
        })
    }
}
```

---

### 6.2. Table-Driven Test for `Score` Plugins

```go
func TestMyScorePlugin(t *testing.T) {
    tests := []struct {
        name          string
        pod           *v1.Pod
        nodes         []*v1.Node
        expectedScores fwk.NodeScoreList
    }{
        {
            name: "scores nodes based on resource packing",
            pod:  st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
            nodes: []*v1.Node{
                st.MakeNode().Name("n1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
                st.MakeNode().Name("n2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
            },
            expectedScores: fwk.NodeScoreList{
                {Name: "n1", Score: 50},
                {Name: "n2", Score: 25},
            },
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            tCtx := ktesting.Init(t)

            var nodeInfos []fwk.NodeInfo
            nodeMap := make(map[string]fwk.NodeInfo)
            for _, n := range tt.nodes {
                ni := framework.NewNodeInfo()
                ni.SetNode(n)
                nodeInfos = append(nodeInfos, ni)
                nodeMap[n.Name] = ni
            }

            snapshot := cache.NewEmptySnapshot()
            snapshot.UpdateNodeInfoMap(nodeMap)

            p := plugintesting.SetupPlugin(tCtx, t, myscoreplugin.New, nil, snapshot)
            scorePlugin := p.(fwk.ScorePlugin)
            state := framework.NewCycleState()

            // Run PreScore if implemented
            if preScorePlugin, ok := p.(fwk.PreScorePlugin); ok {
                status := preScorePlugin.PreScore(tCtx, state, tt.pod, nodeInfos)
                if !status.IsSuccess() {
                    t.Fatalf("PreScore failed: %v", status)
                }
            }

            // Run Score per node
            var gotScores fwk.NodeScoreList
            for _, ni := range nodeInfos {
                score, status := scorePlugin.Score(tCtx, state, tt.pod, ni)
                if !status.IsSuccess() {
                    t.Fatalf("Score failed for node %q: %v", ni.Node().Name, status)
                }
                gotScores = append(gotScores, fwk.NodeScore{Name: ni.Node().Name, Score: score})
            }

            // Run NormalizeScore if implemented
            if scorePlugin.ScoreExtensions() != nil {
                status := scorePlugin.ScoreExtensions().NormalizeScore(tCtx, state, tt.pod, gotScores)
                if !status.IsSuccess() {
                    t.Fatalf("NormalizeScore failed: %v", status)
                }
            }

            if diff := cmp.Diff(tt.expectedScores, gotScores); diff != "" {
                t.Errorf("Unexpected scores (-want +got):\n%s", diff)
            }
        })
    }
}
```

---

### 6.3. Testing `Reserve` & `Unreserve` Rollbacks

```go
func TestReserveRollback(t *testing.T) {
    tCtx := ktesting.Init(t)
    pod := st.MakePod().Name("pod-1").Obj()
    node := st.MakeNode().Name("node-1").Obj()

    p := plugintesting.SetupPlugin(tCtx, t, myreserveplugin.New, nil, cache.NewEmptySnapshot())
    reservePlugin := p.(fwk.ReservePlugin)
    state := framework.NewCycleState()

    // 1. Reserve
    status := reservePlugin.Reserve(tCtx, state, pod, node.Name)
    if !status.IsSuccess() {
        t.Fatalf("Reserve failed: %v", status)
    }

    // 2. Unreserve (Rollback)
    reservePlugin.Unreserve(tCtx, state, pod, node.Name)

    // 3. Verify state cleanup
    // Ensure state keys are cleared or underlying cache released
}
```

---

## 7. Writing Subsystem Tests (Queue, Cache, Preemption)

### 7.1. Queue Subsystem Tests (`backend/queue`)

Use `queue.NewTestQueue` or `queue.NewTestQueueWithObjects` to test active/backoff queues, scheduling gates, and `QueueingHint` callbacks:

```go
func TestSchedulingQueue(t *testing.T) {
    tCtx := ktesting.Init(t)

    // 1. Initialize queue with PrioritySort
    q := queue.NewTestQueue(tCtx, queuesort.Less)
    defer q.Close()

    // 2. Add Pods
    highPod := st.MakePod().Name("high").Priority(100).Obj()
    lowPod := st.MakePod().Name("low").Priority(10).Obj()

    q.Add(tCtx, lowPod)
    q.Add(tCtx, highPod)

    // 3. Pop in priority order
    pInfo1, err := q.Pop(tCtx)
    if err != nil || pInfo1.Pod.Name != "high" {
        t.Errorf("Expected high priority pod, got %v (err: %v)", pInfo1, err)
    }

    pInfo2, err := q.Pop(tCtx)
    if err != nil || pInfo2.Pod.Name != "low" {
        t.Errorf("Expected low priority pod, got %v (err: %v)", pInfo2, err)
    }
}
```

---

### 7.2. Cache Subsystem Tests (`backend/cache`)

Use `fake.Cache` to mock `AssumePod`, `ForgetPod`, and snapshot updates:

```go
func TestCacheAssumeAndForget(t *testing.T) {
    assumedPods := make(map[string]bool)
    fakeCache := &fake.Cache{
        AssumeFunc: func(p *v1.Pod) {
            assumedPods[p.Name] = true
        },
        ForgetFunc: func(p *v1.Pod) {
            delete(assumedPods, p.Name)
        },
    }

    pod := st.MakePod().Name("p1").Node("n1").Obj()
    _ = fakeCache.AssumePod(klog.Background(), pod)
    if !assumedPods["p1"] {
        t.Errorf("Expected pod to be assumed")
    }

    _ = fakeCache.ForgetPod(klog.Background(), pod)
    if assumedPods["p1"] {
        t.Errorf("Expected pod to be forgotten")
    }
}
```

---

## 8. Scheduler Integration Testing (`test/integration/scheduler`)

Integration tests run real scheduler instances against an in-process `kube-apiserver` and `etcd`.

### 8.1. Integration Harness Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Integration Test Context (testutils)                 │
│                                                                        │
│  ┌──────────────────────┐                ┌──────────────────────────┐  │
│  │  In-Process etcd     │◄───────────────┤ In-Process apiserver     │  │
│  └──────────────────────┘                └────────────▲─────────────┘  │
│                                                       │ client-go      │
│                                          ┌────────────┴─────────────┐  │
│                                          │  Real kube-scheduler     │  │
│                                          │  (Informers, Queue, Fwk) │  │
│                                          └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 8.2. Integration Test Example

```go
func TestPodSchedulingIntegration(t *testing.T) {
    // 1. Initialize context and test environment
    testCtx := testutils.InitTestAPIServer(t, "sched-integration", nil)

    // 2. Initialize and run scheduler
    testCtx, shutdownFunc := scheduler.InitTestSchedulerForFrameworkTest(
        t,
        testCtx,
        2,    // Pre-create 2 nodes: test-node-0, test-node-1
        true, // Run scheduler background loop
    )
    defer shutdownFunc()

    cs := testCtx.ClientSet
    ns := testCtx.NS.Name

    // 3. Create Pod via client-go
    pod := st.MakePod().
        Namespace(ns).
        Name("integration-pod").
        Req(map[v1.ResourceName]string{v1.ResourceCPU: "100m"}).
        Obj()

    _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
    if err != nil {
        t.Fatalf("Failed to create pod: %v", err)
    }

    // 4. Wait for Pod to be bound to a node
    err = testutils.WaitForPodToSchedule(testCtx, pod)
    if err != nil {
        t.Fatalf("Pod did not schedule: %v", err)
    }
}
```

---

## 9. Testing Best Practices, Invariants & Common Gotchas

### 9.1. Invariants & Best Practices

1. **Object Isolation via `.Obj()`**:
   - Always call `.Obj()` on fluent builders inside test cases. Never share mutable pointer references across parallel sub-tests.
2. **Context Propagation with `ktesting`**:
   - Use `tCtx := ktesting.Init(t)` to obtain a contextual logger and cancellation context that adheres to Kubernetes testing standards.
3. **Comparing Complex Structures with `cmp.Diff`**:
   - Use `cmp.Diff` instead of `reflect.DeepEqual` for expressive failure diffs.
   - For `fwk.ClusterEvent` comparisons, include `cmpopts.EquateComparable(fwk.ClusterEvent{})`.
4. **Feature Gates Management**:
   - Use `featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MyFeature, true)` to safely enable feature gates during test execution and guarantee cleanup on completion.
5. **Informer Sync Verification**:
   - When using `SetupPluginWithInformers` or custom informer factories, always ensure `informerFactory.WaitForCacheSync(ctx.Done())` returns `true` before initiating scheduling operations.
6. **Parallel Sub-Tests**:
   - Use `t.Parallel()` inside table loops, and capture the loop iteration variable locally (`tt := tt`) to avoid race conditions in Go versions prior to 1.22.

---

### 9.2. Common Gotchas & Troubleshooting

| Symptom / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **`nil pointer dereference` in `NodeInfo.Pods`** | `NodeInfo` initialized without backing pods or nil node reference. | Use `framework.NewNodeInfo(pods...)` and call `nodeInfo.SetNode(node)` before passing to plugins. |
| **Informer events not received in test** | Informer factory was not started or synced. | Call `informerFactory.Start(ctx.Done())` and `informerFactory.WaitForCacheSync(ctx.Done())`. |
| **`Pop()` blocks indefinitely in Queue test** | Queue is empty or `metrics.NewMetricsAsyncRecorder` was omitted. | Use `queue.NewTestQueue` which automatically provisions the required async metrics recorder. |
| **CycleState key collision / panic** | Multiple plugins or test steps write to the same `CycleState` key without unique naming. | Use private typed keys in CycleState or isolate state instances per test case. |
| **Status mismatch in `Filter` vs `PreFilter`** | Plugin failed during `PreFilter` but test only asserted on `Filter`. | Run `PreFilter` before `Filter` in unit test harnesses to mirror the runtime execution pipeline. |
