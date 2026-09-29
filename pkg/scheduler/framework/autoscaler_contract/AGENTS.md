# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/autoscaler_contract`

This guide provides AI agents and human contributors with an architectural overview, interface compatibility guarantees, and testing invariants for the contract verification suite located under `pkg/scheduler/framework/autoscaler_contract`.

---

## 1. High-Level Overview & Subsystem Purpose

The `autoscaler_contract` package serves as an interface compatibility guard between `kube-scheduler` and downstream external consumers—most notably the **Kubernetes Cluster Autoscaler** (`k8s.io/autoscaler/cluster-autoscaler`).

### Why This Package Exists:
Cluster Autoscaler simulates `kube-scheduler` scheduling cycles in-process to determine whether pending/unschedulable pods can fit onto newly provisioned or scaled-up node groups. To perform this simulation, Cluster Autoscaler vendors and directly invokes internal scheduler framework methods (`RunPreFilterPlugins`, `RunFilterPlugins`, `RunReservePluginsReserve`) and implements mock `SharedLister` interfaces.

Because `kube-scheduler` and `cluster-autoscaler` reside in separate repositories, unintended modifications to framework method signatures or lister contracts can break downstream builds or invalidate autoscaling simulation logic. The tests in this package enforce compile-time static type assertions against these external contract definitions.

---

## 2. Directory Architecture & File Map

```
pkg/scheduler/framework/autoscaler_contract/
├── framework_contract_test.go # Static assertions for Framework execution signatures & constructors
├── lister_contract_test.go    # Static assertions for SharedLister and specialized resource listers
└── AGENTS.md                  # This agent guide
```

---

## 3. Contract Specifications

### 3.1. Framework Contract (`framework_contract_test.go`)

The `frameworkContract` interface defines the exact subset of `fwk.Framework` methods required by Cluster Autoscaler:

```go
type frameworkContract interface {
    RunPreFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod) (*fwk.PreFilterResult, *fwk.Status, sets.Set[string])
    RunFilterPlugins(context.Context, fwk.CycleState, *v1.Pod, fwk.NodeInfo) *fwk.Status
    RunReservePluginsReserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status
}
```

#### Verified Invariants:
1. **Framework Implementation**: `frameworkContract(runtime.NewFramework(...))` must compile without type divergence.
2. **CycleState Constructor**: `framework.NewCycleState()` must return a valid, initialized `fwk.CycleState` instance.
3. **PreFilter Return Signature**: Must return `(*fwk.PreFilterResult, *fwk.Status, sets.Set[string])` representing the filtered node list, overall plugin status, and skipped filter plugin set.

### 3.2. Lister & Manager Contracts (`lister_contract_test.go`)

Downstream components construct synthetic `SharedLister` snapshots to simulate prospective node additions. The test suite enforces compatibility across all underlying lister interfaces:

```go
type listerContract interface {
    fwk.NodeInfoLister
    fwk.StorageInfoLister
    fwk.SharedLister
    fwk.ResourceSliceLister
    fwk.PodGroupStateLister
    fwk.PodGroupLister
    fwk.CompositePodGroupStateLister
    fwk.CompositePodGroupLister
    fwk.DeviceClassLister
    fwk.ResourceClaimTracker
    fwk.DeviceClassResolver
    fwk.SharedDRAManager
}
```

#### Tested Interfaces:
- **`fwk.NodeInfoLister`**: Node caching and retrieval (`Get`, `List`, `HavePodsWithAffinityList`).
- **`fwk.StorageInfoLister`**: PV and PVC caching (`IsPVCActive`, `GetStorageClassInfo`).
- **`fwk.SharedDRAManager` / `ResourceSliceLister` / `DeviceClassLister`**: Dynamic Resource Allocation (DRA) claim state and driver resolution.
- **`fwk.PodGroupLister` / `CompositePodGroupLister`**: Workload and gang scheduling hierarchy metadata.

---

## 4. Developer Invariants & Maintenance Rules

1. **Do Not Break Exported Method Signatures**:
   - Any modification to `RunPreFilterPlugins`, `RunFilterPlugins`, `RunReservePluginsReserve`, or `SharedLister` methods will cause compile failures in `autoscaler_contract_test.go`.
   - If a signature change is unavoidable (e.g. adding parameters or returning new metadata), a corresponding synchronization KEP and upgrade path must be coordinated with the `kubernetes/autoscaler` maintainers.
2. **Strict Compile-Time Verification**:
   - The test functions (`TestFrameworkContract`, `TestListerContract`) contain no runtime logic; they assign concrete types to interface variables (`var _ frameworkContract = ...`). If the package compiles, the contracts hold.

---

## 5. Verification Commands

```bash
# Run autoscaler contract tests
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/autoscaler_contract/...
```
