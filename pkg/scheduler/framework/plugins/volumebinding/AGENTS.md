# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/volumebinding`

This guide provides an architectural overview, execution lifecycle, binding state machines, caching invariants, storage capacity scoring mechanics, event handling rules, and testing strategies for the Kubernetes scheduler `VolumeBinding` plugin and its metrics subsystem in `pkg/scheduler/framework/plugins/volumebinding`.

---

## 1. High-Level Purpose & Scope

The `VolumeBinding` plugin (`names.VolumeBinding = "VolumeBinding"`) coordinates PersistentVolumeClaim (PVC) and PersistentVolume (PV) binding decisions with node scheduling feasibility. It ensures that pods requesting persistent or generic ephemeral storage are scheduled onto nodes that satisfy both volume node affinity constraints and storage backend capacity limits.

### Core Objectives:
1. **Topology-Aware Volume Binding**: Delays volume binding for claims using `VolumeBindingWaitForFirstConsumer` until the pod's feasible host nodes are determined, ensuring the selected PV matches the node's topology or the dynamically provisioned volume is created in the correct zone/node.
2. **Static & Dynamic Volume Provisioning**: Matches unbound PVCs with existing pre-provisioned PVs (`StaticBindings`) or selects a node for the external storage provisioner to dynamically create new volumes (`DynamicProvisions`).
3. **Speculative Reservation via Assume Caches**: Uses dedicated `PVAssumeCache` and `PVCAssumeCache` to speculatively reserve PVs and claim annotations in memory, preventing race conditions and double-allocations across back-to-back scheduling cycles before API updates complete.
4. **CSI Storage Capacity Tracking & Scoring**: Evaluates real-time available storage reported via `CSIStorageCapacity` resources during filtering and scores nodes based on volume capacity utilization.
5. **Asynchronous Pre-Binding**: Coordinates API updates and waits for the PV controller to complete volume binding during the asynchronous `PreBind` phase before final pod-to-node binding.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/volumebinding/
├── volume_binding.go          # Plugin struct, PreFilter, Filter, PreScore, Score, Reserve, PreBind, EnqueueExtensions
├── binder.go                  # SchedulerVolumeBinder interface, volumeBinder implementation, FindPodVolumes, AssumePodVolumes, BindPodVolumes
├── assume_cache.go            # PVAssumeCache (with storageclass indexer) and PVCAssumeCache wrappers
├── scorer.go                  # Storage capacity scoring logic (broken linear interpolation & classResourceMap)
├── fake_binder.go             # FakeSchedulerVolumeBinder test implementation
├── test_utils.go              # Test helpers, mock listers, volume generators, and claim factories
├── volume_binding_test.go     # Framework extension point unit tests
├── binder_test.go             # Deep binder algorithm tests (static matching, dynamic provisioning, timeouts)
├── assume_cache_test.go       # Assume cache indexer, assumption, and reversion tests
├── scorer_test.go             # Capacity scorer unit tests
├── metrics/
│   └── metrics.go             # Prometheus metric definitions (binder_cache_requests_total, scheduling_stage_error_total)
└── AGENTS.md                  # This agent documentation
```

---

## 3. Core Data Structures & Interfaces

### 3.1. Key Structures

```
[ stateData (in CycleState) ]
 ├── allBound: bool
 ├── hasStaticBindings: bool
 ├── podVolumeClaims: *PodVolumeClaims
 │    ├── boundClaims: []*v1.PersistentVolumeClaim
 │    ├── unboundClaimsDelayBinding: []*v1.PersistentVolumeClaim
 │    ├── unboundClaimsImmediate: []*v1.PersistentVolumeClaim
 │    └── unboundVolumesDelayBinding: map[string][]*v1.PersistentVolume
 ├── podVolumesByNode: map[string]*PodVolumes (Key: node.Name)
 │    └── PodVolumes:
 │         ├── StaticBindings: []*BindingInfo { pvc, pv }
 │         └── DynamicProvisions: []*DynamicProvision { PVC, NodeCapacity }
 └── Mutex: sync.Mutex (guards podVolumesByNode writes during concurrent Filter)
```

- **`stateData`**: Stored in `CycleState` under key `VolumeBinding`. Created in `PreFilter`, read and populated under a local mutex in `Filter`, and consumed in `Reserve`, `PreScore`, `Score`, `PreBindPreFlight`, `PreBind`, and `Unreserve`.
- **`BindingInfo`**: Represents a static match between an unbound PVC and an existing pre-provisioned PV.
- **`DynamicProvision`**: Represents an unbound PVC requiring dynamic provisioning on a specific node, optionally paired with a matching `CSIStorageCapacity`.
- **`ConflictReasons`**: Explanatory strings returned when a node cannot satisfy volume requirements (`ErrReasonNodeConflict`, `ErrReasonBindConflict`, `ErrReasonNotEnoughSpace`, `ErrReasonPVNotExist`).

### 3.2. `SchedulerVolumeBinder` Interface

```go
type SchedulerVolumeBinder interface {
    GetPodVolumeClaims(logger klog.Logger, pod *v1.Pod) (podVolumeClaims *PodVolumeClaims, err error)
    FindPodVolumes(logger klog.Logger, pod *v1.Pod, podVolumeClaims *PodVolumeClaims, node *v1.Node) (podVolumes *PodVolumes, reasons ConflictReasons, err error)
    AssumePodVolumes(logger klog.Logger, assumedPod *v1.Pod, nodeName string, podVolumes *PodVolumes) (allFullyBound bool, err error)
    RevertAssumedPodVolumes(podVolumes *PodVolumes)
    BindPodVolumes(ctx context.Context, assumedPod *v1.Pod, podVolumes *PodVolumes) error
}
```

---

## 4. End-to-End Scheduling & Binding Lifecycle

```
[ Scheduling Cycle (Synchronous) ]
  │
  ├── 1. PreFilter:
  │      • Validate PVC existence, ClaimLost status, deletion timestamp, ephemeral ownership
  │      • Call GetPodVolumeClaims() to classify claims
  │      • If unboundClaimsImmediate > 0: Reject with UnschedulableAndUnresolvable
  │      • Store initial stateData in CycleState
  │
  ├── 2. Filter (Parallel across nodes):
  │      • Call FindPodVolumes(node):
  │        - checkBoundClaims: verify PV.Spec.NodeAffinity against node labels
  │        - Fast path: verify volume.kubernetes.io/selected-node matches node
  │        - findMatchingVolumes: search unbound pre-provisioned PVs from PVAssumeCache
  │        - checkVolumeProvisions: check StorageClass provisioner & CSIStorageCapacity
  │      • If conflicts found: return UnschedulableAndUnresolvable with ConflictReasons
  │      • Thread-safely store PodVolumes in stateData.podVolumesByNode[node.Name]
  │
  ├── 3. PreScore & Score:
  │      • PreScore: skip scoring if scorer is nil or no static bindings / storage capacity scoring disabled
  │      • Score: calculate storage capacity utilization score per StorageClass
  │
  └── 4. Reserve:
         • Retrieve podVolumes for selected node from CycleState
         • Call AssumePodVolumes():
           - StaticBindings: assume PV pre-bound to PVC in PVAssumeCache
           - DynamicProvisions: assume PVC with AnnSelectedNode in PVCAssumeCache
         • Set state.allBound = allBound
         • On Failure / Preemption: Unreserve invokes RevertAssumedPodVolumes()

[ Binding Cycle (Asynchronous Goroutine) ]
  │
  ├── 5. PreBindPreFlight:
  │      • If state.allBound == true: return Skip
  │      • Return AllowParallel: true (permits parallel PreBind with other plugins)
  │
  └── 6. PreBind:
         • If state.allBound == true: return Success
         • Call BindPodVolumes():
           - bindAPIUpdate: PV update for static bindings, PVC update for dynamic annotations
           - Poll checkBindings() every 1s up to bindTimeout until PV controller finishes binding
         • On Failure / Timeout: return Error, triggers unreserve and requeue
```

---

## 5. Detailed Extension Point Implementations

### 5.1. `PreFilter`
- **In-Place Resize**: If `EnableInPlacePodVerticalScalingSchedulerPreemption` is enabled and `resource.IsPodResizeDeferred(pod)` is true, returns `Skip`.
- **Claim Discovery & Validation (`podHasPVCs`)**:
  - Traverses `pod.Spec.Volumes`. Handles both `PersistentVolumeClaim` and `Ephemeral` volume sources (computing claim name via `ephemeral.VolumeClaimName(pod, &vol)`).
  - Verifies claim exists in informer cache. If ephemeral and `NotFound`, returns error waiting for ephemeral volume controller.
  - Rejects if `pvc.Status.Phase == ClaimLost` or `pvc.DeletionTimestamp != nil`.
  - Verifies generic ephemeral volume ownership (`ephemeral.VolumeIsForPod(pod, pvc)`).
  - If no PVCs referenced, writes empty `stateData` and returns `Skip`.
- **Claim Classification (`GetPodVolumeClaims`)**:
  - Groups claims into:
    - `boundClaims`: `pvc.Spec.VolumeName != ""` and claim phase is `Bound`.
    - `unboundClaimsDelayBinding`: PVC uses a `StorageClass` with `VolumeBindingWaitForFirstConsumer`.
    - `unboundClaimsImmediate`: PVC uses immediate binding mode (or no storage class) but is not yet bound.
  - Pre-fetches all PVs for relevant storage classes into `unboundVolumesDelayBinding` using `PVAssumeCache.ListPVs(storageClassName)`.
  - **Immediate Claim Policy**: If `len(unboundClaimsImmediate) > 0`, the pod cannot be scheduled yet; returns `UnschedulableAndUnresolvable` with reason `"pod has unbound immediate PersistentVolumeClaims"`.

### 5.2. `Filter`
- Executes `FindPodVolumes` for each candidate node:
  1. **Bound Claims Verification (`checkBoundClaims`)**:
     - Fetches PV via `pvCache.Get(pvc.Spec.VolumeName)`.
     - Translates in-tree PV to CSI if migratable.
     - Evaluates node affinity constraints via `storagehelpers.CheckNodeAffinity(pv, node.Labels)`.
     - Sets `boundVolumesSatisfied = false` (`ErrReasonNodeConflict`) or `boundPVsFound = false` (`ErrReasonPVNotExist`) on failure.
  2. **Selected Node Fast Path**:
     - If an unbound claim already has `volume.kubernetes.io/selected-node` set (from a previous scheduling attempt) and it does not match `node.Name`, sets `unboundVolumesSatisfied = false` (`ErrReasonBindConflict`).
  3. **Static PV Matching (`findMatchingVolumes`)**:
     - Iterates through cached PVs in `unboundVolumesDelayBinding`.
     - Filters PVs using `storagehelpers.FindMatchingVolume(pvc, pvs, node, ...)`.
     - Ensures each matching PV satisfies size requests, access modes, storage class, selectors, and node affinity.
     - Tracks allocated PVs in-memory during the cycle to prevent matching the same static PV to multiple PVCs of the same pod.
  4. **Dynamic Provisioning Feasibility (`checkVolumeProvisions`)**:
     - For claims without matching static PVs, verifies the `StorageClass` allows dynamic provisioning on the node.
     - If `CSIStorageCapacity` tracking is enabled, matches the node against driver topology keys and ensures available capacity >= requested capacity.
     - If capacity is insufficient, sets `sufficientStorage = false` (`ErrReasonNotEnoughSpace`).
- **CycleState Recording**:
  - Locks `stateData.Lock()` and stores `podVolumes` in `stateData.podVolumesByNode[node.Name]`.
  - Updates `state.hasStaticBindings`.

### 5.3. `PreScore` & `Score`
- Evaluates capacity utilization when storage capacity scoring is configured:
  - **Static Bindings**: Aggregates requested vs capacity from `BindingInfo.StorageResource()`.
  - **Dynamic Provisions**: Aggregates requested PVC storage vs `provision.NodeCapacity.Capacity`.
  - **Broken Linear Function**: Uses configured `helper.FunctionShape` to map utilization percentage `(requested * 100 / capacity)` to a node score in `[0, MaxNodeScore]`.

### 5.4. `Reserve` & `Unreserve`
- **`Reserve`**:
  - Obtains `podVolumes` for the winning node.
  - Calls `AssumePodVolumes`:
    - **Static PVs**: Clones PV and updates `pv.Spec.ClaimRef` via `volume.GetBindVolumeToClaim()`. Calls `pvCache.Assume(newPV)`.
    - **Dynamic PVCs**: Clones PVC and injects annotation `volume.kubernetes.io/selected-node: <nodeName>`. Calls `pvcCache.Assume(claimClone)`.
  - Marks `state.allBound = true` if all claims are already fully bound, avoiding PreBind execution.
- **`Unreserve`**:
  - Idempotently invokes `Binder.RevertAssumedPodVolumes()`, removing assumed PVs and PVCs from `pvCache` and `pvcCache` using `assumecache.Restore()`.

### 5.5. `PreBindPreFlight` & `PreBind`
- **`PreBindPreFlight`**:
  - Returns `Skip` if `state.allBound == true`.
  - Returns `PreBindPreFlightResult{AllowParallel: true}` enabling parallel execution with other PreBind plugins.
- **`PreBind`**:
  - Executes `BindPodVolumes(ctx, pod, podVolumes)`:
    1. **`bindAPIUpdate`**:
       - Updates PV objects for static bindings via `kubeClient.CoreV1().PersistentVolumes().Update()`.
       - Updates PVC objects for dynamic provisions via `kubeClient.CoreV1().PersistentVolumeClaims().Update()`.
    2. **`wait.PollUntilContextTimeout`**:
       - Periodically checks `checkBindings()` every 1s up to `bindTimeout` (default: `VolumeBindingArgs.BindTimeoutSeconds`, typically 600s).
       - Verifies all PVCs transition to `pvc.Status.Phase == ClaimBound` and `pv.Spec.ClaimRef` is populated by the external PV controller.
  - Returns `fwk.Status` on completion, timeout, or context cancellation.

---

## 6. Assume Cache Mechanics (`assume_cache.go`)

The `VolumeBinding` plugin uses specialized assume caches built on `k8s.io/component-helpers/storage/volume/assumecache`:

```
                 API Server Informer
                         │ (Add / Update / Delete)
                         ▼
             ┌───────────────────────┐
             │      AssumeCache      │
             │  ├── store (informer) │
             │  └── assumeStore      │
             └───────────┬───────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
PVAssumeCache                     PVCAssumeCache
Index: "storageclass"             Key: namespace/name
Methods: ListPVs(scName)          Methods: Get, Assume, Restore
```

1. **`PVAssumeCache`**:
   - Indexed by `storageclass` indexer function (`pvStorageClassIndexFunc`).
   - `ListPVs(storageClassName)` returns both confirmed PVs from the informer store and speculatively assumed PVs.
   - Prevents consecutive pods requiring the same storage class from claiming the same pre-provisioned PV.
2. **`PVCAssumeCache`**:
   - Holds speculatively updated PVC copies containing `volume.kubernetes.io/selected-node`.
   - Prevents re-evaluation of node placement while external dynamic provisioners process the request.
3. **Cache Reconciliation**:
   - When an informer event arrives with a resource version equal to or newer than the assumed object, the assume store automatically evicts the speculative entry.
   - On scheduling failure, `Restore()` explicitly clears the speculative entry.

---

## 7. Event Handling, QueueingHints & EnqueueExtensions

`VolumeBinding` registers cluster events with targeted `QueueingHint` functions to unblock unschedulable pods:

| Resource | Action | QueueingHint Function | Trigger Condition |
|---|---|---|---|
| **`StorageClass`** | `Add \| Update` | `isSchedulableAfterStorageClassChange` | • New StorageClass created.<br>• `AllowedTopologies` updated. |
| **`PersistentVolumeClaim`** | `Add \| Update` | `isSchedulableAfterPersistentVolumeClaimChange` | • PVC created or updated in matching namespace.<br>• PVC name matches pod's PVC or generic ephemeral claim. |
| **`PersistentVolume`** | `Add \| Update` | `nil` (always Queue) | • New PV created or existing PV capacity/affinity modified. |
| **`Node`** | `Add \| UpdateNodeLabel` | `nil` (always Queue) | • Node added or topology/affinity labels changed. |
| **`CSINode`** | `Add \| Update` | `isSchedulableAfterCSINodeChange` | • CSINode created.<br>• `v1.MigratedPluginsAnnotationKey` annotation updated. |
| **`CSIDriver`** | `Update` | `isSchedulableAfterCSIDriverChange` | • CSIDriver updated and `Spec.StorageCapacity` was disabled. |
| **`CSIStorageCapacity`** | `Add \| Update` | `isSchedulableAfterCSIStorageCapacityChange` | • New capacity reported.<br>• `volumeLimit` (`Capacity` / `MaximumVolumeSize`) increased. |

---

## 8. Metrics Subsystem (`metrics/metrics.go`)

Metric subsystem `scheduler_volume`:

```go
const VolumeSchedulerSubsystem = "scheduler_volume"
```

1. **`VolumeBindingRequestSchedulerBinderCache`** (`scheduler_volume_binder_cache_requests_total`):
   - Counter tracking cache operations on `PVAssumeCache` and `PVCAssumeCache`.
   - Labels: `operation` (`"get"`, `"list"`, `"assume"`, `"restore"`).
2. **`VolumeSchedulingStageFailed`** (`scheduler_volume_scheduling_stage_error_total`):
   - Counter tracking failures across volume scheduling stages.
   - Labels: `operation`:
     - `"predicate"`: Failed during `FindPodVolumes` in Filter.
     - `"assume"`: Failed during `AssumePodVolumes` in Reserve.
     - `"bind"`: Failed during `BindPodVolumes` / API updates / timeouts in PreBind.

---

## 9. Testing Guide & Verification

### 9.1. Key Test Files:
- **`volume_binding_test.go`**:
  - Tests extension point lifecycles (`PreFilter`, `Filter`, `Reserve`, `Unreserve`, `PreScore`, `Score`, `PreBindPreFlight`, `PreBind`).
  - Tests immediate vs delayed binding rejections, generic ephemeral volumes, and queueing hints.
- **`binder_test.go`**:
  - Tests `FindPodVolumes` algorithm across bound claims, unbound static claims, dynamic provisioning, multi-PVC pods, PVC selectors, and storage capacity constraints.
  - Tests `AssumePodVolumes` and `BindPodVolumes` timeouts and error recovery.
- **`assume_cache_test.go`**:
  - Tests `PVAssumeCache` storage class indexing, concurrent assumptions, and informer reconciliation.
- **`scorer_test.go`**:
  - Tests linear function scoring shapes for storage capacity utilization.

### 9.2. Running Tests:
```bash
# Run unit tests for volumebinding plugin
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/plugins/volumebinding/...

# Run with race detector
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/volumebinding/...
```

---

## 10. Critical Invariants for Developers & AI Agents

1. **Immediate Claims Cannot Be Scheduled Unbound**:
   - If a pod requests a PVC that does not use `VolumeBindingWaitForFirstConsumer`, that PVC MUST be bound before scheduling. `PreFilter` must reject with `UnschedulableAndUnresolvable`.
2. **CycleState Concurrency in Filter**:
   - `Filter` runs concurrently across nodes. Modifying `stateData.podVolumesByNode` or `stateData.hasStaticBindings` MUST be protected by `stateData.Lock()`.
3. **Assume / Unreserve Symmetry**:
   - Any PV or PVC assumed during `Reserve` via `pvCache.Assume()` / `pvcCache.Assume()` MUST be cleanly reverted in `Unreserve` via `RevertAssumedPodVolumes()` if scheduling or downstream pre-binding fails.
4. **PreBind Timeout Protection**:
   - `BindPodVolumes` must never block indefinitely. Always enforce `bindTimeout` (`wait.PollUntilContextTimeout`) and honour context cancellation.
5. **Generic Ephemeral Volume Validation**:
   - Ephemeral PVCs generated for inline pod specs must verify pod ownership (`ephemeral.VolumeIsForPod`) before processing.
6. **Dynamic Capacity Scorer Non-Additivity**:
   - When scoring dynamic provisions against node capacity, multiple claims against the same storage class must read `provision.NodeCapacity.Capacity.Value()` as the node's total capacity rather than accumulating capacities with `+=`.
