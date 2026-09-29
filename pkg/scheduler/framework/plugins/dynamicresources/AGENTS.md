<!-- Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Dynamic Resources (DRA) Plugin Architecture & Agent Guide

This guide provides AI agents and human contributors with an architectural overview, interface specifications, state machine contracts, lifecycle mechanics, caching and concurrency models, driver communication workflows, and testing strategies for the Dynamic Resource Allocation (DRA) scheduler plugin under `pkg/scheduler/framework/plugins/dynamicresources`.

---

## 1. High-Level Overview & Role

The Dynamic Resources (`DynamicResources`) plugin integrates **Dynamic Resource Allocation (DRA)** into `kube-scheduler`. DRA allows Kubernetes workloads to request specialized, heterogeneous hardware devices (e.g., GPUs, FPGAs, DPUs, fabric interconnects, NVMe disks) with complex allocation constraints, structured parameters, device sharing, and topology awareness.

Unlike classic device plugins (which represent devices as simple integer counters in `node.status.allocatable`), DRA operates on declarative Kubernetes API resources:
- **`ResourceClaim` / `ResourceClaimTemplate`**: Declarative requests for devices, parameters, and sharing models.
- **`ResourceSlice`**: Node-level or cluster-level inventory of available devices, device attributes, capacities, and network/bus topologies published by resource drivers.
- **`DeviceClass`**: Reusable device templates specifying selector criteria, configuration references, and default constraints.

```
                         ┌────────────────────────────────────────┐
                         │   Pod Spec (.spec.resourceClaims /     │
                         │    container extended resources)       │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ DynamicResources Plugin (kube-scheduler)     │
                      │                                              │
                      │  ┌────────────────────────────────────────┐  │
                      │  │ PreEnqueue / PreFilter:                │  │
                      │  │ - Resolve user & extended claims       │  │
                      │  │ - Gather cluster AllocatedState        │  │
                      │  │ - Initialize structured.Allocator      │  │
                      │  └───────────────────┬────────────────────┘  │
                      │                      │                       │
                      │  ┌───────────────────▼────────────────────┐  │
                      │  │ Filter / Score:                        │  │
                      │  │ - Check node affinity & selectors      │  │
                      │  │ - CEL structured parameter allocation  │  │
                      │  │ - Score prioritized subrequests        │  │
                      │  └───────────────────┬────────────────────┘  │
                      │                      │                       │
                      │  ┌───────────────────▼────────────────────┐  │
                      │  │ Reserve / Unreserve:                   │  │
                      │  │ - Signal in-flight allocation          │  │
                      │  │ - Refcount sharers for PodGroups       │  │
                      │  └───────────────────┬────────────────────┘  │
                      │                      │                       │
                      │  ┌───────────────────▼────────────────────┐  │
                      │  │ PreBind:                               │  │
                      │  │ - Write Status.Allocation & ReservedFor│  │
                      │  │ - Update AssumeCache                   │  │
                      │  │ - Poll driver BindingConditions        │  │
                      │  └────────────────────────────────────────┘  │
                      └──────────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼────────────────────┐
                         │ Driver Controller / Node Kubelet       │
                         │ (Prepares hardware & updates status)   │
                         └────────────────────────────────────────┘
```

### Core Responsibilities:
1. **Extension Point Implementation**: Intercepts `SignPod`, `PreEnqueue`, `PreFilter`, `Filter`, `PostFilter`, `PodGroupPostFilter`, `Score`, `Reserve`, `Unreserve`, `PreBindPreFlight`, `PreBind`, and `EnqueueExtensions`.
2. **Structured Parameter Allocation**: Evaluates CEL-based device selectors, match rules, consumable capacities, and device taints directly in-tree via `k8s.io/dynamic-resource-allocation/structured` without external webhook RPCs during node filtering.
3. **In-Flight Allocation & Claim Caching**: Coordinates pending allocations across synchronous scheduling and asynchronous binding cycles using `DefaultDRAManager`, `claimTracker`, and `AssumeCache`.
4. **Gang Scheduling & Workload Sharing**: Coordinates claim allocation and deallocation across `PodGroup` members (`GenericWorkload`).
5. **Asynchronous Driver Handshake (Binding Conditions)**: Blocks in `PreBind` to wait for external DRA driver controllers to complete device initialization on selected nodes before pod binding proceeds.
6. **Extended & Node-Allocatable Resources**: Synthesizes in-memory claims for standard extended resource requests mapped to device classes (`DRAExtendedResource`) and accounts for device-mapped CPU/memory capacity (`DRANodeAllocatableResources`).

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/dynamicresources/
├── dynamicresources.go                   # Main plugin implementation, extension points, and cycle runner
├── dra_manager.go                        # SharedDRAManager, claimTracker, in-flight allocation tracking
├── allocateddevices.go                   # Event-driven informer cache for allocated devices and consumed capacity
├── claims.go                             # claimStore abstraction (user claims vs scheduler-owned claims)
├── extendeddynamicresources.go           # DRAExtendedResource translation, synthesis, and lifecycle
├── nodeallocatabledynamicresources.go    # DRANodeAllocatableResources footprint validation and status patching
├── dynamicresources_test.go              # Comprehensive lifecycle, scoring, timeout, and failure unit tests
├── dra_manager_test.go                   # DRAManager, claimTracker, and informer event tests
├── extendeddynamicresources_test.go      # Extended resource claim creation and cleanup unit tests
├── nodeallocatabledynamicresources_test.go# Node-allocatable footprint and sharing rejection tests
├── prequeueing_race_test.go              # PreQueueingHint race and informer synchronization tests
└── AGENTS.md                             # This architectural guide
```

### Feature Gates Handled:
| Feature Gate | Impact on DynamicResources Plugin |
|---|---|
| `DRAExtendedResource` | Allows standard container `requests[resourceName]` to be satisfied via DRA claims mapped to `DeviceClass`. |
| `DRAConsumableCapacity` | Tracks fine-grained shared device capacity and aggregated consumed metrics. |
| `DRAPrioritizedList` | Enables `firstAvailable` subrequest alternative matching and prioritized scoring. |
| `DRANodeAllocatableResources` | Accounts for CPU/memory capacity mapped to DRA devices and prevents unauthorized cross-pod claim sharing. |
| `DRAPartitionableDevices` | Supports hierarchical and sub-device partitioning. |
| `DRADeviceTaints` | Enforces device-level taints and tolerations during node allocation. |
| `DRADeviceBindingConditions` | Enforces driver-reported status conditions before binding. |
| `DRAResourceClaimDeviceStatus` | Tracks per-device allocated status conditions in `ResourceClaim.Status.Devices`. |
| `DRAAdminAccess` | Enables non-exclusive administrative device sharing across claims. |
| `DRAFractionalCapacityRange` | Supports fractional numeric capacity evaluations in CEL. |
| `DRAListTypeAttributes` | Supports list-type device attributes in CEL matching. |
| `DRAOptionalNodeOperations` | Validates node declared features for claims requiring optional node operations (`SkipNodeOperations`). |
| `DRADerivedAttributes` | Evaluates derived attributes during structured parameter allocation. |
| `DRADeviceCompatibilityGroups` | Validates compatibility constraints across allocated device sets. |
| `DRAWorkloadResourceClaims` | Enables PodGroup-level claim ownership and reservation. |

---

## 3. Core Data Structures & State Management

### 3.1 `DynamicResources` Struct (`dynamicresources.go`)
```go
type DynamicResources struct {
    fts                   feature.Features
    filterTimeout         time.Duration
    bindingTimeout        time.Duration
    fh                    fwk.Handle
    clientset             kubernetes.Interface
    celCache              *cel.Cache
    draManager            fwk.SharedDRAManager
    podIndexer            cache.Indexer
    podResourceClaimIndex string
}
```
- `celCache`: An LRU cache (default size 10) for compiled CEL expressions evaluated during structured allocations.
- `draManager`: The shared DRA manager exposed via `fwk.Handle.SharedDRAManager()`.
- `podIndexer`: Informer indexer mapping `namespace/claimName` to referencing pods for targeted pre-queueing hints.

### 3.2 Ephemeral Cycle State (`stateData` & `podGroupStateData`)
Stored in `fwk.CycleState` under key `"DynamicResources"`:
```go
type stateData struct {
    claims                              claimStore
    draExtendedResource                 draExtendedResource
    allocator                           structured.Allocator
    mutex                               sync.Mutex
    unavailableClaims                   sets.Set[int]
    informationsForClaim                []informationForClaim
    nodeAllocations                     map[string]nodeAllocation
    claimHasNodeAllocatableMappedDevice map[types.UID]bool
}
```
- `claimStore`: Encapsulates user-defined claims alongside optional synthetic extended resource claims.
- `informationsForClaim`: Slices tracking point-in-time node selectors and in-progress allocation results for each claim.
- `nodeAllocations`: Caches candidate allocation results computed in `Filter` for each evaluated node, consumed later by `Reserve` and `PreBind`.
- `unavailableClaims`: Set of claim indices that could not be satisfied on at least one node; analyzed in `PostFilter` for deallocation.

For multi-pod gang scheduling (`GenericWorkload`), `podGroupStateData` is stored in `fwk.PodGroupCycleState`:
```go
type podGroupStateData struct {
    pendingAllocations map[types.UID]sets.Set[types.UID] // claimUID -> set of Pod UIDs
    podsStateData      map[types.NamespacedName]*stateData
}
```

### 3.3 `claimStore` Abstraction (`claims.go`)
Manages the slice of `*resourceapi.ResourceClaim` references:
- Distinguishes between user-owned claims (`numUserOwned`) and the scheduler-generated extended resource claim (stored at `claims[numUserOwned]`).
- Provides iterators:
  - `all()`: Iterates all claims (user + extended).
  - `allUserClaims()`: Iterates user-owned claims only.
  - `toAllocate()`: Filters claims where `claim.Status.Allocation == nil`.

---

## 4. DRA Scheduling Lifecycle Step-by-Step

```
[ Informer Event / Queue Ingress ]
               │
               ▼
[ PreEnqueue ] ──► Check all referenced ResourceClaims exist & are not being deleted
               │
               ▼
[ PreFilter ]  ──► 1. Resolve user claims & synthetic extended resource claims
               │   2. Validate DeviceClasses exist
               │   3. Check pending allocations from gang scheduling (podGroupStateData)
               │   4. GatherAllocatedState() (informer cache + in-flight claims)
               │   5. Instantiate structured.Allocator with CEL cache
               │
               ▼
[ Filter ]     ──► 1. Check allocated claim nodeSelector affinity
 (Parallel)    │   2. Check DRAOptionalNodeOperations node capability
               │   3. Check binding condition timeouts/failures
               │   4. structured.Allocator.Allocate() via CEL across ResourceSlices
               │   5. Check node-allocatable resource footprint (CPU/memory)
               │   6. Cache results in stateData.nodeAllocations[nodeName]
               │
               ▼ (If 0 nodes fit)
[ PostFilter ] ──► Select unavailable allocated claims & trigger API deallocation
               │   (PodGroupPostFilter unreserves group-scoped claims)
               │
               ▼ (If nodes fit)
[ Score ]      ──► Score nodes based on DRAPrioritizedList firstAvailable match ranks
               │
               ▼
[ Reserve ]    ──► 1. Select allocation results for chosen host
               │   2. draManager.SignalClaimPendingAllocation() (increment refcount)
               │   3. Record pendingAllocations in podGroupStateData
               │
               ▼
[ PreBindPreFlight ] ──► Verify claims exist (allows parallel PreBind execution)
               │
               ▼
[ PreBind ]    ──► 1. For extended resources: create real ResourceClaim in API
 (Async)       │   2. RetryOnConflict: Add finalizer, update Status.Allocation & ReservedFor
               │   3. AssumeClaimAfterAPICall() into AssumeCache
               │   4. Remove from in-flight allocations (MaybeRemoveClaimPendingAllocation)
               │   5. Patch Pod status (ExtendedResourceClaimStatus / NodeAllocatable)
               │   6. Poll isPodReadyForBinding() until driver conditions True or timeout
               │
               ▼
[ Unreserve ]  ──► Rollback: remove in-flight claims, AssumedClaimRestore(), API patch unreserve
```

### Detailed Phase Mechanics:

#### 1. `SignPod`
Pods referencing DRA resource claims are excluded from pod signature caching (`Unschedulable`) because resource claim binding dependencies and topology constraints cannot be safely generalized across arbitrary nodes.

#### 2. `PreEnqueue`
Verifies that all `pod.Spec.ResourceClaims` reference existing `ResourceClaim` objects, that no referenced claim has a non-nil `DeletionTimestamp`, and that claim ownership (`IsForPod`) is valid.

#### 3. `PreFilter`
- Resolves all user claims and checks whether extended resources request DRA mapping (`preFilterExtendedResources`).
- If claims are already allocated, verifies `CanBeReserved(claim)` and `IsReservedForPod(pod, claim)`.
- Reuses pending allocations from earlier pods in the same `PodGroup` scheduling cycle (`podGroupState.pendingAllocations`).
- Validates that every requested `DeviceClass` exists.
- Gathers the cluster-wide device allocation snapshot via `draManager.GatherAllocatedState()` (or `ListAllAllocatedDevices()`). Retries on `errClaimTrackerConcurrentModification`.
- Retrieves all `ResourceSlice` objects with device taint rules applied and initializes `structured.NewAllocator(ctx, features, allocatedState, deviceClasses, slices, celCache)`.

#### 4. `Filter`
Executed in parallel across worker goroutines:
- For already-allocated claims:
  - Verifies that `allocation.NodeSelector` matches the node.
  - Verifies that the node declared features include `DRAOptionalNodeOperations` if required by any allocated device.
  - Verifies that device binding conditions have not failed or timed out.
  - Validates node-allocatable direct-mapped claim sharing rules.
- For claims needing allocation:
  - Invokes `state.allocator.Allocate(allocCtx, node, claimsToAllocate)`.
  - Maps device requests to candidate devices in available `ResourceSlice` instances matching class filters and CEL constraints.
  - Validates node-allocatable footprints against node capacity (`calculateAndCheckNodeAllocatableResources`).
- Thread-safely records successful results in `state.nodeAllocations[node.Name]` and records any unavailable claim indices in `state.unavailableClaims`.

#### 5. `PostFilter` & `PodGroupPostFilter`
When no node is feasible:
- `PostFilter`: Randomly selects one unavailable allocated claim reserved exclusively for this pod (or not reserved) and resets `claim.Status.Allocation = nil`, `claim.Status.ReservedFor = nil`, and `claim.Status.Devices = nil` via the API server to trigger reallocation.
- `PodGroupPostFilter`: Iterates all unscheduled pods in the `PodGroup`, deallocates individual pod claims, and unreserves `PodGroup`-level claims via strategic merge patches if no pods in the group are assumed/assigned.

#### 6. `Score` & `NormalizeScore`
When `DRAPrioritizedList` is enabled:
- Computes node scores using `computeScore()`.
- Iterates over allocated subrequests in `claim.Spec.Devices.Requests[].FirstAvailable`:
  - Higher-priority subrequests (lower index in `FirstAvailable`) receive higher score increments: `score += int64(FirstAvailableDeviceRequestMaxSize - i)`.
- Scores are normalized using `helper.DefaultNormalizeScore`.

#### 7. `Reserve`
- Retrieves the cached `nodeAllocation` for the chosen node.
- Populates `state.informationsForClaim[index].allocation`.
- Calls `draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim)`.
- If part of a `PodGroup`, increments the sharer refcount in `podGroupState.pendingAllocations`.

#### 8. `Unreserve`
Rollback handler executed on scheduling or binding failures:
- Calls `draManager.ResourceClaims().MaybeRemoveClaimPendingAllocation(claim.UID, false)`; if deleted, calls `AssumedClaimRestore(namespace, claimName)` to restore the cached claim state.
- Removes the pod from `claim.Status.ReservedFor` via strategic merge patch if already recorded on the API server.
- Cleans up synthetic extended resource claims.

#### 9. `PreBindPreFlight` & `PreBind`
- `PreBindPreFlight`: Returns `AllowParallel: true` if claims exist, permitting concurrent pre-bind preparation across plugins.
- `PreBind`:
  - Iterates claims and invokes `bindClaim()` with retry on conflict (`retry.RetryOnConflict`):
    1. Instantiates synthetic extended resource claims on the API server (`createExtendedResourceClaimInAPI`).
    2. Adds finalizer `resource.kubernetes.io/delete-protection`.
    3. Writes `claim.Status.Allocation` and appends consumer reference to `claim.Status.ReservedFor`.
    4. Calls `draManager.AssumeClaimAfterAPICall(claim)` to populate `AssumeCache`.
    5. Cleans up in-flight tracking on success.
    6. Patches `pod.Status.ExtendedResourceClaimStatus` and `pod.Status.NodeAllocatableResourceClaimStatuses`.
  - **Binding Conditions Check**: If `DRADeviceBindingConditions` is enabled and devices declare `BindingConditions`:
    - Emits event `BindingConditionsPending`.
    - Polls `isPodReadyForBinding(state)` every 5s up to `bindingTimeout` (default 120s).
    - Checks `claim.Status.Devices[].Conditions`:
      - If any condition in `BindingFailureConditions` is `True` -> returns `ErrDeviceBindingFailed`.
      - If all conditions in `BindingConditions` are `True` -> proceeds.
      - If timeout reached -> returns `ErrDeviceBindingTimeout`.
    - Records metrics: `DRABindingConditionsPreBindDuration` and `DRABindingConditionsAllocationsTotal`.

---

## 5. Caching, SharedDRAManager & Concurrency Model

```
                    ┌──────────────────────────────────────────────┐
                    │               DefaultDRAManager              │
                    └──────┬────────────────┬───────────────┬──────┘
                           │                │               │
            ┌──────────────▼──────┐  ┌──────▼──────┐  ┌─────▼─────────────┐
            │ claimTracker        │  │ Resource-   │  │ DeviceClass-      │
            │                     │  │ SliceLister │  │ Lister & Resolver │
            │ ┌─────────────────┐ │  └─────────────┘  └───────────────────┘
            │ │ AssumeCache     │ │
            │ └─────────────────┘ │
            │ ┌─────────────────┐ │
            │ │ inFlight-       │ │
            │ │ Allocations     │ │
            │ └─────────────────┘ │
            │ ┌─────────────────┐ │
            │ │ allocated-      │ │
            │ │ Devices (Events)│ │
            │ └─────────────────┘ │
            └─────────────────────┘
```

### 5.1 `DefaultDRAManager` (`dra_manager.go`)
Implements `fwk.SharedDRAManager` and coordinates DRA state:
- `ResourceClaims()`: Returns `claimTracker`.
- `ResourceSlices()`: Returns `resourceSliceLister` wrapping `resourceslicetracker.Tracker`.
- `DeviceClasses()`: Returns `deviceClassLister`.
- `DeviceClassResolver()`: Returns `extendedresourcecache.ExtendedResourceCache`.

### 5.2 `claimTracker` & Optimistic Concurrency Control
Coordinates claim state across informers, assume cache, and in-flight allocations:
- **`AssumeCache`**: Temporarily stores updated claim objects immediately after API calls in `PreBind`, before informer watch events reflect them.
- **`inFlightAllocations`**: Map of `claimUID -> inFlightAllocation{claim, sharers}`. Tracks claims reserved during scheduling cycles where PreBind API calls have not yet completed. Prevents concurrent pods outside the same `PodGroup` from re-allocating or colliding on the same claim.
- **`allocatedDevices` (`allocateddevices.go`)**: Maintains event-driven sets of dedicated device IDs, shared device IDs, and consumed capacity collections. Every mutation bumps an internal `revision` counter.
- **Optimistic Concurrency & Retry**:
  ```go
  // GatherAllocatedState reads informer state and in-flight claims.
  // If a concurrent mutation bumps the revision during collection,
  // it returns errClaimTrackerConcurrentModification, causing PreFilter
  // to retry via wait.PollUntilContextTimeout.
  ```

### 5.3 Lock Discipline & Invariants
1. `allocatedDevices.mutex` (RWMutex): Protects device sets, shared IDs, capacities, and revision counter. Precomputes additions/deletions outside the write lock to minimize lock contention.
2. `claimTracker.inFlightMutex` (RWMutex): Protects `inFlightAllocations` map and sharer reference counts.
3. `stateData.mutex` (Mutex): Protects `unavailableClaims` and `nodeAllocations` maps during parallel node filtering.
4. Pure cycle isolation: `CycleState` data is read-only during `Filter` and `Score` parallel worker execution, except for fields protected by `stateData.mutex`.

---

## 6. Driver Communication & Structured Parameters

### Structured Parameters vs External Controllers
DRA supports two operational modes:
1. **In-Tree Structured Parameters (Default & Modern)**:
   - Resource drivers publish device inventory and capabilities as `ResourceSlice` objects.
   - The scheduler evaluates selectors, device attributes, capacities, and topology constraints in-tree using `structured.Allocator` and CEL.
   - Eliminates per-pod RPC overhead and external scheduler plugins.
2. **Device Binding Conditions (Driver Handshake)**:
   - For devices requiring physical setup (e.g., programming an FPGA, attaching an SAN LUN, configuring an SR-IOV VF), the driver specifies `BindingConditions` in the allocation result.
   - The scheduler reserves the claim and marks `AllocationTimestamp`.
   - The driver's node/control-plane controller watches `ResourceClaim` objects, performs the hardware preparation, and sets `Status.Devices[].Conditions`.
   - The scheduler's `PreBind` polls these conditions before completing pod binding.

### Binding Conditions Status & Metrics
- Status conditions evaluated:
  - `BindingConditions`: Must all evaluate to `True`.
  - `BindingFailureConditions`: If any evaluates to `True`, binding fails immediately (`ErrDeviceBindingFailed`).
- Metrics recorded:
  - `schedmetrics.DRABindingConditionsPreBindDuration`: Histogram observing duration spent waiting in PreBind per driver and outcome label (`success`, `timeout`, `failed`, `error`).
  - `schedmetrics.DRABindingConditionsAllocationsTotal`: Counter tracking allocations using binding conditions.

---

## 7. Event-Driven Requeueing & PreQueueing

### 7.1 Registered Events (`EventsToRegister`)
```go
events := []fwk.ClusterEventWithHint{
    {Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add | fwk.UpdateNodeLabel | fwk.UpdateNodeAllocatable}},
    {Event: fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.Add | fwk.Update | fwk.Delete},
     QueueingHintFn: pl.isSchedulableAfterClaimChange, PreQueueingHintFn: pl.preQueueingHint},
    {Event: fwk.ClusterEvent{Resource: fwk.TargetPod, ActionType: fwk.UpdatePodGeneratedResourceClaim},
     QueueingHintFn: pl.isSchedulableAfterTargetPodUpdate},
    {Event: fwk.ClusterEvent{Resource: fwk.DeviceClass, ActionType: fwk.Add | fwk.Update}},
    {Event: fwk.ClusterEvent{Resource: fwk.ResourceSlice, ActionType: fwk.Add | fwk.Update}},
}
```

### 7.2 PreQueueingHint & Pod Indexing (`preQueueingHint`)
- Uses `podIndexer` with index `podResourceClaimIndexPrefix + "-" + profileName` indexing pods by referenced `namespace/claimName`.
- When a `ResourceClaim` is deallocated or deleted, it returns `AllPods: true` (since freed devices could unblock any pending pod).
- When a specific claim is modified or created, it returns only the specific pods referencing that claim (`PreQueueingHintResult{Pods: pods}`).

### 7.3 QueueingHint Functions
- `isSchedulableAfterClaimChange`: Evaluates whether a claim deallocation, creation, or status change unblocks the specific pod. Ignores cosmetic metadata updates (e.g. driver finalizers) via `apiequality.Semantic.DeepEqual`.
- `isSchedulableAfterTargetPodUpdate`: Triggers requeueing when a target pod's status is updated with generated resource claim names.

---

## 8. Extended & Node-Allocatable Resources

### 8.1 Extended Resources Backed by DRA (`extendeddynamicresources.go`)
- Allows pods requesting standard container resources (e.g. `limits: { "example.com/gpu": 2 }`) to utilize DRA drivers when a `DeviceClass` maps to that resource name (`spec.extendedResourceName`).
- Lifecycle:
  1. `PreFilter`: Synthesizes an in-memory placeholder claim named `"<extended-resources>"` with a temporary UID.
  2. `Filter`: Creates node-specific requests matching device plugin availability or DRA slice availability.
  3. `Reserve`: Tracks the temporary claim in `inFlightAllocations`.
  4. `PreBind`: Creates the real `ResourceClaim` on the API server with owner references to the Pod and patches `pod.Status.ExtendedResourceClaimStatus`.
  5. `PostFilter` / `Unreserve`: Cleans up temporary or stale API claims.

### 8.2 Node-Allocatable Resources (`nodeallocatabledynamicresources.go`)
- Enables devices to model consumable node-allocatable capacity (e.g., custom memory tiers, offloaded CPU cores) via `Device.NodeAllocatableResources`.
- Enforces strict isolation: directly mapped claims cannot be shared across multiple pods.
- Computes aggregate pod resource demand in `Filter` and validates that `nodeFitsResources` passes for the node's allocatable limits.
- Patches `pod.Status.NodeAllocatableResourceClaimStatuses` during `PreBind`.

---

## 9. Testing Strategy & Verification Guide

### 9.1 Test File Breakdown
| Test File | Primary Coverage Focus |
|---|---|
| `dynamicresources_test.go` | Full lifecycle tests (`TestPlugin`), table-driven tests for PreEnqueue, PreFilter, Filter, Reserve, Unreserve, PreBind, PostFilter, Score, PrioritizedList, CEL errors, BindingConditions timeouts, and claim sharing. |
| `dra_manager_test.go` | Unit tests for `claimTracker`, `allocatedDevices`, assume cache synchronization, sharer refcounting, in-flight allocation tracking, and `errClaimTrackerConcurrentModification` handling. |
| `extendeddynamicresources_test.go` | Unit tests for extended resource translation, synthetic claim creation, device plugin fallback, and cleanup. |
| `nodeallocatabledynamicresources_test.go` | Unit tests for node-allocatable resource calculation, direct-mapped claim sharing rejection, and footprint validation. |
| `prequeueing_race_test.go` | Race condition tests for `PreQueueingHint` under concurrent claim informer updates. |

### 9.2 Running Tests
Execute unit tests from repository root:
```bash
# Run all dynamicresources plugin tests
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/plugins/dynamicresources/...

# Run specific lifecycle test suite
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/plugins/dynamicresources -run TestPlugin

# Run DRAManager concurrency and cache tests
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/plugins/dynamicresources -run TestDRAManager

# Run race detection on prequeueing hints
GOTOOLCHAIN=auto go test -race -v ./pkg/scheduler/framework/plugins/dynamicresources -run TestPreQueueingRace
```
