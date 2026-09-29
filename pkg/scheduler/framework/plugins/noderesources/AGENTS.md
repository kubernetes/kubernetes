# Node Resources Plugins (`pkg/scheduler/framework/plugins/noderesources`)

This guide provides an architectural overview, interface implementations, resource accounting algorithms, scoring strategies, Dynamic Resource Allocation (DRA) integration, queueing hint mechanics, and testing strategies for the `noderesources` plugin suite under `pkg/scheduler/framework/plugins/noderesources`.

---

## 1. High-Level Purpose & Scope

The `noderesources` package provides resource-based feasibility filtering and multi-strategy node prioritization for Kubernetes pods. It encompasses:

1. **`NodeResourcesFit` (`Fit`)**: The primary resource filter and default scoring plugin. Verifies that nodes possess sufficient CPU, memory, ephemeral storage, hugepages, pod count capacity, scalar/extended resources, and DRA extended resources to accommodate a pod's aggregate resource request.
2. **`NodeResourcesBalancedAllocation` (`BalancedAllocation`)**: A scoring plugin that optimizes resource balance across multiple dimensions (CPU, memory, extended resources), preventing resource fragmentation and stranded capacity.
3. **Resource Scoring Strategies**:
   - **`LeastAllocated`**: Favors nodes with lower resource utilization (load spreading).
   - **`MostAllocated`**: Favors nodes with higher resource utilization (bin packing / cluster consolidation).
   - **`RequestedToCapacityRatio`**: Applies custom broken-linear scoring curves based on resource utilization ratios.
4. **Placement Scoring (`ScorePlacement`)**: Evaluates capacity ratios across all nodes within a topology placement for workload/gang scheduling.
5. **Dynamic Resource Allocation (DRA) Extended Resources**: Bridges classical device plugin extended resources with DRA device classes and resource slices.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/noderesources/
├── fit.go                             # NodeResourcesFit plugin (PreFilter, Filter, PreScore, Score, PlacementScore, Queueing Hints)
├── fit_test.go                        # Comprehensive unit tests for Fit (filtering, preemption, in-place resize, queueing hints)
├── resource_allocation.go             # Base resourceAllocationScorer, DRA device/node matching, CEL caching, score pipelines
├── resource_allocation_test.go        # Unit tests for shared scorer logic, DRA calculations, and cache efficiency
├── balanced_allocation.go             # NodeResourcesBalancedAllocation plugin (variance/standard deviation balance scoring)
├── balanced_allocation_test.go        # Unit tests for BalancedAllocation and best-effort pod skipping
├── least_allocated.go                 # LeastAllocated scoring strategy implementation
├── least_allocated_test.go            # Unit tests for LeastAllocated scoring formula and weights
├── most_allocated.go                  # MostAllocated scoring strategy implementation
├── most_allocated_test.go             # Unit tests for MostAllocated scoring formula and weights
├── requested_to_capacity_ratio.go     # RequestedToCapacityRatio scoring strategy with broken-linear shape functions
├── requested_to_capacity_ratio_test.go# Unit tests for custom shape curves and interpolation
├── util_test.go                       # Test utilities, node/pod builder helpers
└── AGENTS.md                          # This agent documentation
```

---

## 3. Core Resource Calculation Model

### 3.1. Pod Resource Request Aggregation (`computePodResourceRequest`)

Pod resource requirements are computed using `k8s.io/component-helpers/resource.PodRequests`:
- **Regular Containers**: Summed across all containers (`Σ requests[i]`).
- **Init Containers**: Iteratively maximized (`max(IC_j)`), because init containers run sequentially.
- **App Containers vs Init Containers**: `MaxResource = max(Σ app_containers, max(init_container_k))`.
- **Pod Overhead (`spec.overhead`)**: Explicitly added to the effective request sum for all tracked resource dimensions.
- **PodLevelResources (KEP-2837)**: When enabled, pod-level resource limits/requests override container-level sums if specified.

```
Pod Resource Request = Overhead + max(
    Σ ContainerRequests,
    max_k(InitContainerRequests_k)
)
```

### 3.2. In-Place Vertical Scaling & Cache Discrepancies (`adjustDeltasToAccomodateCacheDiscrepancy`)

When `EnableInPlacePodVerticalScalingSchedulerPreemption` is active:
- An assigned pod undergoing resize already has its allocated or actual usage accounted for in `NodeInfo.GetRequested()`.
- To avoid double-counting the pod's existing footprint while checking if the additional scale-up fits:
  ```go
  deltaMilliCPU = max(0, podRequest.MilliCPU - cachedPodResource.MilliCPU)
  deltaMemory   = max(0, podRequest.Memory   - cachedPodResource.Memory)
  ```
- Floor at zero (`max(0, ...)`) prevents premature reduction of node usage before the Kubelet completes scale-down.

---

## 4. `NodeResourcesFit` (`Fit`) Plugin

`Fit` implements `PreFilterPlugin`, `FilterPlugin`, `PreScorePlugin`, `ScorePlugin`, `PlacementScorePlugin`, `EnqueueExtensions`, and `SignPlugin`.

```
                                  ┌───────────────────────────────┐
                                  │      PreFilter Execution      │
                                  │  computePodResourceRequest()  │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │    Write to CycleState:       │
                                  │    preFilterStateKey          │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │       Filter Execution        │
                                  │     fitsRequest(podRequest)   │
                                  └───────────────┬───────────────┘
                                                  │
                   ┌──────────────────────────────┴──────────────────────────────┐
                   │                                                             │
        [ All Resources Fit ]                                      [ Insufficient Resources ]
                   │                                                             │
                   ▼                                                             ▼
            ┌───────────────┐                                    ┌───────────────────────────────┐
            │  Return nil   │                                    │ If any unresolvable:          │
            │   (Success)   │                                    │   UnschedulableAndUnresolvable│
            └───────────────┘                                    │ Else:                         │
                                                                 │   Unschedulable (Preemptible) │
                                                                 └───────────────────────────────┘
```

### 4.1. `PreFilter`
- Computes aggregate pod requirements using `computePodResourceRequest(pod, opts)`.
- Stores `preFilterState` (`framework.Resource`) in `CycleState` under `"PreFilterNodeResourcesFit"`.

### 4.2. `Filter`
- Checks capacity and available headroom across all dimensions:
  1. **Pod Count**: `len(nodeInfo.GetPods()) + 1 <= nodeInfo.GetAllocatable().GetAllowedPodNumber()`.
  2. **CPU**: `deltaMilliCPU <= (Allocatable.MilliCPU - Requested.MilliCPU)`.
  3. **Memory**: `deltaMemory <= (Allocatable.Memory - Requested.Memory)`.
  4. **Ephemeral Storage**: `deltaEphemeralStorage <= (Allocatable.EphemeralStorage - Requested.EphemeralStorage)`.
  5. **Scalar / Extended Resources**: Checked against `Allocatable.ScalarResources` unless ignored via `IgnoredResources` / `IgnoredResourceGroups` or delegated to DRA.
- **Resolvability Determination (`InsufficientResource`)**:
  - Sets `Unresolvable = true` if `podRequest.Resource > nodeInfo.GetAllocatable().Resource` (the pod is larger than total node capacity; preemption cannot help).
  - If any insufficient resource is unresolvable, returns `fwk.UnschedulableAndUnresolvable`. Otherwise, returns `fwk.Unschedulable` to allow `DefaultPreemption` to seek victim pods.

### 4.3. `PreScore` & `Score`
- `PreScore` precalculates `podRequests []int64` and gathers DRA pre-score state (`allocatedState`, `resourceSlices`), caching them in `preScoreState`.
- `Score` executes the configured scoring strategy (`LeastAllocated`, `MostAllocated`, or `RequestedToCapacityRatio`).

### 4.4. `EventsToRegister` & Queueing Hints
`Fit` registers fine-grained queueing hints:
- **`AssignedPod (Delete)`**: Queues pods if a deleted assigned pod freed resources.
- **`Node (Add | UpdateNodeAllocatable)`**: Queues pods if a new node is added or allocatable capacity increased.
- **`DeviceClass (Add | Update)`**: Queues pods when a DRA device class matching the pod's extended resource request is created or updated.
- **`AssignedPod / TargetPod (UpdatePodScaleDown)`**: Queues pods if relevant resource dimensions were reduced.
- **`AssignedPod / TargetPod (UpdatePodScaleUp)`**: Requeues deferred-resize pods to trigger preemption if competing workloads consumed headroom.

---

## 5. Scoring Strategies & Resource Allocation Engine

### 5.1. `resourceAllocationScorer` Base Engine (`resource_allocation.go`)

The `resourceAllocationScorer` struct coordinates scoring across plugins:
```go
type resourceAllocationScorer struct {
    Name         string
    useRequested bool // true for BalancedAllocation; false (NonZeroRequested) for Fit
    scorer       func(requested, allocated, allocatable []int64) int64
    resources    []config.ResourceSpec
    draFeatures  structured.Features
    draManager   fwk.SharedDRAManager
    DRACaches    // celCache, deviceMatchCache, nodeMatchCache
}
```

- **DRA Caching**: Uses `sync.Map` caches (`deviceMatchCache`, `nodeMatchCache`) and `cel.Cache` to optimize repeated CEL selector and node selector evaluations across nodes.

### 5.2. `LeastAllocated` (`least_allocated.go`)
- **Philosophy**: Spreads workloads across nodes with the lowest resource utilization.
- **Formula**:
  $$\text{Score} = \frac{\sum_{i} \left( \frac{\text{Allocatable}_i - \text{Requested}_i}{\text{Allocatable}_i} \times \text{MaxNodeScore} \times \text{Weight}_i \right)}{\sum_i \text{Weight}_i}$$

### 5.3. `MostAllocated` (`most_allocated.go`)
- **Philosophy**: Consolidates workloads onto nodes with higher resource utilization (bin packing).
- **Formula**:
  $$\text{Score} = \frac{\sum_{i} \left( \frac{\text{Requested}_i}{\text{Allocatable}_i} \times \text{MaxNodeScore} \times \text{Weight}_i \right)}{\sum_i \text{Weight}_i}$$

### 5.4. `RequestedToCapacityRatio` (`requested_to_capacity_ratio.go`)
- **Philosophy**: Allows custom non-linear prioritization curves defined by `UtilizationShapePoint` slices.
- **Mechanics**: Constructs piecewise linear functions via `helper.BuildBrokenLinearFunction`. Maps utilization percentage to a custom score, scaled to `[0, MaxNodeScore]`.

### 5.5. `NodeResourcesBalancedAllocation` (`balanced_allocation.go`)
- **Philosophy**: Minimizes the difference in resource utilization fractions across dimensions (e.g. avoiding 90% CPU usage with 10% Memory usage).
- **Algorithm**:
  1. Computes resource utilization fractions: $f_i = \frac{\text{Requested}_i}{\text{Allocatable}_i}$.
  2. Computes standard deviation $\sigma$ across fractions.
     - For 2 resources (CPU & Memory): $\sigma = \frac{|f_1 - f_2|}{2}$.
     - For $>2$ resources: $\sigma = \sqrt{\frac{\sum (f_i - \mu)^2}{N}}$.
  3. Computes balance score with and without the pending pod, rewarding nodes whose balance improves:
     $$\text{FinalScore} = \frac{\text{MaxNodeScore}}{2} + \frac{\frac{\text{MaxNodeScore}}{2} + \text{Score}_{\text{with}} - \text{Score}_{\text{without}}}{2}$$
- **Best-Effort Pod Handling**: Returns `fwk.Skip` in `PreScore` for best-effort pods (zero requests) to prevent clustering pods onto the same node.

---

## 6. Testing Strategy & Test Coverage

The test suite in `pkg/scheduler/framework/plugins/noderesources` provides comprehensive coverage:

| Test File | Primary Focus | Key Scenarios Covered |
| :--- | :--- | :--- |
| `fit_test.go` | `NodeResourcesFit` filtering & lifecycle | Container aggregation, init container max calculations, pod overhead, scalar/extended resources, ignored resource groups, pod count limits, cache discrepancy adjustments, and queueing hint event triggers. |
| `balanced_allocation_test.go` | `BalancedAllocation` scoring | 2-resource and multi-resource standard deviation, extended resource balance, score differential before/after placement, and best-effort pod skip behavior. |
| `least_allocated_test.go` | `LeastAllocated` strategy | Inverse utilization calculations, weight summations, zero allocatable guards, and multiple resource weighting. |
| `most_allocated_test.go` | `MostAllocated` strategy | Utilization bin packing, request caps at capacity, weight distributions, and scalar resource packing. |
| `requested_to_capacity_ratio_test.go` | `RequestedToCapacityRatio` strategy | Custom shape curves, linear interpolation between utilization points, and multi-resource weighting. |
| `resource_allocation_test.go` | Shared scoring & DRA | DRA device class resolution, slice allocation matching, CEL selector compilation caching, and node selector evaluations. |
