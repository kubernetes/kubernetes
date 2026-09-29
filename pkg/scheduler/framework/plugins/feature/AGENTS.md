# Scheduler Features Bridge (`pkg/scheduler/framework/plugins/feature`)

This guide provides an architectural overview, feature gate catalog, dependency decoupling model, and testing patterns for the `feature` package in `pkg/scheduler/framework/plugins/feature`.

---

## 1. High-Level Purpose & Scope

The `feature` package provides an architectural abstraction layer between Kubernetes feature gates (`k8s.io/kubernetes/pkg/features` and `k8s.io/component-base/featuregate`) and scheduler framework plugins. By encapsulating feature gate state inside a concrete `Features` struct, it breaks direct compile-time dependencies on the internal `pkg/features` package, avoids global mutable state during scheduling cycles, and allows unit tests to inject custom feature permutations trivially.

### Core Responsibilities:
1. **Feature Gate Decoupling**: Prevents plugin implementations under `pkg/scheduler/framework/plugins/` from importing `k8s.io/kubernetes/pkg/features`.
2. **Snapshot Initialization**: Captures the active boolean state of relevant feature gates at scheduler initialization via `NewSchedulerFeaturesFromGates`.
3. **Deterministic Testing**: Enables unit tests to construct explicit `feature.Features` structs directly without mutating global package variables or synchronizing across parallel test runners.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/feature/
├── feature.go       # Features struct definition and NewSchedulerFeaturesFromGates constructor
└── AGENTS.md        # This agent documentation
```

---

## 3. Supported Feature Gate Catalog

The `Features` struct encapsulates feature flags grouped by functional domain:

### 3.1. Dynamic Resource Allocation (DRA) Features
- **`EnableDRAExtendedResource`**: Enables DRA extended resources matching.
- **`EnableDRAPrioritizedList`**: Enables DRA prioritized resource allocation lists.
- **`EnableDRAAdminAccess`**: Supports administrative access claims in DRA.
- **`EnableDRAConsumableCapacity`**: Tracks consumable capacity for device classes.
- **`EnableDRADeviceCompatibilityGroups`**: Enables device compatibility group matching.
- **`EnableDRAFractionalCapacityRange`**: Supports fractional device capacity ranges.
- **`EnableDRADerivedAttributes`**: Enables derived device attribute evaluations.
- **`EnableDRADeviceTaints`**: Enforces taints on DRA devices.
- **`EnableDRAListTypeAttributes`**: Supports list-typed attributes in CEL filter expressions.
- **`EnableDRAOptionalNodeOperations`**: Supports optional node-level DRA operations.
- **`EnableDRASchedulerFilterTimeout`**: Enforces timeouts during CEL claim evaluations.
- **`EnableDRAResourceClaimDeviceStatus`**: Tracks device status in ResourceClaims.
- **`EnableDRADeviceBindingConditions`**: Enforces binding conditions on DRA devices.
- **`EnableDRAWorkloadResourceClaims`**: Enables workload-level DRA resource claims.
- **`EnableDRAPartitionableDevices`**: Supports partitioned device allocations.
- **`EnableDRANodeAllocatableResources`**: Integrates DRA node allocatable resource capacities.

### 3.2. Workload & Gang Scheduling Features
- **`EnableGenericWorkload`**: Enables generic workload and PodGroup scheduling semantics.
- **`EnableCompositePodGroup`**: Enables multi-tier composite hierarchical pod groups.
- **`EnablePodGroupPreemptionPolicy`**: Enables workload-level preemption policies.
- **`EnableTopologyAwareWorkloadScheduling`**: Enables multi-node topology-aware workload placement.

### 3.3. In-Place Pod Vertical Scaling (VPA) Features
- **`EnableInPlacePodVerticalScaling`**: Supports in-place container resource resizing.
- **`EnableInPlacePodLevelResourcesVerticalScaling`**: Supports in-place resizing at the pod-level resource scope.
- **`EnableInPlacePodVerticalScalingSchedulerPreemption`**: Allows preemption to reclaim resources for resizing pods.

### 3.4. Scheduling Lifecycle & Placement Features
- **`EnableAsyncPreemption`**: Dispatches pod and workload evictions asynchronously (`SchedulerAsyncPreemption`).
- **`EnablePodLevelResources`**: Evaluates aggregate pod-level resource requirements (`pod.Spec.Resources`).
- **`EnableNodeDeclaredFeatures`**: Enables matching against node-declared hardware/software capabilities.
- **`EnableTaintTolerationComparisonOperators`**: Supports operators (`Gt`, `Lt`, `In`, `NotIn`) in taint tolerations.
- **`EnableInterPodAffinityHostnameFastPath`**: Accelerates inter-pod affinity calculations for hostname topologies.
- **`EnableStorageCapacityScoring`**: Prioritizes CSI nodes by available storage capacity.
- **`EnableVolumeAttributesClass`** & **`EnableVolumeLimitScaling`**: Manages volume attributes and scalable volume limits.
- **`EnableNodeInclusionPolicyInPodTopologySpread`** & **`EnableMatchLabelKeysInPodTopologySpread`**: Configures advanced topology spread rules.

---

## 4. Architectural Patterns & Usage Contracts

```
┌──────────────────────────────────────┐
│ k8s.io/component-base/featuregate    │
│ (Command-line flags / API Gates)     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ feature.NewSchedulerFeaturesFromGates│
│ (Snapshot boolean flags)             │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ feature.Features Struct              │
│ (Passed to Plugin New() constructors)│
└──────────────────┬───────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
┌───────────────────────┐   ┌───────────────────────┐
│ DefaultPreemption     │   │ NodeResourcesFit      │
└───────────────────────┘   └───────────────────────┘
```

### 4.1. Factory Ingestion
Scheduler plugin factories (`pkg/scheduler/framework/plugins/registry.go`) pass `Features` to plugin constructors:
```go
func New(ctx context.Context, config runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error)
```

### 4.2. Testing Pattern (Zero Global State)
In unit tests, test authors construct `feature.Features` directly without touching `featuregate.FeatureGate`:
```go
pl, err := defaultpreemption.New(ctx, &args, handle, feature.Features{
    EnableAsyncPreemption: true,
    EnableGenericWorkload: true,
})
```
