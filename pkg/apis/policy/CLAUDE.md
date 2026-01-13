# Package: policy

## Purpose
Defines the internal (unversioned) API types for the policy API group, primarily PodDisruptionBudget (PDB) for controlling voluntary disruptions to applications.

## Key Types

### PodDisruptionBudget
Defines the maximum disruption allowed for a collection of pods, ensuring application availability during voluntary disruptions (node drains, rolling updates, etc.).

### PodDisruptionBudgetSpec
- `MinAvailable`: Minimum pods that must remain available (absolute number or percentage)
- `MaxUnavailable`: Maximum pods that can be unavailable (mutually exclusive with MinAvailable)
- `Selector`: Label selector to identify managed pods
- `UnhealthyPodEvictionPolicy`: Controls eviction of unhealthy pods (IfHealthyBudget or AlwaysAllow)

### PodDisruptionBudgetStatus
- `ObservedGeneration`: Generation observed by controller
- `DisruptedPods`: Map of pods being evicted (name -> eviction time)
- `DisruptionsAllowed`: Number of currently allowed disruptions
- `CurrentHealthy`, `DesiredHealthy`, `ExpectedPods`: Health metrics
- `Conditions`: Standard condition array

### Eviction
Subresource of Pod for evicting pods subject to PDB constraints. Created by POSTing to `/pods/<name>/eviction`.

### UnhealthyPodEvictionPolicyType
- `IfHealthyBudget`: Unhealthy pods evicted only if budget allows
- `AlwaysAllow`: Unhealthy pods always evictable

## Key Constants/Variables
- `PDBV1beta1Label`: Label used for v1beta1 compatibility selector handling
- `NonV1beta1MatchAllSelector`, `NonV1beta1MatchNoneSelector`: Special selectors for v1/internal
- `V1beta1MatchAllSelector`, `V1beta1MatchNoneSelector`: Special selectors for v1beta1 compatibility

## Key Functions
- `StripPDBV1beta1Label`: Removes v1beta1 compatibility label from selectors
- `Kind`, `Resource`: Return qualified GroupKind/GroupResource
- `AddToScheme`: Registers types with a scheme

## Notes
- Empty selector behavior differs between v1beta1 (matches nothing) and v1 (matches all)
- Helper functions manage selector conversion between API versions
