# Package: helper

## Purpose
Utility functions for working with internal core API types, including resource classification, semantic equality, access mode handling, and annotation parsing.

## Key Functions

### Resource Classification
- **IsHugePageResourceName(name)**: Checks if resource has hugepage prefix.
- **IsHugePageResourceValueDivisible(name, quantity)**: Validates hugepage value alignment.
- **HugePageResourceName(pageSize)**: Creates canonical hugepage resource name.
- **HugePageSizeFromResourceName(name)**: Extracts page size from resource name.
- **IsExtendedResourceName(name)**: Checks for extended (non-native) resources.
- **IsNativeResource(name)**: Checks if resource is in kubernetes.io namespace.
- **IsStandardContainerResourceName(name)**: Validates container resource names.
- **IsStandardQuotaResourceName(name)**: Validates quota resource names.
- **IsIntegerResourceName(name)**: Checks if resource must be integer-valued.
- **IsOvercommitAllowed(name)**: Checks if overcommit is allowed for resource.

### Semantic Equality
- **Semantic**: DeepEqual implementation that handles Quantity, Time, Selector comparisons.

### Access Modes
- **GetAccessModesAsString(modes)**: Converts access modes to "RWO,ROX,RWX,RWOP" format.
- **GetAccessModesFromString(modes)**: Parses access mode string back to slice.
- **ContainsAccessMode(modes, mode)**: Checks if mode is in slice.

### Tolerations and Taints
- **GetTolerationsFromPodAnnotations(annotations)**: Parses tolerations from JSON annotation.
- **AddOrUpdateTolerationInPod(pod, toleration)**: Adds/updates toleration in pod.
- **GetTaintsFromNodeAnnotations(annotations)**: Parses taints from JSON annotation.

### Storage
- **GetPersistentVolumeClass(volume)**: Returns storage class name.
- **GetPersistentVolumeClaimClass(claim)**: Returns storage class for claim.
- **PersistentVolumeClaimHasClass(claim)**: Checks if claim has storage class.

### Other
- **IsServiceIPSet(service)**: Checks if ClusterIP is set.
- **IsStandardFinalizerName(str)**: Validates finalizer names.
- **GetDeletionCostFromPodAnnotations(annotations)**: Parses pod-deletion-cost.
