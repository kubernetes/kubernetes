# Package: helper

## Purpose
Utility functions for working with v1 (external) core API types, mirroring the internal helper package but for k8s.io/api/core/v1 types.

## Key Functions

### Resource Classification
- **IsExtendedResourceName(name)**: Checks for extended (non-native) resources.
- **IsNativeResource(name)**: Checks if in kubernetes.io namespace.
- **IsPrefixedNativeResource(name)**: Checks for explicit kubernetes.io prefix.
- **IsHugePageResourceName(name)**: Checks for hugepage prefix.
- **HugePageResourceName(pageSize)**: Creates hugepage resource name.
- **HugePageSizeFromResourceName(name)**: Extracts page size.
- **HugePageUnitSizeFromByteSize(size)**: Converts bytes to KB/MB/GB format.
- **IsHugePageMedium(medium)**: Checks if storage medium is HugePages.
- **HugePageSizeFromMedium(medium)**: Extracts page size from medium.
- **IsOvercommitAllowed(name)**: Checks if overcommit is allowed.
- **IsAttachableVolumeResourceName(name)**: Checks attachable volume prefix.

### Access Modes
- **GetAccessModesAsString(modes)**: Converts to "RWO,ROX,RWX,RWOP" string.
- **GetAccessModesFromString(modes)**: Parses string to access modes.
- **ContainsAccessMode(modes, mode)**: Checks if mode exists.

### Service Helpers
- **IsServiceIPSet(service)**: Checks if ClusterIP is set and not "None".

### Tolerations and Taints
- **AddOrUpdateTolerationInPodSpec(spec, toleration)**: Adds/updates toleration.
- **GetMatchingTolerations(taints, tolerations)**: Returns tolerations matching all taints.

### Topology
- **TopologySelectorRequirementsAsSelector(tsm)**: Converts topology selector to labels.Selector.
- **MatchTopologySelectorTerms(terms, labels)**: Checks if labels match topology terms.
- **NodeSelectorRequirementKeysExistInNodeSelectorTerms(reqs, terms)**: Key existence check.

### Quota
- **ScopedResourceSelectorRequirementsAsSelector(ssr)**: Converts scoped selector to labels.Selector.
