# Package: resource

## Purpose
Provides helper functions for extracting and calculating resource values from v1.Pod objects, used for scheduling, downward API, and resource management.

## Key Functions

### Resource Request Calculation
- `GetResourceRequestQuantity(pod, resourceName) resource.Quantity` - Returns total requested quantity for a resource, including pod overhead when applicable
- `GetResourceRequest(pod, resourceName) int64` - Returns resource request as int64 (MilliValue for CPU, Value for others)

### Resource Value Extraction (Downward API)
- `ExtractResourceValueByContainerName(fs, pod, containerName) (string, error)` - Extracts resource value by container name
- `ExtractResourceValueByContainerNameAndNodeAllocatable(fs, pod, containerName, nodeAllocatable) (string, error)` - Same but merges node allocatable as default limits
- `ExtractContainerResourceValue(fs, container) (string, error)` - Extracts resource value from container spec

### Supported Resource Fields
- `limits.cpu`, `limits.memory`, `limits.ephemeral-storage`
- `requests.cpu`, `requests.memory`, `requests.ephemeral-storage`
- `limits.hugepages-<pageSize>`, `requests.hugepages-<pageSize>`

### Helper Functions
- `MergeContainerResourceLimits(container, allocatable)` - Sets container limits to node allocatable if not specified (for CPU, memory, ephemeral-storage)
- `IsHugePageResourceName(name) bool` - Checks if resource name is a hugepage resource
- `findContainerInPod(pod, containerName)` - Finds container by name (searches both regular and init containers)

## Design Notes
- Resource values are converted to string with ceiling division by the specified divisor
- Supports PodLevelResources feature gate for pod-level resource specifications
- Excludes hugepages from limit merging since they are never overcommitted
