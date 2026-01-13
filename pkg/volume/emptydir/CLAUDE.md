# Package: emptydir

## Purpose
Implements the EmptyDir volume plugin that provides ephemeral storage for pods, creating an empty directory that exists for the lifetime of the pod.

## Key Types/Structs
- `emptyDirPlugin` - VolumePlugin implementation for EmptyDir volumes
- `emptyDir` - Mounter/Unmounter implementation
- `storageMedium` - Enum for storage medium types (default, memory, hugepages)

## Key Functions
- `ProbeVolumePlugins()` - Returns the emptydir plugin
- `SetUpAt()` - Creates the empty directory with appropriate permissions
- `TearDownAt()` - Cleans up the empty directory
- `GetMetrics()` - Returns disk usage metrics for the volume

## Design Patterns
- Supports three storage mediums: default (disk), Memory (tmpfs), HugePages
- Memory medium uses tmpfs with optional size limit
- HugePages medium mounts hugetlbfs for huge page support
- Supports filesystem quota for size limiting (via fsquota)
- Foundation for other volume types (ConfigMap, Secret, DownwardAPI wrap EmptyDir)
