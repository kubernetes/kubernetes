# Package: config

Garbage collector controller configuration types for the kube-controller-manager.

## Key Types

- `GroupResource`: Represents a group/resource pair for ignored resources
- `GarbageCollectorControllerConfiguration`: Contains configuration elements including:
  - `EnableGarbageCollector`: Whether the GC is enabled
  - `ConcurrentGCSyncs`: Number of concurrent GC workers
  - `GCIgnoredResources`: List of resources the GC should ignore

## Purpose

Defines the internal configuration structure used by the garbage collector controller. The ignored resources list allows excluding certain resource types from garbage collection.

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- Must be synced with the kube-apiserver's garbage collection settings
