# Package: config

Contains configuration types for the StatefulSet controller.

## Key Types

- **StatefulSetControllerConfiguration**: Configuration struct containing:
  - `ConcurrentStatefulSetSyncs`: Number of StatefulSets synced concurrently.

## Design Patterns

- Part of the componentconfig pattern for kube-controller-manager.
- Simple configuration focused on concurrency control.
