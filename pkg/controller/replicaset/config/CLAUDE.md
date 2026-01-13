# Package: config

Contains configuration types for the ReplicaSet controller.

## Key Types

- **ReplicaSetControllerConfiguration**: Configuration struct containing:
  - `ConcurrentRSSyncs`: Number of ReplicaSets that are allowed to sync concurrently.

## Design Patterns

- Part of the componentconfig pattern for kube-controller-manager.
- Simple configuration focused on concurrency control.
