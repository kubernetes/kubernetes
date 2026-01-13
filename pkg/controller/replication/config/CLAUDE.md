# Package: config

Contains configuration types for the Replication controller.

## Key Types

- **ReplicationControllerConfiguration**: Configuration struct containing:
  - `ConcurrentRCSyncs`: Number of ReplicationControllers that are allowed to sync concurrently.

## Design Patterns

- Part of the componentconfig pattern for kube-controller-manager.
- Mirrors ReplicaSetControllerConfiguration for the legacy ReplicationController API.
