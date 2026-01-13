# Package: config

Contains configuration types for the ResourceClaim controller.

## Key Types

- **ResourceClaimControllerConfiguration**: Configuration struct containing:
  - `ConcurrentResourceClaimSyncs`: Number of claims that are allowed to sync concurrently (maps to worker count).

## Design Patterns

- Part of the componentconfig pattern for kube-controller-manager.
- Used by Dynamic Resource Allocation (DRA) feature.
