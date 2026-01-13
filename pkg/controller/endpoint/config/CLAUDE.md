# Package: config

Endpoint controller configuration types for the kube-controller-manager.

## Key Types

- `EndpointControllerConfiguration`: Contains configuration elements including:
  - `ConcurrentEndpointSyncs`: Number of concurrent endpoint sync operations
  - `EndpointUpdatesBatchPeriod`: Duration to batch endpoint updates (0 means immediate updates)

## Purpose

Defines the internal configuration structure used by the Endpoint controller. Batching endpoint updates reduces API server load when multiple pods change simultaneously.

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- Batching is useful in large clusters with frequent pod changes
