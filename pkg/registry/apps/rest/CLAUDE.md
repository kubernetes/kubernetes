# Package: rest

## Purpose
Provides the REST storage provider for the entire "apps" API group, wiring up all apps resources (Deployments, StatefulSets, DaemonSets, ReplicaSets, ControllerRevisions) to the API server.

## Key Types

- **StorageProvider**: Implements the storage provider interface for the apps API group

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates and returns APIGroupInfo containing all storage handlers for enabled apps resources
- **v1Storage()**: Creates storage map for apps/v1 API version with:
  - deployments, deployments/status, deployments/scale
  - statefulsets, statefulsets/status, statefulsets/scale
  - daemonsets, daemonsets/status
  - replicasets, replicasets/status, replicasets/scale
  - controllerrevisions
- **GroupName()**: Returns "apps"

## Design Notes

- Central wiring point for all apps API group resources
- Conditionally enables resources based on apiResourceConfigSource
- Each resource checks if it's enabled before creating storage
- Subresources (status, scale) are registered as separate endpoints
- Note: When adding versions, also update aggregator.go priorities
