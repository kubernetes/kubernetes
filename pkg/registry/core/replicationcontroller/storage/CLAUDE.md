# Package: storage

## Purpose
Provides REST storage implementation for ReplicationController objects including status and scale subresources.

## Key Types

- **REST**: Main storage for RC operations.
- **StatusREST**: Storage for /status subresource updates.
- **ScaleREST**: Storage for /scale subresource (replica count updates).

## Key Functions

- **NewREST(optsGetter)**: Returns REST, StatusREST, and ScaleREST instances.
- **ScaleREST.Get()**: Returns current scale (replicas count).
- **ScaleREST.Update()**: Updates replica count via scale subresource.
- **ShortNames()**: Returns `["rc"]` for kubectl.
- **Categories()**: Returns `["all"]` for kubectl get all.

## Design Notes

- Implements scale subresource for horizontal scaling.
- ScaleREST converts between RC and autoscaling.Scale types.
- Uses replicasPathInReplicationController for JSON path patching.
- Returns deleted objects on delete operations.
- Short name "rc" for kubectl convenience.
