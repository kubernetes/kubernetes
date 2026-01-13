# Package: storage

## Purpose
Provides REST storage implementation for ReplicaSet resources with status and scale subresources.

## Key Types

- **ReplicaSetStorage**: Container for all ReplicaSet-related REST endpoints
- **REST**: Main REST storage for ReplicaSet CRUD operations
- **StatusREST**: REST endpoint for /status subresource
- **ScaleREST**: REST endpoint for /scale subresource (autoscaling)
- **scaleUpdatedObjectInfo**: Transforms replicaset->scale->replicaset for scale updates

## Key Functions

- **NewStorage(optsGetter)**: Creates ReplicaSetStorage with all subresource handlers
- **NewREST(optsGetter)**: Creates REST and StatusREST instances
- **ReplicasPathMappings()**: Returns replicas field paths for different API versions
- **ShortNames()**: Returns ["rs"] for kubectl
- **Categories()**: Returns ["all"]
- **ScaleREST.Get()**: Converts ReplicaSet to Scale object
- **ScaleREST.Update()**: Updates replicas via Scale, converts back to ReplicaSet
- **scaleFromReplicaSet()**: Extracts Scale subresource from ReplicaSet

## Design Notes

- Note in code: changes should also be made to ReplicationController
- Scale subresource supports multiple API versions (apps/v1, apps/v1beta2, extensions/v1beta1)
- Uses managed fields handler for proper scale subresource field tracking
- Uses PredicateFunc for efficient watch filtering on labels/fields
- Implements ShortNamesProvider ("rs") and CategoriesProvider ("all")
