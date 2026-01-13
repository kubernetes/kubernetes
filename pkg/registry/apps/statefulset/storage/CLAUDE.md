# Package: storage

## Purpose
Provides REST storage implementation for StatefulSet resources with status and scale subresources.

## Key Types

- **StatefulSetStorage**: Container for all StatefulSet-related REST endpoints
- **REST**: Main REST storage for StatefulSet CRUD operations
- **StatusREST**: REST endpoint for /status subresource
- **ScaleREST**: REST endpoint for /scale subresource (autoscaling)
- **scaleUpdatedObjectInfo**: Transforms statefulset->scale->statefulset for scale updates

## Key Functions

- **NewStorage(optsGetter)**: Creates StatefulSetStorage with all subresource handlers
- **NewREST(optsGetter)**: Creates REST and StatusREST instances
- **ReplicasPathMappings()**: Returns replicas field paths for API versions (v1, v1beta1, v1beta2)
- **ShortNames()**: Returns ["sts"] for kubectl
- **Categories()**: Returns ["all"]
- **ScaleREST.Get()**: Converts StatefulSet to Scale object
- **ScaleREST.Update()**: Updates replicas via Scale, converts back to StatefulSet
- **scaleFromStatefulSet()**: Extracts Scale subresource from StatefulSet

## Design Notes

- Implements ShortNamesProvider ("sts") and CategoriesProvider ("all")
- Scale subresource supports multiple API versions (apps/v1, apps/v1beta1, apps/v1beta2)
- Uses managed fields handler for proper scale subresource field tracking
- Scale includes selector string in status for HPA compatibility
