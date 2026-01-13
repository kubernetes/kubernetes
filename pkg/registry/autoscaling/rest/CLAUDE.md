# Package: rest

## Purpose
Provides the REST storage provider for the "autoscaling" API group, wiring up HorizontalPodAutoscaler resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements the storage provider interface for autoscaling API group
- **hpaStorageGetter**: Function type for lazy HPA storage initialization

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage handlers
- **v1Storage()**: Creates storage map for autoscaling/v1 with horizontalpodautoscalers and status subresource
- **v2Storage()**: Creates storage map for autoscaling/v2 with horizontalpodautoscalers and status subresource
- **GroupName()**: Returns "autoscaling"

## Design Notes

- Uses sync.Once pattern to share single HPA storage instance between v1 and v2 APIs
- Both autoscaling/v1 and autoscaling/v2 versions are supported
- Storage is lazily initialized only when the resource is enabled
- Note: When adding versions, also update aggregator.go priorities
