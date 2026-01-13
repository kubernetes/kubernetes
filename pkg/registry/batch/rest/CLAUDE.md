# Package: rest

## Purpose
Provides the REST storage provider for the "batch" API group, wiring up Job and CronJob resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements the storage provider interface for batch API group

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage handlers
- **v1Storage()**: Creates storage map for batch/v1 with:
  - jobs, jobs/status
  - cronjobs, cronjobs/status
- **GroupName()**: Returns "batch"

## Design Notes

- Only batch/v1 API version is registered (v1beta1 is deprecated/removed)
- Conditionally enables resources based on apiResourceConfigSource
- Each resource checks if it's enabled before creating storage
- Note: When adding versions, also update aggregator.go priorities
